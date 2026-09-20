"""Timeloop GEMM backend: energy and latency of IHA transformer layers on accelerator substrates.

A transformer layer with infinite-head attention decomposes into seven GEMMs, and a SwiGLU MLP adds an
eighth for its gate projection:

    0  QK_gen     n_embd -> n_qk_head_dim * (n_head + n_kv_group)
    1  V_gen      n_embd -> n_v_head_dim * n_kv_group
    2  QK_attn    n_qk_head_dim -> ctx, over n_head / n_kv_group rows, once per KV group
    3  PV_attn    ctx -> n_v_head_dim, over n_head / n_kv_group rows, once per KV group
    4  ATTN_proj  n_v_head_dim * n_head -> n_embd
    5  MLP_FC1    n_embd -> mlp_size
    6  MLP_FC2    mlp_size -> n_embd
    7  MLP_gate   n_embd -> mlp_size, SwiGLU only

evaluate_layer maps every GEMM with the Timeloop mapper, multiplies the two attention GEMMs by the
number of KV groups, and subtracts the DRAM traffic that operator fusion keeps on chip (see
compute_fusion_savings). A layer whose attention_variant is not "infinite" contributes only its
MLP GEMMs. The MLP variant comes from the layer or the individual's globals and defaults to "swiglu",
the rule llmforge.search.individual uses for parameter counts.

Prefill mode maps every GEMM at the prompt length. Decode mode maps the projections at one token and
the attention GEMMs at the KV-cache length, so a decode result is the cost of one generated token.

Timeloop reports cycles at a 1 GHz reference clock, so one cycle is one nanosecond. energy_uJ is in
microjoules.

Mapper results are cached on disk per substrate and GEMM shape under llmforge.paths.TIMELOOP_WORK,
so each shape is mapped once. timeloopfe is imported only when the mapper actually runs.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import sys
from typing import Any, Dict, List, Optional, Tuple

from llmforge import paths
from llmforge.hw.timeloop.stats import parse_dram_dataspace_stats, parse_timeloop_stats

# Suppress verbose timeloopfe / Specification / accelergy logs
for _lg in ("timeloopfe", "Specification", "accelergy"):
    logging.getLogger(_lg).setLevel(logging.WARNING)


class _TimeloopFilter(logging.Filter):
    """Block root-logger messages from timeloopfe internals."""
    _KEYWORDS = ("Loading yaml file", "Found top-key", "Found extra top-key",
                 "Specification:", "Processor ", "parsed-processed",
                 "Parsing extra attributes", "Calculated Specification",
                 "Calling timeloop", "Calling Timeloop",
                 "Dataspace2BranchProcessor", "Branch ", "keeps {", "bypasses {")

    def filter(self, record):
        msg = record.getMessage()
        return not any(kw in msg for kw in self._KEYWORDS)


logging.getLogger().addFilter(_TimeloopFilter())


def _ensure_accelergy_on_path() -> bool:
    """True when the accelergy command resolves, which timeloop-mapper needs to estimate energy.

    timeloop-mapper finds Accelergy with `which accelergy`. A job that runs an environment's Python
    without activating the environment lacks the environment's bin directory on PATH, so that directory
    is added when Accelergy is installed there.
    """
    if shutil.which("accelergy") is None:
        bin_dir = os.path.dirname(sys.executable)
        if os.path.isfile(os.path.join(bin_dir, "accelergy")):
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
    return shutil.which("accelergy") is not None


def timeloop_available() -> bool:
    """True when timeloopfe imports and the timeloop-mapper and accelergy commands resolve."""
    try:
        import timeloopfe.v4  # noqa: F401
    except ImportError:
        return False
    return shutil.which("timeloop-mapper") is not None and _ensure_accelergy_on_path()


def _import_timeloopfe():
    try:
        import timeloopfe.v4 as tl
    except ImportError as e:
        raise ImportError(
            "The Timeloop backend needs timeloopfe and the timeloop-mapper binary. "
            "See docs/hw_simulators.md for installation.") from e
    return tl


# ---------------------------------------------------------------------------
# Architecture configuration registry
# ---------------------------------------------------------------------------

class ArchConfig:
    """Holds all file paths and parameters for a hardware architecture.

    `runs_dir` is the on-disk mapper cache. When it is not given it resolves at call time to
    llmforge.paths.TIMELOOP_WORK / name, so an environment override or a test patch of that root
    applies without rebuilding the registry.
    """
    def __init__(self, name: str, arch_path: str, components_path: str,
                 constraints_path: str, variables_path: str,
                 mapper_path: str, runs_dir: Optional[str] = None,
                 dram_read_bw: float = 4, dram_write_bw: float = 4,
                 d_axis_spatial: Optional[int] = None):
        self.name = name
        self.arch_path = str(arch_path)
        self.components_path = str(components_path)
        self.constraints_path = str(constraints_path)
        self.variables_path = str(variables_path)
        self.mapper_path = str(mapper_path)
        self.runs_dir = runs_dir
        self.dram_read_bw = dram_read_bw
        self.dram_write_bw = dram_write_bw
        # When set, GEMM output dim D is rounded up to the next multiple
        # of `d_axis_spatial` if D > d_axis_spatial AND D % d_axis_spatial != 0.
        # Works around Timeloop's "residual ends not supported for Whoop
        # output" mapper failure on substrates with strict spatial-mesh
        # constraints (currently the four DXE variants, mesh = 8 DXT x 16 VAC = 128).
        # Other substrates (eyeriss / simba / gemmini / flat_edge) leave this
        # unset and the GEMM out_channel passes through unchanged.
        self.d_axis_spatial = d_axis_spatial

    @property
    def runs_dir(self) -> str:
        if self._runs_dir is not None:
            return self._runs_dir
        return str(paths.TIMELOOP_WORK / self.name)

    @runs_dir.setter
    def runs_dir(self, value: Optional[str]) -> None:
        self._runs_dir = None if value is None else str(value)


# Shared paths
_SPECS = paths.TIMELOOP_SPECS
_ARCH_DIR = _SPECS / "arch"
_COMPONENTS = str(_ARCH_DIR / "components" / "*.yaml")
_MAPPER = str(_SPECS / "mapper" / "mapper.yaml")
_PROBLEM_PATH = str(_SPECS / "prob" / "generic_GEMM.yaml")


def _arch_from_dir(name: str, subdir: str, mapper_path: str = _MAPPER, **kwargs) -> ArchConfig:
    d = _ARCH_DIR / subdir
    return ArchConfig(name=name, arch_path=d / "arch.yaml", components_path=_COMPONENTS,
                      constraints_path=d / "constraints.yaml",
                      variables_path=d / "variables.yaml", mapper_path=mapper_path, **kwargs)


ARCH_CONFIGS: Dict[str, ArchConfig] = {
    # Legacy Gemmini configuration with its original technology setting
    "gemmini": ArchConfig(
        name="gemmini",
        arch_path=_ARCH_DIR / "system_gemmini.yaml",
        components_path=_COMPONENTS,
        constraints_path=_SPECS / "constraints" / "constraints.yaml",
        variables_path=_SPECS / "mapper" / "variables.yaml",
        mapper_path=_MAPPER,
    ),
    # The remaining substrates share one technology setting for a like-for-like comparison
    "gemmini_16nm": _arch_from_dir("gemmini_16nm", "gemmini"),
    "eyeriss": _arch_from_dir("eyeriss", "eyeriss"),
    "simba": _arch_from_dir("simba", "simba"),
    # FLAT-Edge: fused attention dataflow (Kao et al., ASPLOS 2023)
    "flat_edge": _arch_from_dir("flat_edge", "flat_edge",
                                mapper_path=str(_ARCH_DIR / "flat_edge" / "mapper.yaml"),
                                dram_read_bw=25, dram_write_bw=25),
    "simba_edge": _arch_from_dir("simba_edge", "simba_edge"),
    # DXE with relaxed constraints for general IHA GEMM evaluation
    "dxe_relaxed": _arch_from_dir("dxe_relaxed", "dxe_relaxed",
                                  mapper_path=str(_ARCH_DIR / "dxe_relaxed" / "mapper.yaml"),
                                  dram_read_bw=4, dram_write_bw=4,
                                  d_axis_spatial=128),   # 8 DXT x 16 VAC, pads to avoid Whoop residual ends
    # DXE relaxed with 2x mac_lane width (4096 MACs)
    "dxe_relaxed_m32": _arch_from_dir("dxe_relaxed_m32", "dxe_relaxed_m32",
                                      mapper_path=str(_ARCH_DIR / "dxe_relaxed_m32" / "mapper.yaml"),
                                      dram_read_bw=4, dram_write_bw=4,
                                      d_axis_spatial=128),   # same DXT/VAC mesh as dxe_relaxed
    # DXE relaxed with 4x mac_lane width (8192 MACs), an edge-NPU target
    "dxe_relaxed_m64": _arch_from_dir("dxe_relaxed_m64", "dxe_relaxed_m64",
                                      mapper_path=str(_ARCH_DIR / "dxe_relaxed_m64" / "mapper.yaml"),
                                      dram_read_bw=4, dram_write_bw=4,
                                      d_axis_spatial=128),   # same DXT/VAC mesh as dxe_relaxed
    # rDXE decoder engine with its original, strict spatial constraints
    "dxe": _arch_from_dir("dxe", "DXE", dram_read_bw=2, dram_write_bw=2,
                          d_axis_spatial=128),   # strict variant, pads the same way as relaxed
}

DEFAULT_ARCH = "gemmini"


def get_arch_config(arch: str = DEFAULT_ARCH) -> ArchConfig:
    if arch not in ARCH_CONFIGS:
        raise ValueError(f"Unknown architecture '{arch}'. Available: {list(ARCH_CONFIGS.keys())}")
    return ARCH_CONFIGS[arch]


def _pad_D_for_arch(D: int, cfg: ArchConfig) -> Tuple[int, Optional[Tuple[int, int]]]:
    """Round GEMM output dim D up to next multiple of `cfg.d_axis_spatial`
    when needed.

    Some substrates (the four DXE variants) hard-code spatial factors on
    the D axis (DXT x VAC = 128 on dxe_relaxed*). When the workload's D
    is `> mesh AND not divisible by mesh`, Timeloop's mapper raises
    "residual ends not supported for Whoop output" and the entire arch
    eval fails. For those substrates we round D up to the next multiple
    of the mesh. This overestimates compute by at most
    `(mesh - D mod mesh) / D` (about 6-11% on common search-space shapes),
    which is much better than losing the data point entirely. For
    `D <= mesh` we pass through unchanged: the mapper handles small-D
    natively (low utilization, real cycles) and padding would inflate.
    Substrates without `d_axis_spatial` set (eyeriss / simba / gemmini /
    flat_edge) always pass through.

    Returns (D_to_use, padding_info or None). padding_info is
    `(D_orig, D_padded)` only when padding was actually applied, so
    callers can surface the inflation in audit fields.
    """
    mesh = getattr(cfg, "d_axis_spatial", None)
    if mesh and D > mesh and D % mesh != 0:
        D_pad = math.ceil(D / mesh) * mesh
        return D_pad, (D, D_pad)
    return D, None


def _prepare_gemm_spec(in_channel: int, out_channel: int, seq_length: int,
                       work_dir: Optional[str], cfg: ArchConfig
                       ) -> Tuple[str, str, Optional[Tuple[int, int]]]:
    """Prepare problem YAML and Timeloop spec files. Returns (out_dir, problem_file, padding_info).

    `work_dir=None` uses the substrate's own cache, `cfg.runs_dir`.
    `padding_info` is `(D_orig, D_padded)` when the substrate has a
    spatial-mesh constraint and the GEMM's output dim was rounded up,
    otherwise None. The on-disk gemm directory is named with the
    *padded* dim so the cache key stays consistent for the actual
    Timeloop run.
    """
    work_dir = cfg.runs_dir if work_dir is None else str(work_dir)
    os.makedirs(work_dir, exist_ok=True)
    out_channel_used, padding_info = _pad_D_for_arch(out_channel, cfg)
    out_dir = os.path.join(work_dir, f"gemm_{in_channel}i_{out_channel_used}o_{seq_length}l")
    os.makedirs(out_dir, exist_ok=True)
    problem_file = os.path.join(out_dir, "generic_GEMM.yaml")
    with open(_PROBLEM_PATH, 'r') as f:
        problem_data = f.read()
        problem_data = problem_data.replace("$IN_CHANNELS", str(in_channel))
        problem_data = problem_data.replace("$OUT_CHANNELS", str(out_channel_used))
        problem_data = problem_data.replace("$OUT_HEIGHT", str(seq_length))
    with open(problem_file, 'w') as f:
        f.write(problem_data)
    if padding_info is not None:
        # Sidecar so the padding is auditable from disk for any cache hit.
        with open(os.path.join(out_dir, "padding.json"), 'w') as f:
            json.dump({"D_orig": padding_info[0], "D_padded": padding_info[1],
                       "mesh": cfg.d_axis_spatial, "arch": cfg.name}, f)
    return out_dir, problem_file, padding_info


def _run_mapper(out_dir: str, problem_file: str, cfg: ArchConfig,
                log_path: Optional[str] = None) -> str:
    """Run Timeloop mapper if not already cached. Returns stats output file path."""
    output_file = os.path.join(out_dir, "timeloop-mapper.stats.txt")
    if not os.path.exists(output_file):
        tl = _import_timeloopfe()
        spec = tl.Specification.from_yaml_files(
            cfg.arch_path, cfg.components_path, cfg.mapper_path,
            problem_file, cfg.constraints_path, cfg.variables_path
        )
        spec.mapspace.template = 'uber'
        if spec.constraints['targets'] is None:
            spec.constraints['targets'] = tl.constraints.ConstraintsList()

        log_path = log_path or os.path.join(cfg.runs_dir, "timeloop.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        if not os.path.exists(log_path):
            with open(log_path, 'w') as f:
                f.write("")
        if not _ensure_accelergy_on_path():
            raise RuntimeError("timeloop-mapper needs the accelergy command. See docs/hw_simulators.md.")
        tl.call_mapper(spec, output_dir=out_dir, log_to=log_path)
        # Without its energy reference table a mapper run prices every action at zero energy.
        if not os.path.exists(os.path.join(out_dir, "timeloop-mapper.ERT.yaml")):
            raise RuntimeError(f"Accelergy wrote no energy reference table in {out_dir}. "
                               "See timeloop-mapper.accelergy.log there.")
    return output_file


def run_GEMM_evaluation(in_channel: int, out_channel: int, seq_length: int,
                        work_dir: Optional[str] = None, log_path: Optional[str] = None,
                        arch: str = DEFAULT_ARCH) -> dict:
    """Map one GEMM and return its summary stats. `work_dir=None` uses the substrate cache."""
    cfg = get_arch_config(arch)
    out_dir, problem_file, padding_info = _prepare_gemm_spec(
        in_channel, out_channel, seq_length, work_dir, cfg)
    output_file = _run_mapper(out_dir, problem_file, cfg, log_path)
    summary = parse_timeloop_stats(output_file)
    if padding_info is not None:
        summary['padded_D'] = list(padding_info)        # [orig, padded]
    return summary


def run_GEMM_evaluation_detailed(in_channel: int, out_channel: int, seq_length: int,
                                  work_dir: Optional[str] = None, log_path: Optional[str] = None,
                                  arch: str = DEFAULT_ARCH) -> Tuple[dict, dict]:
    """Run GEMM evaluation and return both summary stats and per-dataspace DRAM stats.

    Returns:
        (summary_stats, dram_stats) where dram_stats is keyed by dataspace name
        ('Weights', 'Inputs', 'Outputs') with per-dataspace access counts and energy.
    """
    cfg = get_arch_config(arch)
    out_dir, problem_file, padding_info = _prepare_gemm_spec(
        in_channel, out_channel, seq_length, work_dir, cfg)
    output_file = _run_mapper(out_dir, problem_file, cfg, log_path)
    summary = parse_timeloop_stats(output_file)
    dram = parse_dram_dataspace_stats(output_file)
    if padding_info is not None:
        summary['padded_D'] = list(padding_info)        # [orig, padded]
    return summary, dram


# ---------------------------------------------------------------------------
# Fusion savings calculation
# ---------------------------------------------------------------------------

def _dram_output_energy(dram_stats: dict) -> float:
    """Total DRAM energy (pJ) for the Outputs dataspace of a GEMM.

    This is the energy for writing partial sums + reading back for reduction.
    In a fused chain, the producer's output stays on-chip, so this is saved.
    """
    out = dram_stats.get('Outputs', {})
    return out.get('energy_pJ', 0) or 0


def _dram_input_energy(dram_stats: dict) -> float:
    """Total DRAM energy (pJ) for the Inputs dataspace of a GEMM.

    In a fused chain, the consumer reads its inputs from on-chip instead of DRAM.
    """
    inp = dram_stats.get('Inputs', {})
    return inp.get('energy_pJ', 0) or 0


def _dram_output_accesses(dram_stats: dict) -> float:
    """Total DRAM scalar accesses for Outputs (reads + updates)."""
    out = dram_stats.get('Outputs', {})
    reads = out.get('scalar_reads', 0) or 0
    updates = out.get('scalar_updates', 0) or 0
    return reads + updates


def _dram_input_accesses(dram_stats: dict) -> float:
    """Total DRAM scalar reads for Inputs."""
    inp = dram_stats.get('Inputs', {})
    return inp.get('scalar_reads', 0) or 0


def _saved_output_cycles(dram_stats: dict, dram_read_bw: float, dram_write_bw: float) -> float:
    """DRAM cycles of one GEMM's Outputs dataspace: partial sums written and read back."""
    out = dram_stats.get('Outputs', {})
    return ((out.get('scalar_updates', 0) or 0) / dram_write_bw
            + (out.get('scalar_reads', 0) or 0) / dram_read_bw)


def _saved_input_cycles(dram_stats: dict, dram_read_bw: float) -> float:
    """DRAM cycles of one GEMM's Inputs dataspace: inputs read."""
    inp = dram_stats.get('Inputs', {})
    return (inp.get('scalar_reads', 0) or 0) / dram_read_bw


def compute_fusion_savings(
    op_stats: List[Tuple[dict, dict]],
    fusion_edges: List[Tuple[int, int]],
    scale_factors: Optional[Dict[int, float]] = None,
    dram_read_bw: float = 4,
    dram_write_bw: float = 4,
) -> Tuple[float, float]:
    """Compute energy and cycle savings from fusing consecutive operations.

    A fused edge keeps a tensor on chip, which removes the DRAM traffic of the producer's Outputs
    dataspace and of the consumer's Inputs dataspace. A dataspace is removed once even when several
    edges touch it, and its savings scale with the instance count of its own operation only.

    Args:
        op_stats: List of (summary_stats, dram_stats) for one instance of each operation.
        fusion_edges: List of (producer_idx, consumer_idx) pairs defining
            which operations share intermediate data on-chip.
        scale_factors: Optional dict mapping op index to its number of instances,
            n_kv_groups for QK_attn/PV_attn. Operations not listed run once.
        dram_read_bw: DRAM read bandwidth in bytes/cycle.
        dram_write_bw: DRAM write bandwidth in bytes/cycle.

    Returns:
        (saved_energy_uJ, saved_cycles): Total savings from fusion.
    """
    if scale_factors is None:
        scale_factors = {}

    def instances(idx: int) -> float:
        return scale_factors.get(idx, 1.0)

    # Unfused energy of the layer, every instance of every operation counted once.
    total_unfused_energy_pJ = 0.0
    for idx, (summary, _) in enumerate(op_stats):
        e = summary.get('energy_uJ')
        if e is not None:
            total_unfused_energy_pJ += e * 1e6 * instances(idx)

    total_saved_energy_pJ = 0.0
    total_saved_cycles = 0.0
    for idx in sorted({prod_idx for prod_idx, _ in fusion_edges}):
        dram = op_stats[idx][1]
        total_saved_energy_pJ += _dram_output_energy(dram) * instances(idx)
        total_saved_cycles += _saved_output_cycles(dram, dram_read_bw, dram_write_bw) * instances(idx)
    for idx in sorted({cons_idx for _, cons_idx in fusion_edges}):
        dram = op_stats[idx][1]
        total_saved_energy_pJ += _dram_input_energy(dram) * instances(idx)
        total_saved_cycles += _saved_input_cycles(dram, dram_read_bw) * instances(idx)

    # Safety bound: fusion never removes more than 90 percent of the layer's unfused energy.
    total_saved_energy_pJ = min(total_saved_energy_pJ, total_unfused_energy_pJ * 0.9)
    return total_saved_energy_pJ / 1e6, total_saved_cycles


# ---------------------------------------------------------------------------
# Layer evaluation with fusion
# ---------------------------------------------------------------------------

def evaluate_layer(layer: dict, n_embd: int, seq_length: int, work_dir: Optional[str],
                   fused: bool = True, arch: str = DEFAULT_ARCH,
                   mode: str = "prefill", mlp_variant: str = "swiglu") -> dict:
    """Evaluate a single layer's hardware metrics.

    Args:
        mode: "prefill" -- all ops use seq_length (GEMM, batch of tokens)
              "decode"  -- projections use L=1 (GEMV, one token),
                           attention uses L=seq_length as context/KV-cache length
        mlp_variant: "swiglu" adds the gate projection, a GEMM with the shape of MLP_FC1. Any other
              value keeps the two-matrix MLP.
    """
    cfg = get_arch_config(arch)
    try:
        n_head = layer['n_head']
        n_kv_groups = layer['n_kv_group']
        n_qk_head_dim = layer['n_qk_head_dim']
        n_v_head_dim = layer['n_v_head_dim']
        n_cproj = layer['n_cproj']
        attn_variant = layer['attention_variant']
        mlp_size = layer['mlp_size']
    except KeyError as e:
        raise KeyError(f"Missing key in layer definition: {e}")

    # In decode mode: projections are GEMV (L=1), attention uses KV cache length
    proj_seq = 1 if mode == "decode" else seq_length
    attn_ctx = seq_length  # KV cache context length (used in both modes)

    if attn_variant == 'infinite':
        # Run all 7 GEMMs with detailed DRAM stats
        # Op 0: QK_gen  [embd -> qk*(h+kv), proj_seq]
        qk_gen = run_GEMM_evaluation_detailed(
            in_channel=n_embd, out_channel=n_qk_head_dim * (n_head + n_kv_groups),
            seq_length=proj_seq, work_dir=work_dir, arch=arch)
        # Op 1: V_gen   [embd -> v*kv, proj_seq]
        v_gen = run_GEMM_evaluation_detailed(
            in_channel=n_embd, out_channel=n_v_head_dim * n_kv_groups,
            seq_length=proj_seq, work_dir=work_dir, arch=arch)
        # Op 2: QK_attn [qk -> attn_ctx, h//kv]  (scaled by n_kv_groups)
        #   decode: q[1,qk] x K_cache[qk,ctx] -> scores[1,ctx]
        qk_attn = run_GEMM_evaluation_detailed(
            in_channel=n_qk_head_dim, out_channel=attn_ctx,
            seq_length=n_head // n_kv_groups, work_dir=work_dir, arch=arch)
        # Op 3: PV_attn [attn_ctx -> v, h//kv]  (scaled by n_kv_groups)
        #   decode: scores[1,ctx] x V_cache[ctx,v] -> attended[1,v]
        pv_attn = run_GEMM_evaluation_detailed(
            in_channel=attn_ctx, out_channel=n_v_head_dim,
            seq_length=n_head // n_kv_groups, work_dir=work_dir, arch=arch)
        # Op 4: ATTN_proj [v*h -> embd, proj_seq]
        attn_proj = run_GEMM_evaluation_detailed(
            in_channel=n_v_head_dim * n_head, out_channel=n_embd,
            seq_length=proj_seq, work_dir=work_dir, arch=arch)
        # Op 5: MLP_FC1 [embd -> mlp, proj_seq]
        mlp_fc1 = run_GEMM_evaluation_detailed(
            in_channel=n_embd, out_channel=mlp_size,
            seq_length=proj_seq, work_dir=work_dir, arch=arch)
        # Op 6: MLP_FC2 [mlp -> embd, proj_seq]
        mlp_fc2 = run_GEMM_evaluation_detailed(
            in_channel=mlp_size, out_channel=n_embd,
            seq_length=proj_seq, work_dir=work_dir, arch=arch)

        all_ops = [qk_gen, v_gen, qk_attn, pv_attn, attn_proj, mlp_fc1, mlp_fc2]
        if mlp_variant == "swiglu":
            # Op 7: MLP_gate [embd -> mlp, proj_seq], the gate projection of a SwiGLU MLP
            all_ops.append(run_GEMM_evaluation_detailed(
                in_channel=n_embd, out_channel=mlp_size,
                seq_length=proj_seq, work_dir=work_dir, arch=arch))

        if fused:
            # Define the producer->consumer fusion edges:
            #
            # Data flow graph:
            #   hidden -> QK_gen(0) -> Q,K --> QK_attn(2) -> scores --> PV_attn(3)
            #   hidden -> V_gen(1)  -> V   --------------------------> PV_attn(3)
            #   PV_attn(3) -> attended --> ATTN_proj(4)
            #   ATTN_proj(4) -> hidden' --> MLP_FC1(5)
            #   MLP_FC1(5) -> expanded --> MLP_FC2(6)
            #   ATTN_proj(4) -> hidden' --> MLP_gate(7) -> gate --> MLP_FC2(6), SwiGLU only
            #
            # Fusible edges (producer output = consumer input, stays on-chip):
            fusion_edges = [
                (0, 2),  # QK_gen outputs -> QK_attn inputs (Q,K projections)
                (1, 3),  # V_gen outputs -> PV_attn inputs (V values)
                (2, 3),  # QK_attn outputs -> PV_attn inputs (attention scores)
                (3, 4),  # PV_attn outputs -> ATTN_proj inputs (attended values)
                (4, 5),  # ATTN_proj outputs -> MLP_FC1 inputs (hidden states)
                (5, 6),  # MLP_FC1 outputs -> MLP_FC2 inputs (expanded activations)
            ]
            if mlp_variant == "swiglu":
                # The gate activation multiplies MLP_FC1's output elementwise on the way into MLP_FC2.
                fusion_edges += [(4, 7), (7, 6)]
            # Savings read the per-instance stats, so they run before the KV-group scaling below.
            # The two attention GEMMs run once per KV group.
            saved_energy_uJ, saved_cycles = compute_fusion_savings(
                all_ops, fusion_edges, {2: n_kv_groups, 3: n_kv_groups},
                dram_read_bw=cfg.dram_read_bw, dram_write_bw=cfg.dram_write_bw)

        # Apply n_kv_groups scaling to QK_attn (idx 2) and PV_attn (idx 3)
        for idx in [2, 3]:
            summary = all_ops[idx][0]
            for key in ['cycles', 'energy_uJ', 'total_ops', 'total_memory_accesses']:
                if summary[key] is not None:
                    summary[key] *= n_kv_groups

        # Extract summary stats for aggregation
        all_summaries = [op[0] for op in all_ops]

        if fused:
            layer_stats = aggregate_stats(all_summaries)

            # Subtract fusion savings
            if layer_stats['energy_uJ'] is not None:
                layer_stats['energy_uJ'] = max(0, layer_stats['energy_uJ'] - saved_energy_uJ)
            if layer_stats['cycles'] is not None:
                layer_stats['cycles'] = max(0, layer_stats['cycles'] - saved_cycles)
            # Store savings for debugging
            layer_stats['fusion_saved_energy_uJ'] = saved_energy_uJ
            layer_stats['fusion_saved_cycles'] = saved_cycles
        else:
            layer_stats = aggregate_stats(all_summaries)
            layer_stats['fusion_saved_energy_uJ'] = 0.0
            layer_stats['fusion_saved_cycles'] = 0.0

    else:
        # Identity or causal: only MLP
        mlp_fc1 = run_GEMM_evaluation_detailed(
            in_channel=n_embd, out_channel=mlp_size,
            seq_length=proj_seq, work_dir=work_dir, arch=arch)
        mlp_fc2 = run_GEMM_evaluation_detailed(
            in_channel=mlp_size, out_channel=n_embd,
            seq_length=proj_seq, work_dir=work_dir, arch=arch)
        mlp_ops = [mlp_fc1, mlp_fc2]
        fusion_edges = [(0, 1)]  # MLP_FC1 -> MLP_FC2
        if mlp_variant == "swiglu":
            mlp_ops.append(run_GEMM_evaluation_detailed(
                in_channel=n_embd, out_channel=mlp_size,
                seq_length=proj_seq, work_dir=work_dir, arch=arch))
            fusion_edges.append((2, 1))  # MLP_gate -> MLP_FC2

        all_summaries = [op[0] for op in mlp_ops]

        if fused:
            saved_energy_uJ, saved_cycles = compute_fusion_savings(
                mlp_ops, fusion_edges,
                dram_read_bw=cfg.dram_read_bw, dram_write_bw=cfg.dram_write_bw)
            layer_stats = aggregate_stats(all_summaries)
            if layer_stats['energy_uJ'] is not None:
                layer_stats['energy_uJ'] = max(0, layer_stats['energy_uJ'] - saved_energy_uJ)
            if layer_stats['cycles'] is not None:
                layer_stats['cycles'] = max(0, layer_stats['cycles'] - saved_cycles)
            layer_stats['fusion_saved_energy_uJ'] = saved_energy_uJ
            layer_stats['fusion_saved_cycles'] = saved_cycles
        else:
            layer_stats = aggregate_stats(all_summaries)
            layer_stats['fusion_saved_energy_uJ'] = 0.0
            layer_stats['fusion_saved_cycles'] = 0.0

    return layer_stats


def eval_individual(individual: Dict[str, Any], work_dir: Optional[str], fused: bool = True,
                    arch: str = DEFAULT_ARCH, mode: str = "prefill") -> dict:
    """Sum evaluate_layer over the active layers of an Individual dict.

    Uses `globals.block_size` as the sequence length: the prompt length in prefill mode and the
    KV-cache length in decode mode. Callers set it before the call (see HwTimeloop). A layer's
    mlp_variant overrides the one in the globals, which defaults to "swiglu".
    """
    global_spec = individual["globals"]
    layer_spec = individual["layers"]
    n_embd = global_spec["n_embd"]
    seq_length = global_spec["block_size"]
    layer_mask = global_spec.get("layer_mask", None)
    if layer_mask is None:
        raise ValueError("layer_mask is not defined in global_spec")

    hw_eval_list = []
    for i, layer in enumerate(layer_spec):
        if layer_mask[i] == 1:
            mlp_variant = layer.get("mlp_variant", global_spec.get("mlp_variant", "swiglu"))
            layer_stats = evaluate_layer(layer, n_embd, seq_length, work_dir, fused=fused, arch=arch, mode=mode,
                                         mlp_variant=mlp_variant)
            hw_eval_list.append(layer_stats)

    aggregated_stats = aggregate_stats(hw_eval_list)

    # average over sequence length
    aggregated_stats['cycles_per_token'] = aggregated_stats['cycles'] / seq_length if aggregated_stats['cycles'] is not None else None
    aggregated_stats['token_delay'] = aggregated_stats['cycles_per_token'] / 1e9  # assuming 1GHz clock
    aggregated_stats['energy_per_token_uJ'] = aggregated_stats['energy_uJ'] / seq_length if aggregated_stats['energy_uJ'] is not None else None
    aggregated_stats['edp_per_token'] = aggregated_stats['edp'] / seq_length if aggregated_stats['edp'] is not None else None
    return aggregated_stats


def evaluate_population(population: list, base_work_dir: Optional[str], fused: bool = True,
                        arch: str = DEFAULT_ARCH, mode: str = "prefill") -> list:
    n = len(population)
    results = []
    for i, individual in enumerate(population):
        individual_stats = eval_individual(individual, work_dir=base_work_dir, fused=fused, arch=arch, mode=mode)
        results.append(individual_stats)
        print(f"\r  HW eval [{i+1}/{n}]", end="", flush=True)
    print()
    return results


def aggregate_stats(stats_list: list) -> dict:
    aggregated_stats = {}
    for key in stats_list[0].keys():
        # Only sum scalar numeric stats. Non-numeric / structural fields
        # (e.g. `padded_D = [orig, padded]`) are collected separately
        # below so we don't accidentally concatenate lists.
        vals = [s[key] for s in stats_list if s.get(key) is not None]
        if vals and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
            aggregated_stats[key] = sum(vals)
        elif vals:
            aggregated_stats[key] = vals[0]      # carry first non-None as a representative

    # If any input op was padded, surface the per-op padding records.
    # Two cases:
    #   1) GEMM-level: each `s` carries a single `padded_D = [orig, padded]`.
    #   2) Layer-level: each `s` carries `padded_ops` (list of records) +
    #      `padded_op_count` from a previous aggregate_stats call.
    # Concatenate either form into a flat list at this level.
    flat = []
    for s in stats_list:
        if s.get('padded_ops'):
            flat.extend(s['padded_ops'])
        elif s.get('padded_D'):
            flat.append(s['padded_D'])
    if flat:
        aggregated_stats['padded_ops'] = flat
        aggregated_stats['padded_op_count'] = len(flat)

    # recalculate derived metrics
    if aggregated_stats['total_ops'] is not None and aggregated_stats['total_memory_accesses'] is not None and aggregated_stats['total_memory_accesses'] != 0:
        aggregated_stats['algorithmic_intensity_ops_per_access'] = aggregated_stats['total_ops'] / aggregated_stats['total_memory_accesses']
    else:
        aggregated_stats['algorithmic_intensity_ops_per_access'] = None
    aggregated_stats['algorithmic_intensity_ops_per_byte'] = aggregated_stats['algorithmic_intensity_ops_per_access']
    aggregated_stats['edp'] = aggregated_stats['energy_uJ'] * aggregated_stats['cycles'] / 10e6 if aggregated_stats['energy_uJ'] is not None and aggregated_stats['cycles'] is not None else None  # J*ns

    total_cycle = aggregated_stats['cycles']
    aggregated_stats['utilization_pct'] = 0
    aggregated_stats['gflops'] = 0
    for stats in stats_list:
        aggregated_stats['utilization_pct'] += (stats['utilization_pct'] * stats['cycles'] / total_cycle) if stats['utilization_pct'] is not None and stats['cycles'] is not None and total_cycle != 0 else 0
        aggregated_stats['gflops'] += (stats['gflops'] * stats['cycles'] / total_cycle) if stats['gflops'] is not None and stats['cycles'] is not None and total_cycle != 0 else 0

    return aggregated_stats
