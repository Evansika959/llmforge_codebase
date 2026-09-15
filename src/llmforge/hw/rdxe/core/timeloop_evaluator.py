"""Fast Timeloop-backed GEMM evaluator for the rDXE simulator, with disk and in-memory caching.

Key optimizations versus mapping through llmforge.hw.timeloop.gemm with default settings:
  1. Tight mapper settings per call (timeout=120, victory=30, num_threads=4). About 9x faster on
     uncached shapes while keeping the found schedule within a few percent of the slow-mapper
     optimum.
  2. Process-pool `prefetch(shapes)` warms the disk cache for all required shapes in parallel
     before ring simulation, so the ring phase hits the cache for every GEMM.
  3. Persistent on-disk cache per substrate variant and shape, reused across experiments. Each
     variant has its own cache directory, so results never mix across MAC widths.

Substrate specs and cache directories come from llmforge.hw.timeloop.gemm.ARCH_CONFIGS, so the
arch and constraint YAMLs remain the single source of truth.

Without timeloopfe, a shape that has no cached Timeloop result reports an infinite result, and
simulator/layer_eval_timeloop.py falls back to the analytical GEMM model for it. Ring results
record the source of every op (per_op_sources, ops_timeloop, ops_fallback).
"""

import concurrent.futures as _cf
import logging
import os
import re
import shutil
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from llmforge import paths
from llmforge.hw.timeloop import gemm as _gemm
from llmforge.hw.timeloop.stats import (
    parse_buffer_stats, parse_dram_dataspace_stats, parse_timeloop_stats,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fast mapper settings
# ---------------------------------------------------------------------------
# Per-call mapper overrides. Values chosen so that (a) the found mapping's
# energy is within a few % of the default slow mapper and (b) worst-case wall
# time on a DXE-size GEMM is under ~10 s with num_threads=4.
FAST_MAPPER   = dict(num_threads=4, timeout=120,  victory_condition=30)
LOOSE_MAPPER  = dict(num_threads=4, timeout=1200, victory_condition=500)

# Sanity: if the fast mapper's on-chip energy > this x analytical baseline, retry
# with LOOSE_MAPPER.  Small-output GEMVs are the hard cases: the DXE
# D=128 output constraint leaves very little tiling freedom for small N and
# the fast mapper's victory=30 stops early on a bad local optimum.
SANITY_RETRY_MULT = 2.0

# DXE spatial factors (from specs/arch/dxe_relaxed/constraints.yaml):
#   DXT D=8, VAC D=16 -> output-tile = 128
#   mac_lane E=16      -> reduction granularity = 16
# Any GEMM problem fed to Timeloop must have N divisible by 128 and K
# divisible by 16. Unpadded shapes (e.g., out_ch = 320, 1088, 1600) cause
# the mapper to fail. The DXE array runs padded tiles anyway, so
# padding is the physically honest way to evaluate these shapes.
OUT_TILE = 128
RED_TILE = 16

# Shapes above this MAC count are too large for the fast mapper: the
# mapspace blows up and the mapper regularly times out returning cy=0 (no
# valid schedule).  We skip Timeloop for them and let the caller fall back
# to the analytical prefill model instead.  Calibrated from the large-M
# prefill GEMMs that kept failing: (1280,5120,512), (1536,6144,512) etc
# (all > 3G MACs).
MAX_TIMELOOP_MACS = 2_500_000_000

# File sentinel that marks a shape as "known-unmappable", avoids re-running
# the 120s fast mapper every invocation when we already know it failed.
MAPPER_NEGATIVE_SENTINEL = 'timeloop-mapper.UNMAPPABLE'

# File sentinel that marks a loose mapping as final, so a schedule above the sanity baseline is
# served from the cache instead of re-mapped on every call.
MAPPER_LOOSE_SENTINEL = 'timeloop-mapper.LOOSE'

# Reference DXE core count baked into the dxe_relaxed arch YAML
# (8 DXT x 16 VAC = 128 cores). Rescaling cycles for n_cores != 128 is still
# a linear approximation (the mesh sizes in the arch YAML are fixed at 8/16),
# but the per-core MAC width (n_mac_per_vac) is now a proper Timeloop knob
# via the `_arch_variant_for_mac_width` machinery below.
CORES_REF = 128

_TIMELOOP_LOGGER_KEYS = ('timeloop', 'Dataspace', 'Specification', 'accelergy', 'Constraints',
                         'ReferenceLoader', 'Probspace', 'Mapspace', 'Processor')
_WARNED_NO_TIMELOOP = False


def _silence_timeloop_loggers(level: int = logging.WARNING) -> None:
    """Quiet Timeloop's lazily created loggers. They only exist after timeloopfe submodules
    import, so this sweeps the logger registry at call time."""
    for name in list(logging.root.manager.loggerDict.keys()):
        if any(s in name for s in _TIMELOOP_LOGGER_KEYS):
            logging.getLogger(name).setLevel(level)


def _warn_no_timeloop() -> None:
    global _WARNED_NO_TIMELOOP
    if not _WARNED_NO_TIMELOOP:
        log.warning("timeloopfe or timeloop-mapper not found. GEMM shapes without a cached "
                    "Timeloop result use the analytical fallback.")
        _WARNED_NO_TIMELOOP = True


# ---------------------------------------------------------------------------
# Architecture variants: n_mac_per_vac flows into Timeloop's mac_lane.meshX
# ---------------------------------------------------------------------------
_ARCH_VARIANT_LOCK = threading.Lock()


def _arch_variant_for_mac_width(n_mac_per_vac: int, base_arch: str = 'dxe_relaxed') -> str:
    """Return a Timeloop arch name whose `mac_lane.meshX` equals n_mac_per_vac.

    The reference width (16) on dxe_relaxed uses the base arch as is. Widths that ship a spec
    directory (dxe_relaxed_m32, dxe_relaxed_m64) use that registry entry. Any other width clones
    the base spec directory into <TIMELOOP_WORK>/arch_variants/<base>_m<N>/, patches the
    `mac_lane` container's meshX, and registers the clone in gemm.ARCH_CONFIGS. Every variant
    gets its own cache directory, so on-disk Timeloop results never collide across variants.

    The mac_per_vac knob is DXE-specific (it patches `mac_lane.meshX` in the DXE arch YAML). For
    non-DXE archs (eyeriss, flat_edge, etc.) we always return the base arch unchanged.
    """
    if not base_arch.startswith('dxe'):
        return base_arch

    # Default matches the YAML already, no variant needed.
    if n_mac_per_vac == 16 and base_arch == 'dxe_relaxed':
        return base_arch

    variant_name = f"{base_arch}_m{n_mac_per_vac}"
    with _ARCH_VARIANT_LOCK:
        if variant_name in _gemm.ARCH_CONFIGS:
            return variant_name
        base_cfg = _gemm.get_arch_config(base_arch)
        base_dir = os.path.dirname(base_cfg.arch_path)
        variant_dir = os.path.join(str(paths.TIMELOOP_WORK), 'arch_variants', variant_name)
        if not os.path.exists(os.path.join(variant_dir, 'arch.yaml')):
            _write_mac_width_variant(base_dir, variant_dir, n_mac_per_vac)

        def _local(fname: str, fallback: str) -> str:
            p = os.path.join(variant_dir, fname)
            return p if os.path.exists(p) else fallback

        _gemm.ARCH_CONFIGS[variant_name] = _gemm.ArchConfig(
            name=variant_name,
            arch_path=os.path.join(variant_dir, 'arch.yaml'),
            components_path=base_cfg.components_path,
            constraints_path=_local('constraints.yaml', base_cfg.constraints_path),
            variables_path=_local('variables.yaml', base_cfg.variables_path),
            mapper_path=_local('mapper.yaml', base_cfg.mapper_path),
            dram_read_bw=base_cfg.dram_read_bw,
            dram_write_bw=base_cfg.dram_write_bw,
            d_axis_spatial=base_cfg.d_axis_spatial,
        )
    return variant_name


def _write_mac_width_variant(base_dir: str, variant_dir: str, n_mac_per_vac: int) -> None:
    """Clone a DXE spec directory and set its mac_lane meshX. Safe when processes race."""
    os.makedirs(os.path.dirname(variant_dir), exist_ok=True)
    tmp = f"{variant_dir}.tmp{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(base_dir, tmp)
    arch_yaml = os.path.join(tmp, 'arch.yaml')
    with open(arch_yaml) as f:
        src = f.read()
    # Scoped replace: find the `mac_lane` node and set its meshX.
    new_src, n_subs = re.subn(
        r"(name:\s*mac_lane\s*\n\s*spatial:\s*\{meshX:\s*)(\d+)(\s*\})",
        rf"\g<1>{n_mac_per_vac}\g<3>",
        src, count=1,
    )
    if n_subs == 0:
        shutil.rmtree(tmp, ignore_errors=True)
        raise RuntimeError(f"Failed to patch mac_lane meshX in {arch_yaml}")
    with open(arch_yaml, 'w') as f:
        f.write(new_src)
    try:
        os.rename(tmp, variant_dir)
    except OSError:
        shutil.rmtree(tmp, ignore_errors=True)   # another process created the variant first


def pad_for_dxe(in_ch: int, out_ch: int, seq_len: int) -> Tuple[int, int, int]:
    """Round K up to next multiple of RED_TILE, N up to next multiple of OUT_TILE.
    Returns (K_pad, N_pad, M).  Energy/cycles reported for the padded shape
    equal the real HW cost, since the DXE array fires all 128 output lanes
    and 16 reduction lanes every cycle regardless of usefulness."""
    K_pad = ((in_ch  + RED_TILE - 1) // RED_TILE) * RED_TILE
    N_pad = ((out_ch + OUT_TILE - 1) // OUT_TILE) * OUT_TILE
    return K_pad, N_pad, seq_len


@dataclass
class GemmResult:
    M: int         # seq_length
    K: int         # in_channel
    N: int         # out_channel
    energy_pJ: float        # raw Timeloop GEMM energy, DRAM included
    cycles: float           # raw Timeloop cycles on the mapped chip, DRAM included
    dram_stats: dict        # per-dataspace DRAM accesses + energy
    levels: dict = field(default_factory=dict)  # compact_levels of the mapping


def compact_levels(levels: dict) -> Dict[str, dict]:
    """Cycles, dynamic and leakage energy, and per-dataspace accesses of every level of a mapping.

    `levels` is the output of llmforge.hw.timeloop.stats.parse_buffer_stats.
    """
    out = {}
    for name, lv in levels.items():
        out[name] = dict(
            cycles=lv["_stats"].get("cycles") or 0.0,
            dynamic_pJ=lv["_summary"].get("dynamic_energy_pJ") or 0.0,
            leakage_pJ=lv["_specs"].get("leakage_energy_pJ") or 0.0,
            dataspaces={ds: dict(total_reads=v.get("total_reads", 0.0),
                                 total_updates=v.get("total_updates", 0.0),
                                 energy_pJ=v.get("energy_pJ") or 0.0)
                        for ds, v in lv.items() if not ds.startswith("_")},
        )
    return out


def onchip_dynamic_pJ(levels: Dict[str, dict]) -> float:
    """Dynamic energy of every level of a mapping except DRAM."""
    return sum(lv["dynamic_pJ"] for name, lv in levels.items() if name != "DRAM")


def _analytical_baseline_pJ(in_ch: int, out_ch: int, seq_len: int) -> float:
    """Rough on-chip energy baseline (pJ) used only for the sanity-retry decision.

    Precision is not needed, only enough to reject mappings that are ~100x too expensive. The mapper
    reads the weight matrix once per row, so the weight term scales with seq_len.
    """
    # ~0.07 pJ/MAC + ~0.243 pJ/B WMEM reads (weight bytes) + ~0.1 pJ/B act.
    macs = seq_len * in_ch * out_ch
    weight_bytes = seq_len * in_ch * out_ch
    act_bytes    = seq_len * (in_ch + out_ch)
    return macs * 0.07 + weight_bytes * 0.243 + act_bytes * 0.1


def _read_stats(stats_file: str) -> Tuple[dict, dict, dict]:
    """Summary metrics, DRAM dataspaces and compact levels of one mapper statistics file."""
    return (parse_timeloop_stats(stats_file), parse_dram_dataspace_stats(stats_file),
            compact_levels(parse_buffer_stats(stats_file)))


def _run_mapper_with(settings: dict, in_ch: int, out_ch: int, seq_len: int,
                     arch: str) -> Tuple[dict, dict, dict]:
    """Invoke the mapper with the given settings. Returns (summary, dram, levels)."""
    tl = _gemm._import_timeloopfe()
    cfg = _gemm.get_arch_config(arch)
    out_dir, problem_file, _ = _gemm._prepare_gemm_spec(in_ch, out_ch, seq_len, None, cfg)
    stats_file = os.path.join(out_dir, 'timeloop-mapper.stats.txt')

    # Silence the Timeloop INFO chatter emitted during spec parsing.
    _silence_timeloop_loggers(logging.WARNING)

    spec = tl.Specification.from_yaml_files(
        cfg.arch_path, cfg.components_path, cfg.mapper_path,
        problem_file, cfg.constraints_path, cfg.variables_path)
    spec.mapspace.template = 'uber'
    for k, v in settings.items():
        spec.mapper[k] = v
    # Force a fresh run by removing any previous stats
    if os.path.exists(stats_file):
        os.remove(stats_file)
    tl.call_mapper(spec, output_dir=out_dir, log_to=os.path.join(cfg.runs_dir, 'timeloop.log'))
    return _read_stats(stats_file)


def _fast_run_mapper(in_ch: int, out_ch: int, seq_len: int,
                     arch: str) -> Tuple[dict, dict, dict]:
    """Two-pass mapping. The fast mapper runs first, and the loose mapper runs when the fast
    schedule's on-chip energy exceeds SANITY_RETRY_MULT times the analytical baseline. A loose
    mapping is final: it is marked on disk and served from the cache from then on, so repeated calls
    return the same schedule without re-mapping. Returns (summary, dram, levels).
    """
    # Auto-register `<base>_m<N>` variants in the worker process
    # (a spawned worker starts without the parent's runtime registration).
    if arch not in _gemm.ARCH_CONFIGS:
        m = re.match(r'^(.*)_m(\d+)$', arch)
        if m:
            _arch_variant_for_mac_width(int(m.group(2)), base_arch=m.group(1))
    cfg = _gemm.get_arch_config(arch)
    out_dir, _, _ = _gemm._prepare_gemm_spec(in_ch, out_ch, seq_len, None, cfg)
    stats_file    = os.path.join(out_dir, 'timeloop-mapper.stats.txt')
    negative_file = os.path.join(out_dir, MAPPER_NEGATIVE_SENTINEL)
    loose_file    = os.path.join(out_dir, MAPPER_LOOSE_SENTINEL)

    # Fast circuit-break: previously marked unmappable -> fail fast.
    if os.path.exists(negative_file):
        raise RuntimeError(f"shape ({in_ch},{out_ch},{seq_len}) marked UNMAPPABLE")

    # Don't even attempt Timeloop for ridiculously large shapes: the
    # mapper reliably times out and returns cy=0. Persist a sentinel
    # so the next run also short-circuits instead of re-trying.
    macs = max(1, in_ch) * max(1, out_ch) * max(1, seq_len)
    if macs > MAX_TIMELOOP_MACS:
        with open(negative_file, 'w') as f:
            f.write(f"MACs={macs} > MAX_TIMELOOP_MACS={MAX_TIMELOOP_MACS}\n")
        raise RuntimeError(
            f"shape ({in_ch},{out_ch},{seq_len}) exceeds "
            f"MAX_TIMELOOP_MACS; falling back to analytical")

    baseline_pJ = _analytical_baseline_pJ(in_ch, out_ch, seq_len)

    def usable(summary: dict, levels: dict, final: bool) -> bool:
        # A valid schedule whose on-chip energy is within sanity, unless the mapping is final. The
        # check skips DRAM, whose traffic dominates the raw energy of every DXE mapping.
        e_pJ = onchip_dynamic_pJ(levels)
        return (summary.get('cycles') or 0) > 0 and e_pJ > 0 and (
            final or e_pJ <= SANITY_RETRY_MULT * baseline_pJ)

    if os.path.exists(stats_file):
        summary, dram, levels = _read_stats(stats_file)
        if usable(summary, levels, final=os.path.exists(loose_file)):
            return summary, dram, levels

    # Pass 1: fast mapper
    summary, dram, levels = _run_mapper_with(FAST_MAPPER, in_ch, out_ch, seq_len, arch)
    if usable(summary, levels, final=False):
        return summary, dram, levels

    # Pass 2: loose mapper (only for shapes that need it). Its schedule is final.
    summary, dram, levels = _run_mapper_with(LOOSE_MAPPER, in_ch, out_ch, seq_len, arch)
    if not usable(summary, levels, final=True):
        # Mark the shape unmappable so subsequent calls short-circuit instead of re-running mapper.
        with open(negative_file, 'w') as f:
            f.write(f"fast+loose mapper returned cy={summary.get('cycles')}, "
                    f"on-chip e_pJ={onchip_dynamic_pJ(levels)}\n")
        raise RuntimeError(f"shape ({in_ch},{out_ch},{seq_len}) mapper failed; marked UNMAPPABLE")
    with open(loose_file, 'w') as f:
        f.write("loose mapping, final\n")
    return summary, dram, levels


def _worker_eval(args):
    """Process-pool worker. Must be top-level for pickling."""
    in_ch, out_ch, seq_len, arch = args
    try:
        summary, dram, levels = _fast_run_mapper(in_ch, out_ch, seq_len, arch)
        return (in_ch, out_ch, seq_len, True,
                (summary.get('energy_uJ') or 0) * 1e6,
                summary.get('cycles') or 0,
                dram, levels)
    except Exception as e:
        return (in_ch, out_ch, seq_len, False, str(e), 0, {}, {})


class TimeloopEvaluator:
    """Thread-safe memoized evaluator around Timeloop's mapper.

    The first request for a given shape triggers the (fast) mapper; repeat
    calls are served from memory.  Disk cache survives across processes.
    """

    def __init__(self, arch: str = 'dxe_relaxed', verbose: bool = False,
                 n_mac_per_vac: int = 16):
        # n_mac_per_vac drives the arch variant whose mac_lane.meshX is
        # patched to match. Timeloop is then re-mapped for that E-dim, so
        # cycle/energy results reflect the real dataflow change instead of
        # assuming perfect linear speedup.
        self.n_mac_per_vac = n_mac_per_vac
        self.arch     = _arch_variant_for_mac_width(n_mac_per_vac, arch)
        self.verbose  = verbose
        self._mem: Dict[Tuple[int, int, int], GemmResult] = {}
        self._lock    = threading.Lock()
        self.fresh_runs = 0
        self.cache_hits = 0
        self.mapper_available = _gemm.timeloop_available()
        if not self.mapper_available:
            _warn_no_timeloop()
        # Silence Timeloop / accelergy internal INFO chatter at spec-load time.
        _silence_timeloop_loggers(logging.ERROR)
        for _lg in ('timeloopfe', 'Specification', 'accelergy'):
            logging.getLogger(_lg).setLevel(logging.ERROR)

    def evaluate(self, in_ch: int, out_ch: int, seq_len: int) -> GemmResult:
        # Pad to DXE spatial tile boundaries: the chip runs the padded
        # nest; Timeloop would otherwise fail on non-multiples-of-128 outputs.
        K_pad, N_pad, M = pad_for_dxe(in_ch, out_ch, seq_len)
        key = (K_pad, N_pad, M)
        with self._lock:
            if key in self._mem:
                self.cache_hits += 1
                return self._mem[key]
        try:
            summary, dram, levels = _fast_run_mapper(K_pad, N_pad, M, self.arch)
            res = GemmResult(
                M=M, K=K_pad, N=N_pad,
                energy_pJ=(summary.get('energy_uJ') or 0) * 1e6,
                cycles=summary.get('cycles') or 0,
                dram_stats=dram,
                levels=levels,
            )
        except Exception:
            # Unmappable / timed-out shapes: memoize an inf result so the
            # caller's try/except fallback hits instantly on subsequent calls.
            res = GemmResult(M=M, K=K_pad, N=N_pad,
                             energy_pJ=float('inf'),
                             cycles=float('inf'),
                             dram_stats={})
        with self._lock:
            self._mem[key] = res
            self.fresh_runs += 1
        return res

    # ---------- parallel prefetch ----------
    def prefetch(self, shapes: List[Tuple[int, int, int]],
                 n_workers: int = 8) -> Dict[Tuple[int, int, int], GemmResult]:
        """Populate cache for all unique shapes using a process pool."""
        unique = sorted({s for s in shapes})
        todo   = [s for s in unique if s not in self._mem]
        if not todo:
            if self.verbose:
                print(f"  [Timeloop] all {len(unique)} shapes already in memory")
            return self._mem

        if self.verbose:
            print(f"  [Timeloop] prefetch {len(todo)} unique shapes "
                  f"with {n_workers} workers (fast mapper: "
                  f"t={FAST_MAPPER['timeout']}, v={FAST_MAPPER['victory_condition']}, "
                  f"threads={FAST_MAPPER['num_threads']})")

        t0 = time.time()
        done = 0
        # Pad each shape to DXE tile boundaries before dispatching to workers.
        # Also dedup again AFTER padding, since several unpadded shapes may
        # collapse into the same padded shape.
        padded = {}  # padded -> original for reporting
        for s in todo:
            p = pad_for_dxe(*s)
            padded.setdefault(p, []).append(s)
        padded_todo = list(padded.keys())
        args = [(p[0], p[1], p[2], self.arch) for p in padded_todo]
        n_padded = len(padded_todo)
        with _cf.ProcessPoolExecutor(max_workers=n_workers) as ex:
            for (K_pad, N_pad, M, ok, e_or_err, cy, dram, levels) in ex.map(_worker_eval, args):
                done += 1
                pkey = (K_pad, N_pad, M)
                orig_list = padded.get(pkey, [(K_pad, N_pad, M)])
                if ok:
                    gr = GemmResult(M=M, K=K_pad, N=N_pad,
                                    energy_pJ=e_or_err, cycles=cy,
                                    dram_stats=dram, levels=levels)
                    with self._lock:
                        self._mem[pkey] = gr
                        self.fresh_runs += 1
                    if self.verbose:
                        orig_tag = f"({orig_list[0][0]},{orig_list[0][1]},{orig_list[0][2]})"
                        pad_tag  = f"->({K_pad},{N_pad},{M})" if orig_list[0] != pkey else ""
                        print(f"    [{done:>3d}/{n_padded}] {orig_tag:>18s}{pad_tag:<18s}"
                              f"  E={e_or_err/1e6:>7.3f}uJ  cy={int(cy):>7d}")
                else:
                    with self._lock:
                        self._mem[pkey] = GemmResult(
                            M=M, K=K_pad, N=N_pad,
                            energy_pJ=float('inf'), cycles=float('inf'),
                            dram_stats={})
                    if self.verbose:
                        print(f"    [{done:>3d}/{n_padded}] "
                              f"({K_pad},{N_pad},{M})  FAILED: {str(e_or_err)[:300]}")
        if self.verbose:
            print(f"  [Timeloop] prefetch finished in {time.time()-t0:.1f}s")
        return self._mem

    def summary(self) -> str:
        return (f"Timeloop cache: fresh={self.fresh_runs} "
                f"memory-hits={self.cache_hits} "
                f"total-shapes={len(self._mem)}")


# ---------------------------------------------------------------------------
# GEMM-shape enumerator for IHA layers
# ---------------------------------------------------------------------------
def enumerate_gemm_shapes_decode(layer_spec: dict, n_embd: int,
                                 ctx: int) -> List[Tuple[str, int, int, int]]:
    """List the 7 GEMM shapes (name, in_ch, out_ch, seq_len) of one IHA layer
    in decode mode.  'infinite' and MHA variants covered."""
    nh  = layer_spec['n_head']
    nkv = layer_spec['n_kv_group']
    qk  = layer_spec['n_qk_head_dim']
    vd  = layer_spec['n_v_head_dim']
    mlp = layer_spec['mlp_size']
    variant = layer_spec.get('attention_variant', 'infinite')

    # DXE spatial constraint: context dimension >= 32 and in mapper-friendly grid
    def _quantize_ctx(c):
        c = max(c, 32)
        if c <= 128:
            p = 32
            while p < c: p *= 2
            return p
        return ((c + 127) // 128) * 128
    ctx_q = _quantize_ctx(ctx)

    if variant != 'infinite':
        return [
            ('MLP_FC1',  n_embd, mlp,    1),
            ('MLP_FC2',  mlp,    n_embd, 1),
        ]

    return [
        ('QK_gen',    n_embd,       qk * (nh + nkv), 1),
        ('V_gen',     n_embd,       vd * nkv,         1),
        ('QK_attn',   qk,           ctx_q,            nh // max(1, nkv)),
        ('PV_attn',   ctx_q,        vd,               nh // max(1, nkv)),
        ('ATTN_proj', vd * nh,      n_embd,           1),
        ('MLP_FC1',   n_embd,       mlp,              1),
        ('MLP_FC2',   mlp,          n_embd,           1),
    ]


def enumerate_shapes_for_model(model_layer: dict, n_embd: int,
                               ctx: int) -> List[Tuple[int, int, int]]:
    """Return just the (in_ch, out_ch, seq_len) tuples (no name)."""
    return [(ic, oc, sl) for (_, ic, oc, sl)
            in enumerate_gemm_shapes_decode(model_layer, n_embd, ctx)]
