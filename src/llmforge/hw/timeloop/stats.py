"""Parsers for Timeloop mapper statistics files (timeloop-mapper.stats.txt).

    parse_timeloop_stats         summary metrics of one mapping: energy, cycles, area, op counts
    parse_buffer_stats           per-level statistics, split by dataspace for storage levels
    parse_dram_dataspace_stats   the DRAM level of parse_buffer_stats, keyed by dataspace

Energies are picojoules unless a key ends in _uJ. Bandwidths are words per cycle. Access counts
named scalar_* are per instance, exactly as Timeloop prints them. Keys named total_* multiply by
the level's instance count.
"""
import re
from typing import Any, Dict, Optional


def _num(s: str) -> Optional[float]:
    """Parse a number that may contain commas and unit suffixes.

    Returns a float or None if parsing fails.
    """
    if s is None:
        return None
    s = s.strip()
    # remove commas
    s = s.replace(',', '')
    # strip common units (uJ, mm^2, %) and parentheses
    s = re.sub(r"\s*(uJ|mJ|J|mm\^2|mm2|%|GHz)\s*$", '', s, flags=re.IGNORECASE)
    try:
        return float(s)
    except Exception:
        return None


def parse_timeloop_stats(path: str) -> Dict[str, Optional[float]]:
    """Parse a timeloop-mapper.stats.txt file and extract useful metrics.

    Extracted fields (when present):
      - gflops (GFLOPs @1GHz)
      - utilization_pct
      - cycles
      - energy_uJ
      - edp
      - area_mm2
      - total_ops
      - total_memory_accesses
      - optimal_ops_per_byte
      - algorithmic_intensity_ops_per_access = total_ops / total_memory_accesses
      - algorithmic_intensity_ops_per_byte = optimal_ops_per_byte if present, else same as ops_per_access

    Returns a dict mapping keys to floats or None.
    """
    metrics = {
        'gflops': None,
        'utilization_pct': None,
        'cycles': None,
        'energy_uJ': None,
        'edp': None,
        'area_mm2': None,
        'total_ops': None,
        'total_memory_accesses': None,
        'optimal_ops_per_byte': None,
        'algorithmic_intensity_ops_per_access': None,
        'algorithmic_intensity_ops_per_byte': None,
    }

    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()

    # Summary Stats block
    m = re.search(r'GFLOPs\s*\(@1GHz\):\s*([0-9.,Ee+-]+)', text)
    if m:
        metrics['gflops'] = _num(m.group(1))

    m = re.search(r'Utilization:\s*([0-9.,]+)%', text)
    if m:
        metrics['utilization_pct'] = _num(m.group(1))

    m = re.search(r'Cycles:\s*([0-9,]+)', text)
    if m:
        metrics['cycles'] = _num(m.group(1))

    m = re.search(r'Energy:\s*([0-9.,Ee+-]+)\s*uJ', text, flags=re.IGNORECASE)
    if m:
        metrics['energy_uJ'] = _num(m.group(1))

    m = re.search(r'EDP\(J\*cycle\):\s*([0-9.,Ee+-]+)', text)
    if m:
        metrics['edp'] = _num(m.group(1))

    m = re.search(r'Area:\s*([0-9.,Ee+-]+)\s*mm', text)
    if m:
        metrics['area_mm2'] = _num(m.group(1))
    else:
        # try matching "Area: 0.00 mm^2" or similar
        m = re.search(r'Area:\s*([0-9.,Ee+-]+)\s*mm\^2', text)
        if m:
            metrics['area_mm2'] = _num(m.group(1))

    # Operational Intensity & totals
    m = re.search(r'Total ops\s*:\s*([0-9,]+)', text)
    if m:
        metrics['total_ops'] = _num(m.group(1))

    m = re.search(r'Total memory accesses required\s*:\s*([0-9,]+)', text)
    if m:
        metrics['total_memory_accesses'] = _num(m.group(1))

    m = re.search(r'Optimal Op per Byte\s*:\s*([0-9.,Ee+-]+)', text)
    if m:
        metrics['optimal_ops_per_byte'] = _num(m.group(1))

    # Fallbacks: sometimes words are slightly different
    if metrics['total_ops'] is None:
        m = re.search(r'Total elementwise ops\s*:\s*([0-9,]+)', text)
        if m:
            total_elem = _num(m.group(1))
        else:
            total_elem = None
        m = re.search(r'Total reduction ops\s*:\s*([0-9,]+)', text)
        if m:
            total_red = _num(m.group(1))
        else:
            total_red = None
        if total_elem is not None and total_red is not None:
            metrics['total_ops'] = total_elem + total_red

    if metrics['total_memory_accesses'] is None:
        m = re.search(r'Total memory accesses required\s*:\s*([0-9,]+)', text)
        if m:
            metrics['total_memory_accesses'] = _num(m.group(1))

    # Compute algorithmic intensity
    if metrics['total_ops'] and metrics['total_memory_accesses']:
        try:
            metrics['algorithmic_intensity_ops_per_access'] = (
                float(metrics['total_ops']) / float(metrics['total_memory_accesses'])
            )
        except Exception:
            metrics['algorithmic_intensity_ops_per_access'] = None

    # If optimal op/byte present, prefer that as ops/byte metric
    if metrics['optimal_ops_per_byte'] is not None:
        metrics['algorithmic_intensity_ops_per_byte'] = metrics['optimal_ops_per_byte']
    elif metrics['algorithmic_intensity_ops_per_access'] is not None:
        # We don't strictly know whether "memory accesses" is in bytes or words.
        # Use ops_per_access as a fallback for ops/byte.
        metrics['algorithmic_intensity_ops_per_byte'] = metrics['algorithmic_intensity_ops_per_access']

    return metrics


# ---------------------------------------------------------------------------
# Per-level and per-dataspace statistics
# ---------------------------------------------------------------------------

# Fields printed for each dataspace of a storage level, mapped to stable key names.
# Labels that are not listed here still parse, under a generic snake_case key.
_FIELD_KEYS = {
    "Partition size": "partition_size",
    "Utilized capacity": "utilized_capacity",
    "Utilized instances (max)": "utilized_instances_max",
    "Utilized clusters (max)": "utilized_clusters_max",
    "Scalar reads (per-instance)": "scalar_reads",
    "Scalar fills (per-instance)": "scalar_fills",
    "Scalar updates (per-instance)": "scalar_updates",
    "Temporal reductions (per-instance)": "temporal_reductions",
    "Address generations (per-cluster)": "address_generations",
    "Energy (per-scalar-access)": "energy_per_scalar_access_pJ",
    "Energy (per-instance)": "energy_per_instance_pJ",
    "Energy (total)": "energy_pJ",
    "Temporal Reduction Energy (per-instance)": "temporal_reduction_energy_per_instance_pJ",
    "Temporal Reduction Energy (total)": "temporal_reduction_energy_pJ",
    "Address Generation Energy (per-cluster)": "address_generation_energy_per_cluster_pJ",
    "Address Generation Energy (total)": "address_generation_energy_pJ",
    "Bandwidth Consumption Scale": "bandwidth_consumption_scale",
    "Shared Bandwidth (per-instance)": "shared_bandwidth_per_instance",
    "Shared Bandwidth (total)": "shared_bandwidth",
    "Read Bandwidth (per-instance)": "read_bandwidth_per_instance",
    "Read Bandwidth (total)": "read_bandwidth",
    "Write Bandwidth (per-instance)": "write_bandwidth_per_instance",
    "Write Bandwidth (total)": "write_bandwidth",
    # SPECS lines
    "Leakage energy (total)": "leakage_energy_pJ",
    "Per-instance-cycle leakage": "leakage_per_instance_cycle_pJ",
    "Vector access energy": "vector_access_energy_pJ",
}

_SECTION_HEADERS = {
    "Buffer and Arithmetic Levels": "levels",
    "Networks": "networks",
    "Operational Intensity Stats": "intensity",
    "Summary Stats": "summary",
    "fJ/Compute": "fj_per_compute",
}

_LEADING_NUMBER = re.compile(r"^[-+]?(?:\d[\d,]*\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_LEVEL_NAME = re.compile(r"^=== (.+) ===$")
_KEY_VALUE = re.compile(r"^(.+?)\s*:\s*(.*)$")
_DATASPACE_HEADER = re.compile(r"^([A-Za-z_][\w ]*):$")


def _value(raw: str) -> Any:
    """Leading number of a printed value (units dropped), the raw string, or None for '-'."""
    raw = raw.strip()
    m = _LEADING_NUMBER.match(raw)
    if m:
        return float(m.group(0).replace(",", ""))
    return None if raw in ("", "-") else raw


def _key(label: str) -> str:
    label = label.strip()
    if label in _FIELD_KEYS:
        return _FIELD_KEYS[label]
    s = re.sub(r"[^0-9A-Za-z]+", "_", label).strip("_").lower()
    return s


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def parse_buffer_stats(path: str) -> Dict[str, Dict[str, Any]]:
    """Per-level statistics of a Timeloop stats file.

    Returns {level_name: level} for every level listed under "Buffer and Arithmetic Levels". A
    storage level maps each dataspace it keeps ("Weights", "Inputs", "Outputs") to a dict of that
    dataspace's statistics. Every level also carries three dicts:

        _specs     the level's SPECS block, for example instances and technology
        _stats     level-wide STATS lines, for example cycles, or the energy of an arithmetic level
        _summary   total_scalar_accesses and op_per_byte from "Operational Intensity Stats",
                   fj_per_compute from the closing "fJ/Compute" table,
                   energy_from_fj_per_compute_pJ = fj_per_compute * computes / 1000,
                   dynamic_energy_pJ, the dataspace energies or the arithmetic energy, and
                   energy_pJ = dynamic_energy_pJ + the level's leakage energy, which is the
                   level's share of the summary energy

    Each dataspace dict holds the printed fields under the names in _FIELD_KEYS plus derived
    values: instances, total_reads, total_fills, total_updates, total_accesses (all multiplied by
    the level's instance count), and the aliases energy_total_pJ and energy_per_access_pJ.

    An empty stats file, which the mapper leaves behind when it finds no valid mapping, yields {}.
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()

    levels: Dict[str, Dict[str, Any]] = {}
    section = None
    level = None
    block = None
    dataspace = None
    intensity_level = None
    computes = None

    for line in lines:
        text = line.strip()
        if not text or set(text) == {"-"}:
            continue
        indent = _indent(line)

        if indent == 0:
            if text in _SECTION_HEADERS:
                section = _SECTION_HEADERS[text]
                level = dataspace = intensity_level = None
                continue
            if text.startswith("Computes ="):
                computes = _value(text.split("=", 1)[1])
                continue
            m = _LEVEL_NAME.match(text)
            if m and section == "levels":
                level = levels.setdefault(m.group(1), {"_specs": {}, "_stats": {}, "_summary": {}})
                block = dataspace = None
            elif m and section == "intensity":
                intensity_level = m.group(1)
            continue

        if section == "levels" and level is not None:
            if indent == 4 and text in ("SPECS", "STATS"):
                block, dataspace = text, None
                continue
            if block == "STATS" and indent == 4:
                header = _DATASPACE_HEADER.match(text)
                if header:
                    dataspace = level.setdefault(header.group(1), {})
                    continue
            kv = _KEY_VALUE.match(text)
            if not kv:
                continue
            key, value = _key(kv.group(1)), _value(kv.group(2))
            if block == "SPECS":
                level["_specs"][key] = value
            elif block == "STATS":
                target = dataspace if (dataspace is not None and indent >= 8) else level["_stats"]
                target[key] = value

        elif section == "intensity" and intensity_level in levels:
            kv = _KEY_VALUE.match(text)
            if kv:
                levels[intensity_level]["_summary"][_key(kv.group(1))] = _value(kv.group(2))

        elif section == "fj_per_compute" and "=" in text:
            name, value = (s.strip() for s in text.split("=", 1))
            if name in levels:
                levels[name]["_summary"]["fj_per_compute"] = _value(value)

    for lv in levels.values():
        instances = lv["_specs"].get("instances")
        instances = instances if isinstance(instances, float) and instances > 0 else 1.0
        fj = lv["_summary"].get("fj_per_compute")
        if isinstance(fj, float) and isinstance(computes, float):
            lv["_summary"]["energy_from_fj_per_compute_pJ"] = fj * computes / 1000.0
        dataspaces = [ds for name, ds in lv.items() if not name.startswith("_")]
        if dataspaces:
            dynamic = sum((ds.get("energy_pJ") or 0.0)
                          + (ds.get("temporal_reduction_energy_pJ") or 0.0)
                          + (ds.get("address_generation_energy_pJ") or 0.0) for ds in dataspaces)
        else:
            dynamic = lv["_stats"].get("energy_pJ") or 0.0
        leakage = lv["_specs"].get("leakage_energy_pJ")
        lv["_summary"]["dynamic_energy_pJ"] = dynamic
        lv["_summary"]["energy_pJ"] = dynamic + (leakage if isinstance(leakage, float) else 0.0)
        for ds in dataspaces:
            reads = ds.get("scalar_reads") or 0.0
            fills = ds.get("scalar_fills") or 0.0
            updates = ds.get("scalar_updates") or 0.0
            ds["instances"] = instances
            ds["total_reads"] = reads * instances
            ds["total_fills"] = fills * instances
            ds["total_updates"] = updates * instances
            ds["total_accesses"] = (reads + fills + updates) * instances
            if "energy_pJ" in ds:
                ds["energy_total_pJ"] = ds["energy_pJ"]
            if "energy_per_scalar_access_pJ" in ds:
                ds["energy_per_access_pJ"] = ds["energy_per_scalar_access_pJ"]
    return levels


def parse_dram_dataspace_stats(path: str, level: str = "DRAM") -> Dict[str, Dict[str, Any]]:
    """Per-dataspace statistics of the DRAM level, keyed by dataspace name.

    Returns for example {"Inputs": {...}, "Outputs": {...}}. Only the dataspaces the level keeps
    appear, so a substrate whose weights bypass DRAM has no "Weights" entry. The fusion model in
    llmforge.hw.timeloop.gemm and the rDXE corrections read energy_pJ, scalar_reads,
    scalar_updates, and scalar_fills from these dicts. Every substrate in specs/arch names its
    off-chip level DRAM, so the default level name covers all of them.
    """
    stats = parse_buffer_stats(path).get(level, {})
    return {name: ds for name, ds in stats.items() if not name.startswith("_")}


if __name__ == '__main__':
    import json
    import sys
    if len(sys.argv) < 2:
        print('Usage: python -m llmforge.hw.timeloop.stats /path/to/timeloop-mapper.stats.txt')
        raise SystemExit(2)
    path = sys.argv[1]
    print('Parsed Timeloop stats:')
    for k, v in parse_timeloop_stats(path).items():
        print(f'  {k}: {v}')
    print('DRAM dataspaces:')
    print(json.dumps(parse_dram_dataspace_stats(path), indent=2))
