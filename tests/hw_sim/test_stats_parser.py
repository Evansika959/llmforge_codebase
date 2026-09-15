"""Timeloop stats parsers, checked on two cached mapper outputs with different level layouts."""
import pathlib

import pytest

from llmforge.hw.timeloop.stats import (parse_buffer_stats, parse_dram_dataspace_stats,
                                        parse_timeloop_stats)

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
GEMMINI = str(FIXTURES / "gemmini_gemm_128i_896o_512l.stats.txt")
DXE = str(FIXTURES / "dxe_gemm_512i_128o_2l.stats.txt")


def test_summary_metrics():
    s = parse_timeloop_stats(GEMMINI)
    assert s["energy_uJ"] == pytest.approx(107.95)
    assert s["cycles"] == 236548
    assert s["gflops"] == pytest.approx(494.54)
    assert s["total_ops"] == 116981760
    assert s["total_memory_accesses"] == 638976


def test_dram_dataspaces_gemmini_layout():
    dram = parse_dram_dataspace_stats(GEMMINI)
    assert set(dram) == {"Weights", "Inputs", "Outputs"}
    assert dram["Weights"]["energy_pJ"] == pytest.approx(7340032.0)
    assert dram["Inputs"]["energy_pJ"] == pytest.approx(4194304.0)
    assert dram["Outputs"]["energy_pJ"] == pytest.approx(29360128.0)
    assert dram["Weights"]["scalar_reads"] == 114688
    assert dram["Inputs"]["scalar_reads"] == 65536
    assert dram["Outputs"]["scalar_reads"] == 0
    assert dram["Outputs"]["scalar_updates"] == 458752
    for ds in dram.values():
        assert ds["instances"] == 1
        assert ds["energy_per_scalar_access_pJ"] == pytest.approx(64.0)
        # Counts are printed per instance, and energy is per-access energy times all accesses.
        assert ds["energy_pJ"] == pytest.approx(ds["energy_per_scalar_access_pJ"] * ds["total_accesses"])
        assert ds["energy_total_pJ"] == ds["energy_pJ"]
        for key in ("partition_size", "utilized_capacity", "scalar_fills", "temporal_reductions",
                    "address_generations", "read_bandwidth", "write_bandwidth"):
            assert key in ds


def test_dxe_layout_levels_and_instances():
    levels = parse_buffer_stats(DXE)
    assert list(levels) == ["mac", "acc_buffer", "wmem", "head_sram", "global_sram", "DRAM"]
    assert levels["mac"]["_stats"]["energy_pJ"] == pytest.approx(9065.94)
    wmem = levels["wmem"]["Weights"]
    assert wmem["instances"] == 128
    assert wmem["scalar_reads"] == 1024
    assert wmem["total_reads"] == 1024 * 128
    assert levels["head_sram"]["Inputs"]["total_accesses"] == (1024 + 1024) * 8
    dram = parse_dram_dataspace_stats(DXE)
    assert set(dram) == {"Inputs", "Outputs"}      # this mapping keeps weights out of DRAM
    assert dram["Inputs"]["energy_pJ"] == pytest.approx(65536.0)
    assert dram["Outputs"]["energy_pJ"] == pytest.approx(16384.0)
    assert set(parse_dram_dataspace_stats(DXE, level="head_sram")) == {"Inputs", "Outputs"}


@pytest.mark.parametrize("path", [GEMMINI, DXE])
def test_level_energies_match_fj_per_compute_table(path):
    for name, lv in parse_buffer_stats(path).items():
        share = lv["_summary"]["energy_from_fj_per_compute_pJ"]
        computes = share / lv["_summary"]["fj_per_compute"] * 1000.0
        rounding = 0.005 * computes / 1000.0      # fJ/Compute is printed to 0.01 fJ
        assert lv["_summary"]["energy_pJ"] == pytest.approx(share, abs=rounding + 1e-6), name


@pytest.mark.parametrize("path", [GEMMINI, DXE])
def test_levels_sum_to_summary_energy(path):
    total_uJ = sum(lv["_summary"]["energy_pJ"] for lv in parse_buffer_stats(path).values()) / 1e6
    # The summary energy is printed to 0.01 uJ.
    assert total_uJ == pytest.approx(parse_timeloop_stats(path)["energy_uJ"], abs=0.0051)


def test_empty_and_missing_files(tmp_path):
    empty = tmp_path / "timeloop-mapper.stats.txt"
    empty.write_text("")
    assert parse_buffer_stats(str(empty)) == {}
    assert parse_dram_dataspace_stats(str(empty)) == {}
    with pytest.raises(FileNotFoundError):
        parse_dram_dataspace_stats(str(tmp_path / "missing.txt"))
