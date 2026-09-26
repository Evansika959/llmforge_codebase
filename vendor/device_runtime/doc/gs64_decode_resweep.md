# GS64 decode optimization and paired replay (2026-09-21)

## Scope

`src/runq_llmforge.c::matmul` now has a dedicated GS=64 NEON path. It computes four
16-byte vector products per quantization group with two independent int32
accumulation chains, reduces the integer dot, then applies the original weight
and activation scales. Every int16 product is widened before addition (including
the -128×-128 edge case). ARMv7 uses `vmull`/`vpadal`; builds with
`__ARM_FEATURE_DOTPROD` use `vdot`. Non-NEON platforms retain the scalar fallback;
GS16/32 branches are unchanged. `NLF_DISABLE_GS64_DECODE` disables only the new
branch for diagnostic builds.

The existing GS64 **prefill batched** kernel was already vectorized and is not
changed. The shared single-vector `matmul` is also used by the final classifier
in prefill, so TTFT may change too. This is not exclusively a change in the timed
decode interval. Quantization, KV-cache format, attention, sampling, workload and
energy integration are unchanged.

The previous user constraint to preserve `runq_llmforge.c` applied to earlier
measurement work; this explicitly requested kernel optimization changes that
file. Before editing, the old source was saved verbatim and verified against the
frozen 2,000-row measurement contract:

`16c954879607c83a1bb75538b883368d51bef732e340efe1c46a75c4f52362e6`.

## Validation

Artifacts: `scripts/sweep/outputs/gs64_decode_validation_20260921/`.

- `runq_llmforge_baseline.c`: exact pre-optimization source.
- `compile_commands.json`: host and ARMv7 old/new compiler invocations.
- `device_correctness.json`: numerical checks on physical watch
  `67301WRDQW80V6`, ABI `armeabi-v7a`.
- `device_microbenchmark.json`: raw interleaved benchmark observations.
- `validation_summary.json`: source hashes, test summary and device evidence.
- `host_*` and `armv7_*` logits/tokens: complete saved tiny-fixture predictions.

The C harness is `scripts/sweep/layerwise/tests/gs64_kernel_check.c`. It compares
GS16/32/64 against a scalar reference across 13 input widths and random, all
-128, alternating extrema and zero activations. Host ASan/UBSan passed. All
4,096 logits and 32 generated tokens per GS in the two-layer heterogeneous
prefill+31-decode fixture match the original source **exactly**, on both the host
and the watch. This validates the tested cases, not every possible trained LLM.
The DOTPROD path is conditional; the actual watch test exercises ordinary ARMv7
NEON, not DOTPROD.

Seventeen existing layerwise tests and four replay integration tests pass. The
latter operate on temporary copies, verify the exact GS64 subset and reject
header/specification/weight-hash tampering. They skip if the locally prepared
integration fixture is absent.

### Watch microbenchmark (four threads)

Old/new order was baseline, optimized, optimized, baseline, with three 20-call
repeats per shape/block. Values below are medians over six observations per
variant. Watch was unplugged; battery/temperature admission checked before each
block. No voice alert was enabled.

| GEMV input × output | Baseline ms | Optimized ms | Ratio |
|---|---:|---:|---:|
| 384 × 576 | 0.1936 | 0.0406 | 4.77× |
| 576 × 1,536 | 0.8026 | 0.1963 | 4.09× |
| 576 × 50,257 | 27.1475 | 6.1206 | 4.44× |
| 960 × 2,560 | 2.2544 | 0.5565 | 4.05× |
| 960 × 50,257 | 44.7970 | 10.1391 | 4.42× |

These are matrix microbenchmarks, **not whole-model throughput speedups**.

### First full-architecture smoke measurement

`LW_a9fbc3c0410ae4daad50` completed successfully before the remaining replay was
launched. Exact regenerated/uploaded model hashes and the final generated token
match the paired baseline. Old versus new measurements:

| Metric | Baseline | Optimized |
|---|---:|---:|
| Decode throughput | 7.5751 tokens/s | 15.4566 tokens/s |
| TPOT | 132.0110 ms | 64.6974 ms |
| TTFT | 696.4045 ms | 646.8480 ms |

Throughput ratio is 2.0404×. This is a single historical/new pair, not a sweep
aggregate. New dynamic energy is 41.9178 mJ/output token with a
`baseline_drift_over_25pct` warning, which is retained rather than concealed.
The paired report is under
`reports/1790012173232463000/decode_updates.json` in the new campaign.

## New replay dataset

All 664 GS64 architectures are selected from
`diliverable/layerwise_2000_20260921/dataset_snapshot.json`, without looking at
performance labels. Each family contributes 332 architectures. Original splits
are preserved: **536 training / 64 validation / 64 test**. Original IDs, ordered
layer geometry, permutation groups and exact synthetic weights are retained.
Replay order is shuffled with seed 20260921.

The NEW campaign is:

`scripts/sweep/outputs/watch5_gs64_decode_v1_664/`

Key files:

| File/directory | Purpose |
|---|---|
| `baseline/dataset_2000.json` | Unmodified original full dataset copy |
| `baseline/runq_llmforge.c` | Original kernel source |
| `baseline/LW_*.json` | Each paired original row, raw result and model provenance |
| `replay_manifest.json` | Subset, frozen hashes and expected per-architecture model hashes |
| `candidates.sqlite` | NEW jobs and NEW measured labels only |
| `architectures.jsonl` | Frozen shuffled replay schedule |
| `hardware_contract.json` | Exact optimized kernel/build/workload hashes |
| `kernel_validation.json` | Numerical validation and microbenchmark results |
| `source_snapshot/` | Exact code used for the new measurement campaign |
| `attempts/` | Raw optimized traces, phase timestamps and result files |
| `run_state.json`, `progress.json` | Current runner state and saved progress |
| `reports/TIMESTAMP/decode_updates.json` | Versioned paired results and decode update overlay |

The exporter must regenerate the exact baseline model SHA-256 before upload;
the device upload is also checksum-verified. A mismatching final generated token
pauses the replay **before accepting labels**, retaining raw evidence for
investigation. This last-token check is an additional runtime guard, not a
substitute for full numerical tests.

All original sweep databases, the frozen 2,000-row deliverable and all existing
predictor checkpoints remain untouched. **Do not rerun an old campaign with the
new kernel or bypass its source-contract guard.** The frozen baseline retains
its old protocol even though the production kernel is now different.

## Start/resume, status, and reports

The new directory has already been prepared. From the repository root:

```bash
conda activate nanollmforge
adb devices -l
caffeinate -i python -u -m scripts.sweep.layerwise.gs64_resweep run \
  --serial 'YOUR_CURRENT_ADB_SERIAL' --acknowledge-protocol

python -m scripts.sweep.layerwise.gs64_resweep status
python -m scripts.sweep.layerwise.gs64_resweep report
```

Default output is the new 664-point directory above. Resume uses the same
command, skips completed jobs, retains interrupted attempts and restores saved
display settings on exit. A single-point smoke run uses `--max-jobs 1`.
No recurring voice monitor is started.

Protection remains: unplugged watch, battery >30%, admission temperature <45°C,
no temperature polling during inference, four threads, 49 prompt tokens and 32
output tokens (31 decode forwards). Low battery, charging, disconnection or a
validation failure pauses the run rather than weakening the protocol. Resolve
the cause and resume. If restoration is incomplete:

```bash
python -m scripts.sweep.layerwise.gs64_resweep restore --serial 'CURRENT_SERIAL'
```

Reports are automatically emitted when a run exits, including a normal battery
pause. `report` can also be invoked while collecting; it reads only completed,
accepted rows and always writes a **new timestamped report**. It does not
overwrite a prior report or mutate a database.

Each report pairs the old and new complete measurement and separately lists
`decode_tok_s`, `decode_duration_s`, `tpot_ms`, and measured decode gross energy
per forward. The original dynamic-energy label includes prefill and decode;
the new complete dynamic-energy label must therefore be kept with its own
timing/trace, not combined with old TTFT or old energy as a supposedly coherent
fresh measurement. There is no automatic predictor refit or mixed-kernel model
promotion in this task.

Historical versus new observations were not measured at the same time, so their
throughput ratios include environmental variation as well as kernel effects.
Use the interleaved microbenchmark for the isolated GEMV evidence; interpret the
full sweep as paired remeasurement evidence, not a fully controlled causal test.

For a future fresh replay directory (never the already prepared one):

```bash
python -m scripts.sweep.layerwise.gs64_resweep prepare \
  --output scripts/sweep/outputs/NEW_GS64_VERSION \
  --baseline-kernel scripts/sweep/outputs/gs64_decode_validation_20260921/runq_llmforge_baseline.c
```

Preparation refuses to overwrite a directory. A different optimization requires
a new version and fresh validation; kernel hashes are not silently migrated.
