# Layerwise hardware sampling: 500-candidate campaign

## GS64 optimized decode replay (new protocol version)

The original 2,000 measurements are complete and frozen. A **separate 664-point
GS64 replay** now uses an optimized decode GEMV kernel, preserving all original
data and checkpoints. See [the optimization/replay guide](../../../doc/gs64_decode_resweep.md)
for validation, paired update files, and start/resume commands. Its directory is
`outputs/watch5_gs64_decode_v1_664`; use `python -m scripts.sweep.layerwise.gs64_resweep`,
not the original 500-candidate validator. The historical sections below describe
the previous kernel; their frozen source guards must not be bypassed.

## Additional 1,000-point campaign (v3)

`outputs/watch5_layerwise_1000_variable_kv_v3/part1` and `part2` contain
500 new candidates each, with seeds 202609191 and 202609192. Both retain the
variable-KV search space below. Part 1 excludes both earlier 500-point registries;
part 2 additionally excludes part 1, including full layer-multiset permutation
groups. Combined allocation: 500 per family, 800 train / 100 validation / 100 test,
96 uniform controls and 904 heterogeneous architectures.

Start or resume the sequential queue from the repository root:

```bash
caffeinate -i python -u \
  -m scripts.sweep.run_layerwise_1000_v3 \
  --serial 'YOUR_CURRENT_ADB_SERIAL' --acknowledge-protocol
```

The queue skips completed parts and resumes incomplete measurements. Any hardware
pause stops the queue; it does not start part 2 after a failed/paused part 1.
Charge and unplug, then rerun the same command. Each part retains its own logs,
SQLite database, frozen hardware contract and restoration records. Graceful
SIGTERM to the verified queue PID pauses the current runner. No voice monitor is
started. Temperature <45C admission and <=30% battery stop remain unchanged.
The C kernel and existing layerwise generation/measurement sources are unchanged.

## Next batch: per-layer variable KV heads

The completed first batch remains at `outputs/watch5_layerwise_500_v1` (500/500).
The separately prepared second batch is
`outputs/watch5_layerwise_500_variable_kv_v2` (500 planned, initially zero measurements).
Its [specification](../configs/watch5_layerwise_500_variable_kv_space.json) retains
the skeletons, operator profile, workload and unmodified C kernel, while allowing
every positive divisor of a layer's attention-head count as its KV-head count:

| Family | Attention heads | Allowed KV heads per layer |
|---|---:|---|
| smollm2-135m | 3 | 1, 3 |
| smollm2-135m | 6 | 1, 2, 3, 6 |
| smollm2-135m | 9 | 1, 3, 9 |
| smollm2-360m | 5 | 1, 5 |
| smollm2-360m | 10 | 1, 2, 5, 10 |
| smollm2-360m | 15 | 1, 3, 5, 15 |

The legal space grows from 192 to **576/640 layer shapes** respectively. The
prepared batch covers 554/616 distinct shapes; it is not exhaustive. All 452
heterogeneous candidates vary KV heads across layers; 48 uniform controls remain.
The 250-per-family allocation, global GS16/32/64 quotas (84/83/83 per family),
pattern counts, ten-candidate batch balance and 400/50/50 grouped split are retained.
Legal shapes are sampled within the existing pattern/GS constraints; this is not
uniform sampling over head counts, nor a paired causal KV-head experiment.
Actual parameter ranges are 52.24–133.53M and 123.57–353.11M for this sample.

Preparation requires exclusion of previous registries. Both exact architectures
and complete layer-multiset permutation groups are excluded regardless of split,
preventing cross-batch reuse/leakage without consulting performance labels.
The saved search space freezes excluded hashes and provenance, so it does not
depend on the original database remaining at its current filesystem path.

```bash
python -m scripts.sweep.layerwise prepare \
  --spec scripts/sweep/configs/watch5_layerwise_500_variable_kv_space.json \
  --exclude-database scripts/sweep/outputs/watch5_layerwise_500_v1/candidates.sqlite \
  --output scripts/sweep/outputs/watch5_layerwise_500_variable_kv_v2
python -m scripts.sweep.layerwise validate \
  --database scripts/sweep/outputs/watch5_layerwise_500_variable_kv_v2/candidates.sqlite
```

The prepared directory already exists; do not rerun `prepare` into it. After
explicitly authorizing hardware execution, charge/unplug and run this **new**
directory (use the current connected ADB serial):

```bash
caffeinate -i python -u \
  -m scripts.sweep.layerwise run \
  --output scripts/sweep/outputs/watch5_layerwise_500_variable_kv_v2 \
  --serial 'YOUR_CURRENT_ADB_SERIAL' --max-jobs 500 --acknowledge-protocol
```

Preparation performs no ADB calls and starts no voice monitor. Admission remains
strictly <45C, no inference temperature polling, and <=30% battery stop. Host tests
validate varying KV-head descriptors and inference for all three groups; they do
not replace a first-device smoke check or establish trained-model numerical parity.

The generator, registry and CLI were generalized; the first batch's data and
frozen contract were not rewritten. V1 generation still reproduces its original
500 scheduled records exactly. New live runs require a new source contract: do
not bypass the source-hash guard to rerun the completed v1 folder with revised
preparation code. The three original source files are preserved locally as
`outputs/layerwise_v1_source_archive/*.py.txt`, matching the old contract hashes.

The remaining sections describe the **first batch's fixed-KV specification**.

Preparation, validation and synthetic Q8 export work offline. The explicit `run`
command collects hardware measurements into this registry, without modifying or
resuming existing watch sweeps/active-learning workspaces. A fresh prepared
registry has no performance labels; `status` reports how many have been measured.

On 2026-09-14 the user confirmed the fixed skeleton below and authorized collection.
The original preparation snapshot/manifest remains unchanged for provenance;
`confirmation.json` and `hardware_contract.json` record the subsequent execution
decision. That historical kernel had no specialized GS64 **decode** path;
its GS64 batched prefill path was already optimized. Its original source is
preserved in the new replay's `baseline/` directory.

## Search space and pending confirmation

The versioned [specification](../configs/watch5_layerwise_500_space.json) uses the
user's software-search screenshot:

| Family | Independent `d_qk`, `d_v` choices | Attention heads | MLP hidden width |
|---|---|---|---|
| smollm2-135m | 16, 32, 48, 64 | 3, 6, 9 | 384, 768, 1152, 1536 |
| smollm2-360m | 16, 32, 48, 64 | 5, 10, 15 | 640, 1280, 1920, 2560 |

Each family has 192 possible **layer shapes**. One candidate is a complete,
ordered list of layers. The screenshot does not specify the fixed skeleton:
the current draft assumes 30 layers / residual width 576 / 3 KV heads for 135m,
and 32 layers / width 960 / 5 KV heads for 360m. These assumptions must be checked
against the software experiment. The family names do not fix actual parameter
counts, and the previous 50–150M parameter filter is not applied.

The draft operator profile matches the current heterogeneous C format: infinite
attention, concatenated heads, pre- and peri-RMSNorm without epsilon, gated MLP
with exact erf GELU, RoPE, no biases, tied embeddings, vocabulary 50,257 and
context capacity 256. `mlp_variant=swiglu` is the existing configuration spelling;
the selected GELU activation makes this **GeGLU**, not SiLU SwiGLU. These settings
are the retained kernel-compatible measurement profile, not a claim about stock
SmolLM2 or numerical parity with trained software models.

The initial `software_config_confirmed=false` and pending decisions remain in the
preparation specification/manifest. The later confirmation is additive. If shape
or operator settings change, generate a new database rather than editing identities.

## Sampling design

- 500 distinct models: 250 per family; 48 uniform controls and 452 heterogeneous models.
- Per family: effective global GS16 = 84, GS32 = 83, GS64 = 83.
- Patterns across both families: 48 uniform, 96 contiguous-block, 96 alternating,
  164 random, and 96 models in 48 same-layer-multiset/different-order pairs.
- 50 scheduled batches of 10, with five models from each family in each batch;
  order within a batch is shuffled with the recorded seed.
- Provisional fixed splits: 400 train, 50 validation, 50 test. Permutations of the
  same layer multiset stay in one split. Future training/acquisition must not use
  test labels; this module does not train or perform active learning.

GS is **global**, not independently assigned to each layer. Starting from 64, the
exporter-compatible rule halves GS until it divides the residual width and every
layer's `n_h*d_v` and `d_mlp`. Unconstrained layerwise sampling would concentrate
strongly at GS16. Stratified sampling prevents that collapse, but consequently
this is a designed coverage study, **not a uniform draw over the entire search
space and not a causal comparison of group sizes on identical architectures**.

The validator reports distinct layer-shape coverage and parameter ranges. It also
checks SQLite integrity, hashes, ordered layer records, legal geometry, actual
global GS, splits, permutation leakage and batch balance. Analytic memory sizes
are lower bounds, not measured peak RSS or an on-watch memory guarantee.

## Commands

Run from the repository root with the `llmforge` environment activated:

```bash
conda activate llmforge
python -m scripts.sweep.layerwise prepare \
  --output scripts/sweep/outputs/watch5_layerwise_500_v1
python -m scripts.sweep.layerwise validate \
  --database scripts/sweep/outputs/watch5_layerwise_500_v1/candidates.sqlite
```

`prepare` refuses to overwrite an existing directory. The generated outputs are:

- `candidates.sqlite`: `candidates`, ordered `layers`, scheduled `jobs`, empty
  `measurements`, and `metadata` tables.
- `architectures.jsonl`: one complete candidate per line, in scheduled order.
- `search_space.json`: exact specification snapshot.
- `manifest.json`: validation report, provenance hashes and preparation status.

Outputs are Git-ignored; the specification and generator are version-controlled.
The initial database hash in the manifest describes the prepared snapshot, not a
future database after measurement inserts. No performance labels are invented.

Inspect the first scheduled candidates:

```bash
sqlite3 -header -column scripts/sweep/outputs/watch5_layerwise_500_v1/candidates.sqlite \
  'SELECT ordinal,batch,candidate_id,family,q8_group_size,pattern,split FROM jobs JOIN candidates USING(candidate_id) ORDER BY ordinal LIMIT 10;'
```

Export one candidate, replacing `CANDIDATE_ID` with an ID from that query:

```bash
python -m scripts.sweep.layerwise export \
  --database scripts/sweep/outputs/watch5_layerwise_500_v1/candidates.sqlite \
  --candidate CANDIDATE_ID \
  --output scripts/sweep/outputs/watch5_layerwise_500_v1/export/model.q8.rlm
```

Export requires PyTorch. Weights are seeded synthetic fixtures, not trained
language-model weights. The exporter reuses existing Q8 serialization, generates
one matrix at a time, checks the version-2 header/descriptors/file size, and writes
an architecture/seed metadata sidecar. Export is deterministic for the recorded
architecture and software environment; it refuses existing output files. Avoid
exporting all 500 models at once: generate and retire model binaries per job in
the eventual runner. Export failure can leave a partial file; do not use it as a
completed model without successful validation and its metadata sidecar.

## Start, monitor, pause and resume

```bash
conda activate llmforge
adb devices -l
python -u -m scripts.sweep.layerwise run \
  --output scripts/sweep/outputs/watch5_layerwise_500_v1 \
  --serial "$WATCH_ADB_TARGET" --max-jobs 500 --acknowledge-protocol
```

The wireless ADB address can change; use the currently connected identifier.
For long runs on macOS, prefix the run command with `caffeinate -i` to prevent
idle system sleep; this does not disable the watch battery guard. ADB upload
timeouts scale with model file size. The initial live campaign's transport-only
timeout amendment after four candidates is recorded under
`amendments/0001_transfer_timeout/`, retaining the previous contract and original
measurement hashes; inference and energy computations were unchanged.
The contract checks the physical hardware serial, OS/ABI, tokenizer, input
identities, executable and source hashes. Use `--max-jobs 2` for an initial
two-model smoke run. Rerun the same command to skip completed candidates and
continue; this is a fixed 500-model collection queue, not predictor-guided AL.
Raw attempt results are committed transactionally. An interrupted result awaiting
a database commit can be recovered from its trace and provenance. There is a
shared hardware OS lock plus a campaign lock; **do not delete live lock files**.

```bash
python -m scripts.sweep.layerwise status \
  --output scripts/sweep/outputs/watch5_layerwise_500_v1
```

Monitoring reads local state only, including per-target measured count, median,
range, group/family coverage, current candidate and latest admission battery.
`progress.json` updates after each model; `batch_NNN_report.json` is a cumulative
snapshot at each completed ten-model boundary. These are hardware statistics,
**not predictor accuracy scores**: no layerwise predictor has been trained yet.

Ctrl-C or SIGTERM requests an interrupt, terminates only this campaign's verified
device PID, and restores display settings. For a between-job pause, create a file
named `STOP` in the output directory; remove that marker before resuming. At
battery <=30%, charging detection, invalid telemetry, device disconnection, or an
unresolved error, the run pauses with completed results retained. Valid hot
admission samples wait and report every 30 seconds, without a temperature
stability requirement. After three rejections at the final pre-inference check,
the job remains retryable and the runner pauses instead of spinning indefinitely.

If restoration was interrupted, verify no benchmark is active and run:

```bash
python -m scripts.sweep.layerwise restore \
  --output scripts/sweep/outputs/watch5_layerwise_500_v1 --serial "$WATCH_ADB_TARGET"
```

### Measurement contract: `layerwise_aligned_v1`

`measure.c` includes the unchanged `src/runq_llmforge.c` with its CLI entry point
renamed. It loads the model/tokenizer, samples a four-second idle baseline, then
times 49 actual prompt tokens and exactly 32 generated tokens (31 decode forwards).
Synthetic EOS does not end this fixed workload. Sampling uses temperature 0.8,
top-p 0.9 and seed 42; generated bytes are not printed during timed inference.
This wrapper changes the measurement workload implementation and is explicitly a
**new protocol**, not directly interchangeable with legacy AL labels.

The model remains loaded during pre/post baseline sampling. Power is sampled at
100 ms intervals on a separate thread; its timestamps and prefill/decode markers
use the same monotonic clock. Trapezoidal integration uses interpolated exact
phase boundaries. The main energy target includes prefill+decode, divided by 32
generated tokens; a decode-only gross energy per forward is also retained.

Dynamic energy subtracts the median pre-idle power (excluding the first second
and last 200 ms) over the measured interval. Post-idle power and relative drift
are saved; >25% drift is a warning, not automatic removal of latency/throughput.
Nonpositive dynamic energy stays visible as a raw signed value, with the training
energy label null, never silently clamped to zero. Invalid/gapped power traces
also produce a null energy label while retaining valid performance timings.
`measurements.accepted=1` indicates complete valid timing; energy consumers must
additionally require a non-null dynamic label and inspect attempt quality flags.

Admission requires <45C, with one final check immediately before inference; there
are **no temperature reads during inference**. A device-side battery/charging
guard runs once per second, including during inference, and a 300-second watchdog
prevents runaway measurements. The host also has a timeout. Threads=4 and cpuset
`f`; this does not lock the CPU clock or remove OS/DVFS interference. The screen
is awake, brightness=1, automatic brightness disabled; original brightness,
mode and timeout are restored on exit. Wi-Fi is retained for ADB; other OS and
radio activity remains a possible source of measurement noise.

Only one generated model is retained as scratch at a time. After a successful
commit its local/remote binary is deleted (regenerable from seed); raw trace,
timing, model metadata/checksum and provenance are preserved for every attempt.
The original 500-model architecture JSONL is never rewritten by collection.

## Scope and remaining research work

1. Fixed skeleton confirmed; the kernel-compatible operator profile and fixed
   workload are recorded in the hardware contract. Numerical software-model
   parity is not established by synthetic performance sampling.
2. Current decode `matmul` lacks a
   dedicated GS64 branch, although it supports GS64 through its generic path;
   prefill has a GS64 branch. Do not mix measurements across an optimization.
3. The new wrapper aligns power/timing windows; sensor latency, baseline drift
   and clock-frequency changes can still affect energy quality.
4. Smoke-test on the watch before processing the 500 jobs. Keep this campaign
   separate from the old homogeneous predictor/AL dataset; its existing feature
   schema cannot represent an ordered layer list. A new encoder and retraining
   are needed, not simply loading the old checkpoint with these candidates.

## Offline tests

For read-only plots of committed measurements, see the
[reporting guide](reporting/README.md). Reports include performance distributions,
baseline-energy diagnostics, admission conditions and source provenance, without
changing the dataset or the frozen hardware protocol.

```bash
python -m unittest discover -s scripts/sweep/layerwise/tests -v
```

Tests cover deterministic identities, global-group backoff, full database
roundtrips and tampering, profile rejection, all three export groups, and (when a
host C compiler is installed) heterogeneous model loading/prefill/decode with the
unchanged C kernel. Host smoke tests do not establish Android numerical parity,
watch memory feasibility, energy validity or GS64 optimization performance.
