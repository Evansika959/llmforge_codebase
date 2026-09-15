# Accuracy-oriented active learning

The objective is to improve architecture-to-performance prediction under a bounded measurement budget, not to find the fastest or lowest-energy architecture. Initial implementation uses the existing physics32/log-target XGBoost predictor. Transformer and stacked-ensemble training remain separate experiments.

## Checkpoint-driven live execution

### Current continuation: baseline-subtracted dynamic energy

**Delivery snapshot, 2026-09-14:** the dynamic run was paused at the user's request after 9 completed rounds; round 10 remains pending. The latest committed dataset has 1,552 training observations plus the unchanged 150 validation and 314 test observations. Live status is always read from the workspace's `state.json`; these counts describe the export, not an automatically updated status page.

The curated [shipped predictor assets](../../../../../../assets/device/pixel_watch5/README.md) contains the latest dataset/checkpoint and the validation-selected best checkpoint (round 1) with its matching earlier dataset. Selection uses the mean of the three fixed-validation MAPEs within this dynamic stage; it does not evaluate the test set. Exporting a best checkpoint does **not** change the live loop's checkpoint: resuming the existing workspace continues from the latest committed round 9 model and frozen pending round 10 proposal, without remeasuring completed rounds.

`runs/device/` is ignored by Git. Keep the full local workspace for resume and raw-trace audit. The delivery folder alone cannot resume this run. After charging, unplug the watch and use the `active run` command below with the original dynamic workspace, not `assets/device/pixel_watch5/`. No automatic restart is scheduled.

`init` and `replay` now default to **dynamic** energy. The live worker already records baseline power and dynamic energy; the predictor, bootstrap disagreement, acquisition, ingestion, retraining and accuracy reports now use that target end to end. Baseline power is a label-construction input, never a predictor feature.

Dynamic energy is the recorded `max(active_power_w - baseline_power_w, 0) * duration_s * 1000 / 32`, in mJ/output token. It includes prefill. The recorded label is checked against this formula within CSV rounding tolerance, not subtracted twice. Missing/nonpositive/nonfinite baseline or energy, and inconsistent labels, pause ingestion; there is no silent gross fallback or zero-clipping for log training. Subtraction does not guarantee removal of background drift or better predictions.

The prepared continuation is `runs/device/active_learning_watch5_dynamic_under45`. The old gross workspace is retained unchanged. Switching is a **new learning stage**, with a freshly trained dynamic initial checkpoint and committee, fixed holdout identities, and round numbering restarting at 1. Old gross predictions and uncertainty are never relabeled as dynamic. Valid complete measurements from a partial old batch enter the new seed; unmeasured candidates return to the pool. Its provenance records every converted label and input hash; no historical rows are silently filtered.

Migration normally refuses any invalid historical dynamic label. The explicit `--quarantine-invalid-training-energy` option preserves unusable **training** observations and raw rows in `provenance/energy_transition.json`, excludes them from this dynamic fit, and leaves their architectures eligible in the candidate pool. Invalid holdouts always block migration. The prepared watch stage uses this option for the single zero-energy row `AL_56edd3ffbca9b338`; it is not an outlier-error filter or a license to change evaluation data.

To perform such a transition once, into a new directory (no hardware access):

```bash
python -m llmforge.hw.device.prediction active switch-energy \
  --source runs/device/active_learning_watch5_overnight_20260913_noanchors_under45 \
  --workspace runs/device/active_learning_watch5_dynamic_under45 \
  --reason 'User approved baseline-subtracted energy for the actual active learning loop' \
  --accept-source-changes --quarantine-invalid-training-energy
```

To start/resume the prepared dynamic stage, charge and unplug the watch, then run:

```bash
python -u -m llmforge.hw.device.prediction active run \
  --workspace runs/device/active_learning_watch5_dynamic_under45 \
  --execute-hardware --acknowledge-protocol --serial 'YOUR_CURRENT_ADB_SERIAL' --rounds 500
```

`run` wraps the same resumable `step` engine and writes `monitoring/live_*.log`, `monitoring/ACCURACY.md` and `monitoring/accuracy_history.json`. Every completed round reports each target's fixed-validation MAPE and reduction in percentage points relative to this **dynamic stage's initial checkpoint**. It does not evaluate the final test or poll the device for reporting. Hardware stops are not automatically retried. `active monitor --workspace DIRECTORY` refreshes reports without hardware access; `active step` without hardware flags prepares acquisition only when no measurements are available.

The dynamic continuation inherits 10 new candidates/round, no anchors, strict sampled `<45 C` pre-inference admission, no inference temperature polling, and the <=30% battery stop. Existing raw gross fields remain diagnostic data, not the trained energy target. Do not launch the old output-directory-specific gross supervisor for this stage.

### Historical gross continuation: strict <45 C admission

The historical gross continuation directory is `runs/device/active_learning_watch5_overnight_20260913_noanchors_under45`. The following records the earlier temperature migration; use the dynamic stage above for new work.

`active relax-temperature --source OLD_40C_DIRECTORY --workspace NEW_DIRECTORY --reason 'User-approved <45 C admission' --accept-source-changes` creates a reviewed copy without accessing hardware. The adapter installs a strict `<45` sampled check **before** inference, never temperature polling within inference. Exactly 45 C waits. Valid hot/throttled/low-voltage conditions keep waiting with a status line every 30 s, battery checks every polling cycle and Ctrl-C support. Missing/invalid temperature or voltage telemetry still aborts after 180 s. Battery <=30% and charging/busy-device preflight protections are unchanged.

The run contract/settings and each new attempt record `thermal_policy`. New ingested observations retain their accepted attempt's thermal context as metadata, not predictor features. Original datasets and checkpoints remain untouched: their `protocol.temperature_ceiling=40` is the historical admission setting, **not** an assertion that subsequent amended measurements were collected below 40 C. The explicitly versioned operating-policy amendment is authoritative for those measurements. Mixed 40 C / 45 C data may differ in device-state distribution; this change does not establish measurement comparability or remove drift.

Resume with the repository interpreter, the current ADB serial, and the new workspace:

```bash
python -u -m llmforge.hw.device.prediction active step \
  --workspace runs/device/active_learning_watch5_overnight_20260913_noanchors_under45 \
  --execute-hardware --acknowledge-protocol --serial 'YOUR_CURRENT_ADB_SERIAL' --rounds 500
```

No C source or original sweep script changes are required. Cooldown is a sampled admission condition, not a guaranteed peak-temperature bound.

The current operational goal is to exercise the real measurement/retraining loop, not to prove hybrid acquisition beats random sampling. `init --initial-model /path/to/model.joblib --dataset /path/to/dataset.json` copies and hashes a trusted versioned checkpoint. Its dataset fingerprint, protocol and targets must exactly match the initial dataset. The first proposal loads this checkpoint without refitting it. Later proposals load the preceding round's committed `predictor_after.joblib`; the bootstrap acquisition committee is fitted separately. Updating XGBoost after receiving new labels still means fitting on all accumulated training rows, not appending trees to an old booster.

The original overnight workspace `runs/device/active_learning_watch5_overnight_20260913` is preserved as historical evidence (10 candidates plus 6 anchor measurements per round). The reviewed continuation is `runs/device/active_learning_watch5_overnight_20260913_noanchors`: **10 new candidates and no reference measurements per round**, gross energy, the old 40°C sampled pre-inference check and the existing <=30% battery stop. Its migration record documents the removed quality gate, not a successful drift test. The 500-round/5,000-candidate ceiling remains a finite backstop, not permission to continue past low battery.

```bash
python -u -m llmforge.hw.device.prediction active step \
  --workspace runs/device/active_learning_watch5_overnight_20260913_noanchors \
  --execute-hardware --acknowledge-protocol \
  --serial 'YOUR_ADB_SERIAL_OR_IP:PORT' --rounds 500
```

This is the same command for resuming, but do not start a second copy while the current process is alive. Low battery, incomplete restoration, missing labels and failed quality gates still pause execution; the command does not override these safeguards. Battery protection means the experiment ends at 30%, not at a forced device shutdown. The initial device connection and all successful round measurements/checkpoints are recorded under the workspace.

## Rollout plan

1. Offline integration tests: proposal freezing, architecture-disjoint holdouts, missing/invalid results, drift pause, lock release, resume after ingestion, and budget termination. No ADB calls.
2. Retrospective replay: identical initial training sets, held-out architecture pool, fixed validation/test, equal label budgets, hybrid versus random, multiple seeds. A real historical label is revealed only after selecting its architecture. This tests the loop and provides limited learning-curve evidence; it cannot reproduce future hardware noise.
3. One live pilot round: review the protocol, then explicitly launch a candidate-only batch (anchors are optional). Inspect traces, prospective errors and validation metrics before authorizing more rounds.
4. Continue the same workspace one round at a time, or explicitly request a bounded multi-round run. Keep a genuinely prospective final evaluation block untouched before claiming generalization improvements. A plateau is not a reason to collect indefinitely: inspect noise, distribution coverage and model bias first.

No accuracy improvement is guaranteed by the acquisition score. Bootstrap disagreement is an **uncalibrated proxy** for model uncertainty, not a confidence interval or a measurement-noise estimate.

## Modules and round contract

| Module | Responsibility |
|---|---|
| `datasets.py` | Frozen historical-cohort adapter and explicit measurement protocol |
| `committee.py` | Architecture-group bootstrap, train-only hardware calibration, XGBoost committee |
| `sampling.py` | Candidate constraints, deduplication, acquisition and randomized measurement order |
| `engine.py` | Frozen proposals, immutable ingestion, retraining, validation reporting, resume |
| `quality.py` | Schema/target checks and anchor drift reports |
| `migration.py` | Audited no-anchor continuation, preserving old artifacts and measurement protocol |
| `energy.py` | Audited gross-to-dynamic stage, recorded-baseline reconciliation and seed checkpoint |
| `monitoring.py` | Target-specific validation history and explicitly enabled continuous live runner |
| `hardware.py` | Explicit device preflight, bounded attempts, raw-result merge, saved-state recovery |
| `worker.py` | Isolated wrapper around the original sweep script; deterministic weights and trace retention |
| `replay.py` | Offline hidden-label oracle and paired random-sampling comparison |
| `storage.py` | Atomic state/model writes and OS-released locks |
| `cli.py` | Hardware-off-by-default command interface |

Each round freezes predictions **before** measuring. The default batch has 12 disagreement selections, 6 coverage selections and 2 uniformly random selections, shuffled for execution. Each target has equal weight in log space. Predicted throughput/latency/energy values are never rewarded directly. Candidate bounds are 50–150M parameters with the existing generator's supported shape/alignment constraints. No measured temperature, power, battery state or acquisition order enters model features.

**Anchors are disabled by default (`--anchors 0`)**, including live execution. Device-state drift is not assessed automatically. Schema, architecture, token-count and positive/finite target checks remain mandatory. For a separate control experiment, `--anchors 3` enables three reference architectures before and after each batch; these never enter training. New candidate rows enter training only; all validation/test architectures, including aliases, are excluded from acquisition. Physics calibration and bootstrap fitting see training rows only; validation controls stopping. Test reporting is OFF by default (`--evaluate-test` is exploratory).

Default total budget: 5 rounds × 20 new architectures = 100 new training architectures, with no controls. Each round is limited to 3 execution attempts; no-anchor retries measure only missing/invalid candidates. Every successful round fits a fresh single XGBoost predictor on all accumulated training observations. The latest checkpoint is saved, not automatically promoted to a production/best model.

## Prepare without accessing the watch

Run with the repository environment active:

```bash
python -m llmforge.hw.device.prediction active init \
  --workspace runs/device/active_learning_watch5_pilot \
  --energy dynamic --pool-size 5000 --batch-size 20 --members 5 \
  --anchors 0 --max-rounds 5 --temperature 40

python -m llmforge.hw.device.prediction active step \
  --workspace runs/device/active_learning_watch5_pilot

python -m llmforge.hw.device.prediction active status \
  --workspace runs/device/active_learning_watch5_pilot
```

`init` requires a new directory; do not rerun it for the prepared pilot. `step` without `--execute-hardware` fits the initial model/committee and saves the first proposal, then pauses. It makes no ADB calls. Existing complete measurements can still be audited/ingested/retrained by a hardware-off step.

The historical adapter preserves 1,564 architectures: 1,100 training, 150 validation, 314 test. Default energy is **dynamic** (baseline-subtracted). Explicit `--energy gross` selects `total_energy_j * 1000 / 32`, in mJ/output token, for a separate workspace. Both include prefill; gross also includes background power within the inference window. Neither is pure decode-only energy. Both use a nominal 48-token prompt (49 actual tokens), 32 generated outputs and 31 decode forward passes. An explicit normalized dataset supplied with `--dataset` owns its energy protocol and is not changed by `--energy`; the live adapter accepts only its declared compatible protocol.

## Protocol review and live launch

The old CSVs do not establish exact physical-device identity, OS/build or all environmental conditions. The name “watch5” is not hardware verification. `contract.json` deliberately records historical identity as unverified. First live execution pins the connected physical serial, OS fingerprint, ABI, NDK metadata and tokenizer hash; subsequent device changes are rejected.

`--acknowledge-protocol` means you reviewed whether the historical data and new measurements are comparable. It is **not** proof of comparability. New models use deterministic architecture-specific weights, whereas historical random weights were unseeded. With anchors disabled there is no automatic drift assessment. Optional anchor gates test current-session stability; they do not establish that the entire historical cohort matches the new session. If the device/kernel/workload or measurement method differs materially, use a separate reviewed dataset/protocol rather than merging labels blindly.

```bash
python -m llmforge.hw.device.prediction active preflight --serial 'YOUR_ADB_SERIAL_OR_IP:PORT'

python -m llmforge.hw.device.prediction active step \
  --workspace runs/device/active_learning_watch5_pilot \
  --execute-hardware --acknowledge-protocol \
  --serial 'YOUR_ADB_SERIAL_OR_IP:PORT'
```

The default command attempts **one round**, then returns. `--rounds 5` explicitly permits continuation, still bounded by the workspace budget. Review the pilot before doing this. Do not run another sweep concurrently: the legacy runner has shared local/remote paths. Existing local scratch files or running device benchmark processes cause a refusal; inspect them rather than deleting files or killing processes blindly.

The adapter calls `llmforge.hw.device.measurement.sweep.run_sweep_configs` and does not modify the runtime's `src/runq_reallm.c`. It preserves the old runner's awake-screen, four-thread, INT8 measurement behavior. It does not disable wireless services, lock CPU frequency, or use the archived stability-v2 protocol. Screen timeout/stay-awake setting values are saved/restored with read-back; prior wakefulness itself is not restored automatically.

Before inference, the existing runner waits for CPU temperature **<=40°C**, no reported cooling/frequency clamp, and adequate battery voltage. There is no temperature-stability window and no temperature polling during inference. End temperature is recorded. The temperature ceiling is therefore a pre-inference admission condition, not a guaranteed peak temperature. Battery monitoring remains active during inference; <=30% stops the attempt and requests charging. Cooling/telemetry recovery has a finite timeout. Charge, unplug and let the watch cool before resuming. The preflight refuses a connected charger. These are measurement guards, not a guarantee of hardware safety.

## Audit and recovery

```text
workspace/
  contract.json, settings.json, pool.csv, anchors.json, dataset_000.json
  state.json, RUNNING.lock, device_identity.json (after first live attempt)
  rounds/0001/
    proposal.json, schedule.csv, predictor_before.joblib, committee.joblib
    attempts/001/{configs.csv,preflight.json,command.json,hardware.log,raw_results.csv}
    attempts/001/{original_device_state.json,restoration.json,traces/}
    measurements.csv, attempt_audit.json, measurement_audit.json, anchor_report.json
    prequential.json, dataset_after.json, ingested.json
    predictor_after.joblib, completion.json
```

`proposal.json` contains selection reasons, disagreement, coverage, and pre-measurement predictions. `prequential.json` adds actual labels; errors there describe the adaptively selected batch, not unbiased pool-wide accuracy. `completion.json` contains validation before/after and provenance. Raw attempt CSVs are never rewritten; the aggregate is regenerable. Invalid rows are retained in raw artifacts and flagged, not silently cleaned. Completed inference traces are preserved; forced interruption before trace retrieval can leave partial traces only on the watch.

Missing/invalid rows pause the round without partial ingestion. The same live command resumes missing candidates; post-anchors are refreshed only if explicitly enabled. Valid previous rows are retained. A crash after committed ingestion resumes retraining without remeasuring or appending duplicates. OS file locks release when the process exits; a residual `RUNNING.lock` file is normal and need not be removed. A suspended/live process still holds the lock: resume/stop that process normally. Avoid `Ctrl-Z` or `kill -9`; if forced termination occurs, inspect device processes and saved state before resuming.

If restoration was incomplete, use the exact saved state shown in the error:

```bash
python -m llmforge.hw.device.prediction active restore \
  --state /absolute/path/to/attempts/001/original_device_state.json \
  --serial 'YOUR_ADB_SERIAL_OR_IP:PORT'
```

Source files, inputs, kernel and round artifacts are hash checked. Unchanged interrupted workspaces resume; changed code/settings require a **new** workspace. Do not edit contracts or outputs to bypass validation. The scoped migration below removes only the reference-measurement policy, without changing the hardware/energy protocol, resetting attempt budgets or accepting unrelated source edits.

### Discontinue anchors in an existing run

```bash
python -m llmforge.hw.device.prediction active disable-anchors \
  --source runs/device/active_learning_watch5_overnight_20260913 \
  --workspace runs/device/active_learning_watch5_overnight_20260913_noanchors \
  --reason 'User approved candidate-only acquisition; retain validation and battery protection' \
  --accept-source-changes
```

Run once into a **new** directory; this neither calls ADB nor trains a model. It verifies inputs, committed rounds and restoration records, copies the run, preserves the pending proposal/schedule/aggregate in `migrations/0001_disable_anchors/before/`, and records old/new source hashes. Only the enumerated anchor-policy modules may differ. Raw attempts, traces, completed rounds, candidate predictions, device identity and holdouts are retained. The original workspace stays unchanged except for its OS lock file. Historical dataset text may mention an anchor pilot; the explicit quality-policy revision supersedes that requirement without changing target definitions or old dataset/checkpoint hashes.

Then `active step --workspace NEW_DIRECTORY` **without** `--execute-hardware` validates existing candidate labels and retrains if complete. Missing/invalid candidates still pause. Old raw anchor rows are logged as excluded controls, never relabeled as candidates. Future live commands use the new directory; do not migrate again.

### Optional anchor mode only

The following drift checks and manual-review command apply only to workspaces explicitly created with nonzero anchors. With anchors disabled, `anchor_report.json` records `enabled: false` and warns that drift was not assessed; this is not evidence of stable device state. Resume in no-anchor mode measures only missing candidates and never refreshes POST anchors.

For each target, the anchor gate compares absolute log ratios PRE versus POST and against the previous round's geometric-mean reference. It pauses if the median exceeds `log(1.25)`, or any anchor exceeds twice that value. This is a configurable heuristic, not a statistically calibrated drift test. Baseline subtraction is not used for gross-energy gates, but gross energy can still drift with background load. Failed batches remain un-ingested. After investigating traces and device conditions, a deliberate override can be recorded for the exact aggregate hash:

```bash
python -m llmforge.hw.device.prediction active review-drift \
  --workspace runs/device/active_learning_watch5_pilot \
  --accept --reason 'Record the actual evidence and rationale here'
```

Then rerun `step`. Do not accept a flag simply to keep the loop moving. Accepted overrides are recorded in ingestion and completion metadata.

## Offline checks

```bash
python -m unittest discover -s src/llmforge/hw/device/prediction/tests -t src -v

python -m llmforge.hw.device.prediction active replay \
  --output runs/device/active_learning_replay_NEW \
  --initial 300 --batch-size 20 --rounds 3 --members 3 \
  --seeds 42 123 2026 --energy gross
```

Replay uses a smaller initial fit set to leave an existing-label oracle pool; live initialization uses all 1,100 training architectures. Paired strategies use identical initial labels, candidate pools, validation/test splits and budgets. Seeds are summarized individually in `metrics.json` and with mean/seed SD in `summary.json`. The historical test has already been inspected, so these are exploratory comparisons. Pure random sampling is retained as a first-class baseline (`init --strategy random`) rather than assuming adaptive acquisition must win.

Live ADB compilation, transport, screen restoration and measurement timing still require the physical pilot. Offline tests and replay cannot certify them.
