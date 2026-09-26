# Read-only layerwise measurement plots

## All 2,000 architectures and the missing-energy retest

The all-campaign scatter is saved at
`scripts/sweep/outputs/layerwise_all2000_retest_20260921/all_architectures_scatter.png`
(also PDF). It plots throughput, TTFT and dynamic energy against actual parameter
count. Colors indicate global GS16/32/64; marker shapes indicate skeleton family;
the magenta star is the retested point. Warning records remain included.

```bash
python -m scripts.sweep.layerwise.reporting.plot_all_measurements \
  --output scripts/sweep/outputs/NEW_ALL_ARCHITECTURES_REPORT
```

It reads one accepted measurement per architecture across all four registries,
validates labels against artifacts, and exports `plotted_records.json` and
`summary.json` with hashes. It currently contains 2000 timing labels and 1999 energy
labels. The remaining null energy belongs to the older variable-KV batch,
`LW_e11e2a35f693c21d5fa0`; it is omitted only from the energy panel. All 1000 energy
labels in the latest two-part campaign are now available.

The requested point `LW_47d4353f9819966c231b` (v3 part1 ordinal 157) was remeasured
using `python -m scripts.sweep.retest_layerwise_point --campaign CAMPAIGN
--candidate ID --serial SERIAL`. This utility accepts only a completed candidate
with one accepted missing-energy record. It checks the frozen measurement source,
device/build, binary, input files and exact regenerated model-weight hashes.
The original energy was -19.60 mJ/token before invalidation. Attempt 002 returned:

- Dynamic energy: **54.840790 mJ/token**; gross: 59.863191 mJ/token.
- Throughput: **14.807339 tok/s**; TTFT: **949.768750 ms**.
- Pre/post baseline: 0.052810 / 0.058364 W; drift: **10.517%**, no warning.
- Measured admission: 31.0C; host final admission: 33.4C, battery 91%.

After successful display restoration, a SQLite transaction marks attempt 001
`accepted=0` and inserts attempt 002 as `accepted=1`. The **entire measurement row**
is superseded so energy/timing remain from the same run. No attempt files are
deleted or overwritten. The original row, pre-change database backup, selection
policy, new result and outcome are preserved under the campaign's `retests/`.
Future readers must filter `accepted=1`; raw row counts include historical attempts.
Existing frozen training snapshots/checkpoints are unchanged and need a later
explicit refit to incorporate the retest. The large timing difference between the
two attempts is not explained by this one retest and is not a repeatability estimate.

From the repository root, in the `nanollmforge` environment:

```bash
python -m scripts.sweep.layerwise.reporting.plot_measurements \
  --campaign scripts/sweep/outputs/watch5_layerwise_500_v1 \
  --output scripts/sweep/outputs/watch5_layerwise_500_v1/reports/new_snapshot
```

Choose a new output directory. The report reads SQLite in read-only mode, selects
only committed complete jobs, validates architecture/protocol identities and
checks database metrics against saved attempt results. It does not call ADB,
restart collection, change the hardware contract, train a predictor, or remove
warning points. Multiple accepted repeats for one architecture are rejected
until an explicit repeated-measurement reporting policy is implemented.

Artifacts:

- `performance_overview.png` / `.pdf`: each metric vs actual parameter count,
  plus descriptive distributions by effective global GS. Colors encode GS;
  circles/triangles encode the two fixed model skeletons. Boxes show sample
  distributions, not confidence intervals; jitter affects only display positions.
- `energy_quality.png` / `.pdf`: gross vs baseline-subtracted energy, before/after
  idle-power agreement with +/-25% guides, and admission temperature/battery by
  measurement order. Gaps longer than ten minutes are marked rather than joined.
- `summary.json`: counts, ranges, medians, grouped summaries, exploratory rank
  correlations, source hashes and interpretation limitations.
- `plotted_records.json`: the exact derived records plotted, including warnings.

The first report (`reports/first33`) has 33 distinct architectures: 29
heterogeneous and four uniform controls; 16 from the 135M family and 17 from the
360M family. Candidate 34 was not complete and is excluded. GS16/32/64 have
10/13/10 measurements. Two hardware-contract hashes are present because of the
documented host upload-timeout amendment; model/kernel/energy formulas did not
change. No legacy homogeneous AL measurements are included.

Interpretation limits:

- GS groups contain different architectures. Group differences are descriptive,
  not isolated causal estimates of a kernel/group-size effect.
- Energy includes prefill + decode, divided by 32 output tokens. It is not a
  decode-only dynamic-energy estimate.
- Record 3 has approximately 39.7% pre/post baseline drift; it stays visible with
  a red ring. Record 28 has the lowest pre-idle power (~0.049 W), annotated as an
  observation, not automatically rejected or diagnosed as an error.
- A small pre/post difference does not demonstrate consistent operating state
  across architectures. Temperature is an admission sample, not inference peak.
- These are distinct architectures, not repeats. Neither repeatability nor
  prediction error can be inferred from the spread in these charts.

The reporting subpackage is outside the hardware runner's top-level source-hash
set and the old prediction workspace's source set, so adding plots does not
invalidate experiment resume.

To audit a suspected collection-session boundary within a saved snapshot:

```bash
python -m scripts.sweep.layerwise.reporting.session_comparison \
  --report scripts/sweep/outputs/watch5_layerwise_500_v1/reports/first184 \
  --boundary 98
```

This creates `session_comparison.png`, `.pdf` and `.json` without changing the
source data. It compares baseline power and family/GS-stratified medians before
and after the boundary. These are different architectures, not paired repeats:
the comparison flags a potential session effect but does not identify its cause.
In the energy-quality chart, all warned points remain ringed; only the three
largest drift values are text-labelled to keep larger snapshots readable.

## Independent macOS sound monitor

After starting the runner, read its current `pid` from `run_state.json`, then run:

```bash
caffeinate -i python -u -m scripts.sweep.layerwise.reporting.watch_monitor \
  --campaign scripts/sweep/outputs/watch5_layerwise_500_v1 \
  --runner-pid RUNNER_PID --interval 600
```

The monitor checks immediately and every ten minutes thereafter. It reads local
state and SQLite in read-only mode, checks runner liveness and host ADB transport
state, and never polls watch temperature/power or changes device settings. On a
pause (including battery protection), disconnect, unexpected process exit,
monitor error or completed campaign it plays three chimes and a spoken message,
then exits. Detection latency can be up to ten minutes plus check time. This does
not automatically reconnect or restart the experiment. Restart the monitor with
the new runner PID after resuming. Only one monitor per campaign is allowed.

Records append to `sound_monitor.jsonl`, including audio-command failures.
Use Ctrl-C (or SIGTERM to the monitor PID) to stop monitoring without stopping
collection. Audio uses the existing Mac volume/output device; a muted Mac or
headphones may prevent an audible room alert. Test with
`afplay /System/Library/Sounds/Glass.aiff`. This is a local sound alarm, not a
background chat-notification service. Keep the Mac awake for monitoring.

Offline tests (audio is mocked):

```bash
python -m unittest scripts.sweep.layerwise.reporting.test_watch_monitor -v
```
