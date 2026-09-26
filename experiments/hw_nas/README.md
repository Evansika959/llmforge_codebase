# Hardware-aware search experiments

This directory holds the design, the run queue and the analysis of the hardware-aware architecture
search. Every run is one call of `python -m llmforge.search.cosearch` and writes to
`runs/search/<job>`. The search space and the algorithm are specified in `docs/search.md`.

## Questions

1. **Hardware in the loop.** An architecture searched with the target in the loop is compared with
   architectures searched against a hardware-agnostic proxy and then deployed on the same target.
   The proxies are parameter count and FLOPs per token.
2. **Model scale.** The same protocol runs on supernets from 135M to 4B parameters.
3. **Hardware targets.** Three target settings on this machine stress different parts of the model.
   GPU prefill processes batches of 64 prompts of 512 tokens, where matrix multiplications over the
   prompt dominate. GPU decode generates 128 tokens for the same batch from a KV cache, one step at a
   time. The watch target decodes one sequence of 49 prompt and 32 generated tokens on a smartwatch
   processor. The simulator targets need a Timeloop build that this machine lacks.
4. **Heterogeneity.** A per-block allocation is compared with the best uniform slice at equal
   hardware cost.
5. **Search efficiency.** NSGA-II is compared with random sampling at the same evaluation count.
6. **Ground truth.** Searched architectures are trained in place as dedicated models to test whether
   their advantage survives training. A first pair at Qwen3-1.7B is in the queue.

## Software evaluator

Quality is the mean token cross-entropy of a supernet slice on held-out FineWeb-Edu documents,
1024 tokens per document. Searches score documents 0 to 31. Every final front and both anchors are
re-scored on the disjoint documents 32 to 95, reported as `val_loss_heldout`. One slice takes
1.2 to 1.9 seconds on the H100 from 135M to 4B.

### Phase 0, supernet screening

Every candidate checkpoint scored every uniform slice. A checkpoint is kept when larger slices rarely
score worse than slices they contain, and when its slice scores pick well among architectures that
were trained as dedicated models. Regret is the trained-loss gap between the architecture the slice
scores pick and the best one inside each budget, where budgets are the parameter and KV-cache sizes
of the references with at least six references inside. Picking the largest architecture that fits is
the free baseline.

| model | checkpoint | nesting violations | regret, nats | largest that fits | Spearman, slice vs trained | kept |
|---|---|---|---|---|---|---|
| SmolLM2-135M | `sl135_lr8e-3` | 1.27% | 0.012 | 0.019 | 0.984 | yes |
| SmolLM2-135M | `sl135_lr4e-3` | 1.60% | 0.014 | 0.019 | 0.977 | |
| Qwen3-1.7B | `qwen3-1.7b_ab` | 3.48% | 0.000 | 0.253 | 0.917 | yes |
| Qwen3-1.7B | `qwen3-1.7b` | 12.26% | 0.036 | 0.253 | 0.728 | |
| Qwen3-4B | `qwen3-4b_ab_blocks` | 0.19% | none | none | none | yes |
| Qwen3-4B | `qwen3-4b_ab` | 1.02% | none | none | none | |
| Qwen3-0.6B | `qw06_cont` | 2.09% | none | none | none | yes |
| Qwen3-0.6B | `qw06_lr2e-3` | 1.95% | none | none | none | |
| SmolLM2-360M | `sl360_lr4e-3` | 2.01% | 0.018 | 0.073 | 0.946 | yes |
| SmolLM2-360M | `sl360_lr2e-3` | 2.65% | 0.059 | 0.073 | 0.942 | |
| SmolLM2-360M | `sl360_lr1e-3` | 3.91% | 0.137 | 0.073 | 0.941 | |

The 135M references are 37 architectures, the 360M references 48 and the 1.7B references 20, each
trained for 2000 steps. The 360M references come from a training run that fails the identity check,
so the 360M regret numbers are provisional. At 360M the violations, the worst gap and the regret all
fall as the supernet learning rate rises, the same trend as at 135M. The highest learning rate picks
better than the largest architecture that fits by a factor of four, the middle one beats that rule
narrowly, and the lowest loses to it.

No dedicated-training references exist at 4B or 0.6B, so those choices rest on nesting. At 4B the kept
checkpoint has a fifth of the violations, a quarter of those against the smallest slice, and a smaller
worst gap, 0.24 against 0.39 nats. It was also trained with the same layer blocks the 4B search uses.
At 0.6B both checkpoints violate nesting on about 2% of pairs by at most 0.045 nats, so the full-width
loss decides. The kept checkpoint scores 3.06 nats at full width against 3.25, and its loss follows
size more closely, a Spearman correlation of -0.90 against -0.86.

### Early result: heterogeneity without hardware, 2026-09-14

On SmolLM2-135M with the kept supernet, the parameter-proxy NSGA-II front was compared with the uniform
grid under the same objectives, documents and anchor box. This is software-only and provisional.

| front | evaluations | architectures | HV, search | HV, held-out |
|---|---|---|---|---|
| heterogeneous NSGA-II | 832 | 82 | 0.698 | 0.596 |
| uniform grid | 192 | 30 | 0.655 | 0.552 |

On held-out documents the heterogeneous front reaches a median 0.012 nats lower loss at equal
parameters, 0.055 at best, and a median 2.5% fewer parameters at equal loss, 10.7% at best. Two of
the uniform front's points tie or lose slightly. The held-out gap in hypervolume matches the search
gap, so the advantage is not fit to the search documents. The comparison does not separate
heterogeneity from search budget, since NSGA-II saw four times as many architectures. Gains of 0.01 to
0.02 nats are also close to the supernet's ranking noise, which the ground-truth phase has to settle.

The FLOPs-proxy front on the same supernet shares only 4 of its 79 architectures with the
parameter-proxy front, so the two hardware-agnostic baselines pick different architectures. Both fronts
give later blocks more capacity. Averaged over front architectures, the MLP width is about 40% of its
maximum in the first block and about 75% in the last, and the attention widths are smallest in the first
two blocks. These averages mix front composition with allocation, so they describe the fronts rather
than establish a rule.

At Qwen3-1.7B the same comparison shows a larger gap.

| front | evaluations | architectures | HV, search | HV, held-out |
|---|---|---|---|---|
| heterogeneous NSGA-II | 832 | 60 | 0.687 | 0.668 |
| uniform grid | 128 | 20 | 0.610 | 0.591 |

On held-out documents the heterogeneous front reaches a median 0.11 nats lower loss at equal
parameters and a median 4.1% fewer parameters at equal loss. The gain grows as models shrink, from 0.04
to 0.19 nats between 1.46B and 1.02B parameters to 0.65 to 1.27 nats below 1B. Every front architecture
below 1.2B keeps the first attention block, layers 0 to 4, at or near full width and shrinks attention in
the middle blocks instead. A uniform slice has no such option, since shrinking attention shrinks layer 0
as well, and earlier sensitivity measurements put most of this model's attention damage in layer 0. The
135M front shows no such protection of the first block, consistent with layer 0 being far less
sensitive at SmolLM2 scale. The size of the small-model gains still needs dedicated training to
confirm, because the 1.7B dedicated-training references stop at about 1.49B parameters and a supernet
slice can fail in ways a trained model would not. One uniform point near full width, at 1632M
parameters, beats the heterogeneous front, which has no architecture between 1506M and full width.

The 1.7B FLOPs-proxy front shares only 4 of its 63 architectures with the parameter-proxy front. Below
1.2B it also keeps the first attention block at full head count, but it protects the width of those heads
less and keeps more heads in the second block. Averaged over those architectures, the first block keeps
68% of the query and key width and 78% of the value width, against 80% and 92% on the parameter-proxy
front. FLOPs charge attention computed over the prompt, which grows with heads times width, so the FLOPs
proxy spends attention width differently. Both proxies cut the middle block, layers 11 to 15, the most.

At Qwen3-4B the heterogeneous front again beats the uniform grid, by less than at 1.7B.

| front | evaluations | architectures | HV, search | HV, held-out |
|---|---|---|---|---|
| heterogeneous NSGA-II | 832 | 103 | 0.688 | 0.658 |
| uniform grid | 192 | 38 | 0.659 | 0.627 |

On held-out documents the median gain is 0.049 nats at equal parameters, 0.24 at best, and the median
saving is 2.4% of parameters at equal loss. Two uniform points win, one by 0.008 nats at 3.11B and one by
0.088 nats at 1.25B, where the heterogeneous front has no architecture between 1.39B and the smallest
slice. The 4B space uses measured sensitivity blocks, so layer 0 and layers 22 to 24 form their own
attention blocks and the last two layers their own MLP block. On front architectures below the median
size of 2.46B, layer 0 keeps 85% of its heads, 90% of its value width and 85% of its MLP width, and the
last two layers keep 60% of their MLP width. The 30-layer middle MLP block keeps 31%, the heads of
layers 12 to 21 keep 41%, and layers 22 to 24 keep 73% of their heads. Earlier sensitivity measurements
place this model's damage in the same layers, and the search arrives at that pattern from slice scores
alone.

The 4B FLOPs-proxy front shares only 3 of its 109 architectures with the parameter-proxy front, so the
two proxies pick near-disjoint architectures at every scale. Below 2.46B both keep the MLP of layer 0
wide and cut the long middle blocks hardest. The FLOPs front protects the attention of layer 0 less, 78%
of its heads and 76% of its value width, and keeps 93% of the query and key width of layers 22 to 24,
against 63% on the parameter-proxy front. It also cuts the MLP of the last two layers to 46% against
60%. Measuring both fronts on the GPU will show which cost model tracks energy.

### Early result: search on the Pixel Watch 5 space, 2026-09-14

The watch space is the 192 uniform architectures of SmolLM2-135M, so the grid gives the true front, 26
architectures at a normalized hypervolume of 0.871. Each NSGA-II seed recovered 25 or 26 of them.
Because a seed's 174 evaluations cover most of the space, that shows correctness rather than
efficiency. At equal evaluation counts, NSGA-II was compared with random subsets of the grid of the same
size, anchors included, over 2000 draws.

| evaluations | NSGA-II, mean of 3 seeds | random, mean | random, 5th to 95th percentile |
|---|---|---|---|
| 16 | 0.748 | 0.721 | 0.605 to 0.813 |
| 32 | 0.825 | 0.786 | 0.705 to 0.846 |
| 64 | 0.859 | 0.831 | 0.787 to 0.863 |
| 128 | 0.871 | 0.860 | 0.827 to 0.870 |

NSGA-II leads at every budget, by 0.03 to 0.04 between 32 and 64 evaluations, but a random subset of
equal size matches the NSGA-II mean in 10% to 21% of draws at most budgets. Three seeds and a small
space make this weak evidence of efficiency. The random baseline on the heterogeneous GPU space tests
it properly.

### Early result: cost models against three targets, 2026-09-14

All 192 uniform SmolLM2-135M architectures now have GPU prefill and GPU decode energy measured and
Watch energy predicted, with identical slice losses on every target. Prefill energy spans 2.37x and
decode energy 1.43x. Decode uses the grouped schedule. The spread across the three decode windows of one
architecture has a median of 1.5%, and one step up in any knob raises decode energy in 96% to 97% of
cases. Three architectures, all with 9 query heads of dimension 16, spread by more than 20% across
windows, and the worst sits 11% below the calibrated decode model below. The independent second decode
grid in Phase 6 will show whether they reproduce.

Rank correlation of each cost model with each target over the 192 architectures. The fitted rows are
5-fold cross-validated predictions of a linear model on parameters and KV cache size, and of a log-linear
model on parameters and the INT8 group size of the watch runtime.

| cost model | GPU prefill | GPU decode | Watch |
|---|---|---|---|
| parameters | 0.997 | 0.666 | 0.695 |
| FLOPs per token | 0.997 | 0.678 | 0.693 |
| decode MACs | 0.962 | 0.762 | 0.717 |
| KV cache size | 0.150 | 0.816 | 0.428 |
| fitted, parameters and KV cache | 0.998 | 0.979 | 0.754 |
| fitted, parameters and group size | 0.997 | 0.709 | 0.997 |
| GPU prefill energy | 1 | 0.639 | 0.677 |
| GPU decode energy | 0.639 | 1 | 0.684 |

Each cost model then picked its own front, judged on each target at matched loss. Each cell lists the
median extra cost against the target-optimal architecture of equal loss, then the share of front points
paying more than 5%.

| chooser | GPU prefill | GPU decode | Watch |
|---|---|---|---|
| parameters | 0%, 0% | 3%, 29% | 20%, 62% |
| FLOPs per token | 0%, 0% | 3%, 29% | 19%, 54% |
| decode MACs | 0%, 27% | 1%, 14% | 20%, 62% |
| KV cache size | 44%, 88% | 3%, 29% | 103%, 85% |
| fitted, parameters and KV cache | 0%, 0% | 0%, 0% | 28%, 62% |
| fitted, parameters and group size | 0%, 0% | 3%, 29% | 0%, 0% |
| GPU prefill energy | 0%, 0% | 3%, 29% | 10%, 50% |
| GPU decode energy | 15%, 71% | 0%, 0% | 25%, 62% |
| Watch energy | 3%, 32% | 1%, 19% | 0%, 0% |

Each target has its own cost structure. Prefill energy follows parameter count, so any size proxy
already chooses well. Decode energy follows parameters and KV cache size together, which no single proxy
captures. A linear model on both, calibrated on the measured architectures, ranks decode energy at 0.979
with a median error of 0.7% and chooses nearly as well as the measurement. Watch energy follows
parameter count and the INT8 group size of the runtime, which only the second fitted model knows. No
hardware-agnostic proxy serves all three targets, and choices do not transfer between them. The
decode-optimal architectures cost a median 15% more prefill energy and 25% more watch energy than the
optimal ones of equal loss. At 135M the decode range is narrow, so the size proxies lose only a median 3%
there. The 1.7B and 4B decode grids test whether that gap grows with their wider range. Watch energy comes
from the fitted predictor, with the limits described under the Watch target.
`experiments/hw_nas/proxy_fidelity.py` regenerates these tables with
`--fit params_M+kv_cache_MB --fit log:params_M+int8_group_64+int8_group_16`.

The interleaved decode grid over the same architectures reads a median 6.1% higher than the grouped
one, and the two rank architectures with a correlation of 0.90. Its excess does not track prefill power,
so the schedule shifted the level and added noise without favoring any size.
`experiments/hw_nas/pilots/compare_decode_schedules.py` reproduces the comparison.

### Early result: cost models at Qwen3-1.7B, 2026-09-14

All 128 uniform Qwen3-1.7B architectures have GPU prefill and decode energy measured. Prefill energy
spans 2.88x and decode energy 2.33x. One decode measurement, cached a second before a CUDA fault, ran
uniformly slow and read 48% above the calibrated model. It was quarantined and measured again. After
that every one-step increase of any knob raises decode energy, and no architecture sits more than 5.2%
from the calibrated model.

| cost model | Spearman with prefill | Spearman with decode | decode chooser, median extra and share over 5% |
|---|---|---|---|
| parameters | 0.997 | 0.846 | 0%, 14% |
| FLOPs per token | 0.996 | 0.846 | 0%, 14% |
| decode MACs | 0.995 | 0.863 | 0%, 14% |
| KV cache size | 0.198 | 0.667 | 1%, 43% |
| fitted, parameters and KV cache | 0.996 | 0.992 | 0%, 0% |
| GPU prefill energy | 1 | 0.835 | 1%, 36% |

The structure found at 135M holds at 1.7B. Prefill energy follows parameter count. Decode energy follows
parameters and KV cache size together, which the calibrated model captures with a median error of 1.2%.
Parameter count ranks decode better at 1.7B than at 135M, 0.85 against 0.67, and on the uniform grid its
choices pay no median extra decode energy, though 14% of them pay more than 5%. The decode-optimal
architectures cost a median 12% more prefill energy than the prefill-optimal ones of equal loss. Measuring
both proxy fronts on decode tests next whether size proxies misjudge decode on heterogeneous
architectures.

Both proxy fronts, searched without hardware and then measured on the GPU, beat the uniform grid on
prefill energy at this scale.

| front | architectures measured | HV, search | HV, held-out | held-out energy saving at matched loss, median | held-out loss gain at matched energy, median |
|---|---|---|---|---|---|
| uniform grid | 128 | 0.619 | 0.599 | | |
| parameter proxy, measured | 60 | 0.693 | 0.673 | 5.9% | 0.136 nats |
| FLOPs proxy, measured | 63 | 0.698 | 0.677 | 3.8% | 0.123 nats |

The loss gain is in line with the 0.11 nats that slice scores gave at equal parameter count, so measured
energy confirms the heterogeneity result at 1.7B. The proxy searches evaluated 832 architectures against
128 for the grid, and slice scores decide every loss, so the ground-truth pair still has to show that the
gain survives training.

### Early result: the first search with measured hardware, 2026-09-14

NSGA-II on SmolLM2-135M with GPU prefill energy in the loop evaluated 832 architectures in 106
minutes. All were feasible, the spread across repeats stayed at or below 9.2%, and every window lasted
at least 2 s.

| front | evaluations | architectures | HV, search | HV, held-out |
|---|---|---|---|---|
| heterogeneous NSGA-II | 832 | 86 | 0.671 | 0.568 |
| uniform grid | 192 | 41 | 0.623 | 0.524 |

On held-out documents the searched front uses a median 2.7% less energy at equal loss, 9.5% at best,
and reaches a median 0.012 nats lower loss at equal energy. It shares only 4 architectures with the
parameter-proxy front, yet both fronts allocate capacity the same way.

The 832 measured architectures also test whether parameter count keeps predicting prefill energy away
from uniform architectures. It does. The Spearman correlation is 0.998, and a linear fit in parameters
leaves a median residual of 0.6% of energy and a 95th percentile of 1.9%, about the spread across
repeats. FLOPs do nearly as well, decode MACs leave 6.3% at the 95th percentile, and KV cache size leaves
25%. A parameter-proxy search should therefore match the measured search on GPU prefill. This target
shows that the framework runs with real measurements in the loop, but not that measurement beats a
proxy. The Watch target and the decode grid test that.

Both proxy searches, run without any hardware and then measured on the GPU, confirm it.

| front | architectures measured | HV, search | HV, held-out | held-out energy of the measured search at matched loss, median |
|---|---|---|---|---|
| measured NSGA-II | 832 | 0.671 | 0.568 | |
| parameter proxy, measured | 82 | 0.671 | 0.570 | 0.7% more |
| FLOPs proxy, measured | 79 | 0.668 | 0.565 | 0.6% more |

All three fronts use the same anchor box. Each proxy search took 21 minutes of slice scoring plus 8
minutes to measure its front, against 106 minutes for the measured search.

## Hardware targets

### GPU, measured with ZEUS

Each architecture is built as a vendored GPT model with random weights and measured on one NVIDIA
H100 80GB in bf16. The measurement is described in `docs/hw_gpu.md`. The search objective is prefill
energy per prompt token for a batch of 64 prompts of 512 tokens. Every window repeats the prefill
until it lasts at least 2 s, and the median of three windows is reported with the time to first
token. Decode energy per generated token is measured on uniform grids and on searched fronts, with 128
decode steps replayed from CUDA graphs. All prefill windows run first, then 1.5 s of untimed decode,
then three decode windows of at least 6 s.

Three pilots chose this protocol, reproducible with `experiments/hw_nas/pilots/zeus_protocol.py`. The
table lists energy per token in millijoules for the full and the smallest uniform slice, then the
largest spread across three repeats.

| protocol | SmolLM2-135M | Qwen3-1.7B | Qwen3-4B |
|---|---|---|---|
| eager decode, 128 steps | 28.1 / 25.6, 13% | 48.6 / 28.9, 24% | 68.1 / 46.1, 12% |
| CUDA-graph decode, 2 s windows | 8.98 / 6.61, 10% | 31.7 / 15.2, 2% | 57.6 / 23.3, 8% |
| prefill, 2 s windows | 0.750 / 0.323, 9% | 4.60 / 1.60, 2% | 10.7 / 3.14, 2% |

Eager decode is bound by kernel launch overhead. Time per output token stays between 11 and 15 ms at
every scale, and energy moves by only 1.1x to 1.7x. Replaying decode steps from CUDA graphs removes
that overhead and matches eager decoding bit for bit. Time per token then depends on the
architecture, 3.2 to 4.9 ms at 1.7B, but the full slice of 135M costs only 1.36x the smallest.
Prefill energy has the widest range, 2.3x at 135M, 2.9x at 1.7B and 3.4x at 4B. Three independent
passes agree within 1.6% at every scale for prefill, and within 4.6% for graph decode. Windows shorter
than about one second read with up to 40% spread, so every window repeats its pass to 2 s. Two random
heterogeneous architectures per scale landed between the uniform slices of neighboring size under
every protocol.

The choice between graph decode and prefill energy followed criteria written before the graph-decode
pilots. Graph decode had to span 1.5x at 135M and 2x at 1.7B and 4B, keep repeat spread within 10%,
and take about 25 s per architecture at 4B. It missed the 135M ratio and the time budget, 20 to 42 s
at 4B, so the search optimizes prefill energy and graph decode energy is reported for the fronts.
Every measurement holds a cross-process GPU lock, and the queue runs one job at a time.

The first decode grid measured a prefill window and then a decode window in every repeat, as the
pilots did. Its decode windows spread by a median 6% across repeats at 135M, almost twice the prefill
spread. A second pilot, `experiments/hw_nas/pilots/zeus_decode_schedule.py`, ran alongside it with
criteria written before it ran. The reference was three 6 s decode windows after 1.5 s of untimed
decode. Against it the interleaved schedule read 2.8% to 5.4% high on all four architectures measured.
Grouped 2 s windows after the same untimed decode came within 2.6% but missed the 2% tolerance on the
heterogeneous architecture, so decode measurements use the 6 s windows. Their values agree within 1.2%
across independent calls, at 13 to 14 s more per architecture. `docs/hw_gpu.md` lists the pilot table.
The interleaved grid is kept as `gpu-decode-interleaved__smollm2-135m__grid`, and every decode result
below uses the grouped schedule.

**Batched decoding at Qwen3-1.7B and Qwen3-4B.** The GPU searches of the paper optimize loss, energy per
generated token and TPOT under the decode protocol above, as `gpu-serve` runs. In this workload decode takes a
median 78% of request time and 66% of request energy at Qwen3-1.7B, and 91% and 78% at SmolLM2-135M. Energy per
token and TPOT rank the uniform grids with Spearman correlations of 0.955 at Qwen3-1.7B and 0.964 at SmolLM2-135M,
so every record also keeps TTFT for a later check. Qwen3-4B runs on a second worker with the same GPU model and
driver, from `queue_gpu4b.yaml`, with a device label that keeps its measurement caches apart.

### Pixel Watch 5, fitted predictor

The predictor was fitted to on-device measurements of uniform architectures on the nanollmforge.c
runtime, see `docs/hw_device.md`. It returns decode throughput, time to first token and dynamic
energy per generated token for 49 prompt and 32 generated tokens. It accepts uniform architectures
only, so watch searches run on the uniform space of SmolLM2-135M. That model is the only supernet
inside the predictor's 50M to 150M training band. Its 30 layers exceed the 28-layer maximum of the
training data and its vocabulary differs from the 50257 tokens of every training architecture, so
its predictions extrapolate. `device_in_domain` flags that on every record.

An audit of the predictor on the 135M grid explains its irregular costs and bounds what they show. The
predicted energy per token is not monotone in the knobs that set attention width. One step up in head
count lowers it in 38% of cases, by up to 2.7x, and one step up in value head dimension lowers it in
33%, by up to 2.8x. The jumps follow the INT8 group size the runtime derives from the architecture, 64
when the model, attention and MLP widths are all multiples of 64 and 32 or 16 otherwise. The measured
data show the same effect directly. Within each model width, architectures with group size 64 decode
at 0.35 to 0.41 times the throughput of those with group size 32 and use 2.3 to 3.2 times the dynamic
energy per token, after controlling for parameter count and depth. The irregularity is therefore a
property of the watch runtime that no size-based proxy sees. On the grid, a log-linear model on
parameter count and group size, with 5-fold cross-validated predictions, ranks the predicted energy
with a Spearman correlation of 0.997, against 0.689 for parameter count alone. Two limits remain. The measured data
contain query head counts of 1, 2, 4, 6, 8, 12 and 16 only, so the 3 and 9 heads of two thirds of the
grid never appear in them, and `device_on_support` flags such architectures. The shipped predictor's
validation error for energy is 43.6% mean absolute percentage error. Watch results therefore support
the direction and rough size of cost differences, not exact costs.

### Calibrated cost models

A target can also enter a search through a cost model fitted to measurements of other architectures.
`experiments/hw_nas/fit_cost_model.py` fits a linear model on analytic metrics, or a log-linear one with
the `log:` prefix, to a run that measured the target and reports its 5-fold cross-validated error.
`--hw fitted --hw-arg model=PATH` then serves the model to a search, with the objective
`fitted_<cost key>`. The derived indicators `int8_group_64` and `int8_group_16` carry the INT8 group
size that the watch runtime derives from an architecture. A search against a calibrated model measures
only the sample that fits it and the front it returns. It is the baseline that tells whether a target
needs its measurement in the loop or only a calibration.

### Simulators

Timeloop substrates and the rDXE ring accelerator run on a separate worker that has the Timeloop toolchain of
`scripts/setup/install_timeloop.sh`, from `queue_sim.yaml`. No GPU measurement runs on that worker, so the mapper's
CPU load never overlaps an energy window.

| setting | value |
|---|---|
| base models | SmolLM2-135M (`sl135_lr8e-3`) and SmolLM2-360M (`sl360_lr4e-3`), then Qwen3-0.6B (`qw06_cont`) if time allows |
| Timeloop substrates | `gemmini_16nm`, `eyeriss` and `flat_edge`, which share one technology setting |
| rDXE | ring co-search on `dxe_relaxed`, chip selected by energy per token inside the area and power envelope, context 2048, one user |
| workload | 128 prompt tokens and 32 generated tokens, the defaults of `configs/hw/timeloop.yaml` and `configs/hw/rdxe.yaml` |
| objectives | loss, energy per generated token and TPOT |
| S1 | uniform grid of each base model on each target, 192 architectures |
| S2 | NSGA-II, population 32 for 25 generations, on each base model and target |

The uniform grids run first. They show whether energy and latency rank architectures differently on each
substrate before the searches spend their budget, and they give every search its uniform baseline.

## Algorithms and budgets

| algorithm | configuration | evaluations |
|---|---|---|
| NSGA-II | population 32, 32 children per generation, 25 generations | 832 |
| random | 832 distinct random architectures | 832 |
| grid | every uniform architecture | 128 or 192 |
| proxy | NSGA-II as above against parameters or FLOPs, then its front measured on the target | 832 plus the front |
| decode grid | CUDA-graph decode energy of every uniform architecture at 135M, 1.7B and 4B | 192, 128 and 192 |
| decode re-measurement | CUDA-graph decode energy of the NSGA-II and proxy fronts | front size |
| watch NSGA-II | population 16, 10 generations, uniform space | 176 |

Every run first evaluates the full and the smallest architecture. Their objective values fix the box
that normalizes hypervolume, so runs on one scale and target share one scale.

## Ground-truth phase, planned

Slice scores decide every comparison above, so this phase trains a few searched architectures as
dedicated models and compares their trained loss with uniform architectures of equal size. A
heterogeneous architecture has no dense form, so both arms train in place through the elastic forward
with their own configuration pinned, using `experiments/supernet_fidelity/groundtruth_het.py`. Both are
scored on the held-out documents with `experiments/supernet_fidelity/eval_het.py`.

- The first pair is at Qwen3-1.7B near 1B parameters, where the slice scores differ most. The
  heterogeneous front architecture at 998M scores 3.99 nats on held-out documents and the uniform front
  architecture at 986M scores 4.64.
- Both arms train for 2000 steps on FineWeb-Edu with distillation from the frozen base model. A
  learning-rate probe at Qwen3-0.6B, run with dense training, found 3e-4 best, 0.30 nats ahead of 1e-3
  and 0.81 nats ahead of the 3e-3 used for in-place training at 135M, so the Qwen runs use 3e-4. At 135M
  the in-place trainer took as long as dense training, 0.65 against 0.63 hours, so a 1.7B pair should
  take about 5 hours.
- Ten in-place training steps at 1.7B without gradient checkpointing ran at about 4.3 s per step after
  the first and peaked at 43.4 GiB, so the pair trains without checkpointing, about 2.4 hours per arm. A
  dense 1.7B reference took 4.2 s per step. With checkpointing the same steps peaked at 25.9 GiB but ran
  about 14% slower and reached the same loss, so checkpointing is only worth it when memory binds.
- The pair is queued after the 1.7B GPU phase. One scoring job then evaluates both arms on the
  192-document held-out set, which tightens the paired comparison about 1.5x against the 96-document set.
  That set includes the 32 search documents, which can favor slice scores but not trained losses.
- At SmolLM2-135M, `experiments/hw_nas/select_groundtruth.py` picks architectures at matched GPU cost
  from the NSGA-II, uniform and proxy fronts once the GPU runs finish.

## Run matrix

The queue in `queue.yaml` runs in this order.

| phase | content | estimated GPU hours |
|---|---|---|
| 0 | supernet screening, uniform grid per candidate checkpoint, software only | 2 |
| 1 | proxy searches against parameters and FLOPs at 135M, 1.7B and 4B, software only | 2.5 |
| 2 | Pixel Watch 5 on SmolLM2-135M: uniform grid and three NSGA-II seeds | 0.3 |
| gate | holds the queue until the GPU protocol is decided | none |
| 3 | SmolLM2-135M on the GPU: prefill grid, NSGA-II and both proxy fronts, then the decode grid | 4.5 |
| 4 | Qwen3-1.7B on the GPU: decode and prefill grids, both proxy fronts on both targets | 3.5 |
| ground truth | Qwen3-1.7B pair trained in place and scored | 5 |
| 4, continued | Qwen3-1.7B NSGA-II on prefill, then its front and the 135M front on decode | 4.5 |
| 5 | Qwen3-4B on the GPU: decode and prefill grids, both proxy fronts on both targets, NSGA-II on prefill, its front on decode | 11 |
| 6 | random-sampling baseline on SmolLM2-135M and a second, independent 135M decode grid | 3.5 |
| 7 | Qwen3-0.6B: proxy searches, prefill grid, NSGA-II and proxy fronts, fronts on decode | 6 |
| 8 | SmolLM2-360M, same set as Phase 7 | 6 |
| 9 | NSGA-II seeds 1 and 2 at 135M, 1.7B and 4B, random at 1.7B | 20 |

A search with decode energy in the loop at Qwen3-1.7B joins the queue after Phase 4 if the decode grid
and the proxy fronts measured on decode show parameter count and FLOPs misranking decode cost.

Evaluations are cached by architecture across runs, so later seeds reuse every architecture an
earlier run already scored or measured.

Two more workers run `queue_gpu4b.yaml`, Qwen3-4B on the GPU, and `queue_sim.yaml`, the simulator targets.

## Metrics

- Normalized hypervolume of each front inside the anchor box, on search documents and on held-out
  documents.
- Cost saving at matched loss: for each point of a baseline front, the lowest cost the searched front
  reaches at equal or lower loss.
- Loss gain at matched cost, defined the same way along the other axis.
- Hypervolume against evaluation count for NSGA-II and random sampling.
- Cross-target transfer on the uniform space: the architecture that is optimal for one target,
  evaluated on the other.
- Cost-model fidelity per target: the Spearman correlation of each proxy with the measured cost, and
  the extra cost each proxy's front pays at matched loss. A measured front is selected on noisy values,
  so with an independent re-measurement every front is judged on the re-measured values.
- Per-block allocation of the front architectures, compared between targets and proxies.

## Reproduce

```bash
python -m llmforge.search.elastic_space --write-configs configs/search_spaces
python experiments/hw_nas/pilots/zeus_protocol.py --out runs/pilots/zeus_protocol.jsonl
python experiments/hw_nas/queue.py experiments/hw_nas/queue.yaml
python experiments/hw_nas/queue.py experiments/hw_nas/queue.yaml --status
python experiments/hw_nas/screen_supernets.py runs/search/screen__* \
    --refs smollm2-135m=assets/groundtruth/smollm2-135m__ctx1024.json \
    --refs smollm2-360m=assets/groundtruth/smollm2-360m.json \
    --refs qwen3-1.7b=assets/groundtruth/qwen3-1.7b__base_init.json
python experiments/hw_nas/analyze.py --model smollm2-135m --target gpu
# Runs with several cost objectives also report the rank correlation between costs and each cost's winners.
python experiments/hw_nas/analyze.py --model qwen3-1.7b --target gpu-serve
python experiments/hw_nas/analyze.py --model smollm2-135m --target tl-gemmini
python experiments/hw_nas/proxy_fidelity.py --label smollm2-135m \
    --target gpu-prefill=runs/search/gpu__smollm2-135m__grid \
    --target watch=runs/search/watch__smollm2-135m__grid
python experiments/hw_nas/proxy_fidelity.py --label smollm2-135m-decode \
    --target gpu-prefill=runs/search/gpu__smollm2-135m__grid \
    --target gpu-decode=runs/search/gpu-decode__smollm2-135m__grid \
    --target watch=runs/search/watch__smollm2-135m__grid \
    --retest gpu-decode=runs/search/gpu-decode-retest__smollm2-135m__grid \
    --fit params_M+kv_cache_MB
python experiments/hw_nas/fit_cost_model.py --run runs/search/gpu-decode__qwen3-1.7b__grid \
    --fit params_M+kv_cache_MB --out runs/cost_models/gpu-decode__qwen3-1.7b__params-kv.json
```

Supernet checkpoints are expected under `runs/supernet/`, named as in the `supernets` block of
`queue.yaml`. A rerun of an interrupted job replays its cached evaluations exactly and continues.
