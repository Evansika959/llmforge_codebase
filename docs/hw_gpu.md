# GPU target: measured latency and energy

The GPU target measures what an architecture costs to run on a local NVIDIA GPU. The evaluator is
`llmforge.evaluators.hw_zeus` and the measurement is `llmforge.hw.zeus.measure.measure_one`. ZEUS reads
device energy from the NVML cumulative energy counter around each measurement window.

## What is measured

| item | setting |
|---|---|
| model | ReaLLM-Forge GPT with Infinite Head Attention, built from the Individual with random weights |
| precision | bf16 on GPU 0 |
| per-layer shape | query heads, KV groups, query and key head dimension, value head dimension, SwiGLU MLP width |
| fixed structure | RMSNorm, no biases, heads concatenated before the output projection, LM head tied to the embedding over the base vocabulary |
| omitted | rotary embeddings and QK-Norm, both elementwise operations without weights |
| search objective | prefill of 64 prompts of 512 tokens, `prefill_energy_per_token_uJ` |
| decode | the same prefill, then 128 generated tokens per prompt replayed from CUDA graphs, `energy_per_token_uJ`, measured on uniform grids and searched fronts |
| window | a prefill window repeats its pass until it lasts at least 2 s, a decode window at least 6 s |
| decode schedule | all prefill windows first, then 1.5 s of untimed decode, then the decode windows |
| warmup | one full pass of each measured phase, which also calibrates the repeat counts |
| repeats | 3 windows per phase, median reported |

Weights do not change which kernels run, so random weights measure the same compute as trained ones.

## Metrics

| key | definition |
|---|---|
| `ttft_ms` | prefill window time divided by the prefill passes in the window |
| `prefill_energy_per_token_uJ` | prefill window energy divided by passes times batch times prompt length |
| `prefill_power_W` | mean device power over the prefill window |
| `tpot_ms` | decode window time divided by replayed decode steps |
| `energy_per_token_uJ` | decode window energy divided by generated tokens, passes times batch times steps |
| `session_energy_per_token_uJ` | energy of one prefill and one decode pass over every processed token |
| `dynamic_energy_per_token_uJ` | decode energy above the idle power measured at start, recorded but not used |
| `zeus_prefill_energy_cv`, `zeus_repeats_energy_cv` | spread of prefill and decode energy across the three windows, relative to the median |
| `zeus_prefill_passes`, `zeus_decode_passes` | passes repeated inside one window |
| `hw_feasible` | False when the build or the measurement failed, for example out of memory. Any other CUDA fault stops the job instead, and the queue reruns it from the caches |

Dynamic energy is not used as an objective. Idle power read shortly after GPU work stays elevated for
seconds, so the subtracted baseline depends on what ran before.

## Decode implementation

The KV cache is allocated once per prefill with room for the prompt and every generated token. Each
decode step writes one position in place and attends over a view of the filled prefix. Attention uses
native grouped-query attention, so key and value heads are never copied per query head. Infinite Head
Attention assigns query heads to KV groups in contiguous blocks, which is the grouping native
grouped-query attention assumes, so the cached path computes the same attention as the model. In fp32
the final logits of cached decoding match one uncached forward pass to about 5e-6.

Decode steps can be replayed from CUDA graphs, one graph per position, captured after a prefill. Each
graph writes its position of the preallocated cache and attends over the positions before it, so
replaying the graphs in order reproduces cached decoding without per-kernel launch overhead, as
serving engines do. Replayed hidden states match eager cached decoding bit for bit on uniform and
heterogeneous architectures at 135M, 1.7B and 4B. Replaying the graphs again rewrites the same
positions with the same tokens, which lets a window repeat the decode pass without a new prefill.
`tests/hw_gpu/test_zeus_kv_cache.py` checks both equivalences.

Capture warms up on a side CUDA stream, so it first waits for the work queued on the current stream, the
prefill that fills the cache and the draw of the decode tokens. An earlier version skipped that wait. A
long prefill then let the warmup read token ids that were not yet written, and the process failed with a
device-side assert, after about 190 architectures at 135M and about 30 at 1.7B. The race could only reach
warmup outputs, since capture and every measured window run after the streams synchronize. The GPU was
already degraded shortly before one fault, though. The 1.7B architecture measured one second before it
ran uniformly slow, its time to first token 60% and its time per output token 57% above one-knob
neighbors, with a tight window range that no noise check flags. That entry was moved to
`runs/cache/hw/quarantine/` and measured again. After any fault, the last cached entries are
quarantined and re-measured, and each grid is checked against a calibrated cost model before use.

An earlier implementation grew the cache by copying it at every step and expanded key and value heads
for every step. At batch 64 and a 640-token context those copies cost more than attention itself. A
4B-shaped layer spent about 1.25 ms and 672 MB of transient memory on the expansion alone, against
0.07 ms for native grouped-query attention. Measurements from the two implementations are not
comparable. The evaluator records the implementation in its cache setting as `method`, so entries
from different implementations never mix.

## Protocol choice

Three pilots on SmolLM2-135M, Qwen3-1.7B and Qwen3-4B chose the protocol,
`experiments/hw_nas/pilots/zeus_protocol.py` reproduces them and `experiments/hw_nas/README.md` lists
the numbers. Eager decode is bound by kernel launch overhead at every scale, so its cost barely
depends on the architecture. CUDA-graph decode depends on the architecture but spans only 1.36x
between the full and the smallest 135M slice. Prefill energy spans 2.3x to 3.4x and three independent
passes agree within 1.6%. Windows shorter than about one second read with up to 40% spread across
repeats, which is why every window repeats its pass to at least 2 s.

## Decode schedule

The protocol pilots measured one prefill window and then one decode window in every repeat. A decode
window measured that way starts right after prefill, which draws two to three times the decode power.
A second pilot, `experiments/hw_nas/pilots/zeus_decode_schedule.py`, found that such windows read high
and that untimed decode before the windows removes most of the excess. It measured four architectures
under four schedules, three independent calls each, against decision criteria written before it ran.
Each cell lists the median decode energy against the reference schedule, then the median spread
across the windows of one call.

| schedule | 135M full | 135M smallest | 135M heterogeneous | 1.7B smallest | time added at 135M |
|---|---|---|---|---|---|
| interleaved, 2 s decode windows | +4.8%, 0.102 | +5.4%, 0.006 | +2.8%, 0.138 | +3.4%, 0.079 | none |
| interleaved, 1.5 s untimed decode before each window | +0.3%, 0.042 | +1.9%, 0.047 | not run | not run | 5 s |
| grouped, 1.5 s untimed decode, 2 s windows | +0.2%, 0.046 | +1.5%, 0.044 | +2.6%, 0.049 | -1.1%, 0.045 | 2 s |
| grouped, 1.5 s untimed decode, 6 s windows | reference, 0.015 | reference, 0.017 | reference, 0.013 | reference, 0.019 | 13 to 14 s |

The grouped schedule with 2 s windows missed the 2% tolerance on the heterogeneous architecture, so
decode measurements use the grouped schedule with 6 s windows. Its values agree within 1.2% across
independent calls. Prefill-only measurements have no decode window, so the schedule leaves them
unchanged. The schedule, the untimed decode and the decode window length join the cache setting
whenever they differ from the interleaved default, so measurements from the two schedules never mix.

## Isolation and caching

Every measurement holds a cross-process lock on GPU work, `scratch/locks/gpu.lock`. The experiment
queue runs one job at a time, so a measurement window never overlaps other kernels. Results are cached
under `runs/cache/hw` per architecture. The file name encodes the GPU model, prompt and decode lengths,
batch size, precision, repeat count, CUDA-graph decode, window length and any non-default decode
schedule, plus a digest of every setting including `method`. Individual cards of the same model differ
slightly in energy, so a campaign that measures on several GPUs gives each device a label through
`LLMFORGE_DEVICE_LABEL` or `--hw-arg device_label=NAME`, and each device then writes its own cache. Use a
neutral name such as `gpu-b` rather than a host name.

## Reproduce

```bash
python -m llmforge.hw.zeus.measure individual.json --prefill-len 512 --decode-len 0 --batch-size 64
python -m llmforge.hw.zeus.measure individual.json --prefill-len 512 --decode-len 128 --batch-size 64 --cuda-graphs \
    --schedule grouped --settle-s 1.5 --decode-window-s 6
python experiments/hw_nas/pilots/zeus_protocol.py --out runs/pilots/zeus_protocol.jsonl
python experiments/hw_nas/pilots/zeus_decode_schedule.py
python experiments/hw_nas/pilots/zeus_decode_schedule.py --analyze
pytest tests/hw_gpu
```

## Limitations

- Prefill runs in eager PyTorch, so kernel launch overhead is part of its cost. At batch 64 with a
  512-token prompt it is small against the compute.
- Rotary embeddings and QK-Norm are not in the measured graph.
- Energy covers the GPU only. Host CPU energy is not measured.
