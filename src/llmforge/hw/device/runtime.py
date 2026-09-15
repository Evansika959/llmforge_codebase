"""Locations shared by the measurement harness and the active-learning hardware adapter.

The inference runtime is a separate checkout pinned by commit, see docs/hw_device.md. Its root is
llmforge.paths.DEVICE_RUNTIME, which the LLMFORGE_DEVICE_RUNTIME environment variable overrides.
Build products, exported models and raw traces are temporary files under the scratch directory.
"""
import hashlib
import os

from llmforge import paths

RUNTIME = paths.DEVICE_RUNTIME
KERNEL_SOURCE = 'src/runq_reallm.c'
SAMPLER_SOURCE = 'src/power_sampler.c'
EXPORTER = 'reallmforge/export_reallm_hetero.py'
TOKENIZER_CANDIDATES = ('tokenizer_gpt2.bin',
                        'models/nsga_best3_rotary_periln_105M/tokenizer_gpt2.bin',
                        'models/smollm2_135M/tokenizer_gpt2.bin')

WORK = paths.SCRATCH / 'device' / 'sweep'
SWEEP_OUTPUTS = paths.RUNS / 'device' / 'sweeps'
SCRATCH_FILES = ('temp_device_ckpt.pt', 'temp_device_model.q8.rlm', 'runq_reallm_device',
                 'power_sampler_device', 'temp_trace_raw.csv', 'temp_infer_log.txt')


def runtime_file(relative):
    path = RUNTIME / relative
    if not path.is_file():
        raise FileNotFoundError(f'{relative} is missing from the device runtime at {RUNTIME}. '
                                'Check out the pinned runtime there or set LLMFORGE_DEVICE_RUNTIME.')
    return path


def kernel_id():
    """SHA-256 of the inference kernel source, which is the protocol's kernel identity."""
    return hashlib.sha256(runtime_file(KERNEL_SOURCE).read_bytes()).hexdigest()


def child_env():
    """Environment for child Python processes, with this package importable from any directory."""
    env = dict(os.environ)
    source_root = str(paths.PACKAGE.parent)
    existing = [p for p in env.get('PYTHONPATH', '').split(os.pathsep) if p and p != source_root]
    env['PYTHONPATH'] = os.pathsep.join([source_root] + existing)
    return env
