"""Architecture-only features; no measurements or device-state inputs."""
import numpy as np
from scripts.sweep.layerwise.candidates import architecture_summary


def features(arch):
    dim = arch['d_model']
    out = dict(architecture_summary(arch))
    out.update(n_layer=arch['n_layer'], d_model=dim, group_size=arch['q8_group_size'])
    rows = arch['layers']
    arrays = {k: np.array([r[k] for r in rows], dtype=float)
              for k in ('n_h', 'n_kv', 'd_qk', 'd_v', 'd_mlp')}
    arrays['gqa_ratio'] = arrays['n_h'] / arrays['n_kv']
    arrays['attention_out_width'] = arrays['n_h'] * arrays['d_v']
    arrays['kv_bytes_per_token'] = 4 * arrays['n_kv'] * (arrays['d_qk'] + arrays['d_v'])
    arrays['layer_matrix_params'] = dim * (
        arrays['n_h'] * (arrays['d_qk'] + arrays['d_v'])
        + arrays['n_kv'] * (arrays['d_qk'] + arrays['d_v']) + 3 * arrays['d_mlp'])
    arrays['layer_q8_bytes'] = arrays['layer_matrix_params'] * (1 + 4 / arch['q8_group_size'])
    for key, a in arrays.items():
        for label, value in [('sum', a.sum()), ('mean', a.mean()), ('std', a.std()),
                             ('min', a.min()), ('max', a.max()),
                             ('adjacent_change', np.abs(np.diff(a)).mean())]:
            out[f'{key}_{label}'] = float(value)
        for i, block in enumerate(np.array_split(a, 4)):
            out[f'{key}_quarter{i}'] = float(block.mean())
    return out


def matrix(architectures, names=None):
    rows = [features(a) for a in architectures]
    names = sorted(rows[0]) if names is None else names
    x = np.array([[r[k] for k in names] for r in rows], dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite architecture features')
    return x, names


def predict(bundle, architectures):
    x, _ = matrix(architectures, bundle['features'])
    return np.column_stack([np.exp(m.predict(x)) for m in bundle['models']])
