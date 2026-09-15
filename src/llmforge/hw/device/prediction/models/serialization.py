"""Load trusted local bundles, including checkpoints pickled under earlier module names.

Joblib/pickle may execute code. Do not load untrusted model files.
"""
from contextlib import contextmanager
import importlib
import sys
import types

import joblib

_PACKAGE = __name__.rsplit('.', 2)[0]
# Bundles written before the package moved reference these module names.
_LEGACY_MODULES = {
    'physics_surrogate_components': 'models.legacy',
    'compare_surrogates': 'models.legacy',
    'scripts.prediction.features.physics': 'features.physics',
    'scripts.prediction.models.neural': 'models.neural',
    'scripts.prediction.models.legacy': 'models.legacy',
}


@contextmanager
def _legacy_module_names():
    """Temporarily resolve legacy module names to this package, leaving sys.modules unchanged after."""
    added = []
    try:
        for legacy, current in _LEGACY_MODULES.items():
            parts = legacy.split('.')
            for i in range(1, len(parts)):
                parent = '.'.join(parts[:i])
                if parent not in sys.modules:
                    placeholder = types.ModuleType(parent)
                    placeholder.__path__ = []
                    sys.modules[parent] = placeholder
                    added.append(parent)
            if legacy not in sys.modules:
                sys.modules[legacy] = importlib.import_module(f'{_PACKAGE}.{current}')
                added.append(legacy)
        yield
    finally:
        for name in reversed(added):
            sys.modules.pop(name, None)


def load_bundle(path):
    with _legacy_module_names():
        return joblib.load(path)
