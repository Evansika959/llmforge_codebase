"""Load trusted local bundles, including pre-package module names.

Joblib/pickle may execute code. Do not load untrusted model files.
"""
import importlib
import sys
import joblib


def load_bundle(path):
    # Historical profiles were pickled under the former flat module name.
    legacy = importlib.import_module('scripts.prediction.models.legacy')
    for name in ('physics_surrogate_components', 'compare_surrogates'):
        if name not in sys.modules:
            sys.modules[name] = legacy
    return joblib.load(path)
