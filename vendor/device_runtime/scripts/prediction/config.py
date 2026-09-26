"""Shared paths and explicit legacy target schemas (no training side effects)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PREDICTION_ROOT = Path(__file__).resolve().parent
OUTPUTS = PREDICTION_ROOT / 'outputs'
FEATURES = ['n_layer', 'd_model', 'n_h', 'n_kv', 'd_qk', 'd_v', 'd_mlp',
            'total_params_M', 'q8_group_size', 'layer_size_kb', 'mlp_ratio', 'kv_ratio',
            'kv_bytes_per_token', 'attention_width']
TARGETS = ['tpot_ms', 'ttft_ms', 'dynamic_energy_per_token_mj']
PHYSICS_TARGETS = ['decode_tok_s', 'ttft_ms', 'dynamic_energy_per_token_mj']
