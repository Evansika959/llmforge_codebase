"""Hardware evaluator: Pixel Watch 5, from a predictor fitted to on-device measurements.

The predictor was fitted on uniform architectures, so this backend accepts only individuals whose
active layers are identical. Any other individual gets hw_feasible = False.

Workload: 49 prompt tokens then 32 generated tokens on the nanollmforge.c runtime, int8 weights with
per-group scales whose group size follows from the architecture. See docs/hw_device.md for the
measurement protocol, the training domain, and how the bundles were selected.

Emitted keys
    device_decode_tok_s          decode throughput, tokens per second
    device_ttft_ms               time to first token, milliseconds
    device_energy_per_token_mJ   dynamic energy per generated token, millijoules, measured above the
                                 idle baseline over the whole inference including prefill
    device_in_domain             every architecture field and the parameter count lie inside the
                                 range of the predictor's training data
    device_on_support            device_in_domain, and every value of n_h, n_kv, d_qk, d_v and d_model
                                 also occurs in the training data. The sweep drew these from short lists,
                                 so a value inside the range can still be one no observation has
    energy_per_token_uJ          1000 x device_energy_per_token_mJ
    ttft_ms, tpot_ms             tpot_ms = 1000 / device_decode_tok_s
    hw_feasible
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

BUNDLES = {"best": ("predictor_best.joblib", "dataset_best_model.json"),
           "latest": ("predictor_latest.joblib", "dataset_latest.json")}
FIELDS = ("n_layer", "d_model", "n_h", "n_kv", "d_qk", "d_v", "d_mlp")
SUPPORT_FIELDS = ("d_model", "n_h", "n_kv", "d_qk", "d_v")
LAYER_KEYS = ("n_head", "n_kv_group", "n_qk_head_dim", "n_v_head_dim", "mlp_size", "attention_variant")


class HwDevice:
    def __init__(self, bundle: str = "best", asset_dir: Optional[str] = None):
        from ..hw.device.prediction.models.serialization import load_bundle
        from ..paths import DEVICE_ASSETS

        self.asset_dir = Path(asset_dir) if asset_dir else DEVICE_ASSETS / "pixel_watch5"
        if bundle in BUNDLES:
            model_file, data_file = BUNDLES[bundle]
            path = self.asset_dir / "models" / model_file
            self.dataset_path = self.asset_dir / "data" / data_file
        else:
            path, self.dataset_path = Path(bundle), None
        self.bundle = bundle
        self.pack = load_bundle(path)
        self.domain = self._training_domain()

    def _training_domain(self) -> Optional[Dict[str, List[float]]]:
        if self.dataset_path is None or not self.dataset_path.exists():
            return None
        obs = json.loads(self.dataset_path.read_text())["observations"]
        dom = {}
        for f in FIELDS + ("total_params_M",):
            vals = [float(o["architecture"][f]) for o in obs if o["architecture"].get(f) not in (None, "")]
            dom[f] = [min(vals), max(vals)]
        self.support = {f: {int(float(o["architecture"][f])) for o in obs if o["architecture"].get(f) not in (None, "")}
                        for f in SUPPORT_FIELDS}
        return dom

    @staticmethod
    def uniform_config(ind: Dict[str, Any]) -> Optional[Dict[str, int]]:
        g = ind["globals"]
        layers = ind["layers"]
        mask = g.get("layer_mask", [True] * len(layers))
        active = [li for i, li in enumerate(layers) if i < len(mask) and mask[i]]
        if not active:
            return None
        first = tuple(active[0].get(k) for k in LAYER_KEYS)
        if first[-1] != "infinite" or any(tuple(li.get(k) for k in LAYER_KEYS) != first for li in active):
            return None
        return {"n_layer": len(active), "d_model": int(g["n_embd"]), "n_h": int(first[0]),
                "n_kv": int(first[1]), "d_qk": int(first[2]), "d_v": int(first[3]),
                "d_mlp": int(first[4]), "vocab_size": int(g.get("vocab_size", 50257))}

    def in_domain(self, config: Dict[str, int]) -> Optional[bool]:
        if self.domain is None:
            return None
        from ..hw.device.prediction.features.physics import architecture_stats

        params_m = architecture_stats(config)["params"] / 1e6
        ok = all(self.domain[f][0] <= config[f] <= self.domain[f][1] for f in FIELDS)
        return bool(ok and self.domain["total_params_M"][0] <= params_m <= self.domain["total_params_M"][1])

    def on_support(self, config: Dict[str, int]) -> Optional[bool]:
        inside = self.in_domain(config)
        if inside is None:
            return None
        return bool(inside and all(int(config[f]) in self.support[f] for f in SUPPORT_FIELDS))

    def _predict(self, configs: List[Dict[str, int]]):
        from ..hw.device.prediction.inference.predictor import bundle_targets, predict_bundle

        targets = bundle_targets(self.pack)
        try:
            return targets, list(predict_bundle(self.pack, configs))
        except ValueError:
            rows = []
            for c in configs:
                try:
                    rows.append(predict_bundle(self.pack, [c])[0])
                except ValueError as e:
                    rows.append(e)
            return targets, rows

    def evaluate(self, inds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Optional[Dict[str, Any]]] = [None] * len(inds)
        configs, where = [], []
        for i, ind in enumerate(inds):
            c = self.uniform_config(ind)
            if c is None:
                out[i] = {"hw_feasible": False, "device_error": "non-uniform architecture"}
            else:
                configs.append(c)
                where.append(i)
        if configs:
            targets, rows = self._predict(configs)
            for c, i, row in zip(configs, where, rows):
                if isinstance(row, Exception):
                    out[i] = {"hw_feasible": False, "device_error": str(row)}
                    continue
                p = dict(zip(targets, (float(x) for x in row)))
                tok_s, ttft, e_mj = p["decode_tok_s"], p["ttft_ms"], p[targets[2]]
                out[i] = {"device_decode_tok_s": tok_s, "device_ttft_ms": ttft,
                          "device_energy_per_token_mJ": e_mj, "device_in_domain": self.in_domain(c),
                          "device_on_support": self.on_support(c),
                          "energy_per_token_uJ": e_mj * 1e3, "ttft_ms": ttft,
                          "tpot_ms": 1e3 / tok_s, "hw_feasible": True}
        return out
