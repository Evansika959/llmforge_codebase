"""Heterogeneous from-scratch search space, the space of the original LLMForge search.

Samples, repairs, crosses over and mutates Individuals with per-layer attention and MLP choices,
optional layer masks, and bundled layer groups. Searches over a trained supernet use
llmforge.search.elastic_space instead.
"""

import random, math
from typing import Any, Dict, List, Tuple, TypedDict

from .individual import Individual, _mlp_matrix_count

class HeteroSearchSpace:
    def __init__(self, L_max=24, L_min=1, freeze_layer_mask: bool = False,
                 bundle_size: int = 1, num_bundles: int = None,
                 bundled_search: bool = False):
        # `bundled_search` selects a SEPARATE search mode from the default
        # per-layer search:
        #
        #   bundled_search=False (default): the original per-layer, variable-
        #     depth search. layer_mask is free, L_min/L_max gate the depth.
        #
        #   bundled_search=True: gene-tie bundling. The model is `num_bundles`
        #     (B) bundles of `bundle_size` (K) consecutive layers; every layer
        #     in a bundle shares ONE architecture spec (genes shared, NOT
        #     weights — each layer trains its own weights). This shrinks the
        #     genotype from L_max layer-specs to B, cutting the mutation /
        #     sensitivity neighborhood by ~K. Depth is FIXED at K*B layers
        #     (bundles are never added or removed), so the mask is frozen
        #     all-active and L_min/L_max do NOT gate the search — the only
        #     requirement is K*B <= L_max (L_max is a pure capacity cap).
        self.bundled_search = bool(bundled_search)
        if self.bundled_search:
            self.bundle_size = max(1, int(bundle_size))
            if num_bundles is None or int(num_bundles) < 1:
                raise ValueError("bundled_search requires num_bundles>=1.")
            self.num_bundles = int(num_bundles)
            total = self.bundle_size * self.num_bundles
            if total > L_max:
                raise ValueError(
                    f"bundle_size*num_bundles ({self.bundle_size}*"
                    f"{self.num_bundles}={total}) must be <= L_max ({L_max}).")
            self.L_max = total
            self.L_min = total
            self.freeze_layer_mask = True
        else:
            self.bundle_size = 1
            self.L_max = L_max
            self.L_min = L_min  # minimum active layers
            self.freeze_layer_mask = freeze_layer_mask
            self.num_bundles = self.L_max  # each layer is its own bundle

        self.no_repair = False  # if True, sample() does not call repair()

        # Globals
        self.globals = {
            "d_model":      {"type":"int","low":256,"high":2048,"step":256},
            "block_size":      {"type":"int","low":512,"high":512,"step":128},
            "quant_bits":   {"type":"int","low":8,"high":8},
            # replaced active_L with an explicit layer usage mask of length L_max
            #"layer_mask" added later
        }

        # Per-layer fields (heterogeneous)
        self.layer_spec = {
            "n_heads":    {"type":"int","low":1,"high":32,"step":1},   # will be clamped to divisor of d_model
            "mlp_ratio":  {"type":"int","low":1,"high":8,"step":1},
            "attn_type":  {"type":"cat","choices":["mha"]},
        }

    @classmethod
    def from_dicts(
        cls,
        globals_spec: Dict[str, Any],
        layer_spec: Dict[str, Any],
        L_max: int = 24,
        L_min: int = 1,
        no_repair: bool = False,
        freeze_layer_mask: bool = False,
        bundle_size: int = 1,
        num_bundles: int = None,
        bundled_search: bool = False
    ) -> "HeteroSearchSpace":
        """Alternate constructor: build a search space from explicit spec dicts.

        Parameters
        - globals_spec: dict mapping global field name -> spec
            Spec format examples:
              {"type": "int", "low": 256, "high": 2048, "step": 256}
              {"type": "float", "low": 0.0, "high": 1.0}
              {"type": "cat", "choices": ["mha", "flash"]}
        - layer_spec: dict mapping per-layer field name -> spec (same format as above)
        - L_max: maximum number of layers

        Returns a configured HeteroSearchSpace instance.
        """
        inst = cls(L_max=L_max, L_min=L_min, freeze_layer_mask=freeze_layer_mask,
                   bundle_size=bundle_size, num_bundles=num_bundles,
                   bundled_search=bundled_search)
        inst.globals = cls._normalize_spec_dict(globals_spec)
        inst.no_repair = no_repair
        if layer_spec is None:
            inst.layer_spec = {}
        else:
            inst.layer_spec = cls._normalize_spec_dict(layer_spec)
        return inst

    @staticmethod
    def _normalize_spec_dict(spec_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize a spec dict.

        Ensures required keys exist and fills reasonable defaults (e.g., step=1 for ints).
        Raises ValueError on invalid specifications.
        """
        out: Dict[str, Any] = {}
        for k, raw in spec_dict.items():
            if not isinstance(raw, dict):
                raise ValueError(f"Spec for '{k}' must be a dict, got {type(raw)}")
            s = dict(raw)  # shallow copy
            s_type = s.get("type")
            if s_type not in {"int", "float", "cat"}:
                raise ValueError(
                    f"Spec for '{k}' must include a valid 'type' in {{'int','float','cat'}}, got {s_type}"
                )
            if s_type == "int":
                for req in ("low", "high"):
                    if req not in s:
                        raise ValueError(f"Int spec for '{k}' missing '{req}'")
                s.setdefault("step", 1)
                if not isinstance(s["step"], int) or s["step"] <= 0:
                    raise ValueError(f"Int spec for '{k}' has invalid step: {s['step']}")
            elif s_type == "float":
                for req in ("low", "high"):
                    if req not in s:
                        raise ValueError(f"Float spec for '{k}' missing '{req}'")
            elif s_type == "cat":
                if "choices" not in s or not s["choices"]:
                    raise ValueError(f"Cat spec for '{k}' requires non-empty 'choices'")
            out[k] = s
        return out

    # ---------- utils ----------
    def _sample_global(self):
        g = {}
        for k,s in self.globals.items():
            if s["type"]=="int":
                step=s.get("step",1)
                g[k]=random.randrange(s["low"], s["high"]+1, step)
            elif s["type"]=="float":
                g[k]=random.uniform(s["low"], s["high"])
            elif s["type"]=="cat":
                g[k]=random.choice(s["choices"])
        return g

    def _sample_layer(self):
        l = {}
        for k,s in self.layer_spec.items():
            # ensure head is to the power of 2 for efficiency
            if k == "n_heads":
                choices = [h for h in range(s["low"], s["high"]+1) if (h & (h - 1)) == 0]
                l[k] = random.choice(choices) if choices else s["low"]
                continue

            if s["type"]=="int":
                step=s.get("step",1)
                l[k]=random.randrange(s["low"], s["high"]+1, step)
            elif s["type"]=="float":
                l[k]=random.uniform(s["low"], s["high"])
            elif s["type"]=="cat":
                l[k]=random.choice(s["choices"])
        return l
    
    def print_search_space(self) -> None:
        print("Global parameters:")
        for k, s in self.globals.items():
            print(f"  - {k}: {s}")
        for k, s in self.layer_spec.items():
            print(f"  - {k}: {s}")
        print(f"Max number of layers (L_max={self.L_max})")
        print(f"Minimum active layers (L_min={self.L_min})")
        print(f"No repair mode: {self.no_repair}")
        return

    # ---------- public API ----------
    def sample(self) -> Individual:
        g = self._sample_global()
        layers = [self._sample_layer() for _ in range(self.L_max)]
        x: Individual = Individual(g, layers)
        # if globals does not yet have layer_mask (e.g., older serialized), create one
        if self.freeze_layer_mask:
            x["globals"]["layer_mask"] = [True] * self.L_max
        elif "layer_mask" not in x["globals"]:
            active_count = random.randint(self.L_min, self.L_max)
            idxs = set(random.sample(range(self.L_max), active_count))
            x["globals"]["layer_mask"] = [i in idxs for i in range(self.L_max)]
        
        return self.repair(x)

    def repair(self, x: Dict[str, Any]) -> Individual:
        # assume mask always provided; if missing, default to all active
        if self.no_repair:
            return Individual.from_dict(x)
        if "globals" not in x:
            x["globals"] = {}
        if self.freeze_layer_mask:
            x["globals"]["layer_mask"] = [True] * self.L_max
        elif "layer_mask" not in x["globals"]:
            x["globals"]["layer_mask"] = [True]*self.L_max
        mask = list(x["globals"]["layer_mask"])[:self.L_max]
        if len(mask) < self.L_max:
            mask.extend([False]*(self.L_max-len(mask)))
        x["globals"]["layer_mask"] = mask

        y: Dict[str, Any] = {"globals": dict(x["globals"]), "layers": [dict(li) for li in x["layers"]] }
        # Pad/truncate layers to L_max so mutate()/crossover() can safely index
        # up to L_max-1. Padding layers are inactive (mask was padded False).
        # Use an existing active layer as template when possible, else random-sample.
        if len(y["layers"]) > self.L_max:
            y["layers"] = y["layers"][:self.L_max]
        if len(y["layers"]) < self.L_max:
            active_idx = [i for i, m in enumerate(mask) if m and i < len(y["layers"])]
            template = dict(y["layers"][active_idx[-1]]) if active_idx \
                else (dict(y["layers"][-1]) if y["layers"] else self._sample_layer())
            while len(y["layers"]) < self.L_max:
                y["layers"].append(dict(template))
        # clamp globals
        for k,s in self.globals.items():
            if s["type"]=="int":
                step=s.get("step",1)
                lo,hi=s["low"],s["high"]
                y["globals"][k]=max(lo, min(hi, round(y["globals"][k]/step)*step))
            elif s["type"]=="float":
                lo,hi=s["low"],s["high"]
                y["globals"][k]=float(max(lo, min(hi, y["globals"][k])))
            elif s["type"]=="cat":
                if y["globals"][k] not in s["choices"]:
                    y["globals"][k]=s["choices"][0]

        # n_embd = y["globals"]["n_embd"]
        # clamp per-layer + divisibility
        for li in y["layers"]:
            for k,s in self.layer_spec.items():
                if s["type"]=="int":
                    step=s.get("step",1)
                    lo,hi=s["low"],s["high"]
                    li[k]=max(lo, min(hi, round(li[k]/step)*step))
                    # Honor an optional `exclude` list: forbidden integer
                    # values snap to the nearest allowed value (ties -> the
                    # smaller). Used to remove n_head values that collapse
                    # GQA (e.g. odd primes whose only divisor is 1).
                    excl = s.get("exclude")
                    if excl and li[k] in excl:
                        allowed = [v for v in range(lo, hi + 1, step)
                                   if v not in excl]
                        if allowed:
                            li[k] = min(allowed,
                                        key=lambda v: (abs(v - li[k]), v))
                elif s["type"]=="float":
                    lo,hi=s["low"],s["high"]
                    li[k]=float(max(lo, min(hi, li[k])))
                elif s["type"]=="cat":
                    if li[k] not in s["choices"]:
                        li[k]=s["choices"][0]
            
            attn_type = li.get("attention_variant",
                               y["globals"].get("attention_variant", "mha"))
            if attn_type == "mha":
                n_head = li.get("n_head", 8)
                n_embd = y["globals"].get("n_embd", 768)
                if n_embd % n_head != 0:
                    divisors = [h for h in range(1, min(n_head, n_embd) + 1)
                                if n_embd % h == 0]
                    li["n_head"] = (min(divisors, key=lambda h: abs(h - n_head))
                                    if divisors else 1)

            # GQA grouping: n_kv_group must divide n_head. The divisor search
            # ranges over 1..min(n_kv_group, n_head), so this also clamps
            # n_kv_group <= n_head (a value above n_head snaps down to n_head,
            # which divides itself).
            if "n_kv_group" in li:
                n_kv_group = li.get("n_kv_group", 1)
                n_head = li.get("n_head", 8)
                if n_head % n_kv_group != 0:
                    divisors = [g for g in range(1, min(n_kv_group, n_head) + 1)
                                if n_head % g == 0]
                    li["n_kv_group"] = (min(divisors, key=lambda g: abs(g - n_kv_group))
                                        if divisors else 1)


        # Enforce gene-tie bundle homogeneity. Done after per-layer clamps so
        # the representative (first) layer of each bundle is already valid;
        # its bundle-mates inherit that validity by copy.
        if self.bundle_size > 1:
            self._collapse_to_bundles(y)

        # ensure at least one active layer; if mask empty, activate the first
        # bundle (first `bundle_size` layers) so the floor stays homogeneous.
        if not any(y["globals"]["layer_mask"]):
            n_on = self.bundle_size if self.bundle_size > 1 else min(4, self.L_max)
            for i in range(min(n_on, self.L_max)):
                y["globals"]["layer_mask"][i] = True
        return Individual.from_dict(y)

    def _collapse_to_bundles(self, y: Dict[str, Any]) -> None:
        """In-place: force each bundle's `bundle_size` layers (and their mask
        bits) to match the bundle's representative (first) layer. Gene-tie
        only — the layers share an architecture spec but train independent
        weights."""
        B = self.bundle_size
        layers = y["layers"]
        mask = y["globals"]["layer_mask"]
        n = len(layers)
        for b in range(self.num_bundles):
            r = b * B
            if r >= n:
                break
            rep = layers[r]
            rep_active = bool(mask[r]) if r < len(mask) else False
            for off in range(1, B):
                j = r + off
                if j >= n:
                    break
                layers[j] = dict(rep)
                if j < len(mask):
                    mask[j] = rep_active

    # ----- variation: layer-aware -----
    def crossover(self, a: Dict[str,Any], b: Dict[str,Any], crossover_rate: float = 0.9) -> Tuple[Dict[str,Any], Dict[str,Any]]:
        # Align layer/mask lengths to L_max before index-based crossover ops
        a = self.repair(a)
        b = self.repair(b)
        A = {"globals": dict(a["globals"]), "layers":[dict(li) for li in a["layers"]]}
        B = {"globals": dict(b["globals"]), "layers":[dict(li) for li in b["layers"]]}

        # uniform crossover on globals
        for k in self.globals:
            if random.random() < crossover_rate:
                A["globals"][k], B["globals"][k] = B["globals"][k], A["globals"][k]

        # layer usage mask crossover (treat mask as gene string)
        mask_a = list(a["globals"].get("layer_mask", [True]*self.L_max))
        mask_b = list(b["globals"].get("layer_mask", [True]*self.L_max))
       
        # segment crossover on layers
        if random.random() < crossover_rate and self.L_max >= 2:
            # perform crossover only on activated layers
            active_indices_a = [i for i, active in enumerate(mask_a) if active and i < len(A["layers"])]
            active_indices_b = [i for i, active in enumerate(mask_b) if active and i < len(B["layers"])]
            
            shorter_len = min(len(active_indices_a), len(active_indices_b))
            if shorter_len >= 2:
                # randomly choose the length of the segment to swap
                seg_len = random.randint(1, shorter_len - 1)
                # swap the segment from backwards to preserve relative order
                start_idx = random.randint(0, shorter_len - seg_len)
                seg_a = active_indices_a[start_idx:start_idx + seg_len]
                seg_b = active_indices_b[start_idx:start_idx + seg_len]
                for i, j in zip(seg_a, seg_b):
                    A["layers"][i], B["layers"][j] = B["layers"][j], A["layers"][i]
                    # also swap the mask bits to keep consistency
                    mask_a[i], mask_b[j] = mask_b[j], mask_a[i]
                A["globals"]["layer_mask"] = mask_a
                B["globals"]["layer_mask"] = mask_b

        return self.repair(A), self.repair(B)
    
    # return the mutated offspring and the mutation operation applied
    def mutate_v2(self, x: Dict[str,Any]) -> {Dict[str,Any], Dict[str,Any]}:
        y = {"globals": dict(x["globals"]), "layers":[dict(li) for li in x["layers"]]}

        # fix the mutation step size; retry if we would clamp to bounds
        max_attempts = 10
        applied = False
        mutate_op = None

        for _ in range(max_attempts):
            mutate_op = {
                "region": random.choice(["front", "middle", "back"]),
                "type": random.choice(["shrink", "expand"]),
                "param": random.choice(["n_head", "n_kv_group", "n_qk_head_dim", "n_v_head_dim", "mlp_size"]),
            }

            layer_cluter_len = 1
            if mutate_op["region"] == "front":
                layer_indices = range(0, min(layer_cluter_len, self.L_max))
            elif mutate_op["region"] == "middle":
                start = max(0, (self.L_max - layer_cluter_len) // 2)
                end = min(self.L_max, start + layer_cluter_len)
                layer_indices = range(start, end)
            else:  # back
                layer_indices = range(max(0, self.L_max - layer_cluter_len), self.L_max)

            mutated_any = False
            for i in layer_indices:
                li = y["layers"][i]
                k = mutate_op["param"]
                s = self.layer_spec.get(k, None)
                if s is None or s.get("type") != "int":
                    continue
                step = s.get("step", 1)
                lo, hi = s["low"], s["high"]

                if k == "n_head":
                    # make sure the n_head is a multiple of n_kv_group after mutation
                    n_kv_group = li.get("n_kv_group", 1)
                    if n_kv_group > 1:
                        # adjust step to be multiple of n_kv_group
                        step = max(step, n_kv_group)

                if mutate_op["type"] == "shrink":
                    if li[k] - step < lo:
                        continue  # would clamp; try another mutate_op
                    new_val = li[k] - step
                else:  # expand
                    if li[k] + step > hi:
                        continue  # would clamp; try another mutate_op
                    new_val = li[k] + step

                if k == "n_kv_group":
                    n_head = li.get("n_head", 8)
                    if mutate_op["type"] == "shrink":
                        # ensure n_kv_group divides n_head
                        divisors = [g for g in range(1, n_head + 1) if n_head % g == 0 and g <= new_val]
                        if not divisors:
                            continue  # no valid divisor found
                        new_val = max(divisors)
                    else:  # expand
                        divisors = [g for g in range(1, n_head + 1) if n_head % g == 0 and g >= new_val]
                        if not divisors:
                            continue  # no valid divisor found
                        new_val = min(divisors)

                li[k] = new_val
                mutated_any = True
                

            if mutated_any:
                applied = True
                break

        # even if no mutation applied after attempts, return repaired individual
        return self.repair(y), mutate_op

    def mutate(self, x: Dict[str,Any],
        p_glob_int=0.1, p_glob_float=0.1,
        p_layer_int=0.08, p_layer_float=0.08, p_layer_cat=0.05,
        p_swap_layers=0.05) -> Dict[str,Any]:
        # Ensure layers/mask are both length L_max before any L_max-indexed ops
        # (resume from a checkpoint with a different L_max is the common cause
        # of mismatch). repair() pads both sides.
        x = self.repair(x)
        y = {"globals": dict(x["globals"]), "layers":[dict(li) for li in x["layers"]]}

        # mutate globals
        for k,s in self.globals.items():
            if s["type"]=="int" and random.random()<p_glob_int:
                step=s.get("step",1); lo,hi=s["low"],s["high"]
                span_steps=max(1, (hi-lo)//step)
                sigma_steps=max(1.0, span_steps/8.0)
                delta_steps=int(round(random.gauss(0.0, sigma_steps)))
                new_val = y["globals"][k] + delta_steps*step
                # snap back to step grid and clamp
                new_val = int(round(new_val/step)*step)
                y["globals"][k]=max(lo,min(hi, new_val))
            elif s["type"]=="float" and random.random()<p_glob_float:
                lo,hi=s["low"],s["high"]
                sigma=(hi-lo)*0.05
                y["globals"][k]=max(lo,min(hi, y["globals"][k]+random.gauss(0,sigma)))

        # mutate layers
        for li in y["layers"]:
            # make it generic
            for k,s in self.layer_spec.items():
                # add gaussian perturbation for int/float, random resample for cat
                if s["type"]=="int" and random.random()<p_layer_int:
                    step=s.get("step",1); lo,hi=s["low"],s["high"]
                    span_steps=max(1, (hi-lo)//step)
                    sigma_steps=max(1.0, span_steps/8.0)
                    delta_steps=int(round(random.gauss(0.0, sigma_steps)))
                    new_val = li[k] + delta_steps*step
                    new_val = int(round(new_val/step)*step)
                    li[k]=max(lo,min(hi, new_val))
                elif s["type"]=="float" and random.random()<p_layer_float:
                    lo,hi=s["low"],s["high"]
                    sigma=(hi-lo)*0.05
                    li[k]=max(lo,min(hi, li[k]+random.gauss(0,sigma)))
                elif s["type"]=="cat" and random.random()<p_layer_cat:
                    choices=s["choices"]
                    cur=li[k]
                    li[k]=random.choice([c for c in choices if c!=cur] or choices)

        # occasional layer swap (explores schedule if meaningful)
        if random.random()<p_swap_layers and self.L_max>=2:
            i,j = random.sample(range(self.L_max), 2)
            y["layers"][i], y["layers"][j] = y["layers"][j], y["layers"][i]

        # mutate layer usage mask: flip a few bits
        mask = list(x["globals"].get("layer_mask", [True]*self.L_max))
        if self.freeze_layer_mask:
            mask = [True] * self.L_max
        else:
            turn_on_rate = 0.2
            turn_off_rate = 0.1
            for i in range(len(mask)):
                if mask[i]:
                    # currently on, may turn off
                    if random.random() < turn_off_rate:
                        mask[i] = False
                else:
                    # currently off, may turn on
                    if random.random() < turn_on_rate:
                        mask[i] = True
                        # copy the layer_configs from the nearsest active layer
                        left = right = None
                        for j in range(i-1, -1, -1):
                            if mask[j]:
                                left = j
                                break
                        for j in range(i+1, self.L_max):
                            if mask[j]:
                                right = j
                                break
                        if left is not None:
                            y["layers"][i] = y["layers"][left]
                        if right is not None:
                            y["layers"][i] = y["layers"][right]

        # ensure still at least four active
        min_layers = self.L_min
        if sum(mask) < min_layers:
            for _ in range(min_layers - sum(mask)):
                mask[random.randrange(self.L_max)] = True
        y["globals"]["layer_mask"] = mask

        p_rotate_layers = 0.1
        p_mirror_layers = 0.1
        # add the random rotation and mirror operations (Dihedral group D_L)
        if random.random() < p_rotate_layers and self.L_max >= 2:
            # rotation
            k = random.randint(1, self.L_max - 1)
            y["layers"] = y["layers"][k:] + y["layers"][:k]
            y["globals"]["layer_mask"] = y["globals"]["layer_mask"][k:] + y["globals"]["layer_mask"][:k]

        # mirroring
        if random.random() < p_mirror_layers and self.L_max >= 2:
            y["layers"] = y["layers"][::-1]
            y["globals"]["layer_mask"] = y["globals"]["layer_mask"][::-1]

        return self.repair(y)

    def calculate_possible_configs(self) -> int:
        total = 1
        for s in self.globals.values():
            if s["type"] == "int":
                step = s.get("step", 1)
                count = ((s["high"] - s["low"]) // step) + 1
                total *= count
            elif s["type"] == "float":
                # Assuming a reasonable discretization for floats
                total *= round(s["high"] - s["low"]) / 0.1  # arbitrary choice for float discretization
            elif s["type"] == "cat":
                total *= len(s["choices"])
        
        # Layer configurations
        layer_configs = 1
        for s in self.layer_spec.values():
            if s["type"] == "int":
                step = s.get("step", 1)
                count = ((s["high"] - s["low"]) // step) + 1
                layer_configs *= count
            elif s["type"] == "float":
                layer_configs *= round(s["high"] - s["low"]) / 0.1  # arbitrary choice for float discretization
            elif s["type"] == "cat":
                layer_configs *= len(s["choices"])
        
        total_layer_config = 0
        # Each layer can be active or inactive, except we enforce at least L_min active layers
        for layers_active in range(self.L_min, self.L_max + 1):
            total_layer_config += layer_configs * layers_active
        total *= total_layer_config
        return total


