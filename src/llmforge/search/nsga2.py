"""NSGA-II population for the co-search.

Ported from the original LLMForge search and decoupled from remote training and simulator calls,
which now sit behind the evaluator protocols in llmforge.evaluators.

Selection is NSGA-II with constrained domination. A feasible solution dominates an infeasible one,
two infeasible solutions compare by total constraint violation, and two feasible solutions compare
by Pareto dominance. Each generation draws `n_offspring` children. A child comes from two binary
tournaments on constrained domination, crossover, and mutation, and is redrawn up to `max_tries`
times while it duplicates an architecture evaluated before. Parents and children are merged,
duplicates removed, sorted into fronts, ordered inside a front by crowding distance, and truncated
to `n_population`.

The search space supplies sample(), crossover(a, b, rate) and mutate(ind, rate). All randomness
flows through one random.Random, whose state is saved in checkpoints.
"""
from __future__ import annotations

import csv
import json
import math
import os
import random
from typing import Any, Callable, Dict, Iterable, List, Optional

from .individual import Individual


def cons_value(con_key: str, threshold: float, auxs: Dict[str, Any]) -> float:
    """Violation of one constraint, feasible when <= 0.

    A key ending in "_min" is a lower bound on the metric without the suffix. Any other key is an
    upper bound. A missing metric counts as a violation.
    """
    if con_key.endswith("_min"):
        base = con_key[:-4]
        return float(threshold) - float(auxs.get(base, float("-inf")))
    return float(auxs.get(con_key, float("inf"))) - float(threshold)


class EvaluationResult:
    __slots__ = ("objs", "cons", "aux")

    def __init__(self, objs: List[float], cons: List[float], aux: Dict[str, Any]):
        self.objs, self.cons, self.aux = list(objs), list(cons), dict(aux)

    @property
    def feasible(self) -> bool:
        return all(c <= 0 for c in self.cons)

    def to_dict(self) -> Dict[str, Any]:
        return {"objs": self.objs, "cons": self.cons, "aux": self.aux}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "EvaluationResult":
        return EvaluationResult(d["objs"], d["cons"], d.get("aux", {}))


def dominates(o1, c1, o2, c2) -> int:
    """1 if solution 1 dominates 2, -1 if 2 dominates 1, 0 otherwise (constrained domination)."""
    feas1 = all(c <= 0 for c in c1)
    feas2 = all(c <= 0 for c in c2)
    if feas1 and not feas2:
        return 1
    if feas2 and not feas1:
        return -1
    if feas1 and feas2:
        better = worse = False
        for a, b in zip(o1, o2):
            if a < b - 1e-12:
                better = True
            elif a > b + 1e-12:
                worse = True
        if better and not worse:
            return 1
        if worse and not better:
            return -1
        return 0
    v1 = sum(max(0.0, c) for c in c1)
    v2 = sum(max(0.0, c) for c in c2)
    if v1 < v2 - 1e-12:
        return 1
    if v1 > v2 + 1e-12:
        return -1
    return 0


def fast_non_dominated_sort(objs: List[List[float]], cons: List[List[float]]) -> List[List[int]]:
    N = len(objs)
    S = [[] for _ in range(N)]
    n = [0] * N
    F: List[List[int]] = [[]]
    for p in range(N):
        for q in range(N):
            if p == q:
                continue
            d = dominates(objs[p], cons[p], objs[q], cons[q])
            if d == 1:
                S[p].append(q)
            elif d == -1:
                n[p] += 1
        if n[p] == 0:
            F[0].append(p)
    i = 0
    while F[i]:
        Q = []
        for p in F[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    Q.append(q)
        i += 1
        F.append(Q)
    return F[:-1]


def crowding_distance(front: List[int], objs: List[List[float]]) -> Dict[int, float]:
    if not front:
        return {}
    big = 1e300
    safe = {i: [x if math.isfinite(x) else big for x in objs[i]] for i in front}
    dist = {i: 0.0 for i in front}
    for k in range(len(objs[front[0]])):
        order = sorted(front, key=lambda i: safe[i][k])
        lo, hi = safe[order[0]][k], safe[order[-1]][k]
        dist[order[0]] = dist[order[-1]] = float("inf")
        denom = (hi - lo) if abs(hi - lo) > 1e-12 else 1.0
        for j in range(1, len(order) - 1):
            dist[order[j]] += (safe[order[j + 1]][k] - safe[order[j - 1]][k]) / denom
    return dist


def tournament_select(evals: List[EvaluationResult], k: int = 2, rng: random.Random = random) -> int:
    i = rng.randrange(len(evals))
    for _ in range(k - 1):
        j = rng.randrange(len(evals))
        if j == i:
            continue
        d = dominates(evals[i].objs, evals[i].cons, evals[j].objs, evals[j].cons)
        if d == -1 or (d == 0 and rng.random() < 0.5):
            i = j
    return i


class Population:
    def __init__(self, individuals: Iterable[Dict[str, Any]],
                 evaluations: Optional[List[EvaluationResult]] = None, search_space=None,
                 cons_settings: Optional[Dict[str, float]] = None,
                 objs_settings: Optional[List[str]] = None, rng: Optional[random.Random] = None,
                 n_population: int = 16, n_offspring: int = 8, tournament_k: int = 2,
                 mutation_rate: Optional[float] = None, crossover_rate: float = 0.9):
        self.individuals = [ind if isinstance(ind, Individual) else Individual.from_dict(ind)
                            for ind in individuals]
        self.evaluations: List[EvaluationResult] = list(evaluations or [])
        self.offspring: List[Individual] = []
        self.offspring_evaluations: List[EvaluationResult] = []
        self.gen = 0
        self.search_space = search_space
        self.objs_settings = objs_settings
        self.cons_settings = cons_settings or {}
        self.rng = rng or random.Random()
        self.n_population = int(n_population)
        self.n_offspring = int(n_offspring)
        self.tournament_k = int(tournament_k)
        self.mutation_rate = mutation_rate
        self.crossover_rate = float(crossover_rate)

    def __str__(self) -> str:
        return f"Population(gen={self.gen}, size={len(self.individuals)}, evaluated={len(self.evaluations)})"

    # ---- ranking ---------------------------------------------------------------------------

    def fast_non_dominated_sort(self, objs=None, cons=None) -> List[List[int]]:
        if objs is None or cons is None:
            objs = [e.objs for e in self.evaluations]
            cons = [e.cons for e in self.evaluations]
        return fast_non_dominated_sort(objs, cons)

    def reorder_by_non_domination(self) -> List[int]:
        """Order individuals by front, then by crowding distance inside each front."""
        if not self.evaluations:
            return []
        objs = [e.objs for e in self.evaluations]
        cons = [e.cons for e in self.evaluations]
        order: List[int] = []
        for front in fast_non_dominated_sort(objs, cons):
            cd = crowding_distance(front, objs)
            order.extend(sorted(front, key=lambda i: cd[i], reverse=True))
        self.individuals = [self.individuals[i] for i in order]
        self.evaluations = [self.evaluations[i] for i in order]
        return order

    def front_indices(self) -> List[int]:
        fronts = self.fast_non_dominated_sort()
        return fronts[0] if fronts else []

    def delete_duplicates(self, key_fn: Callable[[Dict[str, Any]], str]) -> None:
        seen, keep = set(), []
        for i, ind in enumerate(self.individuals):
            k = key_fn(ind)
            if k not in seen:
                seen.add(k)
                keep.append(i)
        self.individuals = [self.individuals[i] for i in keep]
        if len(self.evaluations) >= len(keep):
            self.evaluations = [self.evaluations[i] for i in keep]

    # ---- variation -------------------------------------------------------------------------

    def _child(self) -> Individual:
        space = self.search_space
        p1 = tournament_select(self.evaluations, self.tournament_k, self.rng)
        p2 = tournament_select(self.evaluations, self.tournament_k, self.rng)
        child, _ = space.crossover(self.individuals[p1], self.individuals[p2], self.crossover_rate)
        return space.mutate(child, self.mutation_rate)

    def _draw(self, make: Callable[[], Individual], key_fn, seen, max_tries: int) -> List[Individual]:
        kids, keys = [], set()
        for _ in range(self.n_offspring):
            child = make()
            if key_fn is not None:
                for _ in range(max_tries - 1):
                    k = key_fn(child)
                    if k not in seen and k not in keys:
                        break
                    child = make()
                keys.add(key_fn(child))
            kids.append(child)
        return kids

    def generate_offspring(self, key_fn=None, seen: Iterable[str] = (), max_tries: int = 50) -> List[Individual]:
        if not self.evaluations:
            raise ValueError("cannot generate offspring without evaluations")
        seen = set(seen)
        self.offspring = self._draw(self._child, key_fn, seen, max_tries)
        self.offspring_evaluations = []
        self.gen += 1
        return self.offspring

    def generate_offspring_random(self, key_fn=None, seen: Iterable[str] = (), max_tries: int = 200) -> List[Individual]:
        seen = set(seen)
        self.offspring = self._draw(self.search_space.sample, key_fn, seen, max_tries)
        self.offspring_evaluations = []
        self.gen += 1
        return self.offspring

    def update_elimination(self, key_fn=None) -> None:
        if len(self.offspring_evaluations) != len(self.offspring):
            raise ValueError("offspring must be evaluated before elimination")
        self.individuals.extend(self.offspring)
        self.evaluations.extend(self.offspring_evaluations)
        self.offspring, self.offspring_evaluations = [], []
        if key_fn is not None:
            self.delete_duplicates(key_fn)
        self.reorder_by_non_domination()
        self.individuals = self.individuals[:self.n_population]
        self.evaluations = self.evaluations[:self.n_population]

    # ---- persistence -----------------------------------------------------------------------

    def save_checkpoint(self, path: str, extra: Optional[Dict[str, Any]] = None) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        state = self.rng.getstate()
        payload = {
            "gen": self.gen,
            "individuals": self.individuals,
            "evaluations": [e.to_dict() for e in self.evaluations],
            "objs_settings": self.objs_settings,
            "cons_settings": self.cons_settings,
            "n_population": self.n_population,
            "n_offspring": self.n_offspring,
            "tournament_k": self.tournament_k,
            "mutation_rate": self.mutation_rate,
            "crossover_rate": self.crossover_rate,
            "rng_state": [state[0], list(state[1]), state[2]],
            **(extra or {}),
        }
        tmp = f"{path}.tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, path)
        return path

    @staticmethod
    def load_checkpoint(path: str, search_space=None, rng: Optional[random.Random] = None) -> "Population":
        with open(path) as f:
            d = json.load(f)
        rng = rng or random.Random()
        s = d.get("rng_state")
        if s:
            rng.setstate((s[0], tuple(s[1]), s[2]))
        pop = Population(d["individuals"], [EvaluationResult.from_dict(e) for e in d["evaluations"]],
                         search_space, d.get("cons_settings"), d.get("objs_settings"), rng,
                         d["n_population"], d["n_offspring"], d["tournament_k"], d["mutation_rate"],
                         d["crossover_rate"])
        pop.gen = int(d["gen"])
        return pop

    def write_to_csv(self, filepath: str) -> None:
        """Individuals with flattened per-layer settings and every metric of their evaluation."""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        if not self.individuals:
            return
        global_keys = [k for k in self.individuals[0]["globals"] if k != "layer_mask"]
        layer_keys = list(self.individuals[0]["layers"][0].keys())
        n_layers = max(len(ind["layers"]) for ind in self.individuals)
        aux_keys: List[str] = []
        for ev in self.evaluations:
            aux_keys.extend(k for k in ev.aux if k not in aux_keys)
        fields = (["idx"] + [f"global_{k}" for k in global_keys]
                  + [f"layer{i}_{k}" for i in range(n_layers) for k in layer_keys] + aux_keys)
        with open(filepath, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for idx, (ind, ev) in enumerate(zip(self.individuals, self.evaluations)):
                row = {"idx": idx}
                row.update({f"global_{k}": ind["globals"].get(k) for k in global_keys})
                for i, layer in enumerate(ind["layers"]):
                    row.update({f"layer{i}_{k}": layer.get(k) for k in layer_keys})
                row.update(ev.aux)
                w.writerow(row)
