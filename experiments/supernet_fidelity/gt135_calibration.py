#!/usr/bin/env python
"""Does the supernet predict trained quality better than anything free?

Rewritten 2026-09-07 after a context-free review destroyed the first version's headline. What that
version got wrong, kept here because both errors are easy to repeat:

  * It reported IN-SAMPLE residual RMSE after fitting `a + b*slice`. That map is scale-free, so a
    predictor whose scores are compressed 20x is not penalised at all, and a rank-only predictor
    (the integers 1..n in truth order, no quality axis whatever) scores 0.059 against the arms'
    0.133-0.187. The metric it chose is an ORDERING metric, which is the opposite of what the
    ground truth exists to measure. Floors: pure noise 0.422, predict-the-mean 0.430.

  * It gave the free baselines one scalar and a straight line, which is the weakest treatment
    available rather than the fairest. Isotonic regression on parameter count alone reaches 0.157
    -- better than three of the four supernet arms.

The baseline that matters is neither: trained CE on the uniform grid is very nearly ADDITIVE in
(d_qk, n_h, d_mlp). Eight coefficients fit it to R^2 0.997, in-sample RMSE 0.023, BELOW the truth's
own measurement SE of 0.041 -- and the slice score adds nothing on top (partial F = 0.00 for every
arm). If that holds at 48 points, the honest result is that this space needs no predictor: a dozen
trained references tabulate it. The supernet's case then has to be made where a table cannot be
tabulated -- per-layer-heterogeneous candidates, where the space is 4^30 x 3^30 x 4^30 rather than
48 -- or with an uncalibrated slice score and zero references. Neither is measured here.

So this script reports OUT-OF-SAMPLE error (leave-one-out) for every model on the same footing,
against both floors, and asks the one question that discriminates: does the slice score add
anything the knobs do not already give away?

  python experiments/supernet_fidelity/gt135_calibration.py --truth runs/supernet/gt135_nll_ctx1024.json
"""
import argparse, itertools, json, os, sys

import numpy as np
from llmforge.supernet.paths import RUNS

KNOBS = ("d_qk", "n_h", "d_mlp")


def knob_design(names, arch, levels):
    cols = [np.ones(len(names))]
    for k in KNOBS:
        for v in levels[k][1:]:                       # first level absorbed by the intercept
            cols.append(np.array([1.0 if arch[n][k] == v else 0.0 for n in names]))
    return np.column_stack(cols)


def loo_rmse(X, y):
    """Leave-one-out RMSE. In-sample RMSE is not reported anywhere: with 9 coefficients on 27
    points it is optimistic by ~5x, and it is the quantity that made the first version wrong."""
    e = []
    for i in range(len(y)):
        m = np.ones(len(y), bool); m[i] = False
        if np.linalg.matrix_rank(X[m]) < X.shape[1]:
            e.append(np.nan); continue          # a knob level held by one architecture only
        b, *_ = np.linalg.lstsq(X[m], y[m], rcond=None)
        e.append(y[i] - X[i] @ b)
    e = np.array(e, float)
    return float(np.sqrt(np.nanmean(e ** 2))), int(np.isnan(e).sum()), e


def isotonic(x, y, decreasing=True):
    """Pool-adjacent-violators. The fair free treatment of a monotone quantity like size."""
    o = np.argsort(x); v = y[o].astype(float).copy()
    if decreasing:
        v = -v
    w = np.ones(len(v)); i = 0
    while i < len(v) - 1:
        if v[i] <= v[i + 1] + 1e-12:
            i += 1; continue
        tot = v[i] * w[i] + v[i + 1] * w[i + 1]; wt = w[i] + w[i + 1]
        v[i] = tot / wt; w[i] = wt
        v = np.delete(v, i + 1); w = np.delete(w, i + 1)
        i = max(i - 1, 0)
    out = np.repeat(v, w.astype(int))
    if decreasing:
        out = -out
    res = np.empty(len(y)); res[o] = out
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", default=f"{RUNS}/gt135_nll_ctx1024.json")
    ap.add_argument("--archs", default=f"{RUNS}/gt135/ARCHS.json")
    ap.add_argument("--arms", nargs="+",
                    default=["lr1e-3", "lr2e-3", "lr4e-3", "lr8e-3", "wdctl", "base"])
    ap.add_argument("--prereg", default=f"{RUNS}/gt135_prereg_predictions.json")
    a = ap.parse_args()

    arch = {r["name"]: r for r in json.load(open(a.archs))}
    truth_raw = json.load(open(a.truth))
    truth = {k: v["nll"] for k, v in truth_raw.items()}
    units_path = a.truth.replace(".json", "_units.json")
    units = json.load(open(units_path)) if os.path.exists(units_path) else {}

    preds = {}
    for t in a.arms:
        p = f"{RUNS}/grid48_{t}.json"
        if os.path.exists(p):
            preds[t] = {k: v["nll"] for k, v in json.load(open(p)).items()}
    names = sorted(set(truth) & set.intersection(*[set(v) for v in preds.values()]))
    y = np.array([truth[n] for n in names])
    levels = {k: sorted({arch[n][k] for n in names}) for k in KNOBS}
    print(f"{len(names)}/{len(arch)} architectures scored both ways. "
          f"trained CE {y.min():.3f}-{y.max():.3f}, sd {y.std():.3f}")
    thin = {k: [v for v in levels[k] if sum(arch[n][k] == v for n in names) < 3] for k in KNOBS}
    if any(thin.values()):
        print(f"  WARNING sparse knob levels (LOO undefined there): "
              + ", ".join(f"{k}={v}" for k, v in thin.items() if v))
    if len(names) < 10:
        print("too few to fit"); return

    # ---- floors, so every number below has a scale ------------------------------------------
    rank = np.argsort(np.argsort(y)).astype(float)
    Zr = np.column_stack([np.ones(len(y)), rank])
    br, *_ = np.linalg.lstsq(Zr, y, rcond=None)
    rng = np.random.default_rng(0)
    nz = []
    for _ in range(2000):
        x = rng.normal(size=len(y)); Z = np.column_stack([np.ones(len(x)), x])
        b, *_ = np.linalg.lstsq(Z, y, rcond=None); nz.append(np.sqrt(((y - Z @ b) ** 2).mean()))
    print(f"  floors: pure-noise x {np.mean(nz):.4f} | predict-the-mean {y.std():.4f} | "
          f"rank-only (no quality axis) {np.sqrt(((y - Zr @ br) ** 2).mean()):.4f}")

    W = np.array([arch[n]["w"] for n in names])
    X_knobs = knob_design(names, arch, levels)

    rows = []
    r_iso = float(np.sqrt(((y - isotonic(W, y)) ** 2).mean()))
    rows.append(("FREE isotonic(params)", r_iso, "in-sample; monotone, 0 linear params"))
    QM = np.array([arch[n]["d_qk"] * arch[n]["d_mlp"] for n in names], float)
    for lab, Z in [("FREE linear(params)", np.column_stack([np.ones(len(W)), W])),
                   ("FREE linear(log params)", np.column_stack([np.ones(len(W)), np.log(W)])),
                   ("FREE additive knobs", X_knobs),
                   ("FREE knobs + d_qk*d_mlp", np.column_stack([X_knobs, QM]))]:
        r_, nan, _ = loo_rmse(Z, y)
        rows.append((lab, r_, f"LOO, {Z.shape[1]} params" + (f", {nan} undefined" if nan else "")))
    for t, p in preds.items():
        x = np.array([p[n] for n in names])
        Z = np.column_stack([np.ones(len(x)), x])
        r_, _, _ = loo_rmse(Z, y)
        rows.append((f"supernet {t}", r_, "LOO, 2 params"))
    print(f"\n{'model':26s} {'LOO RMSE':>9s}   note")
    for lab, r_, note in sorted(rows, key=lambda z: z[1]):
        print(f"{lab:26s} {r_:9.4f}   {note}")

    # ---- the decisive test: does the slice add anything the knobs do not? --------------------
    # Additivity is not a given: d_qk and d_mlp are SUBSTITUTES. Wide attention buys less when the
    # MLP is already wide, and at tied parameter count the contrast flips sign as d_mlp grows
    # (+0.096 -> +0.023 -> -0.015 for (32,9) vs (64,3)). Test it rather than assume it.
    print("\ninteractions on top of the additive table (F > 4.2 is significant at 0.05):")
    b0i, *_ = np.linalg.lstsq(X_knobs, y, rcond=None); r0i = ((y - X_knobs @ b0i) ** 2).sum()
    for lab, z in [("d_qk*d_mlp", QM),
                   ("d_qk*n_h", np.array([arch[n]["d_qk"] * arch[n]["n_h"] for n in names], float)),
                   ("n_h*d_mlp", np.array([arch[n]["n_h"] * arch[n]["d_mlp"] for n in names], float))]:
        XZ = np.column_stack([X_knobs, z])
        b1i, *_ = np.linalg.lstsq(XZ, y, rcond=None); r1i = ((y - XZ @ b1i) ** 2).sum()
        df = len(y) - XZ.shape[1]
        print(f"  {lab:12s} F(1,{df}) = {(r0i - r1i) / (r1i / df):6.2f}")

    print(f"\nincremental value of the slice score OVER the free knob table "
          f"(partial F, df=1,{len(y)-X_knobs.shape[1]-1}):")
    b0, *_ = np.linalg.lstsq(X_knobs, y, rcond=None); r0 = ((y - X_knobs @ b0) ** 2).sum()
    for t, p in preds.items():
        x = np.array([p[n] for n in names])
        XZ = np.column_stack([X_knobs, x])
        b1, *_ = np.linalg.lstsq(XZ, y, rcond=None); r1 = ((y - XZ @ b1) ** 2).sum()
        df = len(y) - XZ.shape[1]
        F = (r0 - r1) / (r1 / df) if r1 > 0 and df > 0 else float("nan")
        print(f"  {t:12s} F = {F:7.2f}   (F > 4.4 would mean the slice carries signal the knobs miss)")

    # ---- matched cost: where a search actually operates ---------------------------------------
    Zw = np.column_stack([np.ones(len(W)), W])
    bw, *_ = np.linalg.lstsq(Zw, y, rcond=None); y_r = y - Zw @ bw
    bk, *_ = np.linalg.lstsq(X_knobs, y, rcond=None); y_k = y - X_knobs @ bk
    print(f"\npartial correlation with the residual truth, after removing:")
    print(f"  {'':12s} {'parameter count':>16s} {'the FULL knob table':>21s}")
    for t, p in preds.items():
        x = np.array([p[n] for n in names])
        bx, *_ = np.linalg.lstsq(Zw, x, rcond=None)
        bxk, *_ = np.linalg.lstsq(X_knobs, x, rcond=None)
        print(f"  {t:12s} {np.corrcoef(x - Zw @ bx, y_r)[0, 1]:+16.3f} "
              f"{np.corrcoef(x - X_knobs @ bxk, y_k)[0, 1]:+21.3f}")
    print("  (removing only the parameter count credits the supernet with knob structure the free "
          "table already has; the right-hand column is the honest one)")

    # ---- pair test, paired SE ------------------------------------------------------------------
    P = [(u, v) for u, v in itertools.combinations(names, 2)
         if abs(arch[u]["w"] - arch[v]["w"]) < 1e-9]
    if P:
        contrasts = {tuple(sorted([(arch[u]["d_qk"], arch[u]["n_h"]),
                                   (arch[v]["d_qk"], arch[v]["n_h"])])) for u, v in P}
        print(f"\nparameter-tied pairs: {len(P)} over {len(contrasts)} distinct contrasts")
        def pair_se(u, v):
            """Paired bootstrap over evaluation units. The unpaired hypot(se_u, se_v) is 5-22x
            larger here and manufactures ties; both models see the same windows."""
            if u not in units or v not in units:
                return None
            au = np.array(units[u]); av = np.array(units[v])
            if au.shape != av.shape:
                return None
            g = np.random.default_rng(0); d = []
            for _ in range(2000):
                k = g.integers(0, len(au), len(au))
                d.append(au[k, 0].sum() / au[k, 1].sum() - av[k, 0].sum() / av[k, 1].sum())
            return float(np.std(d))
        unres = have = 0
        for u, v in P:
            se = pair_se(u, v)
            if se is None:
                continue
            have += 1
            unres += abs(truth[u] - truth[v]) < se
        # Say how many pairs the paired SE actually covered. Silently reporting "x/14" while only
        # 3 pairs had per-window data is how a headline number gets quoted at the wrong n.
        print(f"  unresolved by the truth under the PAIRED SE: {unres}/{have} "
              f"({have} of {len(P)} pairs have per-window data)"
              + ("" if units else "   -- rerun eval_groundtruth to generate _units.json"))
        for t, p in preds.items():
            ok = sum(1 for u, v in P if (p[u] < p[v]) == (truth[u] < truth[v]))
            print(f"  {t:12s} {ok}/{len(P)}")
        for lab, f in [("params", lambda n: arch[n]["w"]),
                       ("n_h*d_qk", lambda n: arch[n]["attn_width"]),
                       ("larger d_qk", lambda n: arch[n]["d_qk"])]:
            ok = n_ = 0
            for u, v in P:
                if f(u) == f(v):
                    continue
                n_ += 1; ok += (f(u) > f(v)) == (truth[u] < truth[v])
            if n_:
                print(f"  FREE {lab:9s} {ok}/{n_}")

    # ---- the pre-registered prospective test ---------------------------------------------------
    if os.path.exists(a.prereg):
        pr = json.load(open(a.prereg))["predictions"]
        held = [n for n in pr if n in truth]
        if held:
            print(f"\nPRE-REGISTERED prospective test: {len(held)}/{len(pr)} predicted shapes now "
                  f"have a trained reference. Fitted on the 27 done before they existed, so this "
                  f"is out-of-sample by construction.")
            for m in sorted(next(iter(pr.values()))):
                e = np.array([truth[n] - pr[n][m] for n in held])
                print(f"  {m:22s} RMSE {np.sqrt((e**2).mean()):.4f}  bias {e.mean():+.4f}")
        else:
            print(f"\nPRE-REGISTERED test: 0/{len(pr)} predicted shapes trained yet.")


if __name__ == "__main__":
    main()
