"""Rank fidelity, reported the way it has to be reported to mean anything.

The headline correlation over all 16 architectures is NOT the result. The four cost levels do not
overlap in weight count (L1 0.812-0.875, L2 0.688-0.750, L3 0.500-0.562, L4 0.375-0.438), so
parameter count alone sorts them into the right blocks. Simulated: a predictor with ZERO ability to
order architectures at matched cost still scores rho ~0.92 over the 16, against a p<0.001 threshold
of 0.756. Quoting that number would announce a working predictor that a zero-GPU-hour parameter
count reproduces.

So three numbers are primary, and the overall rho is decoration:

  W-only baseline   Spearman(weight count, ground truth). The bar the supernet must clear. If it
                    does not, the supernet is a parameter counter.
  within-level      Spearman inside each cost level, and pooled. This is the question a search
                    actually asks -- which shape wins at a fixed budget. Note n=4 per level, so a
                    perfect ordering in one level still only reaches p=0.085; only the pooled
                    figure is testable, and its power is 0.18-0.38 for true correlations of 0.3-0.5.
  partial           Spearman(slice, truth) controlling for W, over all 16.

Both sides must be scored on the SAME benchmark or this measures nothing about weight sharing.
"""
import argparse, json
import itertools
import numpy as np
from scipy.stats import kendalltau, spearmanr

from llmforge.supernet.paths import RUNS


def partial_spearman(x, y, z):
    """Spearman between x and y with z partialled out, on ranks."""
    rx, ry, rz = (np.argsort(np.argsort(v)).astype(float) for v in (x, y, z))
    def resid(a):
        A = np.vstack([rz, np.ones_like(rz)]).T
        return a - A @ np.linalg.lstsq(A, a, rcond=None)[0]
    return spearmanr(resid(rx), resid(ry)).correlation


def perm_p(x, y, n=200000, seed=0):
    rng = np.random.default_rng(seed)
    obs = spearmanr(x, y).correlation
    y = np.asarray(y)
    null = np.array([spearmanr(x, rng.permutation(y)).correlation for _ in range(n // 100)])
    return obs, (np.abs(null) >= abs(obs)).mean()


_NULL_CACHE = {}


def block_null_dist(n):
    """Exact null for concordant-pair count against a fixed reference ordering of n items.

    NOT binomial. The pairs are not independent: transitivity means only n! of the 2^(n choose 2)
    sign patterns are realizable, so a coin-flip null is anticonservative. At n=4 the true
    distribution is (1,3,5,6,5,3,1)/24, giving P(6/6)=0.042 where the binomial claims 0.016 --
    a 2.7x overstatement, which is the difference between "significant" and "marginal".
    """
    if n in _NULL_CACHE:
        return _NULL_CACHE[n]
    from collections import Counter
    ref = list(range(n))
    c = Counter()
    pr = itertools.combinations(range(n), 2)
    pr = list(pr)
    if n <= 8:
        src = itertools.permutations(range(n))
    else:
        # n! stops being enumerable past ~8. The per-level blocks are the ones that must be exact;
        # the overall figure is decoration, so sample the same null there instead of hanging.
        rng = np.random.default_rng(0)
        src = (rng.permutation(n) for _ in range(20000))
    for perm in src:
        c[sum(1 for i, j in pr if (perm[i] < perm[j]) == (ref[i] < ref[j]))] += 1
    tot = sum(c.values())
    _NULL_CACHE[n] = {k: v / tot for k, v in c.items()}
    return _NULL_CACHE[n]


def block_null_p(n, conc):
    d = block_null_dist(n)
    return sum(v for k, v in d.items() if k >= conc)


def pooled_block_p(sizes, conc):
    """Exact pooled null: permute WITHIN each cost level independently, convolve the blocks.

    Treating the pooled pairs as independent coin flips overstates the evidence roughly 9x here,
    because the pairs come from a handful of architectures and inherit transitivity inside every
    block.
    """
    dist = {0: 1.0}
    for n in sizes:
        b = block_null_dist(n)
        nd = {}
        for k, v in dist.items():
            for k2, v2 in b.items():
                nd[k + k2] = nd.get(k + k2, 0.0) + v * v2
        dist = nd
    return sum(v for k, v in dist.items() if k >= conc)


def pairwise_report(name, score, truth, se=None, tag=""):
    """Pair-level concordance, which is the only honest statistic at these sample sizes.

    Reported instead of a bare tau because (a) tau-b silently rewards a predictor for having no
    ties when the baseline it is compared against is tie-dominated -- weight count is constant
    within a cost level, so every within-level pair is a tie for it -- and (b) at n<10 a tau
    carries so few distinct achievable values that quoting three decimals implies precision that
    does not exist. Concordant-out-of-total plus an exact binomial p says exactly what was seen.

    `se` (standard error of each truth value) marks pairs the ground truth cannot actually
    separate. A pair the reference cannot resolve tests nothing, and counting it either way
    inflates or deflates the score.
    """
    score = np.asarray(score, dtype=float)
    truth = np.asarray(truth, dtype=float)
    pairs = list(itertools.combinations(range(len(score)), 2))
    conc = disc = tie = 0
    unres = 0
    for i, j in pairs:
        ds, dt = score[i] - score[j], truth[i] - truth[j]
        if se is not None and abs(dt) < 1.96 * np.hypot(se[i], se[j]):
            unres += 1
        if ds == 0 or dt == 0:
            tie += 1
        elif (ds > 0) == (dt > 0):
            conc += 1
        else:
            disc += 1
    n = conc + disc
    tau_a = (conc - disc) / len(pairs) if pairs else float("nan")
    p = block_null_p(len(score), conc)
    print(f"  {name:22} {conc}/{n} concordant  tau-a {tau_a:+.3f}  "
          f"ties {tie}  one-sided p {p:.3f}  {tag}")
    if unres:
        print(f"  {'':22} ({unres}/{len(pairs)} pairs the GROUND TRUTH cannot resolve at 95%)")
    return conc, n, tau_a


def _counts(score, truth):
    c = d = 0
    for i, j in itertools.combinations(range(len(score)), 2):
        ds, dt = score[i] - score[j], truth[i] - truth[j]
        if ds == 0 or dt == 0:
            continue
        if (ds > 0) == (dt > 0):
            c += 1
        else:
            d += 1
    return c, c + d, (c - d) / max(c + d, 1)


def run_nll(a):
    """Both sides scored as held-out NLL. Converted to goodness = -NLL so that every reported
    correlation is positive-means-agreement, including the weight-count baseline."""
    S, T = json.load(open(a.slices)), json.load(open(a.truth))
    keys = sorted(set(S) & set(T) - {"full"})
    if len(keys) < 3:
        raise SystemExit(f"only {len(keys)} architectures in both files: {keys}")
    W = np.array([S[k]["w"] for k in keys])
    sl = np.array([-S[k]["nll"] for k in keys])
    gt = np.array([-T[k]["nll"] for k in keys])
    se = np.array([T[k].get("se", 0.0) for k in keys])
    lvl = np.array([int(k[1]) for k in keys])

    print(f"metric: held-out NLL (goodness = -NLL)   n={len(keys)}   arms: "
          f"{a.slices.split('/')[-1]} vs {a.truth.split('/')[-1]}\n")
    print(f"{'arch':6} {'W':>6} {'lvl':>4} {'slice_nll':>10} {'truth_nll':>10} {'se':>7}")
    for i, k in enumerate(keys):
        print(f"{k:6} {W[i]:6.3f} {lvl[i]:4d} {-sl[i]:10.4f} {-gt[i]:10.4f} {se[i]:7.4f}")

    print("\n--- OVERALL (decoration: cost levels do not overlap in W, so this is mostly W) ---")
    pairwise_report("W-only -> truth", W, gt, se, "<- zero GPU hours")
    pairwise_report("slice  -> truth", sl, gt, se)
    print(f"  Spearman slice-truth {spearmanr(sl, gt).correlation:+.3f}   "
          f"Kendall tau-b {kendalltau(sl, gt).correlation:+.3f}   "
          f"partial (control W) {partial_spearman(sl, gt, W):+.3f}")

    print("\n--- WITHIN COST LEVEL (W is only NEARLY constant here -- see the baseline below) ---")
    tc = tn = 0
    sizes = []
    for L in sorted(set(lvl)):
        m = lvl == L
        if m.sum() < 2:
            print(f"  L{L}  n={m.sum()}  -- skipped, needs >=2")
            continue
        c, n, _ = pairwise_report(f"L{L} (n={m.sum()})", sl[m], gt[m], se[m],
                                  f"W {W[m].min():.3f}-{W[m].max():.3f}")
        tc += c; tn += n; sizes.append(int(m.sum()))
    if tn:
        p = pooled_block_p(sizes, tc)
        print(f"\n  POOLED within-level: {tc}/{tn} concordant   exact block-permutation p {p:.4f}   "
              f"{'significant' if p < 0.05 else 'NOT significant at 0.05'}")
        wc, wn = 0, 0
        for L in sorted(set(lvl)):
            m = lvl == L
            c, n, _ = _counts(W[m], gt[m])
            wc += c; wn += n
        print(f"  weight count within-level: {wc}/{wn}  <- the real baseline. W is NOT constant "
              f"inside a level, so this is not zero.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", default=str(RUNS / "slice_scores_qwen3-1.7b.json"))
    ap.add_argument("--truth", default=str(RUNS / "groundtruth_eval.json"))
    ap.add_argument("--metric", default="hellaswag")
    ap.add_argument("--nll", action="store_true",
                    help="both files are {arch: {nll, se, w}} instead of benchmark scores")
    a = ap.parse_args()
    if a.nll:
        return run_nll(a)
    S = json.load(open(a.slices)); T = json.load(open(a.truth))
    keys = sorted(set(S) & set(T) - {"full"})
    if len(keys) < 4:
        raise SystemExit(f"only {len(keys)} architectures in both files")
    W = np.array([S[k]["w"] for k in keys])
    sl = np.array([S[k][a.metric] for k in keys])
    gt = np.array([T[k][a.metric] for k in keys])
    lvl = np.array([int(k[1]) for k in keys])

    print(f"metric: {a.metric}   n={len(keys)}\n")
    print(f"{'arch':6} {'W':>6} {'slice':>8} {'truth':>8}")
    for i, k in enumerate(keys):
        print(f"{k:6} {W[i]:6.3f} {sl[i]:8.1f} {gt[i]:8.1f}")

    r_w, p_w = perm_p(W, gt)
    r_s, p_s = perm_p(sl, gt)
    print(f"\n--- the bar ---")
    print(f"  W-only  -> truth : rho {r_w:+.3f}  (p {p_w:.4f})   <- zero GPU hours")
    print(f"  slice   -> truth : rho {r_s:+.3f}  (p {p_s:.4f})")
    print(f"  supernet beats parameter count: {'YES' if r_s > r_w else 'NO'}  "
          f"(delta {r_s - r_w:+.3f})")
    print(f"  partial (control W): rho {partial_spearman(sl, gt, W):+.3f}")

    print(f"\n--- within cost level (the question a search asks) ---")
    rs = []
    for L in sorted(set(lvl)):
        m = lvl == L
        if m.sum() < 3:
            continue
        r = spearmanr(sl[m], gt[m]).correlation
        rs.append(r)
        print(f"  L{L}  n={m.sum()}  rho {r:+.3f}   W {W[m].min():.3f}-{W[m].max():.3f}")
    if rs:
        pooled = float(np.mean(rs))
        print(f"  pooled mean rho {pooled:+.3f}   (null sd 0.29, 95% crit 0.50 at 4 levels of n=4)")
        print(f"  => {'significant' if abs(pooled) > 0.50 else 'NOT significant'}")


if __name__ == "__main__":
    main()
