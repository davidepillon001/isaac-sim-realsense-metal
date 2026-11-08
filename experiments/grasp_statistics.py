"""Statistics of the grasp experiment, with scenes as clusters.

1. Success per model and finish with a 95% cluster-bootstrap interval (resampling scenes) and the design effect.
2. Paired comparisons on the same profiles: exact McNemar test, Holm correction over all comparisons of one
   estimator, difference in success with its bootstrap interval.
3. Logistic GEE on polished profiles (exchangeable correlation within scenes): odds ratios for orientation to
   the baseline, distance and length.

    python -m experiments.grasp_statistics --runs results/grasp_percentile results/grasp_mode --out results/statistics
"""
import argparse
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import binomtest

REALISTIC = ["heuristic", "empirical_metric", "empirical_disparity", "stereo", "stereo_sgbm_strict", "stereo_hdr"]
COMPARE = [("gaussian", "stereo"), ("gaussian", "empirical_metric"), ("gaussian", "empirical_disparity"),
           ("heuristic", "stereo"), ("empirical_metric", "stereo"), ("empirical_disparity", "stereo"),
           ("empirical_metric", "empirical_disparity"), ("stereo", "stereo_sgbm_strict"), ("stereo", "stereo_hdr")]


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def boot_weights(n_scenes, B, rng):
    """How many times each scene is drawn in each of B bootstrap replicates."""
    idx = rng.integers(0, n_scenes, size=(B, n_scenes))
    W = np.zeros((B, n_scenes), np.int32)
    for b in range(B):
        np.add.at(W[b], idx[b], 1)
    return W


def holm(p):
    p = np.asarray(p, float)
    adj = np.empty_like(p)
    run = 0.0
    for r, i in enumerate(np.argsort(p)):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def analyse(df, estimator, B, rng):
    df = df.copy()
    df["success"] = df.success.astype(bool) & df.grasp_found.astype(bool)
    df["key"] = df.scene + "|" + df.object
    scenes = np.array(sorted(df.scene.unique()))
    sidx = {s: i for i, s in enumerate(scenes)}
    W = boot_weights(len(scenes), B, rng)

    rows = []
    for (model, fin), g in df.groupby(["model", "finish"]):
        k = g.groupby("scene").success.sum().reindex(scenes, fill_value=0).values
        n = g.groupby("scene").success.size().reindex(scenes, fill_value=0).values
        K, N = k.sum(), n.sum()
        boot = (W @ k) / np.maximum(W @ n, 1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        wl, wh = wilson(K, N)
        p = K / N
        var_bin = p * (1 - p) / N
        rows.append({"estimator": estimator, "model": model, "finish": fin, "n": int(N), "success_pct": 100 * p,
                     "ci_low": 100 * lo, "ci_high": 100 * hi, "wilson_low": 100 * wl, "wilson_high": 100 * wh,
                     "design_effect": np.var(boot) / var_bin if var_bin > 0 else np.nan})
    S = pd.DataFrame(rows)

    piv = df.pivot_table(index=["key", "scene", "finish"], columns="model", values="success", aggfunc="first").reset_index()
    comp = []
    for fin in ("anodized", "polished"):
        P = piv[piv.finish == fin]
        si = P.scene.map(sidx).values
        for m1, m2 in COMPARE:
            if m1 not in P or m2 not in P:
                continue
            ok = P[m1].notna() & P[m2].notna()
            x1, x2, s = P[m1][ok].astype(bool).values, P[m2][ok].astype(bool).values, si[ok.values]
            n10, n01 = int((x1 & ~x2).sum()), int((~x1 & x2).sum())
            pval = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 > 0 else 1.0
            d = x1.astype(float) - x2.astype(float)
            boot = (W @ np.bincount(s, weights=d, minlength=len(scenes))) / np.maximum(
                W @ np.bincount(s, minlength=len(scenes)), 1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            comp.append({"estimator": estimator, "finish": fin, "model_1": m1, "model_2": m2, "n": int(ok.sum()),
                         "only_1": n10, "only_2": n01, "diff_pt": 100 * d.mean(), "ci_low": 100 * lo,
                         "ci_high": 100 * hi, "p_mcnemar": pval})
    C = pd.DataFrame(comp)
    C["p_holm"] = holm(C.p_mcnemar.values)

    pol = df[(df.finish == "polished") & df.model.isin(REALISTIC)].copy()
    pol["parallel"] = np.cos(np.radians(pol.angle_to_baseline_deg.astype(float))) ** 2  # 1 = parallel to baseline
    pol["distance_dm"] = (pol.distance_m.astype(float) - 0.9) * 10
    pol["length_dm"] = (pol.length_m.astype(float) - pol.length_m.astype(float).mean()) * 10
    pol["y"] = pol.success.astype(int)
    G = []
    for model, g in pol.groupby("model"):
        if g.y.nunique() < 2:
            continue
        m = smf.gee("y ~ parallel + distance_dm + length_dm", groups="scene", data=g,
                    family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
        ci = m.conf_int()
        for term in m.params.index[1:]:
            G.append({"estimator": estimator, "model": model, "factor": term, "odds_ratio": float(np.exp(m.params[term])),
                      "ci_low": float(np.exp(ci.loc[term, 0])), "ci_high": float(np.exp(ci.loc[term, 1])),
                      "p": float(m.pvalues[term]), "within_scene_corr": float(m.cov_struct.dep_params)})
    return S, C, pd.DataFrame(G)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="output folders of experiments.grasp")
    ap.add_argument("--out", default="results/statistics")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    parts = [analyse(pd.read_csv(os.path.join(r, "grasps.csv")), os.path.basename(r.rstrip("/")).split("_")[-1],
                     a.boot, rng) for r in a.runs]
    S, C, G = (pd.concat(t) for t in zip(*parts))
    S.to_csv(os.path.join(a.out, "success.csv"), index=False)
    C.to_csv(os.path.join(a.out, "paired_comparisons.csv"), index=False)
    G.to_csv(os.path.join(a.out, "gee_polished.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(S.round(2).to_string(index=False))
        print(C.round(4).to_string(index=False))
        print(G.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
