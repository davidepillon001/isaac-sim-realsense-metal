"""Does the gap between Gaussian and stereo depth depend on the grasp tolerance? Success on polished profiles is
recomputed from the stored errors with lateral and height tolerance tau, and the two models are compared on the
same profiles with the exact McNemar test.

    python -m experiments.tolerance_sensitivity --runs results/grasp_percentile results/grasp_mode
"""
import argparse
import os

import pandas as pd
from scipy.stats import binomtest

from experiments.grasp import GRIPPER


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--tolerances-mm", nargs="+", type=float, default=[5, 10, 15])
    ap.add_argument("--out", default="results/statistics/tolerance_sensitivity.csv")
    a = ap.parse_args()

    rows = []
    for run in a.runs:
        d = pd.read_csv(os.path.join(run, "grasps.csv"))
        d = d[d.finish == "polished"]
        for tau in a.tolerances_mm:
            ok = (d.grasp_found.astype(bool) & (d.lateral_mm < tau) & (d.height_mm < tau)
                  & (d.angle_deg < GRIPPER["angle_deg"]) & (d.along_mm < GRIPPER["along_mm"]))
            p = d.assign(ok=ok).pivot_table(index=["scene", "object"], columns="model", values="ok", aggfunc="first")
            g, s = p.gaussian.astype(bool), p.stereo.astype(bool)
            n10, n01 = int((g & ~s).sum()), int((~g & s).sum())
            rows.append({"estimator": os.path.basename(run.rstrip("/")).split("_")[-1], "tolerance_mm": tau,
                         "gaussian_pct": 100 * g.mean(), "stereo_pct": 100 * s.mean(),
                         "diff_pt": 100 * (g.mean() - s.mean()), "only_gaussian": n10, "only_stereo": n01,
                         "p_mcnemar": binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0})
    T = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    T.to_csv(a.out, index=False)
    print(T.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
