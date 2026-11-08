"""Sensitivity of the predicted grasp success to the roughness of the polished finish.

The first 30 scenes are re-rendered with another polished roughness (same seed, same geometry), the stereo
depth is computed, and the grasp is evaluated with ideal and stereo depth. Roughness 0.10 is the main dataset.

    python -m experiments.roughness --variants 0.05:data/polished_r005 0.20:data/polished_r020 \
        0.30:data/polished_r030 --main 0.10:data/scenes_main --out results/roughness
"""
import argparse
import os

import numpy as np
import pandas as pd

from experiments.grasp import GRIPPER, run


def boot_ci(g, B=5000, seed=0):
    k = g.groupby("scene").ok.sum()
    n = g.groupby("scene").ok.size()
    idx = np.random.default_rng(seed).integers(0, len(k), (B, len(k)))
    p = k.values[idx].sum(1) / n.values[idx].sum(1)
    return 100 * k.sum() / n.sum(), *np.percentile(100 * p, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", required=True, help="roughness:scene_folder")
    ap.add_argument("--main", required=True, help="roughness:scene_folder of the main dataset")
    ap.add_argument("--n-scenes", type=int, default=30)
    ap.add_argument("--out", default="results/roughness")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    frames, rows = [], []
    for spec in sorted([a.main] + a.variants, key=lambda s: float(s.split(":")[0])):
        rough, root = spec.split(":")
        for est in ("percentile", "mode"):
            d = run(root, est, models=["ideal", "stereo"])
            d = d[d.scene.isin(sorted(d.scene.unique())[:a.n_scenes]) & (d.finish == "polished")].copy()
            d["ok"] = d.success.astype(bool) & d.grasp_found.astype(bool)
            d["roughness"], d["estimator"] = float(rough), est
            frames.append(d)
            for model, g in d.groupby("model"):
                p, lo, hi = boot_ci(g)
                failed = g[g.grasp_found.astype(bool) & ~g.ok]
                rows.append({"estimator": est, "roughness": float(rough), "model": model, "n": len(g),
                             "success_pct": p, "ci_low": lo, "ci_high": hi,
                             "failures_height_pct": 100 * (failed.height_mm >= GRIPPER["height_mm"]).mean()
                             if len(failed) else np.nan})
        print(f"roughness {rough} done", flush=True)
    G = pd.concat(frames)
    G.to_csv(os.path.join(a.out, "grasps.csv"), index=False)
    G["baseline_angle"] = pd.cut(G.angle_to_baseline_deg, [-1, 30, 60, 91], labels=["0-30", "30-60", "60-90"])
    A = G[G.model == "stereo"].pivot_table(index=["estimator", "roughness"], columns="baseline_angle", values="ok",
                                           aggfunc="mean", observed=True) * 100
    A.to_csv(os.path.join(a.out, "stereo_by_baseline_angle.csv"))
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(a.out, "success_vs_roughness.csv"), index=False)
    print(T.round(1).to_string(index=False))
    print(A.round(1).to_string())


if __name__ == "__main__":
    main()
