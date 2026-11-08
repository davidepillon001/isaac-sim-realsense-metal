"""Depth error on metal, simulated vs. real D415.

1. Middlebury v3 metrics of the disparity error (bad 0.5/1/2/4, mean, RMS, A50-A99) and the mm metrics of
   ActiveZero (mean absolute error, share > 4 mm). Errors on valid pixels, invalid pixels reported apart.
2. Wasserstein-1 distance and Kolmogorov-Smirnov statistic between |error| distributions.
3. Sim-vs-real correlation (SRCC, Kadian et al. 2020) of the per-angle curves, with Fisher intervals.
The matte table of ROBI is reported as well: it gives the noise of the real camera on a textured surface.

    python -m experiments.depth_metrics --sim data/scenes_main \
        --robi-zigzag data/robi/Zigzag/Scene_4 data/robi/Zigzag/Scene_5 \
        --robi-chrome data/robi/Chrome_screw/Scene_4 data/robi/Chrome_screw/Scene_5 \
        --xyzibd data/xyzibd --curves results/angle_curves.csv --out results/depth_metrics
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, pearsonr, spearmanr, wasserstein_distance

from d415 import DEPTH_MODELS
from d415.scene import finish_masks, load_scene
from datasets import robi, xyzibd

PER_IMAGE = 20000          # error samples kept per image; rates use all pixels
MODELS = ["gaussian", "heuristic", "empirical_metric", "empirical_disparity", "stereo"]


class Errors:
    def __init__(self):
        self.n = 0
        self.invalid = 0
        self.px, self.mm = [], []

    def add(self, meas, ref, mask, fB, rng, to_mm):
        m = mask & (ref > 0) & np.isfinite(ref)
        if m.sum() == 0:
            return
        valid = m & (meas > 0) & np.isfinite(meas)
        self.n += int(m.sum())
        self.invalid += int(m.sum() - valid.sum())
        e_px = fB / meas[valid] - fB / ref[valid]          # > 0: closer than the truth
        e_mm = (meas[valid] - ref[valid]) * to_mm          # > 0: farther than the truth
        if len(e_px) > PER_IMAGE:
            i = rng.choice(len(e_px), PER_IMAGE, replace=False)
            e_px, e_mm = e_px[i], e_mm[i]
        self.px.append(e_px.astype(np.float32))
        self.mm.append(e_mm.astype(np.float32))

    def arrays(self):
        return (np.concatenate(self.px) if self.px else np.zeros(0),
                np.concatenate(self.mm) if self.mm else np.zeros(0))


def metrics(e):
    px, mm = e.arrays()
    a = np.abs(px)
    r = {"pixels": e.n, "invalid_pct": 100 * e.invalid / max(e.n, 1)}
    if len(a) == 0:
        return r
    for t in (0.5, 1, 2, 4):
        r[f"bad{t}_pct"] = 100 * float((a > t).mean())
    r.update({"avgerr_px": float(a.mean()), "rms_px": float(np.sqrt((a ** 2).mean())),
              "A50_px": float(np.percentile(a, 50)), "A90_px": float(np.percentile(a, 90)),
              "A95_px": float(np.percentile(a, 95)), "A99_px": float(np.percentile(a, 99)),
              "abs_err_mm": float(np.abs(mm).mean()), "over_4mm_pct": 100 * float((np.abs(mm) > 4).mean()),
              "wrong_behind_pct": 100 * float((px[a > 1] < 0).mean()) if (a > 1).any() else np.nan})
    return r


def fisher_ci(r, n):
    if n <= 3 or not np.isfinite(r) or abs(r) >= 1:
        return np.nan, np.nan
    z, se = np.arctanh(r), 1 / np.sqrt(n - 3)
    return float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))


def srcc(curves_csv):
    C = pd.read_csv(curves_csv)
    real = {s: g.set_index("angle") for s, g in C[C.source.str.startswith("real") & (C.region == "metal")].groupby("source")}
    S = C[C.source.str.startswith("sim ") & C.region.str.startswith("metal ")]
    rows = []
    for (src, reg), g in S.groupby(["source", "region"]):
        g = g.set_index("angle")
        for name, R in real.items():
            common = [b for b in g.index if b in R.index]
            s = np.concatenate([g.loc[common, "holes_pct"].values, g.loc[common, "wrong_pct"].values])
            r = np.concatenate([R.loc[common, "holes_pct"].values, R.loc[common, "wrong_pct"].values])
            ok = np.isfinite(s) & np.isfinite(r)
            s, r = s[ok], r[ok]
            if len(s) < 4:
                continue
            pr = pearsonr(s, r)[0]
            lo, hi = fisher_ci(pr, len(s))
            rows.append({"model": src[4:], "finish": reg[6:], "real": name, "points": len(s), "srcc": pr,
                         "ci_low": lo, "ci_high": hi, "spearman": spearmanr(s, r)[0],
                         "mae_pt": float(np.mean(np.abs(s - r)))})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default="data/scenes_main")
    ap.add_argument("--max-scenes", type=int, default=200)
    ap.add_argument("--robi-zigzag", nargs="*", default=[])
    ap.add_argument("--robi-chrome", nargs="*", default=[])
    ap.add_argument("--max-views", type=int, default=125)
    ap.add_argument("--xyzibd", default=None)
    ap.add_argument("--xyzibd-frames", type=int, default=150)
    ap.add_argument("--curves", default="results/angle_curves.csv")
    ap.add_argument("--out", default="results/depth_metrics")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    real, table = {}, Errors()
    rng_table = np.random.default_rng(a.seed + 1)
    for name, scenes in (("ROBI Zigzag", a.robi_zigzag), ("ROBI Chrome", a.robi_chrome)):
        if scenes:
            e = real[name] = Errors()
            for scene in scenes:
                for rs, gt_rs, obj, floor, _ in robi.views(scene, a.max_views):
                    e.add(rs, gt_rs, obj, robi.FB_RS, rng, to_mm=1.0)
                    table.add(rs, gt_rs, floor, robi.FB_RS, rng_table, to_mm=1.0)
    if a.xyzibd:
        e = real["XYZ-IBD"] = Errors()
        for raw, gt, m, _, fB in xyzibd.frames(a.xyzibd, max_frames=a.xyzibd_frames):
            e.add(raw, gt, m, fB, rng, to_mm=1.0)

    sim = {(m, f): Errors() for m in MODELS for f in ("anodized", "polished")}
    for sd in sorted(glob.glob(os.path.join(a.sim, "scene_*")))[:a.max_scenes]:
        g = load_scene(sd)
        ref = np.where(np.isfinite(g["depth"]), g["depth"], 0)
        masks = finish_masks(g)
        for model in MODELS:
            path = f"{sd}/{DEPTH_MODELS[model]}"
            if os.path.exists(path):
                z = np.load(path)
                for fin, m in masks.items():
                    sim[(model, fin)].add(z, ref, m, g["fx"] * g["B"], rng, to_mm=1e3)

    rows = [{"source": f"real {k}", "finish": "metal", **metrics(v)} for k, v in real.items()]
    rows += [{"source": m, "finish": f, **metrics(v)} for (m, f), v in sim.items() if v.n]
    if table.n:
        rows.append({"source": "real ROBI table", "finish": "matte", **metrics(table)})
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(a.out, "middlebury.csv"), index=False)

    pairs = {"anodized": [k for k in real if k != "ROBI Chrome"], "polished": [k for k in real if k == "ROBI Chrome"]}
    D = []
    for (m, fin), v in sim.items():
        if not v.n:
            continue
        sp = np.abs(v.arrays()[0])
        for k in pairs[fin]:
            rp = np.abs(real[k].arrays()[0])
            if len(sp) == 0 or len(rp) == 0:
                continue
            ss = sp if len(sp) <= 200000 else rng.choice(sp, 200000, replace=False)
            rr = rp if len(rp) <= 200000 else rng.choice(rp, 200000, replace=False)
            D.append({"model": m, "finish": fin, "real": k, "wasserstein_px": float(wasserstein_distance(ss, rr)),
                      "ks": float(ks_2samp(ss, rr).statistic),
                      "invalid_diff_pt": 100 * (v.invalid / v.n - real[k].invalid / real[k].n)})
    D = pd.DataFrame(D)
    D.to_csv(os.path.join(a.out, "distances.csv"), index=False)

    # samples for the exceedance figure, at most 50k per source to keep the file small
    keep = np.random.default_rng(1)
    samples = {f"real|{k}": v.arrays()[0] for k, v in real.items()}
    samples.update({f"sim|{m}|{f}": v.arrays()[0] for (m, f), v in sim.items() if v.n})
    samples = {k: (x if len(x) <= 50000 else keep.choice(x, 50000, replace=False)) for k, x in samples.items()}
    np.savez_compressed(os.path.join(a.out, "error_samples_px.npz"), **samples)

    R = srcc(a.curves) if os.path.exists(a.curves) else pd.DataFrame()
    R.to_csv(os.path.join(a.out, "srcc.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(T.round(2).to_string(index=False))
        print(D.sort_values(["finish", "real", "wasserstein_px"]).round(3).to_string(index=False))
        if len(R):
            print(R.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
