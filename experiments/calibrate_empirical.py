"""Calibrate the empirical artefact model on one ROBI scene and check it on another, unseen scene.

Measured on metal pixels of the real D415, per incidence-angle bin: hole rate, rate of wrong pixels
(|disparity error| > 1 px), error distribution of wrong pixels (px and mm) and robust noise of good pixels.

    python -m experiments.calibrate_empirical --fit data/robi/Zigzag/Scene_4 --val data/robi/Zigzag/Scene_5 \
        --out results/calibration/zigzag.json
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from datasets import robi


def measure(scene, max_views, n_err_samples=200000, seed=0):
    nb = len(robi.ANGLE_BINS) - 1
    acc = np.zeros((nb, 3))                     # total, holes, wrong
    errs, errs_mm, good = [], [], []
    used = 0
    for rs, gt_rs, obj, _, cos_t in robi.views(scene, max_views):
        m = obj & (gt_rs > 0) & np.isfinite(cos_t)
        ang = np.degrees(np.arccos(np.clip(cos_t[m], 0, 1)))
        b = np.clip(np.digitize(ang, robi.ANGLE_BINS) - 1, 0, nb - 1)
        meas, ref = rs[m], gt_rs[m]
        valid = meas > 0
        ed = np.zeros_like(ref)
        ed[valid] = robi.FB_RS / meas[valid] - robi.FB_RS / ref[valid]   # > 0: closer than the truth
        wrong = valid & (np.abs(ed) > 1.0)
        np.add.at(acc[:, 0], b, 1)
        np.add.at(acc[:, 1], b, ~valid)
        np.add.at(acc[:, 2], b, wrong)
        errs.append(ed[wrong])
        errs_mm.append(meas[wrong] - ref[wrong])                         # > 0: farther than the truth
        good.append(ed[valid & ~wrong])
        used += 1
    rng = np.random.default_rng(seed)
    errs, errs_mm, good = np.concatenate(errs), np.concatenate(errs_mm), np.concatenate(good)
    if len(errs) > n_err_samples:
        idx = rng.choice(len(errs), n_err_samples, replace=False)
        errs, errs_mm = errs[idx], errs_mm[idx]
    return {"acc": acc, "errs": errs, "errs_mm": errs_mm, "good": good, "views": used}


def rates(acc, min_px=500):
    n = acc[:, 0]
    ok = n >= min_px
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(ok, acc[:, 1] / n, np.nan), np.where(ok, acc[:, 2] / n, np.nan), ok


def fill_nan(x):
    x = x.copy()
    idx = np.where(~np.isnan(x))[0]
    for i in np.where(np.isnan(x))[0]:
        x[i] = x[idx[np.argmin(np.abs(idx - i))]]
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True, help="calibration json")
    ap.add_argument("--max-views", type=int, default=125)
    ap.add_argument("--sim-curves", help="angle curves of the simulated models (experiments.angle_curves), "
                                         "scored on the held-out scene as well")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    fit, val = measure(a.fit, a.max_views), measure(a.val, a.max_views)
    ph_fit, pw_fit, _ = rates(fit["acc"])
    ph_val, pw_val, ok_val = rates(val["acc"])
    centers = (robi.ANGLE_BINS[:-1] + robi.ANGLE_BINS[1:]) / 2
    good = fit["good"]
    calib = {
        "fit_scene": a.fit, "val_scene": a.val, "views_fit": fit["views"], "views_val": val["views"],
        "angle_centers_deg": centers.tolist(),
        "p_hole": fill_nan(ph_fit).tolist(), "p_wrong": fill_nan(pw_fit).tolist(),
        "err_quantiles_px": np.quantile(fit["errs"], np.linspace(0, 1, 201)).tolist(),
        "err_quantiles_mm": np.quantile(fit["errs_mm"], np.linspace(0, 1, 201)).tolist(),
        "s_metal_px": float(1.4826 * np.median(np.abs(good - np.median(good)))),
        "frac_wrong_farther": float((fit["errs"] < 0).mean()),
    }
    json.dump(calib, open(a.out, "w"), indent=1)

    ph, pw = fill_nan(ph_fit), fill_nan(pw_fit)
    m = ok_val
    mae_h = 100 * np.mean(np.abs(ph[m] - ph_val[m]))
    mae_w = 100 * np.mean(np.abs(pw[m] - pw_val[m]))
    labels = [f"{robi.ANGLE_BINS[i]}-{robi.ANGLE_BINS[i + 1]}" for i in range(len(centers))]
    T = pd.DataFrame({"angle": labels, "holes_fit": 100 * ph_fit, "holes_val": 100 * ph_val,
                      "wrong_fit": 100 * pw_fit, "wrong_val": 100 * pw_val})
    T.to_csv(a.out.replace(".json", "_fit_vs_val.csv"), index=False)
    print(T.round(1).to_string(index=False))
    scores = [{"model": "empirical (calibrated)", "mae_holes_pt": mae_h, "mae_wrong_pt": mae_w}]
    if a.sim_curves:
        S = pd.read_csv(a.sim_curves)
        S = S[S.source.str.startswith("sim ") & S.region.str.startswith("metal")]
        for (src, reg), g in S.groupby(["source", "region"]):
            g = g.set_index("angle").reindex(labels)
            sh, sw = g.holes_pct.values / 100, g.wrong_pct.values / 100
            k = ok_val & np.isfinite(sh) & np.isfinite(sw)
            scores.append({"model": f"{src[4:]} ({reg[6:]})", "mae_holes_pt": 100 * np.mean(np.abs(sh[k] - ph_val[k])),
                           "mae_wrong_pt": 100 * np.mean(np.abs(sw[k] - pw_val[k]))})
    scores = pd.DataFrame(scores)
    scores.to_csv(a.out.replace(".json", "_heldout_scores.csv"), index=False)
    print("held-out scene, mean absolute error of the angle curves [pt]:")
    print(scores.round(1).to_string(index=False))
    print(f"good-pixel noise {calib['s_metal_px']:.2f} px, wrong pixels behind the surface "
          f"{100 * calib['frac_wrong_farther']:.0f}%")


if __name__ == "__main__":
    main()
