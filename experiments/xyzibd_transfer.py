"""Does the error of a real D415 on metal stay constant in millimetres or in disparity pixels as the distance
grows? Error quantiles per distance bin on XYZ-IBD (0.5-0.85 m), compared with ROBI (~0.35 m).
If a quantile scales as z^alpha: metric transfer means alpha = 0, disparity transfer alpha = 2.

    python -m experiments.xyzibd_transfer data/xyzibd --calib results/calibration/zigzag.json \
        results/calibration/chrome.json --out results/xyzibd_transfer.csv
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from datasets import xyzibd

DIST_BINS = np.array([500, 650, 750, 850, 950, 1100])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--calib", nargs="*", default=[])
    ap.add_argument("--out", default="results/xyzibd_transfer.csv")
    a = ap.parse_args()

    rows, samples = [], []
    for i, (raw, gt, m, _, fB) in enumerate(xyzibd.frames(a.root), start=1):
        valid = m & (raw > 0)
        ed = np.zeros_like(gt)
        ed[valid] = fB / raw[valid] - fB / gt[valid]
        wrong = valid & (np.abs(ed) > 1.0)
        dbin = np.digitize(gt[m], DIST_BINS) - 1
        for b in range(len(DIST_BINS) - 1):
            sel = dbin == b
            if sel.any():
                rows.append({"bin": b, "n": int(sel.sum()), "holes": int((~valid[m])[sel].sum()),
                             "wrong": int(wrong[m][sel].sum())})
        # all valid pixels, not only wrong ones: the 1 px threshold is a different mm value at each distance
        sub = np.flatnonzero(valid.ravel())
        if len(sub) > 20000:
            sub = np.random.default_rng(i).choice(sub, 20000, replace=False)
        samples.append(pd.DataFrame({"z_mm": gt.ravel()[sub], "err_mm": (raw - gt).ravel()[sub],
                                     "err_px": ed.ravel()[sub], "wrong": wrong.ravel()[sub]}))
    R = pd.DataFrame(rows).groupby("bin").sum()
    E = pd.concat(samples)
    E["bin"] = np.digitize(E.z_mm, DIST_BINS) - 1
    out = []
    for b, r in R.iterrows():
        e = E[E.bin == b]
        ew = e[e.wrong.astype(bool)]
        out.append({"distance_mm": f"{DIST_BINS[b]}-{DIST_BINS[b + 1]}", "pixels": int(r.n),
                    "holes_pct": 100 * r.holes / r.n, "wrong_pct": 100 * r.wrong / r.n,
                    "abs_err_p90_mm": np.percentile(np.abs(e.err_mm), 90),
                    "abs_err_p90_px": np.percentile(np.abs(e.err_px), 90),
                    "abs_err_p97_mm": np.percentile(np.abs(e.err_mm), 97),
                    "abs_err_p97_px": np.percentile(np.abs(e.err_px), 97),
                    "wrong_behind_pct": 100 * float((ew.err_mm > 0).mean()) if len(ew) else np.nan})
    for p in a.calib:
        c = json.load(open(p))
        out.append({"distance_mm": f"ROBI ~350 ({os.path.basename(p)[:-5]}, wrong pixels only)",
                    "abs_err_p90_mm": float(np.median(np.abs(c["err_quantiles_mm"]))),
                    "abs_err_p90_px": float(np.median(np.abs(c["err_quantiles_px"]))),
                    "wrong_behind_pct": 100 * c["frac_wrong_farther"]})
    T = pd.DataFrame(out)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    T.to_csv(a.out, index=False)
    print(T.round(2).to_string(index=False))

    # exponent alpha of q ~ z^alpha between the first and the third distance bin (centres 575 and 800 mm)
    c = (DIST_BINS[:-1] + DIST_BINS[1:]) / 2
    for q in ("p90", "p97"):
        q1, q2 = T.loc[0, f"abs_err_{q}_mm"], T.loc[2, f"abs_err_{q}_mm"]
        print(f"{q}: {q1:.1f} -> {q2:.1f} mm, alpha = {np.log(q2 / q1) / np.log(c[2] / c[0]):.2f}")


if __name__ == "__main__":
    main()
