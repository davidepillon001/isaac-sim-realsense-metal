"""Depth noise and bias of the stereo model on a matte plane, against the D400 datasheet.

Configurations: OpenCV SGBM on a fronto-parallel plane (shows pixel locking), SGBM with sub-pixel refinement on a
plane tilted by 25 deg, and the census + SGM model used everywhere else. Two more show what did not work: SGBM
without padding (empty left columns, lower fill rate) and three refinement steps instead of one.
Statistics on the central 80% of the image, as in Intel's depth-quality tests.

    python -m experiments.plane_noise --fronto data/plane_fronto --tilted data/plane_tilt25 --out results/plane_noise.csv
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np
import pandas as pd

from d415.stereo import depth_from_ir


def plane_stats(root, matcher, refine, **kw):
    rows = []
    for f in sorted(glob.glob(os.path.join(root, "plane_z*"))):
        cam = json.load(open(f"{f}/camera.json"))
        fx, B = cam["intrinsics"]["fx"], cam["baseline_m"]
        L, R = cv2.imread(f"{f}/ir_left.png", 0), cv2.imread(f"{f}/ir_right.png", 0)
        gt = np.load(f"{f}/depth_gt.npy")
        z = depth_from_ir(L, R, fx, B, matcher=matcher, refine=refine, **kw)
        h, w = z.shape
        roi = np.zeros_like(z, bool)
        roi[int(.1 * h):int(.9 * h), int(.1 * w):int(.9 * w)] = True
        v = roi & (z > 0)
        e = z[v] - gt[v]
        d = fx * B / z[v] - fx * B / gt[v]
        rows.append({"distance_m": float(np.median(gt[roi])), "fill_rate": float(v.sum() / roi.sum()),
                     "std_mm": 1e3 * float(np.std(e)), "median_err_mm": 1e3 * float(np.median(e)),
                     "disparity_std_px": float(np.std(d))})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fronto", default="data/plane_fronto")
    ap.add_argument("--tilted", default="data/plane_tilt25")
    ap.add_argument("--out", default="results/plane_noise.csv")
    a = ap.parse_args()

    runs = [("sgbm, fronto-parallel", a.fronto, "sgbm", False, {}),
            ("sgbm + subpixel, tilted 25", a.tilted, "sgbm", True, {}),
            ("census + sgm, tilted 25", a.tilted, "census", True, {}),
            ("sgbm without padding, fronto-parallel", a.fronto, "sgbm", False, {"pad": False}),
            ("census + sgm, 3 refinement steps, tilted 25", a.tilted, "census", True, {"refine_iters": 3})]
    rows = []
    for name, root, matcher, refine, kw in runs:
        rows += [{"config": name, **r} for r in plane_stats(root, matcher, refine, **kw)]
    T = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    T.to_csv(a.out, index=False)
    print(T.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
