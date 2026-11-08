"""Top-down two-finger grasp of every visible profile, evaluated with each depth model.

For each profile: ground-truth instance mask (segmentation assumed perfect), point cloud from the valid depth
pixels, top-face height (90th percentile or histogram mode), axis by PCA of the top face, lateral centre as the
midpoint of the 2-98% extent. Success is judged against the ground-truth pose with the gripper tolerances below.

    python -m experiments.grasp data/scenes_main --estimator percentile --out results/grasp_percentile
    python -m experiments.grasp data/scenes_main --estimator mode --out results/grasp_mode
"""
import argparse
import glob
import json
import math
import os

import numpy as np
import pandas as pd

from d415 import DEPTH_MODELS

PROFILE_W = 0.040
# parallel gripper, 70 mm opening, 20 mm pads, on a 40 mm profile
GRIPPER = {"lateral_mm": 10.0, "height_mm": 10.0, "angle_deg": 10.0, "along_mm": 30.0}
MIN_POINTS = 100


def backproject(z, V, fx):
    h, w = z.shape
    v, u = np.mgrid[0:h, 0:w]
    ok = np.isfinite(z) & (z > 0)
    zz = z[ok]
    pc = np.stack([(u[ok] + 0.5 - w / 2) / fx * zz, -(v[ok] + 0.5 - h / 2) / fx * zz, -zz, np.ones_like(zz)], -1)
    return ok, (pc @ np.linalg.inv(V))[:, :3]


def gt_grasp(T):
    T = np.asarray(T)       # USD row-vector convention: p_w = p_l @ T
    axis = T[0, :3] / np.linalg.norm(T[0, :3])
    center = (np.array([0, 0, PROFILE_W / 2, 1.0]) @ T)[:3]
    return center, axis


def top_height(z, estimator):
    if estimator == "percentile":
        return np.percentile(z, 90)
    z = z[(z > np.percentile(z, 1)) & (z < np.percentile(z, 99))]
    if len(z) < MIN_POINTS:
        return None
    hist, edges = np.histogram(z, bins=np.arange(z.min(), z.max() + 0.002, 0.001))
    hist = np.convolve(hist, np.exp(-0.5 * (np.arange(-6, 7) / 2.0) ** 2), mode="same")
    return edges[np.argmax(hist)] + 0.0005


def estimate_grasp(P, estimator, centre="extent"):
    if len(P) < MIN_POINTS:
        return None
    top = top_height(P[:, 2], estimator)
    if top is None:
        return None
    P = P[(P[:, 2] > top - PROFILE_W - 0.01) & (P[:, 2] < top + 0.01)]
    if len(P) < MIN_POINTS:
        return None
    top_face = P[P[:, 2] > top - 0.006]
    Q = top_face if len(top_face) >= 30 else P
    xy = Q[:, :2] - Q[:, :2].mean(0)
    a = np.linalg.svd(xy, full_matrices=False)[2][0]
    s = (P[:, :2] - Q[:, :2].mean(0)) @ a
    lo, hi = np.percentile(s, [2, 98])
    # midpoint of the extent, not the median: the central slot makes the section bimodal
    lat = xy @ np.array([-a[1], a[0]])
    lat_c = np.median(lat) if centre == "median" else np.mean(np.percentile(lat, [2, 98]))
    c_xy = Q[:, :2].mean(0) + a * (lo + hi) / 2 + np.array([-a[1], a[0]]) * lat_c
    return np.array([c_xy[0], c_xy[1], top - PROFILE_W / 2]), np.array([a[0], a[1], 0.0])


def evaluate(est, gt_c, gt_a):
    if est is None:
        return {"grasp_found": False, "success": False}
    c, a = est
    d = c - gt_c
    a_h = gt_a[:2] / np.linalg.norm(gt_a[:2])
    n_h = np.array([-a_h[1], a_h[0]])
    r = {"grasp_found": True, "lateral_mm": abs(d[:2] @ n_h) * 1e3, "along_mm": abs(d[:2] @ a_h) * 1e3,
         "height_mm": abs(d[2]) * 1e3, "angle_deg": math.degrees(math.acos(min(1.0, abs(float(a[:2] @ a_h)))))}
    r["success"] = all(r[k] < v for k, v in GRIPPER.items())
    return r


def run(root, estimator, min_visible_px=800, models=None, centre="extent"):
    """One row per (profile, depth model)."""
    rows = []
    for sd in sorted(glob.glob(os.path.join(root, "scene_*"))):
        cam = json.load(open(f"{sd}/camera.json"))
        V = np.array(cam["ir_left"]["cameraViewTransform"]).reshape(4, 4)
        fx = cam["intrinsics"]["fx"]
        inst = np.load(f"{sd}/instance.npy")
        id_of = {v: int(k) for k, v in json.load(open(f"{sd}/labels.json"))["instance"].items()}
        scene = json.load(open(f"{sd}/scene.json"))
        depths = {m: np.load(f"{sd}/{f}") for m, f in DEPTH_MODELS.items()
                  if (models is None or m in models) and os.path.exists(f"{sd}/{f}")}
        cam_to_world = np.linalg.inv(V)
        baseline_dir, view_dir = cam_to_world[0, :3], -cam_to_world[2, :3]
        for obj in scene["objects"]:
            if obj["kind"] != "profile" or obj["path"] not in id_of:
                continue
            mask = inst == id_of[obj["path"]]
            if mask.sum() < min_visible_px:
                continue
            if mask[:3].any() or mask[-3:].any() or mask[:, :3].any() or mask[:, -3:].any():
                continue    # cut by the image border
            gc, ga = gt_grasp(obj["T_world_obj"])
            ax = ga - (ga @ view_dir) * view_dir
            angle = math.degrees(math.acos(min(1, abs(ax @ baseline_dir) / (np.linalg.norm(ax) + 1e-9))))
            for model, z in depths.items():
                _, P = backproject(np.where(mask, z, 0), V, fx)
                r = evaluate(estimate_grasp(P, estimator, centre), gc, ga)
                r.update({"scene": os.path.basename(sd), "object": obj["path"], "finish": obj["finish"],
                          "length_m": obj["length_m"], "distance_m": scene["rig_distance_m"],
                          "angle_to_baseline_deg": angle, "model": model,
                          "valid_frac": float((z[mask] > 0).mean()), "visible_px": int(mask.sum())})
                rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--estimator", choices=["percentile", "mode"], required=True)
    ap.add_argument("--min-visible-px", type=int, default=800)
    ap.add_argument("--centre", choices=["extent", "median"], default="extent",
                    help="lateral centre; median was the first version (biased by the central slot)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df = run(a.root, a.estimator, a.min_visible_px, centre=a.centre)
    df.to_csv(os.path.join(a.out, "grasps.csv"), index=False)

    df["ok"] = df.success.astype(bool)
    S = df.groupby(["model", "finish"]).agg(n=("ok", "size"), success_pct=("ok", "mean")).reset_index()
    S["success_pct"] *= 100
    S.to_csv(os.path.join(a.out, "summary.csv"), index=False)
    print(f"{df[['scene', 'object']].drop_duplicates().shape[0]} profiles in {df.scene.nunique()} scenes")
    print(S.pivot_table(index="model", columns="finish", values="success_pct").round(1).to_string())
    ideal = df[df.model == "ideal"]
    print(f"ideal depth: median lateral error {ideal.lateral_mm.median():.1f} mm")

    pol = df[df.finish == "polished"].copy()
    pol["distance"] = pd.cut(pol.distance_m, [0.55, 0.8, 1.0, 1.25])
    pol["baseline_angle"] = pd.cut(pol.angle_to_baseline_deg, [-1, 30, 60, 91], labels=["0-30", "30-60", "60-90"])
    for f in ("distance", "baseline_angle"):
        t = pol.pivot_table(index=f, columns="model", values="ok", aggfunc="mean", observed=True) * 100
        t.to_csv(os.path.join(a.out, f"polished_by_{f}.csv"))
    failed = pol[(pol.model == "stereo") & ~pol.ok]
    if len(failed):
        print("stereo, polished failures exceeding each tolerance [%]: " +
              ", ".join(f"{k} {100 * (failed[k] >= v).mean():.0f}" for k, v in GRIPPER.items()))


if __name__ == "__main__":
    main()
