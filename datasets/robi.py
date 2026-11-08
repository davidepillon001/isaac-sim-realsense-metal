"""ROBI (Yang et al. 2021): real D415 depth of reflective parts with Ensenso reference depth.

Camera parameters from ROBI_Toolkit/cam/*.yml (https://github.com/junyang224/ROBI_Toolkit).
"""
import glob
import json
import os

import cv2
import numpy as np
import pandas as pd

DEPTH_UNIT = 0.03125   # mm per PNG unit
RS = {"fx": 905.905579, "fy": 905.905579, "cx": 631.507141, "cy": 362.560059, "W": 1280, "H": 720}
ENS = {"fx": 1083.097046, "fy": 1083.097046, "cx": 379.326874, "cy": 509.437195, "W": 1280, "H": 1024}
RS_BASELINE_MM = 54.919
FB_RS = RS["fx"] * RS_BASELINE_MM   # [px mm]
ANGLE_BINS = np.array([0, 15, 30, 45, 60, 75, 90])


def cam_pose(path):
    """Camera-to-world pose [mm] stored in the first line of a ROBI view json."""
    with open(path) as f:
        return np.array(json.loads(f.readline())["cam_pose"]).reshape(4, 4)


def backproject(d, K):
    v, u = np.nonzero(d > 0)
    z = d[v, u]
    return np.stack([(u - K["cx"]) / K["fx"] * z, (v - K["cy"]) / K["fy"] * z, z], 1)


def reproject_gt(gt_mm, T_w_ens, T_w_rs):
    """Ensenso reference depth rendered into the RealSense view (z-buffer, 3x3 hole filling)."""
    P = backproject(gt_mm, ENS)
    Pw = P @ T_w_ens[:3, :3].T + T_w_ens[:3, 3]
    T_rs_w = np.linalg.inv(T_w_rs)
    Pr = Pw @ T_rs_w[:3, :3].T + T_rs_w[:3, 3]
    Pr = Pr[Pr[:, 2] > 1]
    u = np.round(Pr[:, 0] / Pr[:, 2] * RS["fx"] + RS["cx"]).astype(int)
    v = np.round(Pr[:, 1] / Pr[:, 2] * RS["fy"] + RS["cy"]).astype(int)
    ok = (u >= 0) & (u < RS["W"]) & (v >= 0) & (v < RS["H"])
    out = np.full((RS["H"], RS["W"]), np.inf, np.float32)
    np.minimum.at(out, (v[ok], u[ok]), Pr[ok, 2].astype(np.float32))
    filled = -cv2.dilate(-np.where(np.isfinite(out), out, 1e9).astype(np.float32), np.ones((3, 3), np.uint8))
    out = np.where(np.isfinite(out), out, np.where(filled < 1e8, filled, np.inf))
    return np.where(np.isfinite(out), out, 0)


def object_mask(gt_rs, T_w_rs=None, centers=None, radius=25.0):
    """Metal parts and bin floor. The floor plane is found with RANSAC; parts are points > 4 mm above it and
    within `radius` mm of an annotated object centre. Both masks are eroded by 3 px."""
    P = backproject(gt_rs, RS)
    v, u = np.nonzero(gt_rs > 0)
    rng = np.random.default_rng(0)
    best, best_n = None, 0
    for _ in range(300):
        s = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        if np.linalg.norm(n) < 1e-6:
            continue
        n /= np.linalg.norm(n)
        cnt = (np.abs((P - s[0]) @ n) < 2.0).sum()
        if cnt > best_n:
            best, best_n = (s[0], n), cnt
    p0, n = best
    h = (P - p0) @ n
    if np.median(h[np.abs(h) > 4]) < 0:
        h = -h
    near = np.ones(len(P), bool)
    if centers is not None:
        Pw = P @ T_w_rs[:3, :3].T + T_w_rs[:3, 3]
        near = np.min(np.linalg.norm(Pw[:, None, :] - centers[None, :, :], axis=-1), axis=1) < radius
    obj = np.zeros(gt_rs.shape, bool)
    floor = np.zeros(gt_rs.shape, bool)
    obj[v, u] = (h > 4.0) & near
    floor[v, u] = np.abs(h) < 2.0
    k = np.ones((7, 7), np.uint8)
    return cv2.erode(obj.astype(np.uint8), k) > 0, cv2.erode(floor.astype(np.uint8), k) > 0


def cos_from_depth(z, K):
    """Cosine of the incidence angle from finite-difference normals of the back-projected depth."""
    h, w = z.shape
    v, u = np.mgrid[0:h, 0:w].astype(np.float32)
    P = np.stack([(u - K["cx"]) / K["fx"] * z, (v - K["cy"]) / K["fy"] * z, z], -1)
    P = cv2.GaussianBlur(P, (5, 5), 1.5)
    n = np.cross(np.gradient(P, axis=1), np.gradient(P, axis=0))
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
    ray = P / (np.linalg.norm(P, axis=-1, keepdims=True) + 1e-9)
    c = np.abs(np.sum(n * ray, -1))
    return np.where(z > 0, c, np.nan)


def views(scene, max_views=125):
    """Yields (realsense_mm, reference_mm, metal_mask, floor_mask, cos_incidence) for every view of a scene."""
    with open(f"{scene}/GT_world2obj.json") as f:
        centers = np.array([np.array(v).reshape(4, 4)[:3, 3] for v in json.load(f).values()])
    paths = sorted(glob.glob(f"{scene}/RealSense/Depth/DEPTH_view_*.png"), key=lambda p: int(p.split("_")[-1][:-4]))
    for p in paths[:max_views]:
        k = p.split("_")[-1][:-4]
        gpath = f"{scene}/GT_Depth/view_{k}.png"
        if not os.path.exists(gpath):
            continue
        rs = cv2.imread(p, -1).astype(np.float32) * DEPTH_UNIT
        gt = cv2.imread(gpath, -1).astype(np.float32) * DEPTH_UNIT
        T_w_rs = cam_pose(p[:-4] + ".json")
        gt_rs = reproject_gt(gt, cam_pose(f"{scene}/GT_Depth/view_{k}.json"), T_w_rs)
        obj, floor = object_mask(gt_rs, T_w_rs, centers)
        yield rs, gt_rs, obj, floor, cos_from_depth(gt_rs, RS)


class AngleCurves:
    """Hole rate and rate of wrong pixels (|disparity error| > 1 px) per incidence-angle bin."""

    def __init__(self):
        self.acc = {}

    def add(self, key, meas, ref, mask, cos_t, fB):
        m = mask & (ref > 0) & np.isfinite(cos_t)
        ang = np.degrees(np.arccos(np.clip(cos_t[m], 0, 1)))
        b = np.clip(np.digitize(ang, ANGLE_BINS) - 1, 0, len(ANGLE_BINS) - 2)
        valid = meas[m] > 0
        ed = np.zeros_like(ang)
        ed[valid] = np.abs(fB / meas[m][valid] - fB / ref[m][valid])
        a = self.acc.setdefault(key, np.zeros((len(ANGLE_BINS) - 1, 3)))
        np.add.at(a[:, 0], b, 1)
        np.add.at(a[:, 1], b, valid)
        np.add.at(a[:, 2], b, valid & (ed > 1.0))

    def table(self, min_px=500):
        rows = []
        for (source, region), a in self.acc.items():
            for i in range(len(ANGLE_BINS) - 1):
                n, v, wr = a[i]
                if n >= min_px:
                    rows.append({"source": source, "region": region, "angle": f"{ANGLE_BINS[i]}-{ANGLE_BINS[i + 1]}",
                                 "holes_pct": 100 * (1 - v / n), "wrong_pct": 100 * wr / n, "n_px": int(n)})
        return pd.DataFrame(rows)
