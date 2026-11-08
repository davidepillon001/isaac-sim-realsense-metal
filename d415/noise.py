"""Depth models that degrade the ideal depth directly, without simulating the IR pair.

gaussian   - i.i.d. disparity noise, no missing pixels (datasheet-level model)
heuristic  - known D415 artefacts with hand-set parameters
empirical  - hole / wrong-pixel rates per incidence angle measured on real D415 data (ROBI)
"""
import cv2
import numpy as np
from scipy.stats import norm

from d415.scene import finish_masks, view_cosine

PARAMS = {
    "s_px": 0.16,                 # disparity noise [px]
    "grazing_deg": (70.0, 85.0),  # hole probability ramps from 0 to 1 over this range
    "edge_jump_m": 0.02,          # depth discontinuity treated as an edge
    "edge_drop_p": 0.6,           # edge pixels: holes with this probability, else flying pixels (p=0.5)
    "metal_hole_frac": {"anodized": 0.05, "polished": 0.30},
    "metal_blob_px": 4.0,         # smoothing scale of the random fields
    "specular_cos": 0.97,         # polished metal saturates near the specular direction
    "z_range": (0.3, 3.0),
    "disp_quant": 1 / 16,
}


def gaussian(depth, fx, B, s_px=0.16, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    z = depth.copy()
    ok = np.isfinite(z) & (z > 0)
    z[ok] += rng.normal(size=ok.sum()) * z[ok] ** 2 * s_px / (fx * B)
    return np.where(ok, z, 0).astype(np.float32)


def occlusion_mask(disp):
    """Pixels of the left image that are hidden in (or outside) the right image."""
    h, w = disp.shape
    xr = np.arange(w, dtype=np.float32)[None, :] - disp
    rev_cummin = np.minimum.accumulate(xr[:, ::-1], axis=1)[:, ::-1]
    nxt = np.concatenate([rev_cummin[:, 1:], np.full((h, 1), np.inf, np.float32)], 1)
    return (xr >= nxt - 0.5) | (xr < 0)


def blobs(shape, frac, sigma, rng):
    """Thresholded smooth random field covering about `frac` of the pixels."""
    f = cv2.GaussianBlur(rng.normal(size=shape).astype(np.float32), (0, 0), sigma)
    f /= f.std() + 1e-9
    return f > norm.ppf(1 - np.clip(frac, 1e-6, 1 - 1e-6))


def correlated_normal(shape, sigma, rng):
    f = cv2.GaussianBlur(rng.normal(size=shape).astype(np.float32), (0, 0), sigma)
    return f / (f.std() + 1e-9)


def _edges(zz, p):
    zmax = cv2.dilate(zz, np.ones((3, 3), np.uint8))
    zmin = cv2.erode(zz, np.ones((3, 3), np.uint8))
    return (zmax - zmin) > p["edge_jump_m"], zmin, zmax


def _finish(d, valid, fx, B, p):
    d = np.round(d / p["disp_quant"]) * p["disp_quant"]
    z = fx * B / np.maximum(d, 1e-3)
    valid &= (z > p["z_range"][0]) & (z < p["z_range"][1])
    return z, valid, d


def heuristic(g, params=None, rng=None):
    p = {**PARAMS, **(params or {})}
    rng = np.random.default_rng() if rng is None else rng
    fx, B = g["fx"], g["B"]
    ok = np.isfinite(g["depth"]) & (g["depth"] > 0)
    zz = np.where(ok, g["depth"], 1.0)
    disp = fx * B / zz
    cos_t = view_cosine(g)
    theta = np.degrees(np.arccos(cos_t))

    d = disp + rng.normal(size=disp.shape) * p["s_px"] / np.maximum(cos_t, 0.2)
    valid = ok.copy()
    lo, hi = p["grazing_deg"]
    valid &= rng.random(disp.shape) > np.clip((theta - lo) / (hi - lo), 0, 1)
    valid &= ~occlusion_mask(np.where(ok, disp, 0))

    edge, zmin, zmax = _edges(zz, p)
    r = rng.random(disp.shape)
    valid &= ~(edge & (r < p["edge_drop_p"]))
    flying = edge & (r >= p["edge_drop_p"]) & (rng.random(disp.shape) < 0.5)
    mix = rng.random(disp.shape)
    d = np.where(flying, fx * B / (mix * zmin + (1 - mix) * zmax), d)

    for fin, m in finish_masks(g).items():
        valid &= ~(m & blobs(disp.shape, p["metal_hole_frac"][fin], p["metal_blob_px"], rng))
        if fin == "polished":
            valid &= ~(m & (cos_t > p["specular_cos"]))

    z, valid, _ = _finish(d, valid, fx, B, p)
    return np.where(valid, z, 0).astype(np.float32)


def empirical(g, calib, calib_polished=None, transfer="metric", params=None, rng=None):
    """Same table/edge/occlusion rules as `heuristic`, but on metal the hole and wrong-pixel rates follow
    the per-angle curves of `calib` (see experiments/calibrate_empirical.py).

    transfer: how the error of wrong pixels measured at ~0.35 m is carried to our distances,
      'metric'    - constant error in mm
      'disparity' - constant error in disparity pixels
    """
    p = {**PARAMS, **(params or {})}
    rng = np.random.default_rng() if rng is None else rng
    fx, B = g["fx"], g["B"]
    ok = np.isfinite(g["depth"]) & (g["depth"] > 0)
    zz = np.where(ok, g["depth"], 1.0)
    disp = fx * B / zz
    cos_t = view_cosine(g)
    theta = np.degrees(np.arccos(cos_t))
    fm = finish_masks(g)
    metal = fm["anodized"] | fm["polished"]
    cal_of = {"anodized": calib, "polished": calib_polished or calib}

    def per_finish(fn):
        out = np.zeros(disp.shape, np.float32)
        for fin, m in fm.items():
            out = np.where(m, fn(cal_of[fin]), out)
        return out

    s = np.where(metal, per_finish(lambda c: c["s_metal_px"]), p["s_px"] / np.maximum(cos_t, 0.2))
    d = disp + rng.normal(size=disp.shape) * s
    valid = ok.copy()
    lo, hi = p["grazing_deg"]
    valid &= ~(~metal & (rng.random(disp.shape) < np.clip((theta - lo) / (hi - lo), 0, 1)))
    valid &= ~occlusion_mask(np.where(ok, disp, 0))

    edge, zmin, zmax = _edges(zz, p)
    r = rng.random(disp.shape)
    valid &= ~(edge & (r < p["edge_drop_p"]))
    flying = edge & (r >= p["edge_drop_p"]) & (rng.random(disp.shape) < 0.5)
    mix = rng.random(disp.shape)
    d = np.where(flying, fx * B / (mix * zmin + (1 - mix) * zmax), d)

    centers = np.array(calib["angle_centers_deg"])
    ph = np.clip(per_finish(lambda c: np.interp(theta, centers, c["p_hole"])), 1e-4, 1 - 1e-4)
    pw = np.clip(per_finish(lambda c: np.interp(theta, centers, c["p_wrong"])), 1e-4, 1 - 1e-4)
    holes = metal & (correlated_normal(disp.shape, p["metal_blob_px"], rng) > norm.ppf(1 - ph))
    pw_cond = np.clip(pw / (1 - ph), 1e-4, 1 - 1e-4)
    wrong = metal & ~holes & (correlated_normal(disp.shape, p["metal_blob_px"], rng) > norm.ppf(1 - pw_cond))
    u = norm.cdf(correlated_normal(disp.shape, 3 * p["metal_blob_px"], rng))  # spatially coherent errors
    if transfer == "disparity":
        err_px = per_finish(lambda c: np.interp(u, np.linspace(0, 1, len(c["err_quantiles_px"])),
                                                c["err_quantiles_px"]))
        d = np.where(wrong, disp + err_px, d)
    else:
        err_m = per_finish(lambda c: np.interp(u, np.linspace(0, 1, len(c["err_quantiles_mm"])),
                                               c["err_quantiles_mm"])) / 1000.0
        d = np.where(wrong, fx * B / np.maximum(zz + err_m, 0.05), d)
    valid &= ~holes

    z, valid, d = _finish(d, valid, fx, B, p)
    valid &= d > 0
    return np.where(valid, z, 0).astype(np.float32)
