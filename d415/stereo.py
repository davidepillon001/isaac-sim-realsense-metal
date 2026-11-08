"""Depth from the rendered IR pair: census+SGM (main model) or OpenCV SGBM, then sub-pixel refinement."""
import cv2
import numpy as np

from d415.census_sgm import census_sgm


def d415_ir_noise(img, rng, k=3.98, theta=0.254, mu=-0.231, sigma=0.83):
    """IR imager noise fitted on a real D415 by Zhang et al. 2023: I = gamma * I + n,
    gamma ~ Gamma(k, theta), n ~ N(mu, sigma). Tested, not used in the final model."""
    x = img.astype(np.float32)
    g = rng.gamma(k, theta, size=x.shape).astype(np.float32)
    n = rng.normal(mu, sigma, size=x.shape).astype(np.float32)
    return np.clip(g * x + n, 0, 255).astype(np.uint8)


def _sgbm(left, right, min_disp, num_disp, block, uniqueness=10, pad=True):
    # OpenCV leaves the first min_disp + num_disp columns empty: pad, match, crop
    pad = min_disp + num_disp if pad else 0
    left = cv2.copyMakeBorder(left, 0, 0, pad, 0, cv2.BORDER_CONSTANT, value=0)
    right = cv2.copyMakeBorder(right, 0, 0, pad, 0, cv2.BORDER_CONSTANT, value=0)
    m = cv2.StereoSGBM_create(minDisparity=min_disp, numDisparities=num_disp, blockSize=block,
                              P1=8 * block * block, P2=32 * block * block, disp12MaxDiff=-1,
                              uniquenessRatio=uniqueness, speckleWindowSize=0, speckleRange=0,
                              mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    return m.compute(left, right).astype(np.float32)[:, pad:] / 16.0


def sgbm_disparity(left, right, min_disp=16, num_disp=176, block=11, lr_tol=1.0, speckle_size=200, speckle_range=2,
                   uniqueness=10, texture_thresh=0.0, pad=True):
    """OpenCV SGBM with left-right check, texture threshold and speckle filter. NaN where rejected."""
    dl = _sgbm(left, right, min_disp, num_disp, block, uniqueness, pad)
    dr = _sgbm(np.ascontiguousarray(right[:, ::-1]), np.ascontiguousarray(left[:, ::-1]), min_disp, num_disp, block,
               uniqueness, pad)[:, ::-1]
    valid = dl >= min_disp
    if texture_thresh > 0:
        f = left.astype(np.float32)
        mu = cv2.boxFilter(f, -1, (block, block))
        sd = np.sqrt(np.maximum(cv2.boxFilter(f * f, -1, (block, block)) - mu * mu, 0))
        valid &= sd >= texture_thresh
    h, w = dl.shape
    xs = np.clip(np.arange(w)[None, :] - np.round(np.where(valid, dl, 0)).astype(int), 0, w - 1)
    valid &= np.abs(dl - np.take_along_axis(dr, xs, axis=1)) <= lr_tol
    d16 = np.where(valid, dl * 16, -16).astype(np.int16)
    cv2.filterSpeckles(d16, -16, speckle_size, speckle_range * 16)
    valid &= d16 >= 0
    return np.where(valid, dl, np.nan)


def refine_subpixel(left, right, disp, iters=1, win=15, min_grad=0.5):
    """One Lucas-Kanade step on the photometric error, to remove pixel locking of the integer matcher."""
    L = left.astype(np.float32)
    R = right.astype(np.float32)
    h, w = L.shape
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    ok = np.isfinite(disp)
    d0 = np.where(ok, disp, 0).astype(np.float32)
    d = d0.copy()
    gLx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    for _ in range(iters):
        Rw = cv2.remap(R, xs - d, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        gx = 0.5 * (cv2.Sobel(Rw, cv2.CV_32F, 1, 0, ksize=3) + gLx) / 8.0
        r = L - Rw
        num = cv2.boxFilter(gx * r, -1, (win, win), normalize=False)
        den = cv2.boxFilter(gx * gx, -1, (win, win), normalize=False)
        delta = np.where(den > min_grad * win * win, -num / np.maximum(den, 1e-6), 0)
        d = d + np.clip(delta, -0.5, 0.5)
    d = np.where(np.abs(d - d0) < 1.0, d, d0)
    return np.where(ok, d, np.nan)


def depth_from_ir(left, right, fx, baseline, matcher="census", refine=True, refine_iters=1, ir_noise=False, rng=None,
                  **kw):
    """IR pair (uint8) -> depth [m], 0 where there is no measurement."""
    if ir_noise:
        rng = np.random.default_rng() if rng is None else rng
        left, right = d415_ir_noise(left, rng), d415_ir_noise(right, rng)
    if matcher == "census":
        d = census_sgm(left, right, **{k: v for k, v in kw.items() if k in ("min_disp", "num_disp", "uniqueness", "lr_tol")})
    else:
        d = sgbm_disparity(left, right, **kw)
    if refine:
        d = refine_subpixel(left, right, d, iters=refine_iters)
    z = fx * baseline / d
    return np.where(np.isfinite(z), z, 0.0).astype(np.float32)
