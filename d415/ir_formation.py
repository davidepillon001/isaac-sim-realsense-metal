"""IR image formation from linear radiance (Isaac Sim HdrColor): shared auto-exposure, optional laser
speckle, thermal noise and 8-bit saturation. Used for the HDR variant of the stereo model.

    I = clip(round(gamma * k * L + n), 0, 255)
"""
import cv2
import numpy as np

SPECKLE_K, SPECKLE_THETA = 3.98, 0.254       # Zhang et al. 2023, real D415
THERMAL_MU, THERMAL_SIGMA = -0.231, 0.83


def speckle_field(shape, grain_px, rng, contrast=None):
    k, th = (SPECKLE_K, SPECKLE_THETA) if contrast is None else (1 / contrast ** 2, contrast ** 2)
    h, w = shape
    if grain_px <= 1:
        return rng.gamma(k, th, size=shape).astype(np.float32)
    g = rng.gamma(k, th, size=(int(np.ceil(h / grain_px)), int(np.ceil(w / grain_px)))).astype(np.float32)
    return cv2.resize(g, (w, h), interpolation=cv2.INTER_LINEAR)


def form_ir(L, gain, rng, grain_px=2, speckle=True, thermal=True, contrast=None):
    x = gain * L.astype(np.float32)
    if speckle:
        x = x * speckle_field(L.shape, grain_px, rng, contrast)
    if thermal:
        x = x + rng.normal(THERMAL_MU, THERMAL_SIGMA, size=L.shape).astype(np.float32)
    return np.clip(np.round(x), 0, 255).astype(np.uint8)


def ir_pair_from_hdr(L_left, L_right, rng, ae_target=0.25, grain_px=2, speckle=False, thermal=True, contrast=None):
    """Both imagers share the exposure, chosen so that the left image has mean ae_target * 255."""
    k = ae_target * 255.0 / max(float(np.mean(L_left)), 1e-9)
    return (form_ir(L_left, k, rng, grain_px, speckle, thermal, contrast),
            form_ir(L_right, k, rng, grain_px, speckle, thermal, contrast))
