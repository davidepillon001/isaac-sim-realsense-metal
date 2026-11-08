"""Centre-symmetric census + 4-path semi-global matching (Hirschmueller 2008), the matching chain of the
Intel D4 stereo processor as described by Keselman et al. 2017 and Zhang et al. 2023."""
import cv2
import numpy as np
from numba import njit, prange


def cs_census(img, wh=4, ww=4):
    """Centre-symmetric census on a (2wh+1)x(2ww+1) window: one bit per symmetric pixel pair."""
    f = np.pad(img.astype(np.int16), ((wh, wh), (ww, ww)), mode="edge")
    h, w = img.shape
    code = np.zeros((h, w), np.uint64)
    bit = 0
    for dy in range(-wh, wh + 1):
        for dx in range(-ww, ww + 1):
            if (dy, dx) <= (0, 0):
                continue
            a = f[wh + dy: wh + dy + h, ww + dx: ww + dx + w]
            b = f[wh - dy: wh - dy + h, ww - dx: ww - dx + w]
            code |= (a > b).astype(np.uint64) << np.uint64(bit)
            bit += 1
    return code, bit


def cost_volume(cl, cr, min_disp, num_disp, nbits):
    h, w = cl.shape
    C = np.full((h, w, num_disp), nbits, np.uint8)
    for k in range(num_disp):
        d = min_disp + k
        if d >= w:
            break
        C[:, d:, k] = np.bitwise_count(cl[:, d:] ^ cr[:, :w - d]).astype(np.uint8)
    return C


@njit(cache=True)
def _aggregate(C, S, P1, P2, dy, dx):
    """Cost aggregation along one direction, accumulated into S."""
    h, w, D = C.shape
    prev = np.empty(D, np.int32)
    cur = np.empty(D, np.int32)
    outer, inner = (h, w) if dy == 0 else (w, h)
    step = dx if dy == 0 else dy
    for i in range(outer):
        first = True
        for jj in range(inner):
            j = jj if step > 0 else inner - 1 - jj
            y, x = (i, j) if dy == 0 else (j, i)
            if first:
                for d in range(D):
                    cur[d] = C[y, x, d]
                first = False
            else:
                m = prev.min()
                for d in range(D):
                    best = prev[d]
                    if d > 0 and prev[d - 1] + P1 < best:
                        best = prev[d - 1] + P1
                    if d < D - 1 and prev[d + 1] + P1 < best:
                        best = prev[d + 1] + P1
                    if m + P2 < best:
                        best = m + P2
                    cur[d] = C[y, x, d] + best - m
            for d in range(D):
                S[y, x, d] += cur[d]
                prev[d] = cur[d]


@njit(parallel=True, cache=True)
def _select(S, min_disp, uniq):
    """Winner-take-all, uniqueness test against the second minimum, parabolic sub-pixel fit."""
    h, w, D = S.shape
    disp = np.full((h, w), np.nan, np.float32)
    for y in prange(h):
        for x in range(w):
            b = 0
            for d in range(1, D):
                if S[y, x, d] < S[y, x, b]:
                    b = d
            s0 = np.int64(S[y, x, b])
            second = np.int64(1 << 30)
            for d in range(D):
                if abs(d - b) > 1 and np.int64(S[y, x, d]) < second:
                    second = np.int64(S[y, x, d])
            if second * (100 - uniq) < s0 * 100:
                continue
            sub = 0.0
            if 0 < b < D - 1:
                a1 = np.int64(S[y, x, b - 1])
                a2 = np.int64(S[y, x, b + 1])
                den = a1 - 2 * s0 + a2
                if den > 0:
                    sub = 0.5 * (a1 - a2) / den
            disp[y, x] = min_disp + b + sub
    return disp


@njit(parallel=True, cache=True)
def _right_disp(S, min_disp):
    """Right-image disparity from the same volume: dR(x') = argmin_d S(y, x' + d, d)."""
    h, w, D = S.shape
    out = np.full((h, w), np.nan, np.float32)
    for y in prange(h):
        for xr in range(w):
            best = 1 << 30
            bd = -1
            for k in range(D):
                x = xr + min_disp + k
                if x >= w:
                    break
                if S[y, x, k] < best:
                    best = S[y, x, k]
                    bd = k
            if bd >= 0:
                out[y, xr] = min_disp + bd
    return out


def census_sgm(left, right, min_disp=16, num_disp=176, P1=15, P2=120, uniqueness=10, lr_tol=1.0, median_tol=1.0,
               win=(4, 4)):
    """Disparity [px] of the left image, NaN where rejected."""
    cl, nb = cs_census(left, *win)
    cr, _ = cs_census(right, *win)
    C = cost_volume(cl, cr, min_disp, num_disp, nb)
    S = np.zeros(C.shape, np.uint16)
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        _aggregate(C, S, P1, P2, dy, dx)
    d = _select(S, min_disp, uniqueness)
    dr = _right_disp(S, min_disp)
    h, w = d.shape
    ok = np.isfinite(d)
    xs = np.clip(np.arange(w)[None, :] - np.round(np.where(ok, d, 0)).astype(int), 0, w - 1)
    ok &= np.abs(d - np.take_along_axis(dr, xs, 1)) <= lr_tol
    dm = cv2.medianBlur(np.where(ok, d, 0).astype(np.float32), 3)
    ok &= np.abs(d - dm) <= median_tol
    return np.where(ok, d, np.nan)
