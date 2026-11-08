"""Figures of the report. Each figure reads only files written by the other experiments.

    python -m experiments.make_figures --out report/figures
    python -m experiments.make_figures --out report/figures --only angle_curves failures
"""
import argparse
import glob
import json
import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from d415 import DEPTH_MODELS  # noqa: E402
from experiments.grasp import backproject  # noqa: E402

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GREEN, YELLOW, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8984"
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
                     "lines.linewidth": 2, "lines.markersize": 5})

NAMES = {"ideal": "ideal depth", "gaussian": "Gaussian noise", "heuristic": "heuristic artefact model",
         "empirical_metric": "empirical, metric transfer", "empirical_disparity": "empirical, disparity transfer",
         "stereo": "physics-based stereo"}
COLORS = {"gaussian": GREY, "heuristic": ORANGE, "empirical_metric": GREEN, "empirical_disparity": YELLOW,
          "stereo": BLUE}
EST = {"percentile": "percentile", "mode": "mode"}


def save(fig, out, name, tight=True):
    if tight:
        fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("  ", name)


def bottom_legend(fig, axes, ncol, fontsize, bottom):
    seen, H, L = set(), [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                seen.add(l)
                H.append(h)
                L.append(l)
    fig.legend(H, L, loc="lower center", ncol=ncol, fontsize=fontsize, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, bottom, 1, 1))


def colorize(z, lo, hi):
    v = np.clip((z - lo) / (hi - lo + 1e-9), 0, 1)
    c = cv2.applyColorMap((255 * (1 - v)).astype(np.uint8), cv2.COLORMAP_TURBO)[..., ::-1]
    c[~(z > 0)] = 0
    return c


def plane_noise(a):
    T = pd.read_csv(os.path.join(a.results, "plane_noise.csv"))
    style = {"sgbm, fronto-parallel": (YELLOW, "--", "SGBM, fronto-parallel"),
             "sgbm + subpixel, tilted 25": (ORANGE, "-", "SGBM + subpixel, tilted 25°"),
             "census + sgm, tilted 25": (BLUE, "-", "census + SGM, tilted 25°")}
    fig, ax = plt.subplots(1, 2, figsize=(3.45, 2.35))
    for cfg, g in T[T.config.isin(style)].groupby("config", sort=False):
        col, ls, lab = style[cfg]
        ax[0].plot(g.distance_m, g.std_mm, ls, color=col, marker="o", ms=3.5, lw=1.5, label=lab)
        ax[1].plot(g.distance_m, g.median_err_mm, ls, color=col, marker="o", ms=3.5, lw=1.5)
    zz = np.linspace(0.45, 1.55, 60)
    fB = 1004.6 * 0.055
    ax[0].plot(zz, 1e3 * zz ** 2 * 0.08 / fB, ":", color=INK2, lw=1.1, label="Eq. (1), 0.08 px (Intel)")
    ax[0].plot(zz, 1e3 * zz ** 2 * 0.31 / fB, ":", color=INK, lw=1.1, label="Eq. (1), 0.31 px")
    ax[0].set(xlabel="distance [m]", ylabel="std of depth error [mm]", title="Noise")
    ax[1].axhline(0, color=INK2, lw=0.8)
    ax[1].set(xlabel="distance [m]", ylabel="median depth error [mm]", title="Bias")
    bottom_legend(fig, ax[:1], 2, 5.6, 0.16)
    save(fig, a.out, "plane_noise", tight=False)


def depth_example(a):
    sd = a.example_scene or sorted(glob.glob(os.path.join(a.scenes, "scene_*")))[0]
    gt = np.load(f"{sd}/depth_gt.npy")
    lo, hi = np.percentile(gt[np.isfinite(gt)], [1, 99])
    panels = [("RGB", cv2.imread(f"{sd}/rgb.png")[..., ::-1]),
              ("left IR (dot projector)", cv2.cvtColor(cv2.imread(f"{sd}/ir_left.png", 0), cv2.COLOR_GRAY2RGB)),
              ("ideal depth", colorize(np.where(np.isfinite(gt), gt, 0), lo, hi))]
    for model, title in (("gaussian", "Gaussian noise"), ("empirical_metric", "empirical model (calibrated on ROBI)"),
                         ("stereo", "physics-based stereo model")):
        panels.append((title, colorize(np.load(f"{sd}/{DEPTH_MODELS[model]}"), lo, hi)))
    fig, axs = plt.subplots(3, 2, figsize=(3.45, 3.0))
    for ax, (t, im) in zip(axs.ravel(), panels):
        ax.imshow(im)
        ax.set_title(t, fontsize=6)
        ax.axis("off")
    save(fig, a.out, "depth_example")


def angle_curves(a):
    C = pd.read_csv(os.path.join(a.results, "angle_curves.csv"))
    order = ["0-15", "15-30", "30-45", "45-60", "60-75", "75-90"]
    x = np.array([7.5, 22.5, 37.5, 52.5, 67.5, 82.5])
    real = (("real ROBI Zigzag", "-", "real, ROBI Zigzag"), ("real ROBI Chrome", "--", "real, ROBI Chrome"),
            ("real XYZ-IBD", ":", "real, XYZ-IBD"))
    short = {"stereo": "physics-based", "heuristic": "heuristic"}
    fig, axs = plt.subplots(1, 2, figsize=(3.45, 2.6), sharey=True)
    for ax, col, title in ((axs[0], "holes_pct", "Holes"), (axs[1], "wrong_pct", "Wrong depth (|e| > 1 px)")):
        for src, ls, lab in real:
            s_ = C[(C.source == src) & (C.region == "metal")].set_index("angle").reindex(order)[col]
            if s_.notna().any():
                ax.plot(x, s_.values, ls, color=INK, marker="o", ms=3, lw=1.5, label=lab)
        for model in ("stereo", "heuristic"):
            for fin, ls in (("anodized", "-"), ("polished", "--")):
                s_ = C[(C.source == f"sim {model}") & (C.region == f"metal {fin}")].set_index("angle").reindex(order)[col]
                ax.plot(x, s_.values, ls, color=COLORS[model], marker="s", ms=2.5, lw=1.3,
                        label=f"{short[model]}, {fin}")
        ax.set(title=title, xlabel="incidence angle [°]", xticks=[0, 30, 60, 90])
    axs[0].set_ylabel("% of metal pixels")
    bottom_legend(fig, axs[:1], 2, 5.6, 0.2)
    save(fig, a.out, "angle_curves", tight=False)


def error_distribution(a):
    D = np.load(os.path.join(a.results, "depth_metrics", "error_samples_px.npz"))
    xs = np.logspace(-1.3, 1.7, 120)
    fig, axs = plt.subplots(1, 2, figsize=(3.45, 2.6), sharey=True)
    panels = ((axs[0], "anodized", (("ROBI Zigzag", "-", "real, ROBI Zigzag"), ("XYZ-IBD", ":", "real, XYZ-IBD")),
               "Anodized"),
              (axs[1], "polished", (("ROBI Chrome", "--", "real, ROBI Chrome"),), "Polished"))
    for ax, fin, reals, title in panels:
        for key, ls, lab in reals:
            if f"real|{key}" in D.files:
                e = np.abs(D[f"real|{key}"])
                ax.plot(xs, [(e > x).mean() for x in xs], ls, color=INK, lw=1.9, label=lab)
        for model in ("gaussian", "heuristic", "empirical_metric", "empirical_disparity", "stereo"):
            k = f"sim|{model}|{fin}"
            if k in D.files:
                e = np.abs(D[k])
                y = np.array([(e > x).mean() for x in xs])
                ax.plot(xs, np.where(y > 0, y, np.nan), ":" if model == "gaussian" else "-", color=COLORS[model],
                        lw=1.3, label=NAMES[model].replace(" artefact model", ""))
        ax.axvline(1, color=INK2, lw=0.8, ls=":")
        ax.set(xscale="log", yscale="log", ylim=(1e-4, 1.2), title=title, xlabel="x [px]")
    axs[0].set_ylabel("P(|e| > x)")
    bottom_legend(fig, axs, 2, 5.6, 0.2)
    save(fig, a.out, "error_distribution", tight=False)


def gee_odds_ratios(a):
    G = pd.read_csv(os.path.join(a.results, "statistics", "gee_polished.csv"))
    models = ["heuristic", "empirical_metric", "empirical_disparity", "stereo"]
    labels = ["heuristic", "empirical, metric", "empirical, disparity", "physics-based"]
    fig, axs = plt.subplots(1, 2, figsize=(3.45, 1.95), sharey=True)
    for ax, factor, title in ((axs[0], "parallel", "Parallel to baseline"), (axs[1], "distance_dm", "Distance +10 cm")):
        for i, m in enumerate(models):
            for k, (est, mk, col) in enumerate((("percentile", "o", BLUE), ("mode", "s", ORANGE))):
                r = G[(G.model == m) & (G.factor == factor) & (G.estimator == est)]
                if r.empty:
                    continue
                r = r.iloc[0]
                y = i + (k - 0.5) * 0.3
                ax.plot([r.ci_low, r.ci_high], [y, y], color=col, lw=1.3)
                ax.plot(r.odds_ratio, y, mk, color=col, ms=4, markeredgecolor="white", markeredgewidth=0.6,
                        label=f"{est} estimator" if i == 0 else None)
        ax.axvline(1, color=INK2, lw=0.9)
        ax.set(xscale="log", title=title, xlabel="odds ratio (95% CI)")
    axs[0].set_yticks(range(len(models)), labels, fontsize=6.5)
    axs[0].invert_yaxis()
    bottom_legend(fig, axs[:1], 2, 6, 0.1)
    save(fig, a.out, "gee_odds_ratios", tight=False)


def failures(a):
    """Polished profiles grasped with ideal depth and missed with stereo depth (mode estimator):
    the most parallel to the baseline, the median one and the most perpendicular."""
    d = pd.read_csv(os.path.join(a.results, "grasp_mode", "grasps.csv"))
    p = d.pivot_table(index=["scene", "object", "finish", "angle_to_baseline_deg"], columns="model",
                      values="success", aggfunc="first").reset_index()
    c = p[(p.finish == "polished") & p.ideal.astype(bool) & ~p.stereo.astype(bool)]
    c = c.sort_values("angle_to_baseline_deg").reset_index(drop=True)
    sel = c.iloc[[0, len(c) // 2, len(c) - 1]]

    fig, axs = plt.subplots(3, 3, figsize=(7.2, 6.45), gridspec_kw={"width_ratios": [1, 1, 1.25]})
    for r, (_, row) in enumerate(sel.iterrows()):
        sd = os.path.join(a.scenes, row.scene)
        cam = json.load(open(f"{sd}/camera.json"))
        V = np.array(cam["ir_left"]["cameraViewTransform"]).reshape(4, 4)
        fx = cam["intrinsics"]["fx"]
        id_of = {v: int(k) for k, v in json.load(open(f"{sd}/labels.json"))["instance"].items()}
        m = np.load(f"{sd}/instance.npy") == id_of[row.object]
        gt = np.load(f"{sd}/depth_gt.npy")
        zs = np.load(f"{sd}/{DEPTH_MODELS['stereo']}")
        ys, xs = np.nonzero(m)
        y0, y1, x0, x1 = max(ys.min() - 25, 0), ys.max() + 25, max(xs.min() - 25, 0), xs.max() + 25
        ir = cv2.imread(f"{sd}/ir_left.png", 0)     # same viewpoint as the mask (RGB is 15 mm away)
        lo, hi = np.nanpercentile(gt[m], [0, 100])
        axs[r, 0].imshow(ir[y0:y1, x0:x1], cmap="gray")
        axs[r, 1].imshow(colorize(np.where(m, zs, 0)[y0:y1, x0:x1], lo - 0.05, hi + 0.05))
        for ax in axs[r, :2]:
            ax.contour(m[y0:y1, x0:x1], [0.5], colors="white", linewidths=0.6)
            ax.set_xticks([])
            ax.set_yticks([])
        bins = np.arange(-20, 70, 1.0)
        for z, col, lab in ((gt, GREY, "ideal depth"), (zs, BLUE, "physics-based stereo depth")):
            axs[r, 2].hist(backproject(np.where(m, z, 0), V, fx)[1][:, 2] * 1e3, bins, color=col, alpha=0.6, label=lab)
        axs[r, 2].axvline(40, color=INK, lw=1, ls="--", label="true top face (40 mm)")
        g = d[(d.scene == row.scene) & (d.object == row.object) & (d.model == "stereo")].iloc[0]
        axs[r, 2].set(xlabel="point height above table [mm]", yticks=[])
        axs[r, 2].set_title(f"angle to baseline {row.angle_to_baseline_deg:.0f}°: height error {g.height_mm:.0f} mm, "
                            f"lateral {g.lateral_mm:.0f} mm", fontsize=7)
    axs[0, 0].set_title("left IR image (crop)", fontsize=8)
    axs[0, 1].set_title("physics-based stereo depth (crop)", fontsize=8)
    bottom_legend(fig, axs[0, 2:], 3, 7, 0.04)
    save(fig, a.out, "failures", tight=False)


def roughness_distance(a):
    T = pd.read_csv(os.path.join(a.results, "roughness", "success_vs_roughness.csv"))
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)

    ax = axs[0]
    for est, col, mk in (("percentile", BLUE, "o"), ("mode", ORANGE, "s")):
        t = T[(T.estimator == est) & (T.model == "stereo")].sort_values("roughness")
        ax.errorbar(t.roughness, t.success_pct, yerr=[t.success_pct - t.ci_low, t.ci_high - t.success_pct],
                    color=col, marker=mk, ms=4, capsize=2, lw=1.6, label=f"{est} estimator")
    ax.axhline(100, color=GREY, ls=":", lw=1, label="ideal depth")
    ax.axvline(0.10, color=INK2, lw=0.6, ls="--")
    ax.set(xlabel="roughness of polished finish", ylabel="success, polished profiles [%]", ylim=(0, 104),
           title="(a) material (30 scenes)")
    ax.legend(fontsize=6, loc="lower right")

    ax = axs[1]
    for est, ls in (("percentile", "-"), ("mode", "--")):
        d = pd.read_csv(os.path.join(a.results, f"grasp_{est}", "grasps.csv"))
        d = d[d.finish == "polished"].copy()
        d["ok"] = d.success.astype(bool) & d.grasp_found.astype(bool)
        d["bin"] = pd.cut(d.distance_m, [0.55, 0.8, 1.0, 1.25])
        for model, lab in (("heuristic", "heuristic"), ("empirical_metric", "empirical, metric"),
                           ("stereo", "physics-based stereo")):
            g = d[d.model == model].groupby("bin", observed=False).ok.mean() * 100
            ax.plot(range(3), g.values, ls, color=COLORS[model], marker="o", ms=3.5, lw=1.4, label=f"{lab}, {est}")
    ax.set_xticks(range(3), ["0.55-0.8", "0.8-1.0", "1.0-1.25"], fontsize=7)
    ax.set(xlabel="camera distance [m]", title="(b) depth model (200 scenes)")
    ax.legend(fontsize=5.4, loc="lower left")

    ax = axs[2]
    R = pd.read_csv(os.path.join(a.results, "roughness", "grasps.csv"))
    R = R[(R.estimator == "mode") & (R.model == "stereo")].copy()
    R["bin"] = pd.cut(R.distance_m, [0.55, 0.9, 1.25])
    for rough, col in zip(sorted(R.roughness.unique()), (INK, BLUE, GREEN, YELLOW)):
        g = R[R.roughness == rough].groupby("bin", observed=False).ok.mean() * 100
        ax.plot(range(2), g.values, "-", color=col, marker="o", ms=4, lw=1.5, label=f"roughness {rough:.2f}")
    ax.set_xticks(range(2), ["0.55-0.9", "0.9-1.25"], fontsize=7)
    ax.set(xlabel="camera distance [m]", title="(c) material x distance (mode est.)")
    ax.legend(fontsize=6, loc="lower left")
    save(fig, a.out, "roughness_distance")


def compact(f):
    """Single-column figures with two panels side by side: smaller fonts."""
    def wrapped(a):
        with plt.rc_context({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7}):
            f(a)
    wrapped.__name__ = f.__name__
    return wrapped


plane_noise, angle_curves, error_distribution, gee_odds_ratios = map(
    compact, (plane_noise, angle_curves, error_distribution, gee_odds_ratios))

FIGURES = {f.__name__: f for f in (plane_noise, depth_example, angle_curves, error_distribution, gee_odds_ratios,
                                   failures, roughness_distance)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--scenes", default="data/scenes_main")
    ap.add_argument("--example-scene", default=None)
    ap.add_argument("--out", default="report/figures")
    ap.add_argument("--only", nargs="*", choices=list(FIGURES), default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for name, f in FIGURES.items():
        if a.only is None or name in a.only:
            f(a)


if __name__ == "__main__":
    main()
