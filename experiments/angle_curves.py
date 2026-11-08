"""Hole rate and wrong-pixel rate on metal vs. incidence angle: real D415 (ROBI, XYZ-IBD) and every depth
model computed on the synthetic scenes.

    python -m experiments.angle_curves --robi data/robi/Zigzag/Scene_4 data/robi/Chrome_screw/Scene_4 \
        --xyzibd data/xyzibd --sim data/scenes_main --out results/angle_curves.csv
"""
import argparse
import glob
import os

import numpy as np

from d415 import DEPTH_MODELS
from d415.scene import finish_masks, load_scene, view_cosine
from datasets import robi, xyzibd


def robi_name(scene):
    obj = os.path.basename(os.path.dirname(scene.rstrip("/")))
    return "real ROBI " + ("Chrome" if obj.lower().startswith("chrome") else obj)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robi", nargs="*", default=[])
    ap.add_argument("--xyzibd", default=None)
    ap.add_argument("--sim", default=None)
    ap.add_argument("--max-views", type=int, default=125)
    ap.add_argument("--max-sim-scenes", type=int, default=200)
    ap.add_argument("--out", default="results/angle_curves.csv")
    a = ap.parse_args()

    curves = robi.AngleCurves()
    for scene in a.robi:
        name = robi_name(scene)
        for rs, gt_rs, obj, floor, cos_t in robi.views(scene, a.max_views):
            curves.add((name, "metal"), rs, gt_rs, obj, cos_t, robi.FB_RS)
            curves.add((name, "floor"), rs, gt_rs, floor, cos_t, robi.FB_RS)
        print(name, "done", flush=True)
    if a.xyzibd:
        for raw, gt, m, cos_t, fB in xyzibd.frames(a.xyzibd):
            curves.add(("real XYZ-IBD", "metal"), raw, gt, m, cos_t, fB)
        print("XYZ-IBD done", flush=True)
    if a.sim:
        for sd in sorted(glob.glob(os.path.join(a.sim, "scene_*")))[:a.max_sim_scenes]:
            g = load_scene(sd)
            ref = np.where(np.isfinite(g["depth"]), g["depth"], 0) * 1e3
            table = np.isin(g["semantic"], [k for k, c in g["labels"].items() if c == "table"])
            regions = {**{f"metal {k}": v for k, v in finish_masks(g).items()}, "table": table}
            cos_t = view_cosine(g)
            fB = g["fx"] * g["B"] * 1e3
            for model, fname in DEPTH_MODELS.items():
                if model == "ideal" or not os.path.exists(f"{sd}/{fname}"):
                    continue
                z = np.load(f"{sd}/{fname}") * 1e3
                for region, m in regions.items():
                    curves.add((f"sim {model}", region), z, ref, m, cos_t, fB)
        print("simulation done", flush=True)

    T = curves.table()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    T.to_csv(a.out, index=False)
    for col in ("holes_pct", "wrong_pct"):
        print(f"\n{col}:")
        print(T.pivot_table(index=["source", "region"], columns="angle", values=col).round(1).to_string())


if __name__ == "__main__":
    main()
