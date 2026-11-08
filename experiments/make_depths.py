"""Apply one depth model to every rendered scene and save depth_<model>.npy next to it.

    python -m experiments.make_depths data/scenes_main --model gaussian
    python -m experiments.make_depths data/scenes_main --model heuristic
    python -m experiments.make_depths data/scenes_main --model empirical_metric \
        --calib results/calibration/zigzag.json --calib-polished results/calibration/chrome.json
    python -m experiments.make_depths data/scenes_main --model stereo
    python -m experiments.make_depths data/scenes_main --model stereo_sgbm_strict
    python -m experiments.make_depths data/scenes_main --model stereo_hdr      # needs scenes rendered with --hdr 1

Ablations (stereo_ir_noise, stereo_hdr_speckle) are only run on data/hdr_test.
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np

from d415 import DEPTH_MODELS
from d415.ir_formation import ir_pair_from_hdr
from d415.noise import empirical, gaussian, heuristic
from d415.scene import load_scene
from d415.stereo import depth_from_ir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--model", required=True, choices=[m for m in DEPTH_MODELS if m != "ideal"])
    ap.add_argument("--calib", help="empirical models: calibration of the anodized finish")
    ap.add_argument("--calib-polished", help="empirical models: calibration of the polished finish")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    cal = json.load(open(a.calib)) if a.calib else None
    cal_pol = json.load(open(a.calib_polished)) if a.calib_polished else None
    for sd in sorted(glob.glob(os.path.join(a.root, "scene_*"))):
        g = load_scene(sd)
        fx, B = g["fx"], g["B"]
        if a.model == "gaussian":
            z = gaussian(g["depth"], fx, B, rng=rng)
        elif a.model == "heuristic":
            z = heuristic(g, rng=rng)
        elif a.model.startswith("empirical"):
            z = empirical(g, cal, cal_pol, transfer=a.model.split("_")[1], rng=rng)
        else:
            L, R = cv2.imread(f"{sd}/ir_left.png", 0), cv2.imread(f"{sd}/ir_right.png", 0)
            if a.model == "stereo":
                z = depth_from_ir(L, R, fx, B, matcher="census")
            elif a.model == "stereo_sgbm_strict":
                z = depth_from_ir(L, R, fx, B, matcher="sgbm", block=11, uniqueness=30, texture_thresh=10)
            elif a.model == "stereo_ir_noise":
                z = depth_from_ir(L, R, fx, B, matcher="census", ir_noise=True, rng=rng)
            else:
                L, R = ir_pair_from_hdr(np.load(f"{sd}/ir_left_hdr.npy"), np.load(f"{sd}/ir_right_hdr.npy"), rng,
                                        speckle=a.model == "stereo_hdr_speckle")
                z = depth_from_ir(L, R, fx, B, matcher="census")
        np.save(f"{sd}/{DEPTH_MODELS[a.model]}", z.astype(np.float32))
        print(os.path.basename(sd), flush=True)


if __name__ == "__main__":
    main()
