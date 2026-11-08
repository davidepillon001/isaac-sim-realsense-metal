"""Everything after rendering, in order: calibration, depth models, depth metrics, grasp, statistics, roughness,
segmentation, figures. Works on Linux, Windows and macOS. Run from the repository root:

    python run_analysis.py                  # all steps
    python run_analysis.py --skip-depths    # reuse the depth files already in data/
"""
import argparse
import subprocess
import sys

ZIG, CHR = "data/robi/Zigzag", "data/robi/Chrome_screw"
CALIB = ["--calib", "results/calibration/zigzag.json", "--calib-polished", "results/calibration/chrome.json"]


def run(module, *args):
    cmd = [sys.executable, "-m", module, *args]
    print("\n>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def calibration(sim_curves=False):
    for name, root in (("zigzag", ZIG), ("chrome", CHR)):
        extra = ["--sim-curves", "results/angle_curves.csv"] if sim_curves else []
        run("experiments.calibrate_empirical", "--fit", f"{root}/Scene_4", "--val", f"{root}/Scene_5",
            "--out", f"results/calibration/{name}.json", *extra)


def depth_models():
    for m in ("gaussian", "heuristic", "stereo", "stereo_sgbm_strict", "stereo_hdr"):
        run("experiments.make_depths", "data/scenes_main", "--model", m)
    for m in ("empirical_metric", "empirical_disparity"):
        run("experiments.make_depths", "data/scenes_main", "--model", m, *CALIB)
    for d in ("polished_r005", "polished_r020", "polished_r030", "anodized_r030", "anodized_r020", "anodized_m070"):
        run("experiments.make_depths", f"data/{d}", "--model", "stereo")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-depths", action="store_true", help="keep the depth files already computed")
    a = ap.parse_args()

    calibration()
    if not a.skip_depths:
        depth_models()
    for m in ("stereo", "stereo_hdr", "stereo_hdr_speckle", "stereo_ir_noise"):   # 3 scenes, always recomputed
        run("experiments.make_depths", "data/hdr_test", "--model", m)

    # depth quality
    run("experiments.plane_noise", "--fronto", "data/plane_fronto", "--tilted", "data/plane_tilt25")
    run("experiments.angle_curves", "--robi", f"{ZIG}/Scene_4", f"{CHR}/Scene_4", "--xyzibd", "data/xyzibd",
        "--sim", "data/scenes_main")
    calibration(sim_curves=True)
    run("experiments.depth_metrics", "--sim", "data/scenes_main", "--robi-zigzag", f"{ZIG}/Scene_4", f"{ZIG}/Scene_5",
        "--robi-chrome", f"{CHR}/Scene_4", f"{CHR}/Scene_5", "--xyzibd", "data/xyzibd")
    run("experiments.xyzibd_transfer", "data/xyzibd", "--calib", "results/calibration/zigzag.json",
        "results/calibration/chrome.json")

    # what did not work: anodized material variants and sensor-physics ablations
    run("experiments.angle_curves", "--sim", "data/scenes_main", "--max-sim-scenes", "30",
        "--out", "results/anodized/curves_r045.csv")
    for v in ("r030", "r020", "m070"):
        run("experiments.angle_curves", "--sim", f"data/anodized_{v}", "--out", f"results/anodized/curves_{v}.csv")
    run("experiments.curve_distance", "--models", "stereo", "--finishes", "anodized",
        "--out", "results/anodized/distance_to_robi.csv", "--sim",
        "metallic 1.0, roughness 0.45 (main):results/anodized/curves_r045.csv",
        "metallic 1.0, roughness 0.30:results/anodized/curves_r030.csv",
        "metallic 1.0, roughness 0.20:results/anodized/curves_r020.csv",
        "metallic 0.7, roughness 0.45:results/anodized/curves_m070.csv")
    run("experiments.angle_curves", "--sim", "data/hdr_test", "--out", "results/ablations/curves_3_scenes.csv")
    run("experiments.curve_distance", "--out", "results/ablations/distance_to_robi.csv",
        "--models", "stereo", "stereo_hdr", "stereo_hdr_speckle", "stereo_ir_noise",
        "--sim", "3 scenes:results/ablations/curves_3_scenes.csv", "200 scenes:results/angle_curves.csv")

    # grasp experiment and statistics
    for est in ("percentile", "mode"):
        run("experiments.grasp", "data/scenes_main", "--estimator", est, "--out", f"results/grasp_{est}")
    run("experiments.grasp_statistics", "--runs", "results/grasp_percentile", "results/grasp_mode")
    run("experiments.tolerance_sensitivity", "--runs", "results/grasp_percentile", "results/grasp_mode")
    run("experiments.grasp", "data/scenes_main", "--estimator", "percentile", "--centre", "median",
        "--out", "results/grasp_median_centre")

    # roughness of the polished finish, segmentation, figures
    run("experiments.roughness", "--main", "0.10:data/scenes_main", "--variants", "0.05:data/polished_r005",
        "0.20:data/polished_r020", "0.30:data/polished_r030")
    run("experiments.segmentation", "prepare")
    run("experiments.segmentation", "train")
    run("experiments.make_figures", "--out", "report/figures")


if __name__ == "__main__":
    main()
