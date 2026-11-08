# Simulating the Intel RealSense D415 on aluminium profiles

Synthetic RGB-D data of aluminium T-slot profiles and L-brackets rendered with Isaac Sim, and a study of how
much the depth-camera model matters for a downstream task. The same 200 scenes are degraded with depth models
of increasing realism, and a top-down grasp of each profile is evaluated with every model. The report is in
[`report/ETH_submission_report_Davide_Pillon.pdf`](report/ETH_submission_report_Davide_Pillon.pdf).

Main result: on polished profiles the physics-based stereo model predicts 85% grasp success with a percentile
height estimator and 45% with a mode estimator, where ideal depth and Gaussian noise predict 100% for both.
A depth-only segmentation network trained on ideal depth drops from 88% to 11% profile IoU when tested on
stereo depth.

## Depth models

| name | file in each scene folder | what it does |
|---|---|---|
| `ideal` | `depth_gt.npy` | rendered depth |
| `gaussian` | `depth_gaussian.npy` | disparity noise, σ = 0.16 px |
| `heuristic` | `depth_heuristic.npy` | hand-tuned artefacts: holes at grazing angles, edges, occlusions |
| `empirical_metric`, `empirical_disparity` | `depth_empirical_*.npy` | hole and false-match rates per incidence angle measured on a real D415 (ROBI); errors transferred in mm or in disparity px |
| `stereo` | `depth_stereo.npy` | rendered IR pair with the dot projector → census transform + semi-global matching (physics-based stereo model) |
| `stereo_sgbm_strict` | `depth_stereo_sgbm_strict.npy` | OpenCV SGBM with stricter uniqueness and texture thresholds (robustness check) |
| `stereo_hdr` | `depth_stereo_hdr.npy` | census + SGM on IR images formed from linear radiance with auto-exposure and thermal noise (robustness check) |

The camera follows the datasheet: 1280×720, 65° horizontal FOV (f = 1004.6 px), 55 mm baseline. The projector
is a point light behind a perforated mask (600×600 cells, 8% open, 40° cone), rendered with path tracing at
32 samples per pixel.

## Layout

```
isaac/          scene generation (runs inside the Isaac Sim container)
d415/           depth models: noise models, census + SGM, IR image formation
datasets/       readers for the real D415 data (ROBI, XYZ-IBD) and the XYZ-IBD downloader
experiments/    one script per result of the report, plus the figures
report/         report (PDF) and its figures
results/        CSV outputs of the experiments
run_analysis.py everything after rendering, in order
```

## Setup

The analysis runs on Linux, Windows and macOS with Python 3.10–3.13 (the range covered by the Open3D, numba
and PyTorch wheels). Create a virtual environment and install the requirements:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` lists what each group of packages is for: Open3D is needed only for the XYZ-IBD reference
depth, PyTorch only for the segmentation experiment, `requests` and `remotezip` only for the XYZ-IBD download.

Rendering needs Isaac Sim 6.2 and an NVIDIA RTX GPU, so it runs on Linux or Windows, not on macOS. On Linux we
use the container `nvcr.io/nvidia/isaac-sim:6.2.0` (`isaac/run_container.sh`); on Windows the generator runs
with the Python of a local Isaac Sim installation (`<isaac-sim>\python.bat isaac\generate_scenes.py ...`, same
arguments as in `isaac/render_all.sh`). Everything else runs on CPU; the segmentation uses a GPU if available.

## Data

**Synthetic.** Start the container from the repository root and render:

```bash
bash isaac/run_container.sh
# inside the container
bash isaac/render_all.sh
```

This writes `data/scenes_main` (200 scenes, seed 1), two calibration planes, `data/hdr_test` (3 scenes for
the sensor-physics ablations) and re-renders of the first 30 scenes with other materials: three roughness values
of the polished finish (`data/polished_r*`) and three anodized variants (`data/anodized_*`). The re-renders use
the same seed, so geometry and camera poses are identical. Rendering takes about 12 s per scene. Each scene
folder contains `rgb.png`, `ir_left.png`, `ir_right.png`, `ir_*_hdr.npy`, `depth_gt.npy`, `semantic.npy`,
`instance.npy`, `normals_world.npy`, `labels.json`, `camera.json` and `scene.json` (object poses).

**Real D415.**
- ROBI ([Yang et al. 2021](https://arxiv.org/abs/2105.04112)): download Zigzag and Chrome Screw from the
  dataset page and place them as `data/robi/Zigzag/Scene_{4,5}` and `data/robi/Chrome_screw/Scene_{4,5}`.
- XYZ-IBD ([BOP](https://huggingface.co/datasets/bop-benchmark/xyzibd)): only the RealSense files of the
  validation split are needed (about 90 MB of the 7.7 GB archive):
  ```bash
  python -m datasets.download_xyzibd --out data/xyzibd --every 5
  # then unzip xyzibd_base.zip and xyzibd_models.zip from the same page into data/xyzibd
  ```

## Reproducing the results

```bash
python run_analysis.py                  # --skip-depths reuses the depth files already in data/
```

Every number and figure of the report comes from one of these scripts:

| report | script (`python -m ...`) | output in `results/` |
|---|---|---|
| noise and pixel locking on a plane, Fig. `plane_noise` | `experiments.plane_noise` | `plane_noise.csv` |
| calibration of the empirical model, held-out scores | `experiments.calibrate_empirical` | `calibration/` |
| depth models | `experiments.make_depths <scenes> --model <name>` | `depth_<name>.npy` per scene |
| holes and wrong depth vs. incidence angle, Fig. `angle_curves` | `experiments.angle_curves` | `angle_curves.csv` |
| Middlebury table, W1, KS, SRCC, ROBI table noise, Fig. `error_distribution` | `experiments.depth_metrics` | `depth_metrics/` |
| error vs. distance on XYZ-IBD (exponent α) | `experiments.xyzibd_transfer` | `xyzibd_transfer.csv` |
| grasp table, success per distance and orientation | `experiments.grasp` | `grasp_percentile/`, `grasp_mode/` |
| bootstrap, McNemar + Holm, design effect, GEE, Fig. `gee_odds_ratios` | `experiments.grasp_statistics` | `statistics/` |
| sensitivity to the grasp tolerance (5 / 10 / 15 mm) | `experiments.tolerance_sensitivity` | `statistics/tolerance_sensitivity.csv` |
| roughness sensitivity, Fig. `roughness_distance` | `experiments.roughness` | `roughness/` |
| segmentation table and paired differences | `experiments.segmentation` | `segmentation/` |
| what did not work: anodized variants, IR noise / speckle / HDR | `experiments.angle_curves` + `experiments.curve_distance` | `anodized/`, `ablations/` |
| what did not work: median lateral centre | `experiments.grasp --centre median` | `grasp_median_centre/` |
| all figures | `experiments.make_figures` | `report/figures/` |

All scripts are run from the repository root as modules (`python -m experiments.<name>`). Seeds are fixed. The
stereo models are deterministic; the random models (`gaussian`, `heuristic`, `empirical_*`) draw from a seeded
generator per run, so regenerating them gives the same statistics but not bit-identical images. Path tracing
itself is not deterministic, so a re-render gives slightly different IR images.

The slowest steps are the stereo models (a few seconds per image with numba) and the segmentation training,
which is much faster on a GPU.

## Notes

- Segmentation is assumed perfect in the grasp experiment (ground-truth instance masks), so the comparison
  isolates the effect of depth.
- The grasp criterion is geometric: lateral and height error < 10 mm, axis error < 10°, error along the axis
  < 30 mm, for a parallel gripper on a 40 mm profile.
- ROBI and XYZ-IBD are used only to calibrate and validate the depth models, never in the grasp test.
