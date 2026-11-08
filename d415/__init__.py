"""Simulated Intel RealSense D415: depth models applied to rendered scenes."""

# depth model name -> file written next to the rendered scene
DEPTH_MODELS = {
    "ideal": "depth_gt.npy",
    "gaussian": "depth_gaussian.npy",
    "heuristic": "depth_heuristic.npy",
    "empirical_metric": "depth_empirical_metric.npy",
    "empirical_disparity": "depth_empirical_disparity.npy",
    "stereo": "depth_stereo.npy",
    "stereo_sgbm_strict": "depth_stereo_sgbm_strict.npy",
    "stereo_hdr": "depth_stereo_hdr.npy",
    # sensor-physics ablations, run on a few scenes only (report: what did not work)
    "stereo_ir_noise": "depth_stereo_ir_noise.npy",
    "stereo_hdr_speckle": "depth_stereo_hdr_speckle.npy",
}

BASELINE_M = 0.055
