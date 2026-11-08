"""Loading a rendered scene and per-pixel geometry (incidence angle, finish masks)."""
import json

import numpy as np


def load_scene(scene_dir):
    """Ground-truth depth, normals, semantics and the left IR camera of one rendered scene."""
    cam = json.load(open(f"{scene_dir}/camera.json"))
    labels = json.load(open(f"{scene_dir}/labels.json"))["semantic"]
    return {
        "depth": np.load(f"{scene_dir}/depth_gt.npy").astype(np.float32),
        "normals": np.load(f"{scene_dir}/normals_world.npy").astype(np.float32),
        "semantic": np.load(f"{scene_dir}/semantic.npy"),
        "labels": {int(k): v["class"] for k, v in labels.items()},
        "fx": cam["intrinsics"]["fx"],
        "B": cam["baseline_m"],
        "V": np.array(cam["ir_left"]["cameraViewTransform"]).reshape(4, 4),
    }


def view_cosine(g):
    """Cosine between the surface normal and the direction to the camera."""
    z, fx = g["depth"], g["fx"]
    h, w = z.shape
    v, u = np.mgrid[0:h, 0:w].astype(np.float32)
    zz = np.where(np.isfinite(z), z, 0)
    p_cam = np.stack([(u + 0.5 - w / 2) / fx * zz, -(v + 0.5 - h / 2) / fx * zz, -zz, np.ones_like(zz)], -1)
    cam_to_world = np.linalg.inv(g["V"])  # USD convention: row vectors
    p_w = p_cam @ cam_to_world
    to_cam = cam_to_world[3, :3][None, None, :] - p_w[..., :3]
    to_cam /= np.linalg.norm(to_cam, axis=-1, keepdims=True) + 1e-9
    n = g["normals"] / (np.linalg.norm(g["normals"], axis=-1, keepdims=True) + 1e-9)
    return np.clip(np.abs(np.sum(n * to_cam, -1)), 0, 1)


def finish_masks(g):
    out = {}
    for fin in ("anodized", "polished"):
        ids = [k for k, c in g["labels"].items() if c.endswith(fin)]
        out[fin] = np.isin(g["semantic"], ids)
    return out
