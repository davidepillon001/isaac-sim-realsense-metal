"""XYZ-IBD (Huang et al. 2025, BOP format): raw D415 depth of metal parts with CAD models and 6-D poses.
The reference depth is obtained by ray casting the CAD models at the annotated poses."""
import glob
import json
import os

import cv2
import numpy as np

BASELINE_MM = 55.0


def load_meshes(models_dir):
    import open3d as o3d
    meshes = {}
    for p in glob.glob(os.path.join(models_dir, "obj_*.ply")):
        m = o3d.io.read_triangle_mesh(p)
        meshes[int(os.path.basename(p)[4:10])] = (np.asarray(m.vertices), np.asarray(m.triangles))
    return meshes


def render_gt(meshes, gts, K, h, w):
    """Depth [mm] and incidence cosine of the posed CAD models."""
    import open3d as o3d
    sc = o3d.t.geometry.RaycastingScene()
    for g in gts:
        v, f = meshes[g["obj_id"]]
        R = np.array(g["cam_R_m2c"]).reshape(3, 3)
        t = np.array(g["cam_t_m2c"]).reshape(3)
        tm = o3d.t.geometry.TriangleMesh()
        tm.vertex.positions = o3d.core.Tensor((v @ R.T + t).astype(np.float32))
        tm.triangle.indices = o3d.core.Tensor(f.astype(np.int32))
        sc.add_triangles(tm)
    Kinv = np.linalg.inv(np.array(K).reshape(3, 3))
    u, vv = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    d = np.stack([u, vv, np.ones_like(u)], -1) @ Kinv.T
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    rays = np.concatenate([np.zeros_like(d), d], -1).reshape(-1, 6).astype(np.float32)
    ans = sc.cast_rays(o3d.core.Tensor(rays))
    t_hit = ans["t_hit"].numpy().reshape(h, w)
    n = ans["primitive_normals"].numpy().reshape(h, w, 3)
    z = np.where(np.isfinite(t_hit), t_hit * d[..., 2], 0)
    cos_t = np.abs(np.sum(n * d, -1))
    return z.astype(np.float32), np.where(z > 0, cos_t, np.nan)


def frames(root, split="val", every=1, max_frames=100000):
    """Yields (raw_depth_mm, reference_mm, visible_object_mask, cos_incidence, f_x * B) per frame."""
    meshes = load_meshes(os.path.join(root, "models"))
    used = 0
    for sd in sorted(glob.glob(os.path.join(root, split, "*"))):
        if not os.path.exists(f"{sd}/scene_gt_realsense.json"):
            continue
        cams = json.load(open(f"{sd}/scene_camera_realsense.json"))
        gts = json.load(open(f"{sd}/scene_gt_realsense.json"))
        for k in sorted(gts, key=int)[::every]:
            if used >= max_frames:
                return
            im = int(k)
            dp = f"{sd}/depth_realsense/{im:06d}.png"
            if not os.path.exists(dp):
                continue
            raw = cv2.imread(dp, -1).astype(np.float32) * float(cams[k].get("depth_scale", 1.0))
            h, w = raw.shape
            K = cams[k]["cam_K"]
            gt, cos_t = render_gt(meshes, gts[k], K, h, w)
            vis = np.zeros((h, w), bool)
            for j in range(len(gts[k])):
                mp = f"{sd}/mask_visib_realsense/{im:06d}_{j:06d}.png"
                if os.path.exists(mp):
                    vis |= cv2.imread(mp, 0) > 0
            if not vis.any():
                vis = gt > 0
            m = cv2.erode((vis & (gt > 0)).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
            if m.sum() < 500:
                continue
            used += 1
            yield raw, gt, m, cos_t, float(np.array(K)[0] * BASELINE_MM)
