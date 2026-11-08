"""Render synthetic D415 scenes of aluminium profiles and brackets with Isaac Sim (path tracing).

The active stereo of the D415 is rendered physically: a point light behind a perforated mask projects the dot
pattern, and two IR cameras 55 mm apart see it. Depth from the IR pair is computed afterwards, outside Isaac Sim
(experiments/make_depths.py).

  --mode scenes  random scenes; per scene: RGB, IR pair (8 bit and linear radiance), ground-truth depth,
                 semantic and instance masks, normals, object poses, camera parameters
  --mode plane   a matte plane at several distances, to check the depth noise against the datasheet

Inside the container (see run_container.sh):
  /isaac-sim/python.sh isaac/generate_scenes.py --mode scenes --num-scenes 200 --seed 1 --out data/scenes_main
  /isaac-sim/python.sh isaac/generate_scenes.py --mode plane --tilt-deg 25 --out data/plane_tilt25
"""
import argparse
import json
import math
import os
import time

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["scenes", "plane"], default="scenes")
parser.add_argument("--out", default="data/scenes_main")
parser.add_argument("--num-scenes", type=int, default=200)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--spp", type=int, default=32, help="path-tracing samples per frame")
parser.add_argument("--denoise", type=int, default=1)
parser.add_argument("--proj-exposure", type=float, default=3.0)
parser.add_argument("--mask-cells", type=int, default=600, help="cells per side of the projector mask")
parser.add_argument("--hole-frac", type=float, default=0.08)
parser.add_argument("--light-radius", type=float, default=1e-5)
parser.add_argument("--distances", type=str, default="0.5,0.75,1.0,1.25,1.5", help="plane mode only")
parser.add_argument("--tilt-deg", type=float, default=0.0, help="plane mode only: rotation about the vertical axis")
parser.add_argument("--hdr", type=int, default=1, help="also save the linear IR radiance (needed by stereo_hdr)")
parser.add_argument("--rough-anodized", type=float, default=0.45)
parser.add_argument("--metallic-anodized", type=float, default=1.0)
parser.add_argument("--rough-polished", type=float, default=0.10)
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp(launch_config={"headless": True})

import carb.settings  # noqa: E402
import isaacsim.core.experimental.utils.stage as stage_utils  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt  # noqa: E402

# D415 (datasheet): 1280x720, HFOV 65 deg, 55 mm baseline, RGB 15 mm from the left IR imager
W, H = 1280, 720
HFOV_DEG = 65.0
FX = 0.5 * W / math.tan(math.radians(HFOV_DEG) / 2)
HA = 20.955                 # horizontal aperture [mm], only sets the focal length in USD units
BASELINE = 0.055
RGB_OFFSET = 0.015
PROJ_OFFSET = BASELINE / 2
MASK_DIST = 0.02
MASK_HALF = 0.018           # covers the field of view also along the diagonal
CONE_DEG = 40.0             # projector housing: no light beyond 40 deg

OUT = os.path.abspath(args.out)
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(args.seed)
settings = carb.settings.get_settings()


def log(msg):
    print(f"[gen] {msg}", flush=True)


# ------------------------------------------------------------------ helper USD
def set_pose(prim, t, yaw_deg=0.0, scale=(1.0, 1.0, 1.0)):
    xf = UsdGeom.Xformable(prim)
    if "xformOp:translate:pose" not in [op.GetOpName() for op in xf.GetOrderedXformOps()]:
        xf.ClearXformOpOrder()
        xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble, "pose")
        xf.AddRotateZOp(UsdGeom.XformOp.PrecisionDouble, "pose")
        xf.AddScaleOp(UsdGeom.XformOp.PrecisionDouble, "pose")
    ops = {op.GetOpName(): op for op in xf.GetOrderedXformOps()}
    ops["xformOp:translate:pose"].Set(Gf.Vec3d(*map(float, t)))
    ops["xformOp:rotateZ:pose"].Set(float(yaw_deg))
    ops["xformOp:scale:pose"].Set(Gf.Vec3d(*map(float, scale)))


def set_matrix(prim, m):
    xf = UsdGeom.Xformable(prim)
    ops = xf.GetOrderedXformOps()
    if len(ops) == 1 and ops[0].GetOpType() == UsdGeom.XformOp.TypeTransform:
        ops[0].Set(m)
    else:
        xf.ClearXformOpOrder()
        xf.AddTransformOp().Set(m)


def look_at_matrix(eye, target):
    """camera-to-world, USD convention (looks along local -Z, Y up)."""
    return Gf.Matrix4d().SetLookAt(Gf.Vec3d(*map(float, eye)), Gf.Vec3d(*map(float, target)),
                                   Gf.Vec3d(0, 0, 1)).GetInverse()


def world_matrix(prim):
    m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return np.array(m, dtype=np.float64)          # USD row vectors: p_w = p_l @ M


def make_material(stage, path, color, metallic, roughness):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def bind(prim, mat):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)


def set_visible(prim, on):
    img = UsdGeom.Imageable(prim)
    img.MakeVisible() if on else img.MakeInvisible()


def quads_to_mesh(stage, path, quads):
    quads = np.asarray(quads, dtype=np.float32)
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(quads.reshape(-1, 3)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(quads), 4, np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.arange(len(quads) * 4, dtype=np.int32)))
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    return mesh.GetPrim()


def box_quads(lo, hi):
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    return [
        [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],
        [(x1, y0, z0), (x1, y0, z1), (x1, y1, z1), (x1, y1, z0)],
        [(x0, y0, z0), (x0, y0, z1), (x1, y0, z1), (x1, y0, z0)],
        [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],
        [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],
        [(x0, y0, z1), (x0, y1, z1), (x1, y1, z1), (x1, y0, z1)],
    ]


def tslot_quads():
    """40x40 T-slot profile, 8 mm slots 12 mm deep; unit length along x (scaled per scene)."""
    s = 0.001
    rects = [(-20, -20, -4, -4), (4, -20, 20, -4), (-20, 4, -4, 20), (4, 4, 20, 20),
             (-4, -8, 4, 8), (-8, -4, -4, 4), (4, -4, 8, 4)]
    q = []
    for (y0, z0, y1, z1) in rects:
        q += box_quads((-0.5, y0 * s, z0 * s + 0.02), (0.5, y1 * s, z1 * s + 0.02))
    return q


def bracket_quads():
    """L-bracket 40x40x40, 4 mm thick."""
    return box_quads((-0.02, -0.02, 0.0), (0.02, 0.02, 0.004)) + box_quads((-0.02, -0.02, 0.0), (-0.016, 0.02, 0.04))


def mask_quads():
    n = args.mask_cells
    keep = rng.random((n, n)) > args.hole_frac
    ii, jj = np.nonzero(keep)
    c = 2 * MASK_HALF / n
    x0, y0 = -MASK_HALF + ii * c, -MASK_HALF + jj * c
    z = np.full_like(x0, -MASK_DIST)
    q = np.stack([np.stack([x0, y0, z], -1), np.stack([x0 + c, y0, z], -1),
                  np.stack([x0 + c, y0 + c, z], -1), np.stack([x0, y0 + c, z], -1)], 1)
    return q, int((~keep).sum()), c


# ------------------------------------------------------------------ static stage
stage_utils.create_new_stage()
rep.orchestrator.set_capture_on_play(False)
settings.set("rtx/post/dlss/execMode", 2)
settings.set("/rtx/rendermode", "PathTracing")
settings.set("/rtx/pathtracing/optixDenoiser/enabled", bool(args.denoise))
stage = omni.usd.get_context().get_stage()
log(f"args={vars(args)}")

rep.functional.create.xform(name="World")
dome = rep.functional.create.dome_light(intensity=150, parent="/World", name="DomeLight")
ceiling = []
for k, x in enumerate((-0.6, 0.0, 0.6)):
    rl = rep.functional.create.rect_light(parent="/World", name=f"Ceiling_{k}")
    L = UsdLux.RectLight(rl)
    L.CreateWidthAttr(1.2)
    L.CreateHeightAttr(0.15)
    L.CreateIntensityAttr(3000.0)
    set_pose(rl, (x, 0.1, 2.0), 90.0)
    ceiling.append(L)

m_table = make_material(stage, "/World/Looks/Table", (0.35, 0.35, 0.37), 0.0, 0.7)
m_wall = make_material(stage, "/World/Looks/CalibTarget", (0.5, 0.5, 0.5), 0.0, 0.9)
m_mask = make_material(stage, "/World/Looks/Mask", (0.0, 0.0, 0.0), 0.0, 1.0)
finishes = {"anodized": make_material(stage, "/World/Looks/AluAnodized", (0.85, 0.85, 0.87), args.metallic_anodized,
                                     args.rough_anodized),
            "polished": make_material(stage, "/World/Looks/AluPolished", (0.91, 0.92, 0.92), 1.0, args.rough_polished)}

table = rep.functional.create.cube(parent="/World", name="Table")
set_pose(table, (0, 0, -0.01), 0, (4.0, 4.0, 0.02))
bind(table, m_table)
rep.functional.modify.semantics(table, {"class": "table"}, mode="add")

# cameras: left IR (depth reference), right IR, RGB
FOCAL = FX * HA / W
cams = {}
for name in ("ir_left", "ir_right", "rgb"):
    c = rep.functional.create.camera(position=(0, -1, 1), look_at=(0, 0, 0), parent="/World", name=name,
                                     focal_length=FOCAL, horizontal_aperture=HA, clipping_range=(0.05, 20.0))
    c.GetAttribute("verticalAperture").Set(HA * H / W)
    cams[name] = c

# projector: point light + perforated mask + emission cone
proj = UsdGeom.Xform.Define(stage, "/World/Projector").GetPrim()
plight = UsdLux.SphereLight.Define(stage, "/World/Projector/Light")
plight.CreateRadiusAttr(args.light_radius)
plight.CreateIntensityAttr(1000.0)
plight.CreateNormalizeAttr(True)
plight.CreateExposureAttr(args.proj_exposure)
shaping = UsdLux.ShapingAPI.Apply(plight.GetPrim())
shaping.CreateShapingConeAngleAttr(CONE_DEG)
shaping.CreateShapingConeSoftnessAttr(0.0)
mq, n_holes, cell = mask_quads()
bind(quads_to_mesh(stage, "/World/Projector/Mask", mq), m_mask)
log(f"mask: {n_holes} holes, cell {cell * 1e3:.3f} mm -> dot ~{cell / MASK_DIST * 1e3:.1f} mm at 1 m")


def place_rig(eye, target):
    """Place cameras and projector as one rigid D415 (parallel axes)."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    fwd = (target - eye) / np.linalg.norm(target - eye)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    for name, off in (("ir_left", 0.0), ("ir_right", BASELINE), ("rgb", RGB_OFFSET)):
        set_matrix(cams[name], look_at_matrix(eye + off * right, target + off * right))
    set_matrix(proj, look_at_matrix(eye + PROJ_OFFSET * right, target + PROJ_OFFSET * right))


# render products and annotators
rps = {n: rep.create.render_product(c, (W, H), name=f"RP_{n}") for n, c in cams.items()}
A = {}
for n in ("ir_left", "ir_right", "rgb"):
    A[f"{n}_rgb"] = rep.AnnotatorRegistry.get_annotator("rgb")
    A[f"{n}_rgb"].attach(rps[n])
for key, name in (("depth", "distance_to_image_plane"), ("semantic", "semantic_segmentation"),
                  ("instance", "instance_id_segmentation"), ("normals", "normals"), ("cam_left", "camera_params")):
    A[key] = rep.AnnotatorRegistry.get_annotator(name)
    A[key].attach(rps["ir_left"])
A["cam_rgb"] = rep.AnnotatorRegistry.get_annotator("camera_params")
A["cam_rgb"].attach(rps["rgb"])
# linear radiance before tone mapping, for the HDR sensor variant; one channel is enough for IR
if args.hdr:
    for n in ("ir_left", "ir_right"):
        A[f"{n}_hdr"] = rep.AnnotatorRegistry.get_annotator("HdrColor")
        A[f"{n}_hdr"].attach(rps[n])


def lighting(mode):
    ir = mode == "ir"
    UsdLux.DomeLight(dome).GetIntensityAttr().Set(40.0 if ir else 150.0)
    for L in ceiling:
        L.GetIntensityAttr().Set(300.0 if ir else 3000.0)
    plight.GetIntensityAttr().Set(1000.0 if ir else 0.0)


def gray(img):
    rgb = np.asarray(img)[..., :3].astype(np.float32)
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).clip(0, 255).astype(np.uint8)


def jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    return x


def capture(folder, with_rgb=True):
    os.makedirs(folder, exist_ok=True)
    t0 = time.time()
    lighting("ir")
    rep.orchestrator.step(rt_subframes=args.spp)
    Image.fromarray(gray(A["ir_left_rgb"].get_data())).save(os.path.join(folder, "ir_left.png"))
    Image.fromarray(gray(A["ir_right_rgb"].get_data())).save(os.path.join(folder, "ir_right.png"))
    if args.hdr:
        for n in ("ir_left", "ir_right"):
            h = np.asarray(A[f"{n}_hdr"].get_data())
            np.save(os.path.join(folder, f"{n}_hdr.npy"), h[..., 0].astype(np.float32))
    np.save(os.path.join(folder, "depth_gt.npy"), np.asarray(A["depth"].get_data(), dtype=np.float32))
    sem, inst = A["semantic"].get_data(), A["instance"].get_data()
    np.save(os.path.join(folder, "semantic.npy"), np.asarray(sem["data"]).astype(np.uint16))
    np.save(os.path.join(folder, "instance.npy"), np.asarray(inst["data"]).astype(np.uint16))
    np.save(os.path.join(folder, "normals_world.npy"), np.asarray(A["normals"].get_data())[..., :3].astype(np.float16))
    with open(os.path.join(folder, "labels.json"), "w") as f:
        json.dump({"semantic": jsonable(sem["info"]["idToLabels"]), "instance": jsonable(inst["info"]["idToLabels"])}, f, indent=1)
    cams_json = {"ir_left": jsonable(A["cam_left"].get_data()), "rgb": jsonable(A["cam_rgb"].get_data()),
                 "intrinsics": {"fx": FX, "fy": FX, "cx": W / 2, "cy": H / 2, "W": W, "H": H},
                 "baseline_m": BASELINE, "rgb_offset_m": RGB_OFFSET}
    with open(os.path.join(folder, "camera.json"), "w") as f:
        json.dump(cams_json, f, indent=1)
    if with_rgb:
        lighting("rgb")
        rep.orchestrator.step(rt_subframes=args.spp)
        Image.fromarray(np.asarray(A["rgb_rgb"].get_data())[..., :3]).save(os.path.join(folder, "rgb.png"))
    return time.time() - t0


# ------------------------------------------------------------------ plane mode
if args.mode == "plane":
    set_visible(table, False)
    wall = rep.functional.create.cube(parent="/World", name="CalibWall")
    bind(wall, m_wall)
    rep.functional.modify.semantics(wall, {"class": "calib_target"}, mode="add")
    place_rig((0, 0, 1.0), (0, 1.0, 1.0))                     # camera 1 m above the floor, looking along +y
    for z in [float(d) for d in args.distances.split(",")]:
        # front face at distance z; a tilted plane spreads the fractional disparities and avoids pixel locking
        t = math.radians(args.tilt_deg)
        set_pose(wall, (-0.01 * math.sin(t), z + 0.01 * math.cos(t), 1.0), args.tilt_deg, (6.0, 0.02, 6.0))
        dt = capture(os.path.join(OUT, f"plane_z{z:.2f}"), with_rgb=False)
        log(f"plane at {z:.2f} m: {dt:.1f} s")

# ------------------------------------------------------------------ scenes mode
else:
    TS, BR = tslot_quads(), bracket_quads()
    MAX_PROFILES, MAX_BRACKETS = 10, 6
    objects = []
    for i in range(MAX_PROFILES + MAX_BRACKETS):
        kind = "profile" if i < MAX_PROFILES else "bracket"
        fin = "polished" if i % 2 else "anodized"
        prim = quads_to_mesh(stage, f"/World/Parts/{kind}_{i:02d}", TS if kind == "profile" else BR)
        bind(prim, finishes[fin])
        rep.functional.modify.semantics(prim, {"class": f"{kind}_{fin}"}, mode="add")
        objects.append({"prim": prim, "kind": kind, "finish": fin, "path": prim.GetPath().pathString})

    def seg_dist(p1, q1, p2, q2):
        """Minimum distance between two 2D segments (sampled)."""
        t = np.linspace(0, 1, 15)
        a = p1[None] + t[:, None] * (q1 - p1)[None]
        b = p2[None] + t[:, None] * (q2 - p2)[None]
        return np.min(np.linalg.norm(a[:, None] - b[None], axis=-1))

    def randomize_scene():
        """Parts lying on the table without overlaps (rejection sampling)."""
        n_p = rng.integers(4, MAX_PROFILES + 1)
        n_b = rng.integers(2, MAX_BRACKETS + 1)
        active = [o for o in objects if o["kind"] == "profile"][:n_p] + [o for o in objects if o["kind"] == "bracket"][:n_b]
        placed, meta = [], []
        for o in objects:
            set_visible(o["prim"], False)
        for o in active:
            for _ in range(50):
                c = rng.uniform([-0.3, -0.25], [0.3, 0.25])
                yaw = rng.uniform(0, 180)
                L = rng.uniform(0.12, 0.4) if o["kind"] == "profile" else 0.04
                d = np.array([math.cos(math.radians(yaw)), math.sin(math.radians(yaw))]) * L / 2
                seg = (c - d, c + d, 0.03 if o["kind"] == "profile" else 0.035)
                if all(seg_dist(seg[0], seg[1], s[0], s[1]) > seg[2] + s[2] + 0.005 for s in placed):
                    placed.append(seg)
                    if o["kind"] == "profile":
                        set_pose(o["prim"], (c[0], c[1], 0.0), yaw, (L, 1, 1))
                    else:
                        set_pose(o["prim"], (c[0], c[1], 0.0), yaw)
                    set_visible(o["prim"], True)
                    meta.append({"path": o["path"], "kind": o["kind"], "finish": o["finish"],
                                 "length_m": L if o["kind"] == "profile" else None, "yaw_deg": yaw})
                    break
        return meta

    def random_camera():
        dist = rng.uniform(0.6, 1.2)
        elev = math.radians(rng.uniform(50, 80))
        az = math.radians(rng.uniform(-120, -60))         # from the operator side of the cell
        target = np.array([*rng.uniform([-0.05, -0.05], [0.05, 0.05]), 0.0])
        eye = target + dist * np.array([math.cos(elev) * math.cos(az), math.cos(elev) * math.sin(az), math.sin(elev)])
        return eye, target, dist

    for s in range(args.num_scenes):
        folder = os.path.join(OUT, f"scene_{s:04d}")
        meta = randomize_scene()
        eye, target, dist = random_camera()
        place_rig(eye, target)
        dt = capture(folder)
        for m in meta:
            m["T_world_obj"] = world_matrix(stage.GetPrimAtPath(m["path"])).tolist()
        with open(os.path.join(folder, "scene.json"), "w") as f:
            json.dump({"objects": meta, "rig_eye": eye.tolist(), "rig_target": target.tolist(),
                       "rig_distance_m": dist, "seed": args.seed, "args": vars(args)}, f, indent=1)
        log(f"scene {s:04d}: {len(meta)} objects, camera at {dist:.2f} m, {dt:.1f} s")

rep.orchestrator.wait_until_complete()
log("DONE")
simulation_app.close()
