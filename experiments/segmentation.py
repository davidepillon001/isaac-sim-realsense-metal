"""Train on one depth model, test on another: a small U-Net segments background / profile / bracket from depth.

Input: 320x180, two channels (height relative to the image median, validity mask). Scenes 0-159 for training,
160-199 for testing, 3 seeds per configuration. Training is deterministic for a given device (CPU or GPU). Metrics: IoU of the profile class and recall of profile pixels
per finish, with bootstrap intervals over test scenes.

    python -m experiments.segmentation prepare --root data/scenes_main --cache data/seg_cache.npz
    python -m experiments.segmentation train --cache data/seg_cache.npz --out results/segmentation
"""
import argparse
import glob
import json
import os
import time

import cv2
import numpy as np
import pandas as pd

from d415 import DEPTH_MODELS

H, W = 180, 320
SOURCES = ["ideal", "gaussian", "stereo"]


def prepare(a):
    X = {k: [] for k in SOURCES}
    Y, F, names = [], [], []
    for sd in sorted(glob.glob(os.path.join(a.root, "scene_*"))):
        if not all(os.path.exists(f"{sd}/{DEPTH_MODELS[k]}") for k in SOURCES):
            continue
        lab = json.load(open(f"{sd}/labels.json"))["semantic"]
        sem = np.load(f"{sd}/semantic.npy")
        y = np.zeros(sem.shape, np.uint8)      # 0 background, 1 profile, 2 bracket
        fin = np.zeros(sem.shape, np.uint8)    # 1 anodized, 2 polished
        for k, v in lab.items():
            c, m = v["class"], sem == int(k)
            y[m] = 1 if c.startswith("profile") else 2 if c.startswith("bracket") else y[m]
            fin[m] = 1 if c.endswith("anodized") else 2 if c.endswith("polished") else fin[m]
        Y.append(cv2.resize(y, (W, H), interpolation=cv2.INTER_NEAREST))
        F.append(cv2.resize(fin, (W, H), interpolation=cv2.INTER_NEAREST))
        for k in SOURCES:
            z = np.load(f"{sd}/{DEPTH_MODELS[k]}").astype(np.float32)
            X[k].append(cv2.resize(np.where(np.isfinite(z) & (z > 0), z, 0), (W, H), interpolation=cv2.INTER_NEAREST))
        names.append(os.path.basename(sd))
    os.makedirs(os.path.dirname(a.cache) or ".", exist_ok=True)
    np.savez_compressed(a.cache, Y=np.array(Y), F=np.array(F), names=np.array(names),
                        **{f"X_{k}": np.array(v) for k, v in X.items()})
    print(f"{len(names)} scenes -> {a.cache}")


def to_input(z):
    valid = (z > 0).astype(np.float32)
    med = np.array([np.median(zi[zi > 0]) if (zi > 0).any() else 1.0 for zi in z], np.float32)[:, None, None]
    rel = np.where(z > 0, (med - z) * 10.0, 0.0).astype(np.float32)   # decimetres
    return np.stack([rel, valid], 1)


def make_unet(torch, base=16):
    nn = torch.nn

    def block(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                             nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            c = [base, base * 2, base * 4, base * 8]
            self.e = nn.ModuleList([block(2, c[0]), block(c[0], c[1]), block(c[1], c[2]), block(c[2], c[3])])
            self.u = nn.ModuleList([nn.ConvTranspose2d(c[i + 1], c[i], 2, stride=2) for i in range(3)])
            self.d = nn.ModuleList([block(c[i] * 2, c[i]) for i in range(3)])
            self.head = nn.Conv2d(c[0], 3, 1)

        def forward(self, x):
            skips = []
            for i, e in enumerate(self.e):
                x = e(x)
                if i < 3:
                    skips.append(x)
                    x = nn.functional.max_pool2d(x, 2)
            for i in (2, 1, 0):
                x = self.u[i](x)
                s = skips[i]
                x = nn.functional.pad(x, (0, s.shape[3] - x.shape[3], 0, s.shape[2] - x.shape[2]))
                x = self.d[i](torch.cat([x, s], 1))
            return self.head(x)

    return UNet()


def weighted_cross_entropy(torch, logits, y, w):
    """Same as F.cross_entropy(logits, y, weight=w), written element-wise: the built-in 2-D loss has no
    deterministic CUDA implementation."""
    onehot = torch.nn.functional.one_hot(y, logits.shape[1]).permute(0, 3, 1, 2).to(logits.dtype)
    wp = w[y]
    return -(wp * (onehot * torch.log_softmax(logits, 1)).sum(1)).sum() / wp.sum()


def train_one(torch, X, Y, epochs, seed, device, bs=8):
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = make_unet(torch).to(device)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    w = torch.tensor([0.2, 1.0, 1.5], device=device)   # background dominates
    Xt, Yt = torch.from_numpy(X), torch.from_numpy(Y.astype(np.int64))
    for _ in range(epochs):
        net.train()
        perm = np.random.permutation(len(Xt))
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            x, y = Xt[idx].to(device), Yt[idx].to(device)
            if np.random.rand() < 0.5:
                x, y = x.flip(-1), y.flip(-1)
            if np.random.rand() < 0.5:
                x, y = x.flip(-2), y.flip(-2)
            loss = weighted_cross_entropy(torch, net(x), y, w)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    return net


def predict(torch, net, X, device, bs=8):
    net.eval()
    with torch.no_grad():
        return np.concatenate([net(torch.from_numpy(X[i:i + bs]).to(device)).argmax(1).cpu().numpy()
                               for i in range(0, len(X), bs)])


def counts(P, Y, F):
    """Per scene (numerator, denominator): profile IoU, and recall per finish."""
    rows = []
    for p, y, f in zip(P, Y, F):
        gt, pr = y == 1, p == 1
        r = {"all": (np.logical_and(gt, pr).sum(), np.logical_or(gt, pr).sum())}
        for name, k in (("anodized", 1), ("polished", 2)):
            g = gt & (f == k)
            r[name] = (np.logical_and(g, pr).sum(), g.sum())
        rows.append(r)
    return rows


def ratio_boot(c, key, B=5000, seed=0):
    num = np.array([x[key][0] for x in c], float)
    den = np.array([x[key][1] for x in c], float)
    idx = np.random.default_rng(seed).integers(0, len(num), (B, len(num)))
    b = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1)
    return 100 * num.sum() / max(den.sum(), 1), *(100 * np.percentile(b, [2.5, 97.5]))


def train(a):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # deterministic cuBLAS, must precede CUDA init
    import torch
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    D = np.load(a.cache)
    n = len(D["names"])
    tr, te = np.arange(0, int(n * 0.8)), np.arange(int(n * 0.8), n)
    Y, F = D["Y"], D["F"]
    X = {k: to_input(D[f"X_{k}"]) for k in SOURCES}
    os.makedirs(a.out, exist_ok=True)
    rows, cnt = [], {}
    for train_on in a.train_on:
        for seed in range(a.seeds):
            t0 = time.time()
            net = train_one(torch, X[train_on][tr], Y[tr], a.epochs, seed, device)
            for test_on in a.test_on:
                c = cnt[(train_on, test_on, seed)] = counts(predict(torch, net, X[test_on][te], device), Y[te], F[te])
                for key in ("all", "anodized", "polished"):
                    v, lo, hi = ratio_boot(c, key)
                    rows.append({"train_on": train_on, "test_on": test_on, "seed": seed, "profiles": key,
                                 "value_pct": v, "ci_low": lo, "ci_high": hi})
            print(f"train on {train_on}, seed {seed}: {time.time() - t0:.0f} s", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(a.out, "per_seed.csv"), index=False)
    S = R.groupby(["train_on", "test_on", "profiles"]).value_pct.agg(["mean", "std"]).reset_index()
    S.to_csv(os.path.join(a.out, "summary.csv"), index=False)

    # paired difference on the same test scenes and seeds: trained on stereo minus trained on another model
    comp = []
    for other in [t for t in a.train_on if t != "stereo"]:
        for key in ("all", "anodized", "polished"):
            diffs = []
            for seed in range(a.seeds):
                cs, co = cnt.get(("stereo", "stereo", seed)), cnt.get((other, "stereo", seed))
                if cs is None or co is None:
                    continue
                idx = np.random.default_rng(seed).integers(0, len(cs), (5000, len(cs)))

                def ratio(c):
                    num = np.array([x[key][0] for x in c], float)[idx].sum(1)
                    return num / np.maximum(np.array([x[key][1] for x in c], float)[idx].sum(1), 1)
                diffs.append(100 * (ratio(cs) - ratio(co)))
            if diffs:
                d = np.concatenate(diffs)
                comp.append({"comparison": f"stereo - {other}", "profiles": key, "diff_pt": float(d.mean()),
                             "ci_low": float(np.percentile(d, 2.5)), "ci_high": float(np.percentile(d, 97.5))})
    C = pd.DataFrame(comp)
    C.to_csv(os.path.join(a.out, "paired_differences.csv"), index=False)
    print(S.pivot_table(index=["train_on", "test_on"], columns="profiles", values="mean").round(1).to_string())
    print(C.round(1).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "train"])
    ap.add_argument("--root", default="data/scenes_main")
    ap.add_argument("--cache", default="data/seg_cache.npz")
    ap.add_argument("--out", default="results/segmentation")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--train-on", nargs="+", default=SOURCES)
    ap.add_argument("--test-on", nargs="+", default=["ideal", "stereo"])
    a = ap.parse_args()
    prepare(a) if a.cmd == "prepare" else train(a)


if __name__ == "__main__":
    main()
