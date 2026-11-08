"""Distance between simulated and real per-angle curves (holes and wrong pixels on metal), as the mean absolute
difference over the angle bins. Anodized aluminium is compared with ROBI Zigzag, polished with ROBI Chrome.

Used for two checks of the report: the anodized material variants, and the sensor-physics ablations
(IR noise, speckle, HDR) on a few scenes against the full dataset.

    python -m experiments.curve_distance --real results/angle_curves.csv \
        --sim r0.45:results/anodized/curves_r045.csv r0.30:results/anodized/curves_r030.csv \
        --models stereo --out results/anodized/distance_to_robi.csv
"""
import argparse
import os

import numpy as np
import pandas as pd

REAL_FOR = {"anodized": "real ROBI Zigzag", "polished": "real ROBI Chrome"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default="results/angle_curves.csv")
    ap.add_argument("--sim", nargs="+", required=True, help="label:angle_curves.csv")
    ap.add_argument("--models", nargs="*", default=None, help="only these depth models")
    ap.add_argument("--finishes", nargs="+", default=["anodized", "polished"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    R = pd.read_csv(a.real)
    R = R[R.region == "metal"]
    rows = []
    for spec in a.sim:
        label, path = spec.split(":", 1)
        S = pd.read_csv(path)
        S = S[S.source.str.startswith("sim ")]
        for (src, region), g in S.groupby(["source", "region"]):
            model, fin = src[4:], region.replace("metal ", "")
            if fin not in a.finishes or (a.models and model not in a.models):
                continue
            real = R[R.source == REAL_FOR[fin]].set_index("angle")
            g = g.set_index("angle")
            common = [b for b in g.index if b in real.index]
            dh = np.abs(g.loc[common, "holes_pct"] - real.loc[common, "holes_pct"])
            dw = np.abs(g.loc[common, "wrong_pct"] - real.loc[common, "wrong_pct"])
            rows.append({"variant": label, "model": model, "finish": fin, "real": REAL_FOR[fin], "bins": len(common),
                         "mae_holes_pt": dh.mean(), "mae_wrong_pt": dw.mean(),
                         "mae_total_pt": np.concatenate([dh.values, dw.values]).mean()})
    T = pd.DataFrame(rows).sort_values(["finish", "mae_total_pt"])
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    T.to_csv(a.out, index=False)
    print(T.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
