"""A-D0: at which spatial frequency does the frame-to-frame scale change live?

For every adjacent pair, r(x) = log(warp(d_t+1)(x) / d_t(x)) on reliable foreground pixels.
r is summarised on a gh x gw grid (median per cell), upsampled bilinearly and removed from
warp(d_t+1). The share of the flow-compensated error that disappears tells how much of the flicker
a correction field of that resolution could remove at best. It is an oracle (it looks at the next
frame), so it is an upper bound, not a method.

    python part3/improve_a/diagnose_frequency.py
"""
import argparse
import json
import os
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as C  # noqa: E402

GRIDS = [(1, 1), (2, 4), (3, 5), (6, 10), (12, 20), (24, 40), (45, 80)]


def cell_field(r, valid, gh, gw, min_px=30):
    h, w = r.shape
    iy = np.minimum((np.arange(h) * gh) // h, gh - 1)
    ix = np.minimum((np.arange(w) * gw) // w, gw - 1)
    cell = iy[:, None] * gw + ix[None, :]
    g = np.full(gh * gw, np.nan, np.float32)
    cv, rv = cell[valid], r[valid]
    order = np.argsort(cv, kind="stable")
    cv, rv = cv[order], rv[order]
    ids, starts, counts = np.unique(cv, return_index=True, return_counts=True)
    for i, s, c in zip(ids, starts, counts):
        if c >= min_px:
            g[i] = np.median(rv[s:s + c])
    g[np.isnan(g)] = np.median(rv)
    g = g.reshape(gh, gw)
    return cv2.resize(g, (w, h), interpolation=cv2.INTER_LINEAR) if (gh, gw) != (1, 1) else np.full((h, w), g[0, 0])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="results/figures/part3/improve_a")
    args = p.parse_args()

    rgb = C.load_frames()
    fw, bw = C.farneback_cached(rgb)
    xx, yy, MX, MY, OK = C.geometry(fw, bw)
    methods = {"DA V2-S": (C.load_depth(C.DA2_RAW), C.norm_range(C.DA2_KEY)),
               "VDA-S": (C.load_depth(C.VDA_RAW), C.norm_range(C.VDA_KEY))}
    U = {k: (d - lo) / (hi - lo) for k, (d, (lo, hi)) in methods.items()}
    common_fg = np.logical_and.reduce([C.fg_masks(u) for u in U.values()])
    n = len(rgb)

    res = {k: {f"{gh}x{gw}": [] for gh, gw in GRIDS} for k in U}
    raw = {k: [] for k in U}
    for t in range(n - 1):
        mx, my = MX[t], MY[t]
        valid = common_fg[t] & (C.remap(common_fg[t + 1].astype(np.float32), mx, my, True) > 0.5) & OK[t]
        for k, u in U.items():
            a = u[t]
            b = C.remap(u[t + 1], mx, my)
            raw[k].append(float(np.mean(np.abs(a - b)[valid])))
            r = np.log(np.maximum(b, 1e-4) / np.maximum(a, 1e-4))
            for gh, gw in GRIDS:
                R = cell_field(r, valid, gh, gw)
                res[k][f"{gh}x{gw}"].append(float(np.mean(np.abs(a - b * np.exp(-R))[valid])))

    S = {}
    for k in U:
        e0 = float(np.mean(raw[k]))
        S[k] = {"raw_err": e0,
                "after": {g: float(np.mean(v)) for g, v in res[k].items()},
                "removed_pct": {g: float(100 * (1 - np.mean(v) / e0)) for g, v in res[k].items()}}
    os.makedirs(args.out, exist_ok=True)
    json.dump({"summary": S, "grids": [f"{a}x{b}" for a, b in GRIDS],
               "note": "oracle upper bound: correction field estimated from the same pair"},
              open(os.path.join(args.out, "a_d0_frequency.json"), "w"), indent=2)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    cells = [a * b for a, b in GRIDS]
    for i, k in enumerate(U):
        ax[0].plot(cells, [S[k]["removed_pct"][f"{a}x{b}"] for a, b in GRIDS], "o-", color=f"C{i}", label=k)
        ax[1].plot(cells, [S[k]["after"][f"{a}x{b}"] for a, b in GRIDS], "o-", color=f"C{i}", label=k)
        ax[1].axhline(S[k]["raw_err"], color=f"C{i}", ls=":", lw=1)
    for a in ax:
        a.set_xscale("log")
        a.set_xticks(cells)
        a.set_xticklabels([f"{a_}x{b_}" for a_, b_ in GRIDS], rotation=30)
        a.set_xlabel("grid of the scale-correction field (rows x cols)")
        a.legend(fontsize=8)
    ax[0].set_ylabel("flow-comp. error removed (%)")
    ax[1].set_ylabel("flow-comp. error after correction (dotted: raw)")
    fig.suptitle("A-D0: oracle scale correction at different spatial resolutions (upper bound, uses the next frame)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "a_d0_frequency.png"), dpi=120)
    print(json.dumps(S, indent=2))


if __name__ == "__main__":
    main()
