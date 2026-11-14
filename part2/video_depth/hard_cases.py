"""D5: three representative difficult cases from the required clip (DA V2-S vs VDA-S).

Each row is one case at frame t: RGB crop, both depth crops (fixed clip-level normalization,
same colormap as the other figures), and both flow-compensated residuals |v_t - warp(v_t+1)|
(same flow, same valid-pixel mask as temporal_flicker.py). Crops are upsampled with nearest
neighbour so pixels stay visible. Per-case numbers go to hard_cases.json.

    python part2/video_depth/hard_cases.py
    python part2/video_depth/hard_cases.py --contact 60 69   # helper: contact sheet to pick crops
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from temporal_flicker import NAMES, pair_flow  # noqa: E402

# name, frame t, crop (x0, y0, x1, y1) in 640x360 frame coordinates, one-line reason
CASES = [
    ("Thin truss and rail ties", 31, (260, 0, 460, 200),
     "Members a few pixels wide: depth edges bleed into the sky and flow is unreliable, "
     "so many of these pixels fail the consistency check and are not evaluated."),
    ("Train and riders", 10, (110, 70, 310, 170),
     "Small, articulated objects are predicted as one smooth blob; individual riders are not separated."),
    ("Far content entering", 67, (430, 230, 640, 360),
     "Distant trees and a tower enter at the border late in the clip: both models put them barely "
     "above the sky (near 0); the flag pole is only a faint line, and the far end of the range is compressed."),
]


def load(args):
    frames = np.load(args.frames)["frames"]
    norm = json.load(open(args.norm))
    V = {}
    for k, path in zip(NAMES, [args.da2, args.vda]):
        z = np.load(path)
        d = z["depth" if "depth" in z.files else z.files[0]].astype(np.float32)
        lo, hi = norm[k]
        V[k] = np.clip((d - lo) / (hi - lo + 1e-12), 0, 1)
    return frames, V


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames", default="results/raw/davis_frames/frames.npz")
    p.add_argument("--da2", default="results/raw/part1b_da2/depth.npz")
    p.add_argument("--vda", default="results/raw/part2b_vda/depth.npz")
    p.add_argument("--norm", default="results/figures/part1b_2b_davis/norm_params.json")
    p.add_argument("--cmap", default="inferno")
    p.add_argument("--fg", type=float, default=0.05)
    p.add_argument("--fb", type=float, default=2.0)
    p.add_argument("--contact", type=int, nargs=2, default=None, help="only write a contact sheet of frames a..b")
    p.add_argument("--out", default="results/figures/part1b_2b_davis")
    args = p.parse_args()

    rgb, V = load(args)
    n, h, w, _ = rgb.shape
    os.makedirs(args.out, exist_ok=True)

    if args.contact:
        a, b = args.contact
        idx = list(range(a, b + 1))
        fig, ax = plt.subplots(len(idx), 3, figsize=(15, 2.9 * len(idx)))
        for r, t in enumerate(idx):
            ax[r, 0].imshow(rgb[t])
            ax[r, 0].set_title(f"#{t}")
            for j, k in enumerate(NAMES):
                ax[r, j + 1].imshow(V[k][t], cmap=args.cmap, vmin=0, vmax=1)
            for c in ax[r]:
                c.set_xticks(range(0, w, 80))
                c.set_yticks(range(0, h, 60))
                c.grid(alpha=0.4)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f"contact_{a}_{b}.png"), dpi=70)
        return

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    kernel = np.ones((3, 3), np.uint8)
    short = ["DA V2-S", "VDA-S"]
    fig, ax = plt.subplots(len(CASES), 5, figsize=(22, 3.9 * len(CASES)), layout="constrained")
    stats = {}
    for r, (name, t, (x0, y0, x1, y1), why) in enumerate(CASES):
        g0 = cv2.cvtColor(rgb[t], cv2.COLOR_RGB2GRAY)
        g1 = cv2.cvtColor(rgb[t + 1], cv2.COLOR_RGB2GRAY)
        mx, my, fb, inside = pair_flow(g0, g1, xx, yy)

        def fgmask(i):
            m = np.ones((h, w), bool)
            for k in NAMES:
                m &= V[k][i] > args.fg
            return cv2.erode(m.astype(np.uint8), kernel).astype(bool)

        fg1 = cv2.remap(fgmask(t + 1).astype(np.float32), mx, my, cv2.INTER_NEAREST) > 0.5
        valid = fgmask(t) & fg1 & inside & (fb < args.fb)
        fg_any = np.zeros((h, w), bool)
        for k in NAMES:
            fg_any |= V[k][t] > args.fg
        sl = (slice(y0, y1), slice(x0, x1))
        s = {"frame": t, "crop_xyxy": [x0, y0, x1, y1],
             "fg_px_in_crop": int(fg_any[sl].sum()),
             "valid_px_in_crop": int(valid[sl].sum()),
             "valid_fraction_of_fg": float(valid[sl].sum() / max(fg_any[sl].sum(), 1))}
        ax[r, 0].imshow(rgb[t][sl], interpolation="nearest")
        for j, k in enumerate(NAMES):
            ax[r, j + 1].imshow(V[k][t][sl], cmap=args.cmap, vmin=0, vmax=1, interpolation="nearest")
            res = np.abs(V[k][t] - cv2.remap(V[k][t + 1], mx, my, cv2.INTER_LINEAR))
            res[~valid] = np.nan
            im = ax[r, j + 3].imshow(res[sl], cmap="magma", vmin=0, vmax=0.15, interpolation="nearest")
            ax[r, j + 3].set_facecolor("#9a9a9a")
            s[k] = {"median_value_fg_crop": float(np.median(V[k][t][sl][fg_any[sl]])) if fg_any[sl].any() else None,
                    "residual_mean_valid_crop": float(np.nanmean(res[sl])) if valid[sl].any() else None}
        stats[name] = s
        ax[r, 0].set_ylabel(f"({r + 1}) {name}\nframe {t}", fontsize=11)
        ax[r, 0].text(0, -0.03, why, transform=ax[r, 0].transAxes, fontsize=9, va="top").set_in_layout(False)
        titles = [f"RGB #{t}"] + [f"{k} #{t}" for k in NAMES] + [f"{sn} residual {t}->{t + 1}" for sn in short]
        for j, c in enumerate(ax[r]):
            c.set_xticks([])
            c.set_yticks([])
            c.set_title(titles[j], fontsize=10)
    fig.get_layout_engine().set(h_pad=0.25)
    fig.colorbar(im, ax=ax[:, 4], shrink=0.6, label="|v_t - warp(v_t+1)|  (gray = not evaluated)")
    fig.suptitle("Difficult cases in davis_rollercoaster (relative depth, fixed clip-level normalization per method)")
    fig.savefig(os.path.join(args.out, "hard_cases.png"), dpi=110, bbox_inches="tight", pad_inches=0.4)
    plt.close(fig)
    with open(os.path.join(args.out, "hard_cases.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
