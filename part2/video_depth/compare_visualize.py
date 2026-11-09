"""Part 1b + 2b: side-by-side video and temporal diagnostics, DA V2-S (frame-wise) vs VDA-S.

Rules implemented (project PDF, Sec. 5.2 and 6.2):
  * same frames, same colormap for both methods;
  * ONE fixed normalization per method over the whole clip (never per frame);
  * outputs are labelled "relative depth"; colors are not a shared metric scale.
The normalization parameters are saved to norm_params.json so a Part 3 improvement can
reuse the unchanged method's normalization (--norm_from).

Outputs:
  results/videos/<name>.mp4             RGB | DA V2-S | VDA-S
  results/figures/<name>/frames.png     selected frames (+ zoomed crop if --crop is given)
  results/figures/<name>/xt_slice.png   one image row stacked over time (flicker shows as horizontal bands)
  results/figures/<name>/pixel_curves.png  displayed value of a few pixels over time
  results/figures/<name>/temporal_change.png + metrics.json  optional rough indicator

Example (from the repo root):
    python part2/video_depth/compare_visualize.py --frames_idx 10 31 32 70 --row 240
"""
import argparse
import json
import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

LABEL = "relative depth"


def load(path):
    z = np.load(path)
    key = "depth" if "depth" in z.files else z.files[0]
    return z[key].astype(np.float32)


def resize_stack(depth, h, w):
    if depth.shape[1:] == (h, w):
        return depth
    return np.stack([cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR) for d in depth])


def normalize(depth, lo, hi):
    return np.clip((depth - lo) / (hi - lo + 1e-12), 0.0, 1.0)


def colorize(v01, cmap):
    """v01: [N,H,W] in [0,1] -> [N,H,W,3] uint8 RGB (bright = larger value = nearer)."""
    lut = (plt.get_cmap(cmap)(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    return lut[(v01 * 255).round().astype(np.uint8)]


def header(img, text):
    bar = np.full((28, img.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return np.concatenate([bar, img], 0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames", default="results/raw/davis_frames/frames.npz")
    p.add_argument("--meta", default="results/raw/davis_frames/meta.json")
    p.add_argument("--da2", default="results/raw/part1b_da2/depth.npz")
    p.add_argument("--vda", default="results/raw/part2b_vda/depth.npz")
    p.add_argument("--name", default="part1b_2b_davis")
    p.add_argument("--cmap", default="inferno")
    p.add_argument("--pct", type=float, nargs=2, default=[2.0, 98.0], help="clip-level percentiles per method")
    p.add_argument("--norm_from", default=None, help="reuse norm_params.json from an earlier (unchanged-method) run")
    p.add_argument("--frames_idx", type=int, nargs="+", default=[10, 31, 32, 70])
    p.add_argument("--crop", type=int, nargs=4, default=None, metavar=("X", "Y", "W", "H"), help="zoom box")
    p.add_argument("--row", type=int, default=None, help="image row for the x-t slice (default: middle)")
    p.add_argument("--pixels", type=int, nargs="+", default=None, help="x1 y1 x2 y2 ... for time curves")
    args = p.parse_args()

    frames = np.load(args.frames)["frames"]
    n, h, w, _ = frames.shape
    fps = json.load(open(args.meta))["fps"]
    methods = {
        "DA V2-S (frame-wise)": resize_stack(load(args.da2), h, w),
        "VDA-S (video)": resize_stack(load(args.vda), h, w),
    }
    for k, d in methods.items():
        assert d.shape[0] == n, f"{k}: {d.shape[0]} frames vs {n}"

    # one fixed normalization per method over the whole clip
    if args.norm_from:
        norm = json.load(open(args.norm_from))
    else:
        norm = {k: [float(x) for x in np.percentile(d, args.pct)] for k, d in methods.items()}
    vis01 = {k: normalize(d, *norm[k]) for k, d in methods.items()}
    color = {k: colorize(v, args.cmap) for k, v in vis01.items()}

    fig_dir = os.path.join("results/figures", args.name)
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs("results/videos", exist_ok=True)
    with open(os.path.join(fig_dir, "norm_params.json"), "w") as f:
        json.dump({**norm, "_percentiles": args.pct, "_note": "clip-level, per method"}, f, indent=2)

    # 1) side-by-side video
    video_path = os.path.join("results/videos", f"{args.name}.mp4")
    names = list(methods)
    first = np.concatenate(
        [header(frames[0], "RGB")] + [header(color[k][0], f"{k} | {LABEL}") for k in names], 1
    )
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (first.shape[1], first.shape[0]))
    for i in range(n):
        tiles = [header(frames[i], f"RGB  frame {i}")] + [header(color[k][i], f"{k} | {LABEL}") for k in names]
        writer.write(cv2.cvtColor(np.concatenate(tiles, 1), cv2.COLOR_RGB2BGR))
    writer.release()

    # 2) selected frames (+ crop)
    idx = [i for i in args.frames_idx if 0 <= i < n]
    rows = len(idx) * (2 if args.crop else 1)
    fig, axes = plt.subplots(rows, 3, figsize=(12, 2.6 * rows), squeeze=False)
    r = 0
    for i in idx:
        panels = [("RGB", frames[i])] + [(k, color[k][i]) for k in names]
        for c, (title, img) in enumerate(panels):
            axes[r, c].imshow(img)
            axes[r, c].set_title(f"{title}  #{i}", fontsize=9)
            axes[r, c].axis("off")
            if args.crop:
                x, y, cw, ch = args.crop
                axes[r, c].add_patch(plt.Rectangle((x, y), cw, ch, fill=False, ec="cyan", lw=1.2))
                axes[r + 1, c].imshow(img[y : y + ch, x : x + cw])
                axes[r + 1, c].set_title("zoom", fontsize=9)
                axes[r + 1, c].axis("off")
        r += 2 if args.crop else 1
    fig.suptitle(f"Same frames, fixed clip-level normalization per method ({LABEL})", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "frames.png"), dpi=150)
    plt.close(fig)

    # 3) x-t slice: one row over time
    row = args.row if args.row is not None else h // 2
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].imshow(frames[:, row, :, :], aspect="auto")
    axes[0].set_title(f"RGB, row y={row}")
    for ax, k in zip(axes[1:], names):
        ax.imshow(color[k][:, row, :, :], aspect="auto")
        ax.set_title(f"{k}, row y={row}")
    for ax in axes:
        ax.set_xlabel("x (pixel)")
        ax.set_ylabel("frame")
    fig.suptitle("x-t slice: flicker appears as horizontal bands", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "xt_slice.png"), dpi=150)
    plt.close(fig)

    # 4) pixel curves
    if args.pixels:
        pts = list(zip(args.pixels[0::2], args.pixels[1::2]))
    else:
        pts = [(int(0.2 * w), int(0.2 * h)), (int(0.8 * w), int(0.2 * h)), (int(0.5 * w), int(0.1 * h)), (int(0.2 * w), int(0.85 * h))]
    fig, axes = plt.subplots(1, len(pts), figsize=(4 * len(pts), 3), squeeze=False)
    for ax, (x, y) in zip(axes[0], pts):
        for k, ls in zip(names, ["-", "--"]):
            ax.plot(vis01[k][:, y, x], ls, label=k)
        ax.set_title(f"pixel (x={x}, y={y})", fontsize=9)
        ax.set_xlabel("frame")
        ax.set_ylabel("normalized value")
        ax.set_ylim(0, 1)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "pixel_curves.png"), dpi=150)
    plt.close(fig)

    # 5) optional rough indicator: mean |v(t+1) - v(t)| of the normalized maps
    metrics = {}
    fig, ax = plt.subplots(figsize=(8, 3))
    for k in names:
        diff = np.abs(np.diff(vis01[k], axis=0)).mean(axis=(1, 2))
        metrics[k] = {"mean_abs_frame_change": float(diff.mean())}
        ax.plot(np.arange(1, n), diff, label=k)
    ax.set_xlabel("frame t")
    ax.set_ylabel("mean |v(t) - v(t-1)|")
    ax.set_title("Rough indicator only: includes real camera/object motion", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "temporal_change.png"), dpi=150)
    plt.close(fig)
    metrics["_note"] = "optional, not required by the project; includes true motion, use only as a rough comparison"
    with open(os.path.join(fig_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"video   -> {video_path}")
    print(f"figures -> {fig_dir}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
