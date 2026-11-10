"""Part 1b/2b: flicker-specific temporal metrics, DA V2-S (frame-wise) vs VDA-S.

Why: mean |v(t+1) - v(t)| (compare_visualize.py) is dominated by real camera motion of thin
structures over a sky that both models set to 0, so it cannot separate flicker from motion.
This script removes motion first with optical flow, then measures what is left.

Per consecutive pair (t, t+1), on pixels that are foreground in both methods, eroded,
inside the image and pass a forward-backward flow check (Farnebäck on the RGB frames):
  * warped error   mean |v_t - W(v_{t+1})|              (lower = more stable)
  * scale jump     median W(v_{t+1}) / v_t, as log ratio (0 = no global rescale)
  * median curve   per-frame median foreground value; flicker = jagged, motion = smooth
Values are the clip-level normalized maps from norm_params.json, so both methods live in [0, 1].
No ground truth: these quantify consistency, not accuracy.

Inputs (from the repo root):
  raw (preferred, for the report):
    python part2/video_depth/temporal_flicker.py --source raw
  side-by-side video fallback (8-bit, compressed; colormap inverted -> approximate):
    python part2/video_depth/temporal_flicker.py --source video
"""
import argparse
import json
import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

NAMES = ["DA V2-S (frame-wise)", "VDA-S (video)"]


def load_raw(args):
    frames = np.load(args.frames)["frames"]
    n, h, w, _ = frames.shape
    norm = json.load(open(args.norm))
    out = {}
    for name, path in zip(NAMES, [args.da2, args.vda]):
        z = np.load(path)
        d = z["depth" if "depth" in z.files else z.files[0]].astype(np.float32)
        if d.shape[1:] != (h, w):
            d = np.stack([cv2.resize(x, (w, h), interpolation=cv2.INTER_LINEAR) for x in d])
        lo, hi = norm[name]
        out[name] = np.clip((d - lo) / (hi - lo + 1e-12), 0, 1)
    reliable = {k: np.ones_like(v, bool) for k, v in out.items()}
    return frames, out, reliable


def load_video(args):
    from scipy.spatial import cKDTree

    cap = cv2.VideoCapture(args.video)
    F = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        F.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    F = np.stack(F)[:, args.header:]
    tw = F.shape[2] // 3
    lut = (plt.get_cmap(args.cmap)(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.float32)
    tree = cKDTree(lut)
    out, reliable = {}, {}
    for j, name in enumerate(NAMES):
        tile = F[:, :, (j + 1) * tw:(j + 2) * tw]
        dist, idx = tree.query(tile.reshape(-1, 3).astype(np.float32))
        out[name] = (idx / 255.0).reshape(tile.shape[:3]).astype(np.float32)
        reliable[name] = dist.reshape(tile.shape[:3]) < 10  # drop compression-blended edge colors
    return F[:, :, :tw], out, reliable


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["raw", "video"], default="raw")
    p.add_argument("--frames", default="results/raw/davis_frames/frames.npz")
    p.add_argument("--da2", default="results/raw/part1b_da2/depth.npz")
    p.add_argument("--vda", default="results/raw/part2b_vda/depth.npz")
    p.add_argument("--norm", default="results/figures/part1b_2b_davis/norm_params.json")
    p.add_argument("--video", default="results/videos/part1b_2b_davis.mp4")
    p.add_argument("--header", type=int, default=28, help="title bar height in the side-by-side video")
    p.add_argument("--cmap", default="inferno")
    p.add_argument("--fg", type=float, default=0.05, help="foreground threshold on normalized value (sky ~ 0)")
    p.add_argument("--fb", type=float, default=2.0, help="forward-backward flow consistency threshold (px)")
    p.add_argument("--seams", type=int, nargs="*", default=[22, 44, 66], help="VDA window starts to shade")
    p.add_argument("--out", default="results/figures/part1b_2b_davis")
    args = p.parse_args()

    rgb, V, rel = (load_raw if args.source == "raw" else load_video)(args)
    n, h, w = V[NAMES[0]].shape
    gray = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in rgb]
    kernel = np.ones((3, 3), np.uint8)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    def fg(i):
        m = np.ones((h, w), bool)
        for k in NAMES:
            m &= (V[k][i] > args.fg) & rel[k][i]
        return cv2.erode(m.astype(np.uint8), kernel).astype(bool)

    fgs = [fg(i) for i in range(n)]
    err = {k: [] for k in NAMES}
    logs = {k: [] for k in NAMES}
    npx = []
    for t in range(n - 1):
        fw = cv2.calcOpticalFlowFarneback(gray[t], gray[t + 1], None, 0.5, 5, 15, 5, 7, 1.5, 0)
        bw = cv2.calcOpticalFlowFarneback(gray[t + 1], gray[t], None, 0.5, 5, 15, 5, 7, 1.5, 0)
        mx, my = xx + fw[..., 0], yy + fw[..., 1]
        bx = cv2.remap(bw[..., 0], mx, my, cv2.INTER_LINEAR)
        by = cv2.remap(bw[..., 1], mx, my, cv2.INTER_LINEAR)
        fb = np.hypot(fw[..., 0] + bx, fw[..., 1] + by)
        inside = (mx >= 0) & (mx < w - 1) & (my >= 0) & (my < h - 1)
        fg1 = cv2.remap(fgs[t + 1].astype(np.float32), mx, my, cv2.INTER_NEAREST) > 0.5
        valid = fgs[t] & fg1 & inside & (fb < args.fb)
        npx.append(int(valid.sum()))
        for k in NAMES:
            a = V[k][t][valid]
            b = cv2.remap(V[k][t + 1], mx, my, cv2.INTER_LINEAR)[valid]
            err[k].append(float(np.mean(np.abs(a - b))) if a.size else np.nan)
            logs[k].append(float(np.log(np.median(b / np.maximum(a, 1e-3)))) if a.size else np.nan)
    med = {k: np.array([np.median(V[k][i][fgs[i]]) if fgs[i].any() else np.nan for i in range(n)]) for k in NAMES}

    metrics = {}
    for k in NAMES:
        e, s = np.array(err[k]), np.array(logs[k])
        metrics[k] = {
            "flow_warped_abs_err_mean": float(np.nanmean(e)),
            "scale_jump_mean_abs_log": float(np.nanmean(np.abs(s))),
            "scale_jump_max_abs_log": float(np.nanmax(np.abs(s))),
            "scale_jump_max_at_pair": int(np.nanargmax(np.abs(s))),
            "median_curve_mean_abs_2nd_diff": float(np.nanmean(np.abs(np.diff(med[k], 2)))),
        }
    e0, e1 = np.array(err[NAMES[0]]), np.array(err[NAMES[1]])
    metrics["_pairs_where_VDA_lower_err"] = float(np.mean(e1 < e0))
    metrics["_valid_px_per_pair_median"] = int(np.median(npx))
    metrics["_source"] = args.source
    metrics["_note"] = ("flow = Farneback on RGB, fb<%.1fpx, fg>%.2f, 3x3 erosion; values in clip-normalized "
                        "units; consistency only, no GT. video source = approximate (8-bit, compressed)."
                        % (args.fb, args.fg))

    os.makedirs(args.out, exist_ok=True)
    tag = "" if args.source == "raw" else "_approx"
    with open(os.path.join(args.out, f"flicker_metrics{tag}.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    tt = np.arange(1, n)
    for k, c in zip(NAMES, ["C0", "C1"]):
        ax[0].plot(tt, err[k], c, label=k)
        ax[1].plot(tt, logs[k], c, label=k)
        ax[2].plot(np.arange(n), med[k], c, label=k)
    ax[0].set_ylabel("|v_t - warp(v_t+1)|")
    ax[1].set_ylabel("log scale ratio")
    ax[1].axhline(0, color="k", lw=0.5)
    ax[2].set_ylabel("median fg value")
    ax[2].set_xlabel("frame")
    for a in ax:
        for s in args.seams:
            a.axvspan(s, min(s + 9.5, n - 1), color="gray", alpha=0.12)
    ax[0].legend(fontsize=8)
    ax[0].set_title("Flow-compensated temporal consistency (relative depth, clip-normalized). "
                    "Gray: VDA window overlap zones" + (" [approx. from mp4]" if tag else ""), fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, f"flicker_analysis{tag}.png"), dpi=130)
    plt.close(fig)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
