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

VDA blend zones (shaded in the plots) are read from the upstream stitching code
(video_depth_anything/video_depth.py, INFER_LEN=32, OVERLAP=10, INTERP_LEN=8): for 70 frames the
windows start at input frames 0/22/44/66 and the linear cross-fade covers frames 24-31, 46-53, 68-69.

Outputs (in --out; suffix _approx for --source video):
  flicker_metrics.json     summary numbers quoted in the report
  flicker_curves.json      per-pair / per-frame curves behind the plots
  flicker_analysis.png     error, scale-jump and median curves
  flicker_qualitative.png  residual maps at one pair + x-t zoom through the train

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


def pair_flow(g0, g1, xx, yy):
    """Farneback flow g0->g1 and the backward check. Returns sampling maps (mx, my), fb error, inside mask."""
    h, w = g0.shape
    fw = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 5, 15, 5, 7, 1.5, 0)
    bw = cv2.calcOpticalFlowFarneback(g1, g0, None, 0.5, 5, 15, 5, 7, 1.5, 0)
    mx, my = xx + fw[..., 0], yy + fw[..., 1]
    bx = cv2.remap(bw[..., 0], mx, my, cv2.INTER_LINEAR)
    by = cv2.remap(bw[..., 1], mx, my, cv2.INTER_LINEAR)
    fb = np.hypot(fw[..., 0] + bx, fw[..., 1] + by)
    inside = (mx >= 0) & (mx < w - 1) & (my >= 0) & (my < h - 1)
    return mx, my, fb, inside


def blend_zones(spec):
    return [tuple(int(v) for v in z.split("-")) for z in spec]


def qualitative(rgb, V, valid_t, maps_t, t, args, path, tag):
    """Row 1: RGB and |v_t - warp(v_t+1)| at pair t. Row 2: x-t slice through the train."""
    mx, my = maps_t
    y = args.xt_y
    (x0, x1), (t0, t1) = args.xt_x, args.xt_t
    t1 = min(t1, rgb.shape[0] - 1)
    fig, ax = plt.subplots(2, 3, figsize=(18, 7.6))
    ax[0, 0].imshow(rgb[t])
    ax[0, 0].set_title(f"RGB #{t}")
    vmax = 0.15
    for j, k in enumerate(NAMES):
        r = np.abs(V[k][t] - cv2.remap(V[k][t + 1], mx, my, cv2.INTER_LINEAR))
        r[~valid_t] = 0
        im = ax[0, j + 1].imshow(r, cmap="magma", vmin=0, vmax=vmax)
        ax[0, j + 1].set_title(f"{k}: |v_t - warp(v_t+1)|, t={t} (valid px only)")
    fig.colorbar(im, ax=ax[0, 2], fraction=0.03)
    ext = [x0, x1, t1 + 0.5, t0 - 0.5]
    ax[1, 0].imshow(rgb[t0:t1 + 1, y, x0:x1], aspect="auto", extent=ext, interpolation="nearest")
    ax[1, 0].set_title(f"RGB x-t, y={y}, frames {t0}-{t1}")
    for j, k in enumerate(NAMES):
        ax[1, j + 1].imshow(V[k][t0:t1 + 1, y, x0:x1], cmap=args.cmap, vmin=0.4, vmax=1.0, aspect="auto",
                            extent=ext, interpolation="nearest")
        ax[1, j + 1].set_title(f"{k} x-t zoom (contrast 0.4-1.0, same for both)")
    for a in ax[0]:
        a.axis("off")
    for a in ax[1]:
        a.set_xlabel("x")
        a.set_ylabel("frame")
    fig.suptitle("Relative depth, fixed clip-level normalization per method" + (" [approx. from mp4]" if tag else ""))
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


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
    p.add_argument("--blend", nargs="*", default=["24-31", "46-53", "68-69"],
                   help="VDA cross-fade zones (first-last frame) to shade; see module docstring")
    p.add_argument("--qual_t", type=int, default=-1,
                   help="pair t for the residual maps; -1 = pair with the largest DA V2 scale jump")
    p.add_argument("--xt_y", type=int, default=180, help="row of the x-t slice")
    p.add_argument("--xt_x", type=int, nargs=2, default=[270, 420], help="column range of the x-t zoom")
    p.add_argument("--xt_t", type=int, nargs=2, default=[30, 69], help="frame range of the x-t zoom")
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
    npx, valids, maps = [], [], []
    for t in range(n - 1):
        mx, my, fb, inside = pair_flow(gray[t], gray[t + 1], xx, yy)
        fg1 = cv2.remap(fgs[t + 1].astype(np.float32), mx, my, cv2.INTER_NEAREST) > 0.5
        valid = fgs[t] & fg1 & inside & (fb < args.fb)
        npx.append(int(valid.sum()))
        valids.append(valid)
        maps.append((mx, my))
        for k in NAMES:
            a = V[k][t][valid]
            b = cv2.remap(V[k][t + 1], mx, my, cv2.INTER_LINEAR)[valid]
            err[k].append(float(np.mean(np.abs(a - b))) if a.size else np.nan)
            logs[k].append(float(np.log(np.median(b / np.maximum(a, 1e-3)))) if a.size else np.nan)
    med = {k: np.array([np.median(V[k][i][fgs[i]]) if fgs[i].any() else np.nan for i in range(n)]) for k in NAMES}

    zones = blend_zones(args.blend)
    in_blend = np.array([any(a <= t + 1 <= b for a, b in zones) for t in range(n - 1)])  # pair t -> frame t+1
    metrics = {}
    for k in NAMES:
        e, s = np.array(err[k]), np.array(logs[k])
        top = np.argsort(-np.abs(s))[:3]
        metrics[k] = {
            "flow_warped_abs_err_mean": float(np.nanmean(e)),
            "scale_jump_mean_abs_log": float(np.nanmean(np.abs(s))),
            "scale_jump_max_abs_log": float(np.nanmax(np.abs(s))),
            "scale_jump_max_at_pair": int(np.nanargmax(np.abs(s))),
            "median_curve_mean_abs_2nd_diff": float(np.nanmean(np.abs(np.diff(med[k], 2)))),
            "scale_jump_mean_pct": float(100 * np.nanmean(np.expm1(np.abs(s)))),
            "scale_jump_max_pct": float(100 * np.expm1(np.nanmax(np.abs(s)))),
            "scale_jump_top3_pairs": [int(i) for i in top],
            "scale_jump_top3_log": [float(s[i]) for i in top],
            "scale_jump_sign_changes": int(np.sum(np.diff(np.sign(s[np.abs(s) > 1e-6])) != 0)),
            "pairs_scale_jump_gt_2pct": int(np.sum(np.abs(s) > np.log(1.02))),
            "flow_err_blend_pairs_mean": float(np.nanmean(e[in_blend])),
            "flow_err_other_pairs_mean": float(np.nanmean(e[~in_blend])),
            "median_curve_min": [float(np.nanmin(med[k])), int(np.nanargmin(med[k]))],
            "median_curve_max": [float(np.nanmax(med[k])), int(np.nanargmax(med[k]))],
            "median_curve_range": float(np.nanmax(med[k]) - np.nanmin(med[k])),
        }
    e0, e1 = np.array(err[NAMES[0]]), np.array(err[NAMES[1]])
    metrics["_pairs_where_VDA_lower_err"] = float(np.mean(e1 < e0))
    metrics["_pairs_where_VDA_lower_err_count"] = [int(np.sum(e1 < e0)), int(n - 1)]
    metrics["_flow_err_ratio_DA2_over_VDA"] = float(np.nanmean(e0) / np.nanmean(e1))
    metrics["_num_frames"] = int(n)
    metrics["_pair_index"] = "pair t compares frame t with frame t+1"
    metrics["_blend_zones"] = args.blend
    metrics["_valid_px_per_pair_median"] = int(np.median(npx))
    metrics["_source"] = args.source
    metrics["_note"] = ("flow = Farneback on RGB, fb<%.1fpx, fg>%.2f, 3x3 erosion; values in clip-normalized "
                        "units; consistency only, no GT." % (args.fb, args.fg)
                        + (" Approximate: inverted from the 8-bit compressed mp4." if args.source == "video" else ""))

    os.makedirs(args.out, exist_ok=True)
    tag = "" if args.source == "raw" else "_approx"
    with open(os.path.join(args.out, f"flicker_metrics{tag}.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    curves = {"valid_px": npx, **{k: {"flow_err": err[k], "log_scale": logs[k], "median_fg": med[k].tolist()}
                                  for k in NAMES}}
    with open(os.path.join(args.out, f"flicker_curves{tag}.json"), "w") as f:
        json.dump(curves, f)

    qt = metrics[NAMES[0]]["scale_jump_max_at_pair"] if args.qual_t < 0 else args.qual_t
    qualitative(rgb, V, valids[qt], maps[qt], qt, args, os.path.join(args.out, f"flicker_qualitative{tag}.png"), tag)

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
        for z0, z1 in zones:
            a.axvspan(z0 - 0.5, min(z1 + 0.5, n - 1), color="gray", alpha=0.12)
    ax[0].legend(fontsize=8)
    ax[0].set_title("Flow-compensated temporal consistency (relative depth, clip-normalized). "
                    "Gray: VDA cross-fade zones" + (" [approx. from mp4]" if tag else ""), fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, f"flicker_analysis{tag}.png"), dpi=130)
    plt.close(fig)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
