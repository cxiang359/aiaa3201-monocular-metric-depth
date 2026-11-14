"""Part 1b/2b (stage 2): stronger temporal measurements M1-M5 for relative video depth.

One evaluation script for the baselines and any Part 3 variant, so all of them are measured the
same way. No ground truth exists for the clip: everything here measures temporal consistency, not
accuracy. All statistics use the UNCLIPPED predictions (M5). Visualization elsewhere keeps the
p2-p98 clip normalization.

Shared geometry (per optical-flow backend, Farneback or RAFT):
  adjacent forward/backward flow, forward-backward check (< --fb px), inside-image test, and a
  common foreground mask (normalized value > --fg in every method, 3x3 erosion). The same pixels
  are used for every method.

M1  long-range drift: frames t and t+k (k = 1, 5, 10, 20) linked by chaining adjacent flows; a
    pixel must pass the consistency check at every step. Reports |log scale ratio| and the
    flow-compensated error as a function of k, plus a cumulative log-scale trajectory L(t) and
    its change across each VDA cross-fade zone vs. elsewhere.
M2  scale vs. shift: robust (Huber IRLS) fit warp(d_t+1) ~ s * d_t + b on raw values per pair;
    also how much of the residual a scale-only or a scale+shift correction removes.
M3  edge stability: depth edges = neighbour ratio > --edge_ratio, only where both sides are
    foreground and away from the sky boundary (internal edges). Edges of t+1 are warped to t and
    matched with a 1 px tolerance (F1). Canny edges of the RGB frames, matched the same way, give
    the F1 a perfectly stable signal reaches under this flow.
M4  error floor: (a) unit-free: residual left after warping / residual without warping, for RGB
    and for depth; (b) same units as depth: round-trip resampling error of v_t through the
    forward and backward flow; (c) all main numbers recomputed with RAFT.
M5  saturation: the old clipped metrics next to the unclipped ones, and the share of foreground
    pixels above the clip range.

    python part2/video_depth/temporal_eval.py                  # Farneback + RAFT
    python part2/video_depth/temporal_eval.py --flows farneback
    # Part 3 variant, normalized with the unchanged method's parameters:
    python part2/video_depth/temporal_eval.py --methods \\
        "DA V2-S (frame-wise)|results/raw/part1b_da2/depth.npz" \\
        "DA V2-S + A|results/raw/p3_a/depth.npz|DA V2-S (frame-wise)"
"""
import argparse
import json
import os
import time

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DEFAULT_METHODS = [
    "DA V2-S (frame-wise)|results/raw/part1b_da2/depth.npz",
    "VDA-S (video)|results/raw/part2b_vda/depth.npz",
]
COLORS = ["C0", "C1", "C2", "C3", "C4"]


# ----------------------------------------------------------------------------- loading
def load_methods(specs, norm_path, h, w):
    norm = json.load(open(norm_path))
    out = []
    for spec in specs:
        parts = spec.split("|")
        name, path = parts[0], parts[1]
        key = parts[2] if len(parts) > 2 else name
        z = np.load(path)
        d = z["depth" if "depth" in z.files else z.files[0]].astype(np.float32)
        if d.shape[1:] != (h, w):
            d = np.stack([cv2.resize(x, (w, h), interpolation=cv2.INTER_LINEAR) for x in d])
        lo, hi = norm[key]
        out.append({"name": name, "d": d, "lo": float(lo), "hi": float(hi), "norm_key": key,
                    "u": (d - lo) / (hi - lo)})  # unclipped normalized value
    return out


# ----------------------------------------------------------------------------- flow
def flows_farneback(gray):
    fw, bw = [], []
    for t in range(len(gray) - 1):
        fw.append(cv2.calcOpticalFlowFarneback(gray[t], gray[t + 1], None, 0.5, 5, 15, 5, 7, 1.5, 0))
        bw.append(cv2.calcOpticalFlowFarneback(gray[t + 1], gray[t], None, 0.5, 5, 15, 5, 7, 1.5, 0))
    return np.stack(fw), np.stack(bw)


def flows_raft(rgb, batch=8):
    import torch
    from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

    wts = Raft_Large_Weights.DEFAULT
    model = raft_large(weights=wts).cuda().eval()
    tf = wts.transforms()
    h, w = rgb.shape[1:3]
    assert h % 8 == 0 and w % 8 == 0, "RAFT needs H, W divisible by 8"
    x = torch.from_numpy(rgb).permute(0, 3, 1, 2)

    def run(a, b):
        out = []
        with torch.no_grad():
            for i in range(0, len(a), batch):
                p, q = tf(a[i:i + batch], b[i:i + batch])
                out.append(model(p.cuda(), q.cuda())[-1].permute(0, 2, 3, 1).cpu().numpy())
        return np.concatenate(out)

    fw, bw = run(x[:-1], x[1:]), run(x[1:], x[:-1])
    del model
    torch.cuda.empty_cache()
    return fw.astype(np.float32), bw.astype(np.float32)


def remap(img, mx, my, nearest=False):
    return cv2.remap(img, mx, my, cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR)


# ----------------------------------------------------------------------------- helpers
def huber_fit(x, y, iters=10, c=1.345):
    """Robust y ~ s x + b (IRLS, Huber weights, MAD scale)."""
    wgt = np.ones_like(x)
    s, b = 1.0, 0.0
    for _ in range(iters):
        sw = np.sqrt(wgt)
        s, b = np.linalg.lstsq(np.stack([x * sw, sw], 1), y * sw, rcond=None)[0]
        r = y - (s * x + b)
        u = np.abs(r) / (c * (1.4826 * np.median(np.abs(r)) + 1e-9))
        wgt = np.where(u <= 1, 1.0, 1.0 / np.maximum(u, 1e-9))
    return float(s), float(b)


def huber_scale(x, y, iters=10, c=1.345):
    """Robust y ~ s x (no shift)."""
    wgt = np.ones_like(x)
    s = 1.0
    for _ in range(iters):
        s = float(np.sum(wgt * x * y) / (np.sum(wgt * x * x) + 1e-12))
        r = y - s * x
        u = np.abs(r) / (c * (1.4826 * np.median(np.abs(r)) + 1e-9))
        wgt = np.where(u <= 1, 1.0, 1.0 / np.maximum(u, 1e-9))
    return s


def log_ratio(a, b):
    return float(np.log(np.median(b / np.maximum(a, 1e-6))))


def depth_edges(d, fg_inner, ratio):
    """Edge where the neighbour ratio exceeds `ratio`; both pixels must be inner foreground."""
    d = np.maximum(d, 1e-6)
    e = np.zeros(d.shape, bool)
    rx = d[:, 1:] / d[:, :-1]
    ry = d[1:, :] / d[:-1, :]
    jx = ((rx > ratio) | (rx < 1 / ratio)) & fg_inner[:, 1:] & fg_inner[:, :-1]
    jy = ((ry > ratio) | (ry < 1 / ratio)) & fg_inner[1:, :] & fg_inner[:-1, :]
    e[:, 1:] |= jx
    e[1:, :] |= jy
    return e


def edge_f1(e0, e1w, region, tol_kernel):
    e0 = e0 & region
    e1w = e1w & region
    if e0.sum() == 0 or e1w.sum() == 0:
        return np.nan
    d0 = cv2.dilate(e0.astype(np.uint8), tol_kernel) > 0
    d1 = cv2.dilate(e1w.astype(np.uint8), tol_kernel) > 0
    p = (e1w & d0).sum() / e1w.sum()
    r = (e0 & d1).sum() / e0.sum()
    return float(2 * p * r / (p + r + 1e-12))


def blend_zones(spec):
    return [tuple(int(v) for v in z.split("-")) for z in spec]


# ----------------------------------------------------------------------------- evaluation
def evaluate(rgb, gray, M, fw, bw, args):
    n, h, w = M[0]["d"].shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    k3 = np.ones((3, 3), np.uint8)
    k5 = np.ones((5, 5), np.uint8)
    names = [m["name"] for m in M]

    # common foreground (unclipped normalized value > fg in every method)
    cfg_raw = np.ones((n, h, w), bool)
    for m in M:
        cfg_raw &= m["u"] > args.fg
    cfg = np.stack([cv2.erode(c.astype(np.uint8), k3).astype(bool) for c in cfg_raw])

    # adjacent geometry
    MX, MY, OK = [], [], []
    for t in range(n - 1):
        mx, my = xx + fw[t][..., 0], yy + fw[t][..., 1]
        bx, by = remap(bw[t][..., 0], mx, my), remap(bw[t][..., 1], mx, my)
        fbe = np.hypot(fw[t][..., 0] + bx, fw[t][..., 1] + by)
        inside = (mx >= 0) & (mx < w - 1) & (my >= 0) & (my < h - 1)
        MX.append(mx), MY.append(my), OK.append(inside & (fbe < args.fb))

    R = {"adjacent": {k: {} for k in names}, "valid_px": [], "rgb": {}}
    adj = {k: {key: [] for key in ["err_unclipped", "err_clipped", "log_ratio_raw", "log_ratio_clipped",
                                   "s", "b_norm", "err_after_scale", "err_after_affine", "nowarp_err",
                                   "roundtrip_floor", "edge_f1", "sat_frac"]} for k in names}
    rgb_warp, rgb_nowarp, rgb_edge = [], [], []
    for t in range(n - 1):
        mx, my = MX[t], MY[t]
        valid = cfg[t] & (remap(cfg[t + 1].astype(np.float32), mx, my, True) > 0.5) & OK[t]
        R["valid_px"].append(int(valid.sum()))
        # backward sampling maps (frame t+1 -> t) for the round-trip floor
        bmx, bmy = xx + bw[t][..., 0], yy + bw[t][..., 1]
        # RGB floor (M4a) and RGB edge reference (M3)
        g0, g1 = gray[t].astype(np.float32), gray[t + 1].astype(np.float32)
        rgb_warp.append(float(np.mean(np.abs(g0 - remap(g1, mx, my))[valid]) / 255))
        rgb_nowarp.append(float(np.mean(np.abs(g0 - g1)[valid]) / 255))
        edge_region = cfg[t] & (remap(cfg[t + 1].astype(np.float32), mx, my, True) > 0.5) & \
            (mx >= 0) & (mx < w - 1) & (my >= 0) & (my < h - 1)
        c0, c1 = cv2.Canny(gray[t], 50, 150) > 0, cv2.Canny(gray[t + 1], 50, 150) > 0
        c1w = remap(c1.astype(np.uint8), mx, my, True) > 0
        rgb_edge.append(edge_f1(c0, c1w, edge_region, k3))

        for m in M:
            k, d, u = m["name"], m["d"], m["u"]
            A = adj[k]
            a_raw, b_raw = d[t][valid], remap(d[t + 1], mx, my)[valid]
            a_u, b_u = u[t][valid], remap(u[t + 1], mx, my)[valid]
            rng = m["hi"] - m["lo"]
            A["err_unclipped"].append(float(np.mean(np.abs(a_u - b_u))))
            A["err_clipped"].append(float(np.mean(np.abs(np.clip(a_u, 0, 1) - np.clip(b_u, 0, 1)))))
            A["log_ratio_raw"].append(log_ratio(a_raw, b_raw))
            A["log_ratio_clipped"].append(log_ratio(np.clip(a_u, 1e-3, 1), np.clip(b_u, 1e-3, 1)))
            s, b = huber_fit(a_raw, b_raw)
            s0 = huber_scale(a_raw, b_raw)
            A["s"].append(s)
            A["b_norm"].append(b / rng)
            A["err_after_scale"].append(float(np.mean(np.abs(s0 * a_raw - b_raw)) / rng))
            A["err_after_affine"].append(float(np.mean(np.abs(s * a_raw + b - b_raw)) / rng))
            A["nowarp_err"].append(float(np.mean(np.abs(u[t] - u[t + 1])[valid])))
            rt = remap(remap(u[t], bmx, bmy), mx, my)
            A["roundtrip_floor"].append(float(np.mean(np.abs(u[t] - rt)[valid]) / 2))
            fg_inner = cv2.erode((u[t] > args.fg).astype(np.uint8), k5).astype(bool)
            fg_inner1 = cv2.erode((u[t + 1] > args.fg).astype(np.uint8), k5).astype(bool)
            e0 = depth_edges(d[t], fg_inner, args.edge_ratio)
            e1 = depth_edges(d[t + 1], fg_inner1, args.edge_ratio)
            e1w = remap(e1.astype(np.uint8), mx, my, True) > 0
            A["edge_f1"].append(edge_f1(e0, e1w, edge_region, k3))
            A["sat_frac"].append(float(np.mean(u[t][cfg_raw[t]] > 1)))
    R["adjacent"] = adj
    R["rgb"] = {"warp_err": rgb_warp, "nowarp_err": rgb_nowarp, "canny_edge_f1": rgb_edge}

    # M1: chained flow, k steps
    ks = args.ks
    K = max(ks)
    longr = {k: {str(s): {"log_ratio": [], "err": [], "px": []} for s in ks} for k in names}
    for t in range(n - 1):
        cx, cy = xx.copy(), yy.copy()
        chain = np.ones((h, w), bool)
        for j in range(min(K, n - 1 - t)):
            step = t + j
            chain &= remap(OK[step].astype(np.float32), cx, cy, True) > 0.5
            dx, dy = remap(fw[step][..., 0], cx, cy), remap(fw[step][..., 1], cx, cy)
            cx, cy = cx + dx, cy + dy
            chain &= (cx >= 0) & (cx < w - 1) & (cy >= 0) & (cy < h - 1)
            kk = j + 1
            if kk in ks:
                tk = t + kk
                vm = cfg[t] & (remap(cfg[tk].astype(np.float32), cx, cy, True) > 0.5) & chain
                for m in M:
                    L = longr[m["name"]][str(kk)]
                    if vm.sum() < 500:
                        L["log_ratio"].append(np.nan), L["err"].append(np.nan), L["px"].append(int(vm.sum()))
                        continue
                    a, b = m["d"][t][vm], remap(m["d"][tk], cx, cy)[vm]
                    au, bu = m["u"][t][vm], remap(m["u"][tk], cx, cy)[vm]
                    L["log_ratio"].append(log_ratio(a, b))
                    L["err"].append(float(np.mean(np.abs(au - bu))))
                    L["px"].append(int(vm.sum()))
    R["long_range"] = longr
    return R


def summarize(R, names, args, n):
    zones = blend_zones(args.blend)
    S = {}
    for k in names:
        A = R["adjacent"][k]
        lr = np.array(A["log_ratio_raw"])
        traj = np.concatenate([[0.0], np.cumsum(lr)])  # L(t), log scale of frame t relative to frame 0
        # change of L across each blend zone vs the same-length change elsewhere
        zone_rows = []
        for z0, z1 in zones:
            if z1 >= n:
                z1 = n - 1
            length = z1 - z0 + 1
            delta = float(traj[z1] - traj[z0 - 1])
            others = [abs(traj[s + length] - traj[s]) for s in range(0, n - length)
                      if not any(not (s + length < a0 - 1 or s > a1) for a0, a1 in zones)]
            zone_rows.append({"zone": f"{z0}-{z1}", "delta_log": delta,
                              "same_length_outside_mean_abs": float(np.mean(others)) if others else None})
        err_u = np.array(A["err_unclipped"])
        S[k] = {
            "M5_err_unclipped": float(np.nanmean(err_u)),
            "M5_err_clipped_old": float(np.nanmean(A["err_clipped"])),
            "M5_mean_abs_log_ratio_raw": float(np.nanmean(np.abs(lr))),
            "M5_mean_abs_log_ratio_clipped_old": float(np.nanmean(np.abs(A["log_ratio_clipped"]))),
            "M5_max_abs_log_ratio_raw": [float(np.nanmax(np.abs(lr))), int(np.nanargmax(np.abs(lr)))],
            "M5_saturated_fg_frac_mean": float(np.mean(A["sat_frac"])),
            "M2_s_mean_abs_dev": float(np.mean(np.abs(np.array(A["s"]) - 1))),
            "M2_s_std": float(np.std(A["s"])),
            "M2_b_norm_mean_abs": float(np.mean(np.abs(A["b_norm"]))),
            "M2_b_norm_std": float(np.std(A["b_norm"])),
            "M2_err_after_scale": float(np.mean(A["err_after_scale"])),
            "M2_err_after_affine": float(np.mean(A["err_after_affine"])),
            "M2_removed_by_scale_pct": float(100 * (1 - np.mean(A["err_after_scale"]) / np.mean(err_u))),
            "M2_removed_by_affine_pct": float(100 * (1 - np.mean(A["err_after_affine"]) / np.mean(err_u))),
            "M3_edge_f1_mean": float(np.nanmean(A["edge_f1"])),
            "M4_depth_warp_over_nowarp": float(np.mean(err_u) / np.mean(A["nowarp_err"])),
            "M4_roundtrip_floor": float(np.mean(A["roundtrip_floor"])),
            "M4_err_over_floor": float(np.mean(err_u) / np.mean(A["roundtrip_floor"])),
            "M1_by_k": {kk: {"mean_abs_log_ratio": float(np.nanmean(np.abs(v["log_ratio"]))),
                             "mean_abs_scale_change_pct": float(100 * np.nanmean(np.expm1(np.abs(v["log_ratio"])))),
                             "mean_err": float(np.nanmean(v["err"])),
                             "median_px": int(np.median(v["px"])) if v["px"] else 0}
                        for kk, v in R["long_range"][k].items()},
            "M1_trajectory_range_log": float(traj.max() - traj.min()),
            "M1_blend_zone_change": zone_rows,
        }
    S["_rgb"] = {"M4_rgb_warp_over_nowarp": float(np.mean(R["rgb"]["warp_err"]) / np.mean(R["rgb"]["nowarp_err"])),
                 "M4_rgb_warp_err": float(np.mean(R["rgb"]["warp_err"])),
                 "M3_canny_edge_f1_mean": float(np.nanmean(R["rgb"]["canny_edge_f1"]))}
    S["_valid_px_median"] = int(np.median(R["valid_px"]))
    return S


# ----------------------------------------------------------------------------- figures
def shade(ax, zones, n):
    for z0, z1 in zones:
        ax.axvspan(z0 - 0.5, min(z1 + 0.5, n - 1), color="gray", alpha=0.12)


def figures(R, S, names, args, n, out, tag):
    zones = blend_zones(args.blend)
    tt = np.arange(1, n)
    # M1
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.2))
    ks = [int(k) for k in args.ks]
    for i, k in enumerate(names):
        c = COLORS[i]
        ax[0].plot(ks, [S[k]["M1_by_k"][str(kk)]["mean_abs_scale_change_pct"] for kk in ks], "o-", color=c, label=k)
        ax[1].plot(ks, [S[k]["M1_by_k"][str(kk)]["mean_err"] for kk in ks], "o-", color=c, label=k)
        traj = np.concatenate([[0.0], np.cumsum(R["adjacent"][k]["log_ratio_raw"])])
        ax[2].plot(np.arange(n), traj, color=c, label=k)
    ax[0].set(xlabel="frame gap k", ylabel="mean |scale change| t -> t+k (%)", title="M1: scale change vs gap")
    ax[1].set(xlabel="frame gap k", ylabel="flow-compensated error", title="M1: error vs gap")
    ax[2].set(xlabel="frame", ylabel="cumulative log scale L(t)", title="M1: scale trajectory (gray: VDA cross-fade)")
    shade(ax[2], zones, n)
    ax[2].axhline(0, color="k", lw=0.5)
    ax[0].legend(fontsize=8)
    fig.suptitle(f"Long-range consistency, unclipped values, flow = {tag}. Includes true depth change from camera motion.",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"m1_drift_{tag}.png"), dpi=120)
    plt.close(fig)
    # M2
    fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for i, k in enumerate(names):
        A = R["adjacent"][k]
        ax[0].plot(tt, A["s"], color=COLORS[i], label=k)
        ax[1].plot(tt, A["b_norm"], color=COLORS[i], label=k)
        ax[2].plot(tt, A["err_unclipped"], color=COLORS[i], label=f"{k}: raw")
        ax[2].plot(tt, A["err_after_affine"], color=COLORS[i], ls="--", label=f"{k}: after s,b")
    ax[0].axhline(1, color="k", lw=0.5)
    ax[1].axhline(0, color="k", lw=0.5)
    ax[0].set_ylabel("scale s")
    ax[1].set_ylabel("shift b / (p98 - p2)")
    ax[2].set_ylabel("flow-comp. error")
    ax[2].set_xlabel("frame t+1")
    for a in ax:
        shade(a, zones, n)
    ax[0].legend(fontsize=8)
    ax[2].legend(fontsize=7, ncol=2)
    ax[0].set_title(f"M2: robust fit warp(d_t+1) = s d_t + b per pair (flow = {tag})", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"m2_scale_shift_{tag}.png"), dpi=120)
    plt.close(fig)
    # M3
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for i, k in enumerate(names):
        ax.plot(tt, R["adjacent"][k]["edge_f1"], color=COLORS[i], label=f"{k} (mean {S[k]['M3_edge_f1_mean']:.3f})")
    ax.plot(tt, R["rgb"]["canny_edge_f1"], color="k", ls=":",
            label=f"RGB Canny edges, flow ceiling (mean {S['_rgb']['M3_canny_edge_f1_mean']:.3f})")
    shade(ax, zones, n)
    ax.set(xlabel="frame t+1", ylabel="edge F1 (1 px tol.)",
           title=f"M3: internal depth-edge consistency between frames (flow = {tag})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"m3_edges_{tag}.png"), dpi=120)
    plt.close(fig)


def write_md(all_S, names, args, path, timing):
    flows = list(all_S)
    L = ["# Temporal evaluation M1-M5 (generated by temporal_eval.py)", "",
         "Relative depth, no ground truth: consistency only. All statistics on unclipped values.",
         f"Settings: fg > {args.fg}, forward-backward < {args.fb} px, edge ratio {args.edge_ratio}, "
         f"edge tolerance 1 px, k = {', '.join(args.ks)}, blend zones {', '.join(args.blend)}.", ""]
    for f in flows:
        S = all_S[f]
        L += [f"## Flow: {f}  (valid px per pair, median {S['_valid_px_median']}; flow time {timing[f]:.1f} s)", "",
              "| Metric | " + " | ".join(names) + " |", "| --- |" + " --- |" * len(names)]

        def row(label, fn):
            L.append(f"| {label} | " + " | ".join(fn(S[k]) for k in names) + " |")

        row("Adjacent flow-comp. error, unclipped (M5)", lambda s: f"{s['M5_err_unclipped']:.4f}")
        row("Adjacent flow-comp. error, clipped (old)", lambda s: f"{s['M5_err_clipped_old']:.4f}")
        row("Mean abs scale change, unclipped (M5)", lambda s: f"{100 * np.expm1(s['M5_mean_abs_log_ratio_raw']):.2f}%")
        row("Mean abs scale change, clipped (old)", lambda s: f"{100 * np.expm1(s['M5_mean_abs_log_ratio_clipped_old']):.2f}%")
        row("Max abs scale change (pair)", lambda s: f"{100 * np.expm1(s['M5_max_abs_log_ratio_raw'][0]):.2f}% ({s['M5_max_abs_log_ratio_raw'][1]})")
        row("Foreground share above p98 (saturated)", lambda s: f"{100 * s['M5_saturated_fg_frac_mean']:.1f}%")
        for kk in args.ks:
            row(f"M1 mean abs scale change, k={kk}", lambda s, kk=kk: f"{s['M1_by_k'][kk]['mean_abs_scale_change_pct']:.2f}%")
        for kk in args.ks:
            row(f"M1 flow-comp. error, k={kk}", lambda s, kk=kk: f"{s['M1_by_k'][kk]['mean_err']:.4f}")
        row("M1 range of scale trajectory L(t)", lambda s: f"{100 * np.expm1(s['M1_trajectory_range_log']):.1f}%")
        for zi, z in enumerate(args.blend):
            row(f"M1 change of L across zone {z} (outside, same length)",
                lambda s, zi=zi: f"{100 * np.expm1(s['M1_blend_zone_change'][zi]['delta_log']):+.2f}% "
                                 f"({100 * np.expm1(s['M1_blend_zone_change'][zi]['same_length_outside_mean_abs'] or 0):.2f}%)")
        row("M2 mean abs (s - 1)", lambda s: f"{s['M2_s_mean_abs_dev']:.4f}")
        row("M2 mean abs shift b / range", lambda s: f"{s['M2_b_norm_mean_abs']:.4f}")
        row("M2 error removed by scale only", lambda s: f"{s['M2_removed_by_scale_pct']:.1f}%")
        row("M2 error removed by scale + shift", lambda s: f"{s['M2_removed_by_affine_pct']:.1f}%")
        row("M3 internal edge F1 (1 px)", lambda s: f"{s['M3_edge_f1_mean']:.3f}")
        row("M4 error after warp / without warp", lambda s: f"{s['M4_depth_warp_over_nowarp']:.3f}")
        row("M4 round-trip floor (same units)", lambda s: f"{s['M4_roundtrip_floor']:.4f}")
        row("M4 error / floor", lambda s: f"{s['M4_err_over_floor']:.1f}x")
        r = S["_rgb"]
        L += ["", f"RGB reference: warp/no-warp error ratio {r['M4_rgb_warp_over_nowarp']:.3f}, "
                  f"Canny edge F1 {r['M3_canny_edge_f1_mean']:.3f} (what a perfectly stable signal reaches with this flow).", ""]
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames", default="results/raw/davis_frames/frames.npz")
    p.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, help="name|depth.npz[|norm key]")
    p.add_argument("--norm", default="results/figures/part1b_2b_davis/norm_params.json")
    p.add_argument("--flows", nargs="+", default=["farneback", "raft"], choices=["farneback", "raft"])
    p.add_argument("--fg", type=float, default=0.05)
    p.add_argument("--fb", type=float, default=2.0)
    p.add_argument("--edge_ratio", type=float, default=1.05)
    p.add_argument("--ks", nargs="+", default=["1", "5", "10", "20"])
    p.add_argument("--blend", nargs="*", default=["24-31", "46-53", "68-69"])
    p.add_argument("--out", default="results/figures/part1b_2b_davis/temporal_eval")
    args = p.parse_args()
    args.ks = [str(int(k)) for k in args.ks]
    args.ks_int = [int(k) for k in args.ks]

    rgb = np.load(args.frames)["frames"]
    n, h, w, _ = rgb.shape
    gray = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in rgb]
    M = load_methods(args.methods, args.norm, h, w)
    names = [m["name"] for m in M]
    os.makedirs(args.out, exist_ok=True)

    all_S, timing = {}, {}
    for flow in args.flows:
        t0 = time.time()
        fw, bw = flows_farneback(gray) if flow == "farneback" else flows_raft(rgb)
        timing[flow] = time.time() - t0
        a = argparse.Namespace(**vars(args))
        a.ks = args.ks_int
        R = evaluate(rgb, gray, M, fw, bw, a)
        S = summarize(R, names, args, n)
        S["_flow_seconds"] = timing[flow]
        S["_methods"] = {m["name"]: {"norm_key": m["norm_key"], "lo": m["lo"], "hi": m["hi"]} for m in M}
        all_S[flow] = S
        with open(os.path.join(args.out, f"temporal_eval_{flow}.json"), "w") as fh:
            json.dump({"summary": S, "curves": R}, fh)
        figures(R, S, names, args, n, args.out, flow)
        print(f"[{flow}] done in {time.time() - t0:.0f} s")
        del fw, bw, R
    write_md(all_S, names, args, os.path.join(args.out, "temporal_eval.md"), timing)
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(all_S, fh, indent=2)
    print(open(os.path.join(args.out, "temporal_eval.md")).read())


if __name__ == "__main__":
    main()
