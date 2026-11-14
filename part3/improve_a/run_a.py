"""Improvement A (first samples): post-process DA V2-S frame-wise depth to reduce flicker.

  A1  global per-frame scale + shift      y_t = exp(g_t) d_t + b_t                (1x1 grid)
  A2  low-frequency scale field           y_t = exp(up(G_t))(x) d_t(x) + b_t      (gh x gw grid)
  A3  A2 + bidirectional, confidence-weighted flow fusion

A1/A2 are solved jointly over the clip (L1 on flow correspondences t -> t+k, k in --ks, plus
spatial / temporal smoothness of G), Adam on GPU. Gauge: the time mean of every grid cell of G is 0
and the time mean of b is 0, so the correction removes only fluctuations and keeps the clip-level
depth layout. Depth is handled in DA V2 normalized units (d / p98 range) and saved back in raw units,
so temporal_eval.py can use the DA V2 normalization key.

    python part3/improve_a/run_a.py
"""
import argparse
import json
import os
import sys
import time

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as C  # noqa: E402


# ----------------------------------------------------------------------------- correspondences
def build_pairs(u, fg, fw, OK, xx, yy, ks, per_pair, rng):
    n, h, w = u.shape
    T0, T1, A, B, P0, P1 = [], [], [], [], [], []
    for k in ks:
        for t in range(n - k):
            cx, cy, m = C.chain(fw, OK, t, k, xx, yy) if k > 1 else (xx + fw[t][..., 0], yy + fw[t][..., 1], OK[t])
            m = m & fg[t] & (C.remap(fg[t + k].astype(np.float32), cx, cy, True) > 0.5)
            idx = np.flatnonzero(m)
            if len(idx) < 200:
                continue
            idx = rng.choice(idx, min(per_pair, len(idx)), replace=False)
            py, px = np.divmod(idx, w)
            T0.append(np.full(len(idx), t)), T1.append(np.full(len(idx), t + k))
            A.append(u[t].ravel()[idx]), B.append(C.remap(u[t + k], cx, cy).ravel()[idx])
            P0.append(np.stack([px, py], 1).astype(np.float32)), P1.append(np.stack([cx.ravel()[idx], cy.ravel()[idx]], 1))
    cat = np.concatenate
    return dict(t0=cat(T0), t1=cat(T1), a=cat(A), b=cat(B), p0=cat(P0), p1=cat(P1))


def bilinear_index(p, t, gh, gw, h, w):
    """Flat indices into G[n, gh, gw] and weights for points p (x, y) of frame t (align_corners=True)."""
    fx = p[:, 0] / (w - 1) * (gw - 1)
    fy = p[:, 1] / (h - 1) * (gh - 1)
    x0 = np.clip(np.floor(fx), 0, max(gw - 2, 0)).astype(np.int64)
    y0 = np.clip(np.floor(fy), 0, max(gh - 2, 0)).astype(np.int64)
    ax, ay = fx - x0, fy - y0
    if gw == 1:
        x0, ax = np.zeros_like(x0), np.zeros_like(ax)
    if gh == 1:
        y0, ay = np.zeros_like(y0), np.zeros_like(ay)
    x1, y1 = np.minimum(x0 + 1, gw - 1), np.minimum(y0 + 1, gh - 1)
    base = t * gh * gw
    I = np.stack([base + y0 * gw + x0, base + y0 * gw + x1, base + y1 * gw + x0, base + y1 * gw + x1], 1)
    W = np.stack([(1 - ax) * (1 - ay), ax * (1 - ay), (1 - ax) * ay, ax * ay], 1)
    return I, W.astype(np.float32)


# ----------------------------------------------------------------------------- A1 / A2 solver
def solve_field(pairs, n, h, w, gh, gw, lam_s, lam_t, steps, lr, dev):
    I0, W0 = bilinear_index(pairs["p0"], pairs["t0"], gh, gw, h, w)
    I1, W1 = bilinear_index(pairs["p1"], pairs["t1"], gh, gw, h, w)
    tt = lambda x, dt=torch.float32: torch.as_tensor(x, dtype=dt, device=dev)  # noqa: E731
    I0, W0, I1, W1 = tt(I0, torch.long), tt(W0), tt(I1, torch.long), tt(W1)
    a, b = tt(pairs["a"]), tt(pairs["b"])
    t0, t1 = tt(pairs["t0"], torch.long), tt(pairs["t1"], torch.long)
    G = torch.zeros(n, gh, gw, device=dev, requires_grad=True)
    beta = torch.zeros(n, device=dev, requires_grad=True)
    opt = torch.optim.Adam([G, beta], lr=lr)
    hist = []
    for it in range(steps):
        Gf = G.reshape(-1)
        s0 = torch.exp((Gf[I0] * W0).sum(1))
        s1 = torch.exp((Gf[I1] * W1).sum(1))
        data = (s1 * b + beta[t1] - (s0 * a + beta[t0])).abs().mean()
        reg = torch.zeros((), device=dev)
        if lam_s > 0 and gw > 1:
            reg = reg + lam_s * (G[:, :, 1:] - G[:, :, :-1]).pow(2).mean()
        if lam_s > 0 and gh > 1:
            reg = reg + lam_s * (G[:, 1:, :] - G[:, :-1, :]).pow(2).mean()
        if lam_t > 0:
            reg = reg + lam_t * (G[1:] - G[:-1]).pow(2).mean()
        loss = data + reg
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():  # gauge: no static correction, only fluctuations around the clip mean
            G -= G.mean(0, keepdim=True)
            beta -= beta.mean()
        if it % 100 == 0 or it == steps - 1:
            hist.append([it, float(data), float(reg)])
    return G.detach(), beta.detach(), hist


def apply_field(u, G, beta, dev):
    n, h, w = u.shape
    S = torch.exp(F.interpolate(G[:, None], size=(h, w), mode="bilinear", align_corners=True))[:, 0]
    y = S * torch.as_tensor(u, device=dev) + beta[:, None, None]
    return y.cpu().numpy(), S.cpu().numpy()


# ----------------------------------------------------------------------------- A3 fusion
def fuse(y, fw, bw, gray, w_max, s_f=1.0, s_i=10.0, s_d=0.05):
    n, h, w = y.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    def one_dir(order, flow_to_prev, flow_back):
        z = np.empty_like(y)
        z[order[0]] = y[order[0]]
        wsum = []
        for i, t in enumerate(order[1:]):
            prev = order[i]
            f = flow_to_prev(t)          # flow from frame t to frame prev
            mx, my = xx + f[..., 0], yy + f[..., 1]
            fb = flow_back(t)            # flow from prev back to t, sampled at the target
            bx, by = C.remap(fb[..., 0], mx, my), C.remap(fb[..., 1], mx, my)
            e = np.hypot(f[..., 0] + bx, f[..., 1] + by)
            inside = (mx >= 0) & (mx < w - 1) & (my >= 0) & (my < h - 1)
            zw = C.remap(z[prev], mx, my)
            gw_ = C.remap(gray[prev], mx, my)
            c = np.exp(-(e / s_f) ** 2) * np.exp(-((gray[t] - gw_) / s_i) ** 2) * np.exp(-((y[t] - zw) / s_d) ** 2)
            wt = w_max * c * inside
            z[t] = wt * zw + (1 - wt) * y[t]
            wsum.append(float(wt.mean()))
        return z, wsum

    # forward pass: frame t looks at t-1 through bw[t-1] (t -> t-1); its check uses fw[t-1]
    zf, wf = one_dir(list(range(n)), lambda t: bw[t - 1], lambda t: fw[t - 1])
    # backward pass: frame t looks at t+1 through fw[t]; its check uses bw[t]
    zb, wb = one_dir(list(range(n - 1, -1, -1)), lambda t: fw[t], lambda t: bw[t])
    return 0.5 * (zf + zb), {"mean_weight_fwd": float(np.mean(wf)), "mean_weight_bwd": float(np.mean(wb))}


# ----------------------------------------------------------------------------- fidelity
def fidelity(y, u, fg_thr, edge_ratio=1.05):
    """Structure change vs the input after a per-frame affine fit, and same-frame edge F1."""
    k3, k5 = np.ones((3, 3), np.uint8), np.ones((5, 5), np.uint8)
    se, f1 = [], []
    for t in range(len(u)):
        m = u[t] > fg_thr
        A = np.stack([y[t][m], np.ones(m.sum())], 1)
        coef = np.linalg.lstsq(A, u[t][m], rcond=None)[0]
        se.append(float(np.mean(np.abs(A @ coef - u[t][m]))))
        inner = cv2.erode(m.astype(np.uint8), k5).astype(bool)
        e_in = C.depth_edges(u[t], inner, edge_ratio)
        e_out = C.depth_edges(np.maximum(y[t], 1e-6), inner, edge_ratio)
        f1.append(C.edge_f1(e_in, e_out, inner, k3))
    return {"structure_err_mean": float(np.mean(se)), "same_frame_edge_f1_mean": float(np.nanmean(f1))}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid", type=int, nargs=2, default=[6, 10])
    p.add_argument("--ks", type=int, nargs="+", default=[1, 5])
    p.add_argument("--per_pair", type=int, default=4000)
    p.add_argument("--lam_s", type=float, default=1.0)
    p.add_argument("--lam_t", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--w_max", type=float, default=0.5)
    p.add_argument("--fg", type=float, default=0.05)
    p.add_argument("--raw_out", default="results/raw")
    p.add_argument("--out", default="results/figures/part3/improve_a")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--suffix", default="", help="appended to variant names, e.g. -adj")
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    rgb = C.load_frames()
    n, h, w, _ = rgb.shape
    gray = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in rgb]).astype(np.float32)
    d = C.load_depth(C.DA2_RAW)
    lo, hi = C.norm_range(C.DA2_KEY)
    rng_ = hi - lo
    u = (d - lo) / rng_
    t_flow = time.time()
    fw, bw = C.farneback_cached(rgb)
    xx, yy, MX, MY, OK = C.geometry(fw, bw)
    t_flow = time.time() - t_flow
    fg = C.fg_masks(u, args.fg)
    t0 = time.time()
    pairs = build_pairs(u, fg, fw, OK, xx, yy, args.ks, args.per_pair, np.random.default_rng(args.seed))
    t_pairs = time.time() - t0

    os.makedirs(args.out, exist_ok=True)
    out_y, logs = {}, {}
    for name, (gh, gw), lam_s in [("A1", (1, 1), 0.0), ("A2", tuple(args.grid), args.lam_s)]:
        t0 = time.time()
        G, beta, hist = solve_field(pairs, n, h, w, gh, gw, lam_s, args.lam_t, args.steps, args.lr, dev)
        y, S = apply_field(u, G, beta, dev)
        if dev == "cuda":
            torch.cuda.synchronize()
        out_y[name] = y
        logs[name] = {"grid": [gh, gw], "lam_s": lam_s, "lam_t": args.lam_t, "solve_s": time.time() - t0,
                      "loss_hist": hist, "scale_field_range": [float(S.min()), float(S.max())],
                      "per_frame_mean_scale_pct": float(100 * np.mean(np.abs(np.expm1(G.mean((1, 2)).cpu().numpy())))),
                      "beta_mean_abs": float(beta.abs().mean())}
    t0 = time.time()
    out_y["A3"], finfo = fuse(out_y["A2"], fw, bw, gray, args.w_max)
    logs["A3"] = {"base": "A2", "w_max": args.w_max, "fuse_s": time.time() - t0, **finfo}

    out_y = {k + args.suffix: v for k, v in out_y.items()}
    logs = {k + args.suffix: v for k, v in logs.items()}
    for name, y in out_y.items():
        logs[name].update(fidelity(y, u, args.fg))
        logs[name].update({"flow_s": t_flow, "pairs_s": t_pairs, "n_pairs_px": int(len(pairs["a"])), "ks": args.ks,
                           "device": dev, "command": " ".join(sys.argv)})
        C.save_depth(os.path.join(args.raw_out, f"p3_a_{name}", "depth.npz"), y * rng_ + lo, logs[name])
    json.dump(logs, open(os.path.join(args.out, f"run_a_log{args.suffix}.json"), "w"), indent=2)

    # x-t slice: row 180 through the train, every frame stacked top to bottom
    vda = C.load_depth(C.VDA_RAW)
    vlo, vhi = C.norm_range(C.VDA_KEY)
    sf = args.suffix
    cols = [("DA V2-S (input)", u), (f"A1{sf} global", out_y["A1" + sf]),
            (f"A2{sf} {args.grid[0]}x{args.grid[1]} field", out_y["A2" + sf]),
            (f"A3{sf} = A2 + fusion", out_y["A3" + sf]), ("VDA-S (reference)", (vda - vlo) / (vhi - vlo))]
    fig, ax = plt.subplots(2, len(cols), figsize=(4 * len(cols), 6.5),
                           gridspec_kw={"height_ratios": [1, 1.2]})
    for j, (title, v) in enumerate(cols):
        ax[0, j].imshow(np.clip(v[43], 0, 1), cmap="inferno", vmin=0, vmax=1)
        ax[0, j].axhline(180, color="c", lw=0.6)
        ax[0, j].set_title(f"{title}\nframe 43", fontsize=9)
        ax[1, j].imshow(np.clip(v[:, 180, :], 0, 1), cmap="inferno", vmin=0, vmax=1, aspect="auto")
        ax[1, j].set_title("x-t slice of row 180 (time goes down)", fontsize=8)
        ax[1, j].set_xlabel("x")
        ax[0, j].set_xticks([]), ax[0, j].set_yticks([])
    ax[1, 0].set_ylabel("frame")
    fig.suptitle("Improvement A samples: horizontal stripes in the x-t slice = flicker", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, f"a_xt_slices{sf}.png"), dpi=110)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "loss_hist"} for k, v in logs.items()}, indent=2))


if __name__ == "__main__":
    main()
