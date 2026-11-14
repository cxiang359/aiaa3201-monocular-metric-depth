"""Improvement B (first samples): re-stitch the VDA-S windows; the network output is untouched.

  B0       official stitching (must equal results/raw/part2b_vda/depth.npz bitwise)
  B1-ov    sequential, align on the 8 real overlap frames (instead of frame 0 + position 12), all pixels, LS
  B1-fg    as B1-ov, foreground pixels only (sky excluded)
  B1-rob   as B1-fg, Huber IRLS instead of least squares
  B2       joint L1 over all windows: every pair of predictions of the same frame (overlap frames,
           frame 0, the position-12 keyframes), foreground only, first window fixed
  B3       B2 + one sparse anchor window (32 frames spread over the whole clip, one extra forward);
           the anchor only takes part in the alignment, the output still comes from the windows

All variants keep the upstream 8-frame linear cross-fade.

    python part3/improve_b/run_b.py
"""
import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
import vda_windows as V  # noqa: E402


def huber(x, y, rng, max_px=300000):
    if len(x) > max_px:
        sel = rng.choice(len(x), max_px, replace=False)
        x, y = x[sel], y[sel]
    return C.huber_fit(x.astype(np.float64), y.astype(np.float64))


def sequential(W, real, how, fg_thr, css, rng):
    params = [(1.0, 0.0)]
    for w in range(1, len(W)):
        s0, b0 = params[-1]
        prev = np.maximum(W[w - 1][24:32] * s0 + b0, 0)
        cur = W[w][2:10]
        keep = real[w][2:10]
        x, y = cur[keep], prev[keep]
        m = (y > fg_thr) if fg_thr is not None else np.ones_like(y, bool)
        if how == "ls":
            s, b = css(x, y, m)
        else:
            s, b = huber(x[m], y[m], rng)
        params.append((float(s), float(b)))
    return params


def joint(entries, ref_out, fg_thr, n_free_fixed0, per_pair, rng, iters=30):
    """entries: dict frame -> list of (unit, depth map). unit 0 is fixed (s=1, t=0).
    Minimise sum |s_u x_u + t_u - s_v x_v - t_v| over all pairs sharing a frame (IRLS for L1)."""
    n_units = n_free_fixed0
    rows_x, rows_u, rows_v, rows_y = [], [], [], []
    n_pairs = 0
    for f, E in entries.items():
        m = ref_out[f] > fg_thr
        idx = np.flatnonzero(m)
        if len(idx) < 200:
            continue
        for (u, xu), (v, xv) in itertools.combinations(E, 2):
            if u == v:
                continue
            sel = rng.choice(idx, min(per_pair, len(idx)), replace=False)
            rows_u.append(np.full(len(sel), u)), rows_v.append(np.full(len(sel), v))
            rows_x.append(xu.ravel()[sel]), rows_y.append(xv.ravel()[sel])
            n_pairs += 1
    U, Vv = np.concatenate(rows_u), np.concatenate(rows_v)
    X, Y = np.concatenate(rows_x).astype(np.float64), np.concatenate(rows_y).astype(np.float64)
    N, P = len(X), 2 * (n_units - 1)  # unknowns: (s_u, t_u) for u >= 1
    A = np.zeros((N, P))
    c = np.zeros(N)
    r_ = np.arange(N)
    for side, units, vals in ((1, U, X), (-1, Vv, Y)):
        free = units > 0
        A[r_[free], 2 * (units[free] - 1)] += side * vals[free]
        A[r_[free], 2 * (units[free] - 1) + 1] += side
        c[~free] -= side * vals[~free]  # fixed unit: s=1, t=0 -> constant term moves to the right
    theta = np.linalg.lstsq(A, c, rcond=None)[0]
    eps = 1e-3 * np.percentile(np.abs(X), 98)
    for _ in range(iters):
        r = A @ theta - c
        wgt = 1.0 / np.maximum(np.abs(r), eps)
        sw = np.sqrt(wgt)
        theta = np.linalg.lstsq(A * sw[:, None], c * sw, rcond=None)[0]
    params = [(1.0, 0.0)] + [(float(theta[2 * i]), float(theta[2 * i + 1])) for i in range(n_units - 1)]
    return params, {"n_pairs": n_pairs, "n_px": int(N), "l1_mean": float(np.mean(np.abs(A @ theta - c)))}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vda_root", default="external/Video-Depth-Anything")
    p.add_argument("--ckpt", default="data/weights/video_depth_anything_vits.pth")
    p.add_argument("--fg", type=float, default=0.05, help="foreground: normalized VDA value above this")
    p.add_argument("--per_pair", type=int, default=20000)
    p.add_argument("--raw_out", default="results/raw")
    p.add_argument("--out", default="results/figures/part3/improve_b")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)

    vd, css, interp = V.import_upstream(args.vda_root)
    model = V.load_model(args.vda_root, args.ckpt)
    frames = C.load_frames()
    n = len(frames)
    lo, hi = C.norm_range(C.VDA_KEY)
    fg_thr = lo + args.fg * (hi - lo)

    V.predict_windows(model, vd, frames[:32])  # warm-up
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    depth_list, idx, real = V.predict_windows(model, vd, frames)
    torch.cuda.synchronize()
    t_pred = time.perf_counter() - t0
    nw = len(depth_list) // vd.INFER_LEN
    W = [np.stack(depth_list[w * 32:(w + 1) * 32]) for w in range(nw)]
    IDX = idx.reshape(nw, 32)
    REAL = real.reshape(nw, 32)

    t0 = time.perf_counter()
    off, off_scales = V.stitch_official(depth_list, n, vd, css, interp)
    t_off = time.perf_counter() - t0
    upstream, _ = model.infer_video_depth(frames, 30, input_size=518, device="cuda", fp32=False)
    stored = C.load_depth(C.VDA_RAW)
    check = {"official_vs_infer_video_depth_bitwise": bool(np.array_equal(off, np.asarray(upstream, np.float32))),
             "official_vs_stored_part2b_bitwise": bool(np.array_equal(off, stored)),
             "max_abs_diff_vs_stored": float(np.abs(off - stored).max())}
    print("instrumentation check:", check)

    layout = [{"window": w, "frames": IDX[w].tolist(), "real": REAL[w].tolist()} for w in range(nw)]
    logs = {"check": check, "predict_s": t_pred, "official_stitch_s": t_off, "windows": nw, "layout": layout,
            "official_params": [[1.0, 0.0]] + off_scales}
    variants = {}

    for name, how, use_fg in [("B1-ov", "ls", False), ("B1-fg", "ls", True), ("B1-rob", "huber", True)]:
        t0 = time.perf_counter()
        prm = sequential(W, REAL, how, fg_thr if use_fg else None, css, rng)
        variants[name] = (prm, {"align_s": time.perf_counter() - t0})

    # entries per frame: every real prediction of that frame, keyed by window
    def frame_entries():
        E = {}
        for w in range(nw):
            seen = set()
            for pos in range(32):
                f = int(IDX[w, pos])
                if REAL[w, pos] and f not in seen:
                    seen.add(f)
                    E.setdefault(f, []).append((w, W[w][pos]))
        return {f: e for f, e in E.items() if len({u for u, _ in e}) > 1}

    t0 = time.perf_counter()
    E = frame_entries()
    prm, info = joint(E, off, fg_thr, nw, args.per_pair, rng)
    variants["B2"] = (prm, {"align_s": time.perf_counter() - t0, "shared_frames": sorted(E), **info})

    sel = np.round(np.linspace(0, n - 1, vd.INFER_LEN)).astype(int)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    anchor = V.forward_frames(model, vd, frames, sel)
    torch.cuda.synchronize()
    t_anchor = time.perf_counter() - t0
    t0 = time.perf_counter()
    E3 = {}
    for w in range(nw):
        seen = set()
        for pos in range(32):
            f = int(IDX[w, pos])
            if REAL[w, pos] and f not in seen:
                seen.add(f)
                E3.setdefault(f, []).append((w, W[w][pos]))
    for j, f in enumerate(sel):
        E3.setdefault(int(f), []).append((nw, anchor[j]))
    E3 = {f: e for f, e in E3.items() if len({u for u, _ in e}) > 1}
    prm, info = joint(E3, off, fg_thr, nw + 1, args.per_pair // 2, rng)
    variants["B3"] = (prm[:nw], {"align_s": time.perf_counter() - t0, "anchor_forward_s": t_anchor,
                                 "anchor_frames": sel.tolist(), "anchor_params": prm[nw], **info})

    for name, (prm, info) in variants.items():
        out = V.stitch_with(W, n, prm, interp)
        logs[name] = {"params": prm, **info, "max_abs_diff_vs_official_over_range": float(np.abs(out - off).max() / (hi - lo))}
        C.save_depth(os.path.join(args.raw_out, f"p3_b_{name}", "depth.npz"), out, logs[name])
    os.makedirs(args.out, exist_ok=True)
    json.dump(logs, open(os.path.join(args.out, "run_b_log.json"), "w"), indent=2)
    print(json.dumps({k: v for k, v in logs.items() if k != "layout"}, indent=2))


if __name__ == "__main__":
    main()
