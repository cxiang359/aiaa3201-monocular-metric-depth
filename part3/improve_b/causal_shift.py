"""B step 0: is the VDA scale drop inside the cross-fade zones caused by the stitching or by the content?

Run the unchanged upstream VDA on frames s..69 (s = 0, 5, 11, 16). The windows, and with them the
cross-fade zones, move by s frames; the image content stays where it was. For each run, the
cumulative log-scale trajectory L(t) (median log ratio of flow-matched foreground pixels, same rules
as temporal_eval.py) is measured, and its change across
  (a) the zones of this run (global frame index 24+s..31+s, 46+s..53+s), and
  (b) the original zones 24-31 and 46-53,
is compared with the same-length change elsewhere. If the drop follows (a), stitching causes it.

    python part3/improve_b/causal_shift.py
"""
import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
import vda_windows as V  # noqa: E402

ORIG = [(24, 31), (46, 53)]


def trajectory(d, s, MX, MY, OK, fg_frac):
    """L over global frames s..n-1 (L[s] = 0)."""
    hi = np.percentile(d, 98)
    fg = C.fg_masks(d / hi, fg_frac)
    lr = []
    for j in range(len(d) - 1):
        t = s + j
        valid = fg[j] & (C.remap(fg[j + 1].astype(np.float32), MX[t], MY[t], True) > 0.5) & OK[t]
        lr.append(C.log_ratio(d[j][valid], C.remap(d[j + 1], MX[t], MY[t])[valid]))
    return np.concatenate([[0.0], np.cumsum(lr)])


def zone_change(L, s, z, n, all_zones):
    """Change of L across global zone z, and mean |change| over same-length spans touching no zone."""
    z0, z1 = z
    g = lambda f: L[f - s]  # noqa: E731
    delta = float(g(z1) - g(z0 - 1))
    length = z1 - z0 + 1
    others = [abs(g(a + length) - g(a)) for a in range(s, n - length)
              if not any(not (a + length < b0 - 1 or a > b1) for b0, b1 in all_zones)]
    return delta, float(np.mean(others)) if others else None


def pct(x):
    return 100 * np.expm1(x)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shifts", type=int, nargs="+", default=[0, 5, 11, 16])
    p.add_argument("--vda_root", default="external/Video-Depth-Anything")
    p.add_argument("--ckpt", default="data/weights/video_depth_anything_vits.pth")
    p.add_argument("--fg", type=float, default=0.05)
    p.add_argument("--out", default="results/figures/part3/improve_b")
    args = p.parse_args()

    frames = C.load_frames()
    n = len(frames)
    fw, bw = C.farneback_cached(frames)
    _, _, MX, MY, OK = C.geometry(fw, bw)
    model = V.load_model(args.vda_root, args.ckpt)

    res, curves = {}, {}
    for s in args.shifts:
        d, _ = model.infer_video_depth(frames[s:], 30, input_size=518, device="cuda", fp32=False)
        d = np.asarray(d, np.float32)
        L = trajectory(d, s, MX, MY, OK, args.fg)
        own = [(a + s, b + s) for a, b in C.zones_for(n - s)]
        own = [(a, min(b, n - 1)) for a, b in own]
        row = {"own_zones": [], "original_zones": []}
        # reference: same-length spans that touch none of this run's own zones
        for z in own:
            dlt, oth = zone_change(L, s, z, n, own)
            row["own_zones"].append({"zone": f"{z[0]}-{z[1]}", "change_pct": pct(dlt),
                                     "outside_mean_abs_pct": None if oth is None else pct(oth)})
        for z in ORIG:
            dlt, _ = zone_change(L, s, z, n, own)
            row["original_zones"].append({"zone": f"{z[0]}-{z[1]}", "change_pct": pct(dlt), "is_own_zone": z in own})
        res[s] = row
        curves[s] = (np.arange(s, n), L, own)
        print(s, json.dumps(row))

    os.makedirs(args.out, exist_ok=True)
    json.dump({"results": res, "curves": {s: c[1].tolist() for s, c in curves.items()}},
              open(os.path.join(args.out, "causal_shift.json"), "w"), indent=2)

    fig, ax = plt.subplots(len(args.shifts), 1, figsize=(10, 2.3 * len(args.shifts)), sharex=True)
    for i, s in enumerate(args.shifts):
        x, L, own = curves[s]
        # align the curves at frame 20 so they can be compared by eye
        a = ax[i]
        a.plot(x, pct(L - L[20 - s]), color="C1")
        for z0, z1 in own:
            a.axvspan(z0 - 0.5, z1 + 0.5, color="C1", alpha=0.18)
        for z0, z1 in ORIG:
            a.axvline(z0 - 0.5, color="k", ls=":", lw=0.8), a.axvline(z1 + 0.5, color="k", ls=":", lw=0.8)
        txt = ", ".join(f"{r['zone']}: {r['change_pct']:+.1f}%" for r in res[s]["own_zones"])
        a.set_title(f"start s = {s}: zones of this run (orange) {txt}", fontsize=9)
        a.set_ylabel("scale vs frame 20 (%)")
        a.axhline(0, color="k", lw=0.4)
    ax[-1].set_xlabel("original frame index (dotted: original zones 24-31, 46-53)")
    fig.suptitle("Causal test: do the VDA scale drops move with the cross-fade zones?", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "causal_shift.png"), dpi=120)


if __name__ == "__main__":
    main()
