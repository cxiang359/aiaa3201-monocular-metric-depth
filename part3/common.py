"""Shared helpers for Part 3 (improve_a / improve_b).

Reuses the flow, fitting and edge code of part2/video_depth/temporal_eval.py so that the
methods and the evaluation use exactly the same correspondence rules.
"""
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "part2", "video_depth"))
from temporal_eval import (depth_edges, edge_f1, flows_farneback, huber_fit, log_ratio,  # noqa: E402,F401
                           remap)

FRAMES = "results/raw/davis_frames/frames.npz"
NORM = "results/figures/part1b_2b_davis/norm_params.json"
DA2_RAW = "results/raw/part1b_da2/depth.npz"
VDA_RAW = "results/raw/part2b_vda/depth.npz"
DA2_KEY = "DA V2-S (frame-wise)"
VDA_KEY = "VDA-S (video)"
FLOW_CACHE = "results/raw/p3_flows_farneback.npz"


def load_frames(path=FRAMES):
    return np.load(path)["frames"]


def load_depth(path):
    z = np.load(path)
    return z["depth" if "depth" in z.files else z.files[0]].astype(np.float32)


def norm_range(key):
    lo, hi = json.load(open(NORM))[key]
    return float(lo), float(hi)


def save_depth(path, d, log=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, depth=d.astype(np.float32))
    if log is not None:
        with open(os.path.join(os.path.dirname(path), "run_log.json"), "w") as f:
            json.dump(log, f, indent=2)


def farneback_cached(rgb, path=FLOW_CACHE):
    """Adjacent forward/backward Farneback flow, same parameters as temporal_eval.py, cached."""
    if os.path.exists(path):
        z = np.load(path)
        if z["fw"].shape[0] == len(rgb) - 1:
            return z["fw"], z["bw"]
    gray = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in rgb]
    fw, bw = flows_farneback(gray)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, fw=fw, bw=bw)
    return fw, bw


def geometry(fw, bw, fb_thr=2.0):
    """Per adjacent pair t->t+1: sampling maps into frame t+1 and the reliability mask."""
    n1, h, w, _ = fw.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    MX, MY, OK = [], [], []
    for t in range(n1):
        mx, my = xx + fw[t][..., 0], yy + fw[t][..., 1]
        bx, by = remap(bw[t][..., 0], mx, my), remap(bw[t][..., 1], mx, my)
        fbe = np.hypot(fw[t][..., 0] + bx, fw[t][..., 1] + by)
        inside = (mx >= 0) & (mx < w - 1) & (my >= 0) & (my < h - 1)
        MX.append(mx), MY.append(my), OK.append(inside & (fbe < fb_thr))
    return xx, yy, MX, MY, OK


def chain(fw, OK, t, k, xx, yy):
    """Chain k adjacent flows from frame t; returns sampling maps into frame t+k and the mask."""
    h, w = xx.shape
    cx, cy = xx.copy(), yy.copy()
    m = np.ones((h, w), bool)
    for j in range(k):
        s = t + j
        m &= remap(OK[s].astype(np.float32), cx, cy, True) > 0.5
        dx, dy = remap(fw[s][..., 0], cx, cy), remap(fw[s][..., 1], cx, cy)
        cx, cy = cx + dx, cy + dy
        m &= (cx >= 0) & (cx < w - 1) & (cy >= 0) & (cy < h - 1)
    return cx, cy, m


def fg_masks(u, thr=0.05):
    k3 = np.ones((3, 3), np.uint8)
    return np.stack([cv2.erode((x > thr).astype(np.uint8), k3).astype(bool) for x in u])


def zones_for(n, step=22, a=2, b=9):
    """VDA cross-fade zones (frame indices) for a clip of n frames: [22w+2, 22w+9]."""
    return [(step * w + a, min(step * w + b, n - 1)) for w in range(1, (n - 1) // step + 1)
            if step * w + a <= n - 1]
