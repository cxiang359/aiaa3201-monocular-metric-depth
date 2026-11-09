"""Part 1b: Depth Anything V2-Small run independently on every frame (frame-wise baseline).

Output is RELATIVE depth (affine-invariant inverse depth, larger = nearer). It is never
converted to meters or aligned to any ground truth.

Saves raw float predictions (no per-frame normalization) plus a run log with
per-frame latency and peak GPU memory.

Example (from the repo root):
    python part1/mono_baseline/run_da2_framewise.py --run_id p1b_da2s_v1
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np
import torch

MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
}


def git_commit(path="."):
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames", default="results/raw/davis_frames/frames.npz")
    p.add_argument("--da2_root", default="external/Depth-Anything-V2")
    p.add_argument("--ckpt", default="data/weights/depth_anything_v2_vits.pth")
    p.add_argument("--encoder", default="vits", choices=list(MODEL_CONFIGS))
    p.add_argument("--input_size", type=int, default=518, help="DA V2 default; keep fixed for fair comparison")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--out_dir", default="results/raw/part1b_da2")
    p.add_argument("--run_id", default="p1b_da2s")
    args = p.parse_args()

    sys.path.insert(0, os.path.abspath(args.da2_root))
    from depth_anything_v2.dpt import DepthAnythingV2

    assert torch.cuda.is_available(), "CUDA GPU required for the efficiency numbers"
    device = "cuda"

    model = DepthAnythingV2(**MODEL_CONFIGS[args.encoder])
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model = model.to(device).eval()

    frames_rgb = np.load(args.frames)["frames"]  # [N,H,W,3] uint8 RGB
    frames_bgr = frames_rgb[..., ::-1].copy()  # infer_image expects cv2-style BGR

    with torch.no_grad():
        for _ in range(args.warmup):
            model.infer_image(frames_bgr[0], args.input_size)

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        depths, times = [], []
        for fr in frames_bgr:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            d = model.infer_image(fr, args.input_size)  # HxW float, same size as the input frame
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
            depths.append(d.astype(np.float32))

    depth = np.stack(depths)
    peak_mb = torch.cuda.max_memory_allocated() / 2**20
    os.makedirs(args.out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(args.out_dir, "depth.npz"), depth=depth)

    times_ms = np.array(times) * 1000
    log = {
        "run_id": args.run_id,
        "part": "1b",
        "method": f"Depth Anything V2-{args.encoder} (frame-wise)",
        "output": "relative depth (affine-invariant inverse depth, larger = nearer)",
        "repo_commit": git_commit("."),
        "upstream_commit": git_commit(args.da2_root),
        "checkpoint": args.ckpt,
        "frames": args.frames,
        "num_frames": int(depth.shape[0]),
        "frame_hw": list(depth.shape[1:]),
        "input_size": args.input_size,
        "precision": "fp32",
        "latency_ms_mean": float(times_ms.mean()),
        "latency_ms_std": float(times_ms.std()),
        "latency_note": "per frame, model forward incl. resize, after warm-up, cuda-synchronized",
        "peak_gpu_mem_mb": float(peak_mb),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "command": " ".join(sys.argv),
        "depth_range": [float(depth.min()), float(depth.max())],
    }
    os.makedirs("results/logs", exist_ok=True)
    for path in (os.path.join(args.out_dir, "run_log.json"), os.path.join("results/logs", f"{args.run_id}.json")):
        with open(path, "w") as f:
            json.dump(log, f, indent=2)
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
