"""Part 2b: Video Depth Anything-Small on the same frames as the Part 1b baseline.

Feeds the frames saved by part1/mono_baseline/extract_frames.py (same frames, same
resolution as the DA V2 run) into VideoDepthAnything.infer_video_depth, which is what the
official run.py calls. Default precision is FP16 (autocast), as the project requires.

Output is RELATIVE depth with one scale/shift shared across the clip. No metric conversion.

Example (from the repo root):
    python part2/video_depth/run_vda.py --run_id p2b_vdas_v1
If GPU memory is short: add --input_size 392 (and report it).

Cross-check with the official script (should give matching depth):
    cd external/Video-Depth-Anything && python run.py \
      --input_video ./assets/example_videos/davis_rollercoaster.mp4 --output_dir ../../results/raw/part2b_vda_official \
      --encoder vits --max_len 100 --max_res 640 --save_npz
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


def infer(model, frames, fps, input_size, fp32):
    try:
        depths, _ = model.infer_video_depth(frames, fps, input_size=input_size, device="cuda", fp32=fp32)
    except TypeError:  # older upstream versions without the fp32 argument
        depths, _ = model.infer_video_depth(frames, fps, input_size=input_size, device="cuda")
    return np.asarray(depths, dtype=np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames", default="results/raw/davis_frames/frames.npz")
    p.add_argument("--meta", default="results/raw/davis_frames/meta.json")
    p.add_argument("--vda_root", default="external/Video-Depth-Anything")
    p.add_argument("--ckpt", default="data/weights/video_depth_anything_vits.pth")
    p.add_argument("--encoder", default="vits", choices=list(MODEL_CONFIGS))
    p.add_argument("--input_size", type=int, default=518, help="lower it only if memory is short, and report it")
    p.add_argument("--fp32", action="store_true", help="default is FP16 autocast")
    p.add_argument("--no_warmup", action="store_true")
    p.add_argument("--out_dir", default="results/raw/part2b_vda")
    p.add_argument("--run_id", default="p2b_vdas")
    args = p.parse_args()

    sys.path.insert(0, os.path.abspath(args.vda_root))
    from video_depth_anything.video_depth import VideoDepthAnything

    assert torch.cuda.is_available(), "CUDA GPU required"
    model = VideoDepthAnything(**MODEL_CONFIGS[args.encoder])
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"), strict=True)
    model = model.to("cuda").eval()

    frames = np.load(args.frames)["frames"]  # [N,H,W,3] uint8 RGB, same as Part 1b
    fps = json.load(open(args.meta))["fps"]

    if not args.no_warmup:  # one short clip so CUDA kernels/autocast are initialised before timing
        infer(model, frames[:32], fps, args.input_size, args.fp32)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    depth = infer(model, frames, fps, args.input_size, args.fp32)
    torch.cuda.synchronize()
    total_s = time.perf_counter() - t0
    peak_mb = torch.cuda.max_memory_allocated() / 2**20

    assert depth.shape[0] == frames.shape[0], (depth.shape, frames.shape)
    os.makedirs(args.out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(args.out_dir, "depth.npz"), depth=depth)

    log = {
        "run_id": args.run_id,
        "part": "2b",
        "method": f"Video Depth Anything-{args.encoder} (offline, relative)",
        "output": "relative depth, one scale/shift for the whole clip",
        "repo_commit": git_commit("."),
        "upstream_commit": git_commit(args.vda_root),
        "checkpoint": args.ckpt,
        "frames": args.frames,
        "num_frames": int(depth.shape[0]),
        "frame_hw": list(depth.shape[1:]),
        "input_size": args.input_size,
        "precision": "fp32" if args.fp32 else "fp16 (autocast)",
        "latency_ms_per_frame": float(total_s * 1000 / depth.shape[0]),
        "latency_note": "whole clip through infer_video_depth (windows, stitching, resize back) / num_frames, after warm-up",
        "total_seconds": float(total_s),
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
