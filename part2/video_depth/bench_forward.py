"""D6: put DA V2-S and VDA-S latency on the same footing.

The end-to-end numbers in results/logs/*_timing*.json are measured differently:
  * DA V2-S: per frame, infer_image (CPU resize/normalize + GPU forward + resize back), FP32
  * VDA-S:   whole clip through infer_video_depth (windows of 32, stitching), divided by #frames, FP16
This script times only the network forward on GPU tensors that are already preprocessed,
cuda-synchronized, after warm-up, and splits out the CPU preprocessing time.

    python part2/video_depth/bench_forward.py
"""
import argparse
import json
import os
import platform
import sys
import time

import numpy as np
import torch

CFG = {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]}


def timed(fn, reps):
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(ts)), float(np.std(ts))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames", default="results/raw/davis_frames/frames.npz")
    p.add_argument("--da2_root", default="external/Depth-Anything-V2")
    p.add_argument("--vda_root", default="external/Video-Depth-Anything")
    p.add_argument("--da2_ckpt", default="data/weights/depth_anything_v2_vits.pth")
    p.add_argument("--vda_ckpt", default="data/weights/video_depth_anything_vits.pth")
    p.add_argument("--input_size", type=int, default=518)
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--out", default="results/logs/p1b2b_forward_bench.json")
    args = p.parse_args()

    frames = np.load(args.frames)["frames"]
    res = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda,
           "python": platform.python_version(), "input_size": args.input_size, "frame_hw": list(frames.shape[1:3]),
           "reps": args.reps, "note": "GPU forward only, preprocessed tensors already on GPU, after warm-up"}

    # ---- DA V2-S -------------------------------------------------------------
    sys.path.insert(0, os.path.abspath(args.da2_root))
    from depth_anything_v2.dpt import DepthAnythingV2

    m = DepthAnythingV2(**CFG)
    m.load_state_dict(torch.load(args.da2_ckpt, map_location="cpu"))
    m = m.cuda().eval()
    bgr = frames[0][..., ::-1].copy()
    t0 = time.perf_counter()
    for _ in range(args.reps):
        x, _ = m.image2tensor(bgr, args.input_size)
    pre_ms = (time.perf_counter() - t0) * 1000 / args.reps
    res["da2_tensor_hw"] = list(x.shape[-2:])
    res["da2_cpu_preprocess_ms"] = pre_ms
    with torch.no_grad():
        for prec, ac in (("fp32", False), ("fp16_autocast", True)):
            def f():
                with torch.autocast("cuda", enabled=ac):
                    m(x)
            for _ in range(5):
                f()
            torch.cuda.reset_peak_memory_stats()
            mean, std = timed(f, args.reps)
            res[f"da2_forward_ms_per_frame_{prec}"] = [mean, std]
            res[f"da2_peak_mem_mb_{prec}"] = torch.cuda.max_memory_allocated() / 2**20
    del m
    torch.cuda.empty_cache()

    # ---- VDA-S ---------------------------------------------------------------
    sys.path.pop(0)
    for k in [k for k in sys.modules if k.startswith("depth_anything_v2") or k.startswith("utils")]:
        del sys.modules[k]
    sys.path.insert(0, os.path.abspath(args.vda_root))
    from video_depth_anything.video_depth import VideoDepthAnything, INFER_LEN

    v = VideoDepthAnything(**CFG)
    v.load_state_dict(torch.load(args.vda_ckpt, map_location="cpu"), strict=True)
    v = v.cuda().eval()
    h, w = x.shape[-2:]  # same resize rule (short side -> 518, multiple of 14) as DA V2
    clip = torch.randn(1, INFER_LEN, 3, h, w, device="cuda")
    with torch.no_grad():
        for prec, ac in (("fp16_autocast", True), ("fp32", False)):
            def g():
                with torch.autocast("cuda", enabled=ac):
                    v(clip)
            for _ in range(3):
                g()
            torch.cuda.reset_peak_memory_stats()
            mean, std = timed(g, args.reps)
            res[f"vda_forward_ms_per_window_{prec}"] = [mean, std]
            res[f"vda_forward_ms_per_input_frame_{prec}"] = mean / INFER_LEN
            # each new window contributes INFER_LEN - OVERLAP (=22) new frames
            res[f"vda_forward_ms_per_new_frame_{prec}"] = mean / 22
            res[f"vda_peak_mem_mb_{prec}"] = torch.cuda.max_memory_allocated() / 2**20
    res["vda_window"] = {"frames_per_window": INFER_LEN, "new_frames_per_window": 22,
                         "windows_for_70_frames": 4}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
