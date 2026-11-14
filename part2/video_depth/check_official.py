"""D3: check that run_vda.py gives the same depth as the official Video-Depth-Anything run.py.

Step 1 (server, inside external/Video-Depth-Anything, exactly the command from the project PDF):
    python run.py --input_video ./assets/example_videos/davis_rollercoaster.mp4 \
      --output_dir ../../results/raw/part2b_vda_official --encoder vits --max_len 100 --max_res 640 --save_npz
Step 2 (repo root):
    python part2/video_depth/check_official.py

Compares
  * input frames: read_video_frames(max_len=100, max_res=640) vs results/raw/davis_frames/frames.npz
  * depth: official *_depths.npz vs our results/raw/part2b_vda/depth.npz
  * noise floor: two of our own FP16 runs (part2b_vda vs timing_vda), same code, same frames
Differences are reported relative to the clip's depth range, since the output is relative depth.
"""
import argparse
import json
import os
import sys

import numpy as np


def compare(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    rng = float(np.percentile(a, 98) - np.percentile(a, 2))
    d = np.abs(a - b)
    return {
        "shape": list(a.shape),
        "max_abs_diff": float(d.max()),
        "mean_abs_diff": float(d.mean()),
        "max_abs_diff_over_p2_p98_range": float(d.max() / rng),
        "mean_abs_diff_over_p2_p98_range": float(d.mean() / rng),
        "p99_abs_diff_over_p2_p98_range": float(np.percentile(d, 99) / rng),
        "pearson_r": float(np.corrcoef(a.ravel(), b.ravel())[0, 1]),
        "bitwise_equal": bool(np.array_equal(a, b)),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--official", default="results/raw/part2b_vda_official/davis_rollercoaster_depths.npz")
    p.add_argument("--ours", default="results/raw/part2b_vda/depth.npz")
    p.add_argument("--ours_rerun", default="results/raw/timing_vda/depth.npz")
    p.add_argument("--frames", default="results/raw/davis_frames/frames.npz")
    p.add_argument("--video", default="external/Video-Depth-Anything/assets/example_videos/davis_rollercoaster.mp4")
    p.add_argument("--vda_root", default="external/Video-Depth-Anything")
    p.add_argument("--out", default="results/logs/p2b_vda_official_check.json")
    args = p.parse_args()

    res = {"official_cmd": "run.py --encoder vits --max_len 100 --max_res 640 --save_npz (default FP16, input_size 518)"}

    sys.path.insert(0, os.path.abspath(args.vda_root))
    from utils.dc_utils import read_video_frames

    off_frames, fps = read_video_frames(args.video, 100, -1, 640)
    ours_frames = np.load(args.frames)["frames"]
    res["frames"] = {
        "official_shape": list(off_frames.shape),
        "ours_shape": list(ours_frames.shape),
        "fps": float(fps),
        "identical": bool(off_frames.shape == ours_frames.shape and np.array_equal(off_frames, ours_frames)),
    }

    off = np.load(args.official)["depths"]
    ours = np.load(args.ours)["depth"]
    res["official_vs_ours"] = compare(off, ours)
    if os.path.exists(args.ours_rerun):
        res["noise_floor_ours_vs_ours_rerun"] = compare(ours, np.load(args.ours_rerun)["depth"])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
