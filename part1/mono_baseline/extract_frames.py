"""Extract the first N frames of the required clip once, shared by Part 1b and Part 2b.

Both methods then read exactly the same RGB frames at the same resolution, which is
what the project's fair-comparison rule asks for.

Frames are read with Video Depth Anything's own `read_video_frames` (the reader used by
its run.py with --max_len/--max_res), falling back to OpenCV if that import fails.

Example (from the repo root):
    python part1/mono_baseline/extract_frames.py --num_frames 100 --max_res 640
"""
import argparse
import json
import os
import sys

import numpy as np


def read_frames(video, num_frames, max_res, vda_root):
    """Return (frames uint8 RGB [N,H,W,3], fps, reader name)."""
    try:
        sys.path.insert(0, os.path.abspath(vda_root))
        from utils.dc_utils import read_video_frames  # Video-Depth-Anything

        frames, fps = read_video_frames(video, num_frames, -1, max_res)
        return np.asarray(frames), float(fps), "Video-Depth-Anything utils.dc_utils.read_video_frames"
    except ImportError as err:
        print(f"[warn] could not import VDA reader ({err}); falling back to OpenCV")
        import cv2

        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            raise FileNotFoundError(video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = []
        while len(out) < num_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            if max_res > 0 and max(h, w) > max_res:
                s = max_res / max(h, w)
                frame = cv2.resize(frame, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
            out.append(frame)
        return np.stack(out), float(fps), "OpenCV fallback"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", default="external/Video-Depth-Anything/assets/example_videos/davis_rollercoaster.mp4")
    p.add_argument("--vda_root", default="external/Video-Depth-Anything")
    p.add_argument("--num_frames", type=int, default=100)
    p.add_argument("--max_res", type=int, default=640, help="same value as VDA --max_res; -1 keeps original size")
    p.add_argument("--out_dir", default="results/raw/davis_frames")
    p.add_argument("--save_png", action="store_true", help="also write each frame as PNG for inspection")
    args = p.parse_args()

    frames, fps, reader = read_frames(args.video, args.num_frames, args.max_res, args.vda_root)
    if len(frames) != args.num_frames:
        raise RuntimeError(f"expected {args.num_frames} frames, got {len(frames)}")

    os.makedirs(args.out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(args.out_dir, "frames.npz"), frames=frames)
    meta = {
        "video": args.video,
        "num_frames": int(len(frames)),
        "fps": fps,
        "height": int(frames.shape[1]),
        "width": int(frames.shape[2]),
        "max_res": args.max_res,
        "color_order": "RGB",
        "reader": reader,
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    if args.save_png:
        import cv2

        for i, fr in enumerate(frames):
            cv2.imwrite(os.path.join(args.out_dir, f"{i:03d}.png"), cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))

    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
