# Part 1b · Monocular frame baseline (and shared setup for Part 2b)

Depth Anything V2-Small runs independently on every frame of `davis_rollercoaster.mp4`.
The project asks for the first 100 frames, but the clip shipped with Video-Depth-Anything has
only **70 frames** (24 fps, resized to 640×360 by `--max_res 640`), so all 70 are used. The
extraction and the official VDA `run.py --max_len 100` both stop at 70 (checked:
`results/logs/p2b_vda_official_check.json`). Its output is **relative depth** (affine-invariant inverse
depth, larger = nearer); it is never converted to meters or aligned to ground truth.
The same frames feed the Video Depth Anything run in `part2/video_depth/`.

## 1. Environment (WSL, CUDA GPU)

```bash
nvidia-smi                                   # note the GPU model for the efficiency table
conda create -n p3depth python=3.10 -y && conda activate p3depth
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # match your driver
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 2. Upstream code and weights (not tracked by git)

```bash
mkdir -p external data/weights
git clone https://github.com/DepthAnything/Depth-Anything-V2 external/Depth-Anything-V2
git clone https://github.com/DepthAnything/Video-Depth-Anything external/Video-Depth-Anything
pip install -r external/Depth-Anything-V2/requirements.txt
pip install -r external/Video-Depth-Anything/requirements.txt
pip install matplotlib
wget -O data/weights/depth_anything_v2_vits.pth \
  "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true"
wget -O data/weights/video_depth_anything_vits.pth \
  "https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth?download=true"
```

If the two `requirements.txt` files conflict, create a second environment for VDA.
Record both upstream commits (`git -C external/<repo> rev-parse --short HEAD`); the run logs do this automatically.

## 3. Run (from the repo root)

```bash
# shared: first 100 frames (the clip has 70, all used) at the original frame rate, resized like VDA --max_res 640
python part1/mono_baseline/extract_frames.py --num_frames 100 --max_res 640

# Part 1b: frame-wise DA V2-S, raw float output + latency / peak memory log
python part1/mono_baseline/run_da2_framewise.py --run_id p1b_da2s_v1

# Part 2b and the comparison: see part2/video_depth/README.md
```

Outputs (local only, ignored by git):

| Path | Content |
| --- | --- |
| `results/raw/davis_frames/frames.npz`, `meta.json` | 70 RGB frames (640×360, 24 fps), reader used |
| `results/raw/part1b_da2/depth.npz` | `depth`: float32 `[70, 360, 640]`, no normalization |
| `results/logs/<run_id>.json` | GPU, precision (FP32), input size 518, ms/frame, peak MB, commits, command |

## Notes

- Do not use upstream `run_video.py` for figures: it min-max normalizes every frame on its own, which hides flicker.
- Keep `--input_size 518` fixed between this baseline and any Part 3 variant of it.
