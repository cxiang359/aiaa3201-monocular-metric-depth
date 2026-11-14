# Part 2b · Video depth (Video Depth Anything-Small) and the Part 1b/2b comparison

Run the shared setup and frame extraction in `part1/mono_baseline/README.md` first.

## Run (from the repo root)

```bash
# VDA-S, offline, default FP16, same frames as Part 1b
python part2/video_depth/run_vda.py --run_id p2b_vdas_v1
#   out of memory?  add --input_size 392 (or lower) and report the value

# cross-check with the official script and the project's required flags (D3)
mkdir -p external/Video-Depth-Anything/checkpoints
ln -sf ../../../data/weights/video_depth_anything_vits.pth external/Video-Depth-Anything/checkpoints/
pip install imageio-ffmpeg   # run.py writes its mp4s before the npz and fails without it
(cd external/Video-Depth-Anything && python run.py \
  --input_video ./assets/example_videos/davis_rollercoaster.mp4 \
  --output_dir ../../results/raw/part2b_vda_official \
  --encoder vits --max_len 100 --max_res 640 --save_npz)
python part2/video_depth/check_official.py      # -> results/logs/p2b_vda_official_check.json

# side-by-side video + temporal diagnostics (Part 1b and 2b)
python part2/video_depth/compare_visualize.py --frames_idx 10 31 32 54 66

# flicker metrics on the raw float depth (flow-compensated), curves and qualitative figure
python part2/video_depth/temporal_flicker.py --source raw
# three difficult cases (crops + flow-compensated residuals)
python part2/video_depth/hard_cases.py
# stage-2 measurements M1-M5 (long-range drift, scale/shift, edges, flow floor, unclipped stats),
# with Farneback and RAFT; RAFT weights download to $TORCH_HOME (set it if ~/.cache is full)
python part2/video_depth/temporal_eval.py
# efficiency: GPU-forward-only timing for both models (complements the end-to-end logs)
python part2/video_depth/bench_forward.py
```

`temporal_flicker.py --source video` is a fallback that inverts the colormap of the side-by-side
mp4. It is approximate (8-bit, compressed) and is not used for any reported number.

### Official script vs `run_vda.py` (D3)

Same 70 input frames (bitwise), and the official `run.py` depth is **bitwise identical** to
`run_vda.py` (max abs diff 0, Pearson r = 1). Two of our own FP16 runs are also bitwise
identical, so this setup is deterministic. `run_vda.py` is used for the timing logs because it
times only inference and reads the shared `frames.npz`.

Outputs:

| Path | Content | Git |
| --- | --- | --- |
| `results/raw/part2b_vda/depth.npz` | `depth`: float32 `[70, 360, 640]` | ignored |
| `results/logs/<run_id>.json` | GPU, FP16, input size, ms/frame, peak MB, commits | ignored |
| `results/videos/part1b_2b_davis.mp4` | RGB, DA V2-S, VDA-S side by side, labelled relative depth | ignored (goes into `depth_results.zip`) |
| `results/figures/part1b_2b_davis/frames.png` | selected frames, optional zoom | commit |
| `results/figures/part1b_2b_davis/xt_slice.png` | one row over time; flicker = horizontal bands | commit |
| `results/figures/part1b_2b_davis/pixel_curves.png` | displayed value of fixed pixels over time | commit |
| `results/figures/part1b_2b_davis/temporal_change.png`, `metrics.json` | optional rough indicator | commit |
| `results/figures/part1b_2b_davis/norm_params.json` | clip-level normalization per method | commit |
| `results/figures/part1b_2b_davis/flicker_metrics.json`, `flicker_curves.json` | flow-compensated metrics (raw depth) and the per-pair curves | commit |
| `results/figures/part1b_2b_davis/flicker_analysis.png`, `flicker_qualitative.png` | curves with VDA cross-fade zones; residual maps + x-t zoom | commit |
| `results/figures/part1b_2b_davis/hard_cases.png`, `hard_cases.json` | three difficult cases | commit |
| `results/figures/part1b_2b_davis/temporal_eval/` | M1–M5: `temporal_eval.md` (tables), `summary.json`, per-flow curves, `m1/m2/m3_*.png` | commit |
| `results/logs/p2b_vda_official_check.json` | official `run.py` vs `run_vda.py` | commit |
| `results/logs/p1b2b_forward_bench.json` | GPU-forward-only timing | commit |

## Rules the scripts enforce

- Same frames and colormap for both methods; one fixed normalization per method over the whole clip, never per frame.
- Labels say "relative depth"; colors are not a shared metric scale.
- A Part 3 improvement reuses the unchanged method's normalization: `--norm_from results/figures/part1b_2b_davis/norm_params.json`.
- No depth ground truth exists for this clip: report "more stable / more consistent", never "more accurate".

## Efficiency (D6)

NVIDIA RTX A6000, torch 2.1.1 + CUDA 12.1, Small (ViT-S) models, 640×360 frames, input size 518
(network tensor 518×924 for both models). All times after warm-up, CUDA-synchronized.

| Method | Precision | End-to-end ms/frame | Peak GPU MB | GPU forward only | Forward peak MB |
| --- | --- | --- | --- | --- | --- |
| DA V2-S, frame-wise | FP32 | 63.4 ± 1.1 | 375 | 28.1 ms/frame | 376 |
| DA V2-S, frame-wise | FP16 autocast | — | — | 21.2 ms/frame | 330 |
| VDA-S, offline | FP16 autocast (default) | 83.3 | 2689 | 239 ms / 32-frame window | 2492 |
| VDA-S, offline | FP32 | 107.1 | 3867 | 802 ms / 32-frame window | 3670 |

What each column measures:

- **End-to-end, DA V2-S** (`results/logs/p1b_da2s_timing.json`): one call of `infer_image` per frame,
  i.e. CPU resize/normalize + host-to-device copy + forward + resize back. The CPU part alone
  is about 44 ms (`da2_cpu_preprocess_ms`), so most of the 63 ms is not the network.
- **End-to-end, VDA-S** (`p2b_vdas_timing*.json`): the whole clip through `infer_video_depth`
  divided by 70. That includes CPU preprocessing of 98 frames (the clip is padded to 4 windows),
  4 forward passes, resizing back and scale/shift stitching. It is a throughput number, not a
  per-frame latency: the first depth map is only available after a full 32-frame window.
- **GPU forward only** (`results/logs/p1b2b_forward_bench.json`, `bench_forward.py`): the network
  on a preprocessed tensor already on the GPU. A VDA window is 32 input frames but only 22 of them are
  new (10 are reference frames from the previous window), so FP16 costs 7.5 ms per input frame or
  10.9 ms per new frame. The 4 windows take about 0.96 s of the 5.83 s end-to-end time.
- **Peak memory** is `torch.cuda.max_memory_allocated` (tensors only, not the CUDA context).
