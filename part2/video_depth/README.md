# Part 2b · Video depth (Video Depth Anything-Small) and the Part 1b/2b comparison

Run the shared setup and frame extraction in `part1/mono_baseline/README.md` first.

## Run (from the repo root)

```bash
# VDA-S, offline, default FP16, same frames as Part 1b
python part2/video_depth/run_vda.py --run_id p2b_vdas_v1
#   out of memory?  add --input_size 392 (or lower) and report the value

# optional cross-check with the official script and the project's required flags
(cd external/Video-Depth-Anything && python run.py \
  --input_video ./assets/example_videos/davis_rollercoaster.mp4 \
  --output_dir ../../results/raw/part2b_vda_official \
  --encoder vits --max_len 100 --max_res 640 --save_npz)
# the official run.py expects its weight in external/Video-Depth-Anything/checkpoints/

# side-by-side video + temporal diagnostics (Part 1b and 2b)
python part2/video_depth/compare_visualize.py --frames_idx 10 31 32 70 --crop 200 120 160 120
```

Outputs:

| Path | Content | Git |
| --- | --- | --- |
| `results/raw/part2b_vda/depth.npz` | `depth`: float32 `[100, H, W]` | ignored |
| `results/logs/<run_id>.json` | GPU, FP16, input size, ms/frame, peak MB, commits | ignored |
| `results/videos/part1b_2b_davis.mp4` | RGB, DA V2-S, VDA-S side by side, labelled relative depth | ignored (goes into `depth_results.zip`) |
| `results/figures/part1b_2b_davis/frames.png` | selected frames, optional zoom | commit |
| `results/figures/part1b_2b_davis/xt_slice.png` | one row over time; flicker = horizontal bands | commit |
| `results/figures/part1b_2b_davis/pixel_curves.png` | displayed value of fixed pixels over time | commit |
| `results/figures/part1b_2b_davis/temporal_change.png`, `metrics.json` | optional rough indicator | commit |
| `results/figures/part1b_2b_davis/norm_params.json` | clip-level normalization per method | commit |

## Rules the scripts enforce

- Same frames and colormap for both methods; one fixed normalization per method over the whole clip, never per frame.
- Labels say "relative depth"; colors are not a shared metric scale.
- A Part 3 improvement reuses the unchanged method's normalization: `--norm_from results/figures/part1b_2b_davis/norm_params.json`.
- No depth ground truth exists for this clip: report "more stable / more consistent", never "more accurate".

## Efficiency numbers for the report

Both logs give ms/frame and peak GPU memory measured after warm-up with CUDA synchronization.
DA V2-S latency is per frame; VDA-S latency is the whole clip through `infer_video_depth`
divided by the number of frames (it processes 32-frame windows).
