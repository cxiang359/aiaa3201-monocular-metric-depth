# Part 1b / 2b 观察段落（报告草稿）+ 证据对照

2026-11-14 修订。所有数字都来自服务器上的原始浮点深度（`results/raw/part1b_da2/depth.npz`、`results/raw/part2b_vda/depth.npz`），由 `python part2/video_depth/temporal_flicker.py --source raw` 等脚本生成，每个数字都能在第二节列出的 JSON 里找到。11/10 版本用的是从 mp4 反推的近似值（`*_approx`），其中“第 66–67 帧跳 14%”“2.6 倍”“2.3%”在原始数据里不成立，已全部替换。图号 Fig. X/Y/Z/W、Table N/E 是占位符。

## 一、报告英文段落（可直接改写进 Experiments）

**Setup.**
The required clip `davis_rollercoaster.mp4` has 70 frames (24 fps); the project asks for the first 100, so all 70 are used. Frames are read with the Video-Depth-Anything reader at `--max_res 640` (640×360) and shared by both methods. Both outputs are affine-invariant relative inverse depth (larger = nearer). They are never converted to meters or aligned to any ground truth. Each method is visualized with one fixed 2nd–98th percentile normalization over the whole clip and the same colormap, so colors are not a shared metric scale.

**Part 1b: Depth Anything V2-Small, frame-wise.**
DA V2-S runs independently on every frame (input size 518, FP32). Within a frame, the predictions are sharp: truss members, rails and the train are cleanly separated from the sky, which the model assigns zero disparity (Fig. X). Over time, however, the output flickers. Each frame's scale and shift are inferred independently, so the same structure gets brighter or darker between consecutive frames even though it barely moves. In an x–t slice through the train (Fig. Y, bottom) the train appears as horizontal bands instead of a smooth gradient. After optical-flow compensation, the residual between frames 43 and 44 covers the whole track and train, not just their edges (Fig. Y, top), which points to a global rescaling rather than local errors. Measured as the median ratio between consecutive flow-aligned frames, the global scale changes by 1.4% per frame on average. It exceeds 2% in 19 of 69 pairs and peaks at 6.0% (frames 43→44) and 5.2% (frames 41→42). The direction flips 40 times: the change is a zero-mean jitter, not a drift, as expected when every frame picks its own scale and shift.

**Part 2b: Video Depth Anything-Small vs. frame-wise DA V2-S.**
VDA-S processes the same frames offline with the repository's default FP16 and input size 518 (four 32-frame windows). The official `run.py` with the project's flags (`--encoder vits --max_len 100 --max_res 640 --save_npz`) gives bitwise-identical depth to our script. Single frames look almost the same as DA V2-S (Fig. X), consistent with VDA reusing the DA V2 encoder. Temporally, VDA-S is much more consistent (Table N). The flow-compensated temporal error is 2.0× lower (0.0145 vs. 0.0284 in clip-normalized units) and lower on all 69 consecutive pairs. The mean frame-to-frame scale change drops from 1.4% to 0.3%, the maximum from 6.0% to 1.4%, and no pair exceeds 2%. In the x–t slice the train region changes smoothly as the train moves away.

VDA-S is not free of artifacts. Its three largest scale changes (frames 28→29, 29→30 and 68→69, about 1.3–1.4% each) all fall inside the cross-fade zones where the upstream code blends consecutive windows (frames 24–31, 46–53, 68–69; Fig. Z, shaded). Its flow-compensated error is slightly higher there than elsewhere (0.0151 vs. 0.0143). The median foreground value also dips gradually from 0.705 at frame 20 to 0.631 at frame 34 and recovers by frame 46. The dip starts in the first cross-fade zone, but this curve also contains real scene motion, so we treat the link to window stitching as a hypothesis to test in Part 3, not a finding. No depth ground truth exists for this clip, so these results show that VDA-S is more temporally consistent, not that it is more accurate.

**Why a naive frame-difference metric fails here.**
The mean absolute difference between consecutive normalized maps is almost identical for the two methods (0.1035 vs. 0.1027). On this clip the metric is dominated by real motion: about three quarters of each frame is sky, predicted as zero by both models, and thin structures sweeping across it cause large zero-to-high changes unrelated to flicker. We therefore compensate motion with optical flow before measuring consistency.

**Metric definition (Method / Experiments).**
For each consecutive pair (t, t+1) we compute Farnebäck optical flow on the RGB frames and warp the prediction of frame t+1 to frame t. We keep pixels that:

- are foreground (normalized value > 0.05) in both methods;
- stay inside the image after warping;
- survive a 3×3 erosion;
- pass a forward–backward consistency check (< 2 px).

The same pixels are used for both methods (median 43,549 per pair, 38,916–53,203). On these pixels we report the mean absolute residual |v_t − W(v_{t+1})| and, as the global scale change, the median ratio W(v_{t+1}) / v_t. Limitations: flow is least reliable on thin structures, and genuine depth change between frames is included in the residual; it is small at 24 fps and identical for both methods.

**Difficult cases (Fig. W).**
(1) *Thin truss and rail ties* (frame 31). Members a few pixels wide bleed into the sky at their edges, and flow is unreliable there: only 71% of the foreground pixels in the crop pass the consistency check. Inside the crop, DA V2-S has twice the temporal residual of VDA-S (0.028 vs. 0.014).
(2) *Train and riders* (frame 10). Both models predict the riders as one smooth blob without individual heads or seats. This is where frame-wise flicker is most visible: the residual is 0.041 for DA V2-S vs. 0.010 for VDA-S.
(3) *Far content entering* (frame 67). Distant trees and a tower enter at the bottom-right late in the clip. Both models place them barely above the sky (median normalized value 0.11 vs. ≈ 0.7 for the coaster), and the flag pole is only a faint line, so the far end of the depth range is compressed. This is a limitation of relative inverse depth plus a clip-level normalization fitted mostly to near structure, and video modeling does not fix it.

**Table N（草稿）**

| Metric (clip-normalized, ↓) | DA V2-S (frame-wise) | VDA-S (video) |
| --- | --- | --- |
| Flow-compensated temporal error | 0.0284 | **0.0145** |
| Mean frame-to-frame scale change | 1.4% | **0.3%** |
| Max frame-to-frame scale change | 6.0% (frames 43→44) | **1.4%** (frames 68→69) |
| Pairs with scale change > 2% | 19 / 69 | **0 / 69** |
| Consecutive pairs where more stable | 0 / 69 | **69 / 69** |
| Naive mean \|Δv\| (motion-dominated) | 0.1035 | 0.1027 |

**Table E（效率，草稿）** — RTX A6000, input size 518 (tensor 518×924), 640×360 frames.

| Method | Precision | End-to-end ms/frame | Peak GPU MB | GPU forward only |
| --- | --- | --- | --- | --- |
| DA V2-S | FP32 | 63.4 | 375 | 28.1 ms/frame |
| VDA-S | FP16 | 83.3 | 2689 | 239 ms per 32-frame window (10.9 ms per new frame) |
| VDA-S | FP32 | 107.1 | 3867 | 802 ms per window |

Footnote for the table: DA V2-S end-to-end is the per-frame `infer_image` call, of which about 44 ms is CPU preprocessing. VDA-S end-to-end is the whole clip through `infer_video_depth` divided by 70 frames, so it is throughput, not per-frame latency.

## 二、每个结论的证据来源

文件都在 `results/figures/part1b_2b_davis/`（简称 F）和 `results/logs/`（简称 L）。pair t 表示第 t 帧与第 t+1 帧。

| 结论 | 证据文件 | 看哪里 / 怎么得到 |
| --- | --- | --- |
| 片段 70 帧，640×360，24 fps；官方读帧与我们相同 | L `p2b_vda_official_check.json` | `frames.official_shape` = `ours_shape` = [70, 360, 640, 3]，`identical: true`，`fps: 24.0` |
| 官方 `run.py` 与 `run_vda.py` 输出逐位相同 | L `p2b_vda_official_check.json` | `official_vs_ours.bitwise_equal: true`，`max_abs_diff: 0`；两次自跑也逐位相同（`noise_floor_...`） |
| 两者单帧都很锐利，天空为 0 | F `frames.png` | 第 #10、#31、#32、#54、#66 帧，中间列与右列 |
| DA V2 逐帧闪烁：同一物体亮度逐帧跳 | F `flicker_qualitative.png` 下排 | 车厢 x–t 放大（y=180，x 270–420，第 30–69 帧）：DA V2 是横向色带，VDA 是平滑渐变，两者同一对比度拉伸 |
| DA V2 的闪烁是整体尺度变化 | F `flicker_qualitative.png` 上排 | pair 43 的残差：DA V2 整条轨道和车厢都亮，VDA 只在边缘有亮点 |
| 光流补偿误差 0.0284 对 0.0145，约 2.0 倍 | F `flicker_metrics.json` | `flow_warped_abs_err_mean`；`_flow_err_ratio_DA2_over_VDA` = 1.95 |
| 69/69 帧对 VDA 误差更低 | 同上 | `_pairs_where_VDA_lower_err_count` = [69, 69] |
| 平均尺度变化 1.4% 对 0.3% | 同上 | `scale_jump_mean_pct` = 1.42 / 0.31 |
| DA V2 最大 6.0%（43→44），其次 5.2%（41→42） | 同上 | `scale_jump_max_pct` = 5.95；`scale_jump_top3_pairs` = [43, 41, 2]，log = −0.058、−0.050；exp(0.050) − 1 ≈ 5.2% |
| DA V2 19 对 > 2%，VDA 0 对 | 同上 | `pairs_scale_jump_gt_2pct` = 19 / 0 |
| DA V2 是无方向抖动 | 同上；F `flicker_analysis.png` 中图 | `scale_jump_sign_changes` = 40（VDA 14）；蓝线围绕 0 上下交替 |
| VDA 最大 1.4%（68→69），前三都在交叉渐变区 | 同上；F `flicker_analysis.png` 灰色区域 | VDA `scale_jump_top3_pairs` = [68, 28, 29]，分别对应第 69、29、30 帧，落在 `_blend_zones` 68–69、24–31 内 |
| 渐变区来自上游拼接代码 | `external/Video-Depth-Anything/video_depth_anything/video_depth.py` | `INFER_LEN=32, OVERLAP=10, INTERP_LEN=8`；70 帧对应窗口起点 0/22/44/66，渐变区 24–31、46–53、68–69（`part1b-2b-workflow.md` 2.5 节） |
| VDA 渐变区误差略高 | F `flicker_metrics.json` | VDA `flow_err_blend_pairs_mean` = 0.0151，`flow_err_other_pairs_mean` = 0.0143（DA V2 两者几乎相同：0.0283 / 0.0284） |
| VDA 中位数第 20→34 帧下降，第 46 帧回升 | F `flicker_curves.json`；F `flicker_analysis.png` 下图 | VDA `median_fg[20]` = 0.705，`[34]` = 0.631（`median_curve_min` = [0.631, 34]），`[46]` = 0.671 |
| 有效像素数 | F `flicker_metrics.json`、`flicker_curves.json` | `_valid_px_per_pair_median` = 43549；`valid_px` 最小 38916，最大 53203 |
| 朴素指标失效 | F `metrics.json`、`temporal_change.png` | 0.1035 对 0.1027，两条曲线几乎重合 |
| 三个困难案例的数字 | F `hard_cases.json`、`hard_cases.png` | 细桁架 `valid_fraction_of_fg` = 0.71，残差 0.028 / 0.014；车厢残差 0.041 / 0.010；远景中位值 0.110 / 0.107 |
| 效率 | L `p1b_da2s_timing.json`、`p2b_vdas_timing.json`、`p2b_vdas_timing_fp32.json`、`p1b2b_forward_bench.json` | `latency_ms_mean` / `latency_ms_per_frame`、`peak_gpu_mem_mb`；`da2_cpu_preprocess_ms` = 43.8；`vda_forward_ms_per_window_fp16_autocast` = 239 |

## 三、注意事项

- 措辞只用 "more temporally consistent / more stable"，不要说 "more accurate"。
- VDA 中位数下凹“始于渐变区”只是吻合，曲线里含真实运动；报告里写成假设，Part 3 改进 B 用受控实验检验。
- “edge instability”：两种方法的残差都集中在边缘，但混有 Farnebäck 在细杆件上的误差，不宜单独下强结论。
- `*_approx` 文件（mp4 反推的近似版）已从仓库删除，报告不引用；需要时可用 `--source video` 重新生成。
- 报告里写明：所需片段共 70 帧，全部使用（PDF 写的是前 100 帧）。
