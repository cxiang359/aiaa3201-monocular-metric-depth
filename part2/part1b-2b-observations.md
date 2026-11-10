# Part 1b / 2b 观察段落（报告草稿）+ 证据对照

2026-11-10。数字来自对比视频反解色表的近似值（`*_approx`），交稿前用服务器原始 npz 运行 `python part2/video_depth/temporal_flicker.py --source raw` 后替换。图号 Fig. X/Y/Z、Table N 是占位符。

## 一、报告英文段落（可直接改写进 Experiments）

**Part 1b: Depth Anything V2-Small, frame-wise (relative depth).**
We run DA V2-S independently on every frame of the required clip. The provided `davis_rollercoaster.mp4` contains 70 frames, all of which are used (640×360, input size 518, FP32). Outputs are affine-invariant relative inverse depth; they are shown without metric conversion or ground-truth alignment, using one 2nd–98th percentile normalization computed over the whole clip. Within a single frame the predictions are sharp: thin truss members, rails and the train silhouette are cleanly separated from the sky, which the model assigns zero disparity (Fig. X). Over time, however, the output is unstable. Because each frame's scale and shift are inferred independently, the same structure changes brightness between consecutive frames although it barely moves. In an x–t slice through the train (Fig. Yb), the train region appears as horizontal bands instead of a smooth gradient. After optical-flow compensation, the residual between consecutive frames is spread along the whole track rather than confined to edges (Fig. Ya), which points to a global scale/shift change rather than local errors. The effect is strongest when new content enters the view: at frames 66–67, as trees and a distant tower appear in the lower right, the global scale of the already visible structure jumps by about 14% and falls back within two frames (Fig. Z, middle).

**Part 2b: Video Depth Anything-Small vs. frame-wise DA V2-S.**
VDA-S processes the same frames offline (FP16 autocast, input size 518, 32-frame windows; 4 windows for 70 frames), with its own clip-level normalization and the same colormap. Single-frame appearance is nearly indistinguishable from DA V2-S (Fig. X), consistent with VDA keeping the DA V2 encoder frozen. Temporally, VDA-S is far more consistent: the train region in the x–t slice changes smoothly as the train moves away (Fig. Yb), and the flow-compensated residual is concentrated at object boundaries (Fig. Ya). Quantitatively (Table N), the flow-compensated temporal error is 2.6× lower (0.0082 vs. 0.0215 in clip-normalized units), the mean frame-to-frame global scale change drops from 2.3% to 0.3%, and VDA-S is more stable on all 69 consecutive frame pairs. The scale jump at frames 66–67 does not occur in VDA-S (maximum 1.4% over the clip). We observe no discontinuities at the window overlaps (frames 22–31, 44–53, 66–69); the only visible drift is a small, gradual dip of the median foreground value around frames 28–37 (Fig. Z, bottom). Since the video has no depth ground truth, these results show that VDA-S is more temporally consistent, not that it is more accurate.

**Why a naive frame-difference metric fails here.**
The mean absolute difference between consecutive normalized maps is almost identical for the two methods (0.1035 vs. 0.1027), and the two curves follow each other closely. On this clip the metric is dominated by real motion: about three quarters of each frame is sky, predicted as zero by both models, and thin structures sweeping across it create large zero-to-high changes unrelated to flicker. We therefore compensate motion with optical flow before measuring consistency.

**Metric definition (Method / Experiments).**
For each consecutive pair (t, t+1) we compute Farnebäck optical flow on the RGB frames and warp the prediction of frame t+1 to frame t. We keep pixels that are foreground (normalized value > 0.05) in both methods, remain inside the image after warping, survive a 3×3 erosion, and pass a forward–backward consistency check (< 2 px); the same pixels are used for both methods (median 4,366 per pair). On these pixels we report the mean absolute residual |v_t − W(v_{t+1})| and, as a global scale change, the median ratio W(v_{t+1}) / v_t. Limitations: flow is least reliable on thin structures, and genuine depth change between frames is included in the residual; it is small at 24 fps and identical for both methods.

**Table N（草稿）**

| Metric (clip-normalized, ↓) | DA V2-S (frame-wise) | VDA-S (video) |
| --- | --- | --- |
| Flow-compensated temporal error | 0.0215 | **0.0082** |
| Mean frame-to-frame scale change | 2.3% | **0.3%** |
| Max frame-to-frame scale change | 14.3% (frames 66–67) | **1.4%** |
| Consecutive pairs where more stable | 0 / 69 | **69 / 69** |
| Naive mean \|Δv\| (motion-dominated) | 0.1035 | 0.1027 |

## 二、每个结论的证据来源

文件都在 `results/figures/part1b_2b_davis/`。

| 结论 | 证据文件 | 看哪里 / 怎么得到 |
| --- | --- | --- |
| 两者单帧都很锐利，细杆件、车厢轮廓清楚，天空为 0 | `frames.png` | 第 #10、#31、#32、#54、#66 帧中间列和右列；天空全黑 |
| 两者单帧几乎一样 | `frames.png` | 同一行中间列和右列逐行对比，结构和明暗分布基本一致 |
| DA V2 逐帧闪烁：同一物体亮度逐帧跳 | `flicker_qualitative_approx.png` 下排 | 车厢区域 x-t 放大（x≈280–420，第 30–69 帧）：DA V2 是一条条横向色带，VDA 是平滑渐变。两张用同一对比度拉伸（0.4–1.0） |
| DA V2 的闪烁是整体尺度变化，不只是边缘 | `flicker_qualitative_approx.png` 上排 | t=22 的残差图：DA V2 整条左侧轨道都是均匀紫色，VDA 只在边缘有亮点 |
| 第 66–67 帧 DA V2 尺度跳约 14% | `flicker_analysis_approx.png` 中图；`flicker_metrics_approx.json` | 中图蓝线在第 66–67 帧冲到 +0.11、+0.13，随后回落到 −0.07、−0.08；json 里 `scale_jump_max_abs_log` = 0.1335，位置 66。exp(0.1335) − 1 ≈ 14.3%。新内容（树、塔）见 `frames.png` 的 #66 右下角 |
| 光流补偿误差 2.6 倍差距、69/69 帧对 | `flicker_analysis_approx.png` 上图；json | 上图橙线全程低于蓝线；json 中 0.0215 / 0.0082 = 2.6，`_pairs_where_VDA_lower_err` = 1.0 |
| 平均尺度变化 2.3% 对 0.3% | json | `scale_jump_mean_abs_log` = 0.0231 / 0.0031，换算 exp(x) − 1 |
| VDA 窗口接缝无跳变 | `flicker_analysis_approx.png` 三张图的灰色区域 | 灰色是重叠区（22–31、44–53、66–69），橙线在其中没有尖峰；VDA 最大跳变在第 39 帧，不在灰色区 |
| VDA 第 28–37 帧有小幅下凹 | `flicker_analysis_approx.png` 下图 | 橙线从约 0.76 缓降到约 0.71，再回到 0.76，变化是渐进的 |
| 旧指标失效 | `temporal_change.png`；`metrics.json` | 两条曲线几乎重合，0.1035 对 0.1027；天空占比由反解数据统计（前景约 24–27%） |
| VDA 的设置（FP16、518、4 个窗口） | `results/logs/p2b_vdas_v1.json`；HANDOFF | precision、input_size 字段；窗口数来自运行时进度条 4/4 |

## 三、注意事项

- 近似值的来源：并排视频是 8 位 mp4（有压缩），色表反解有误差；颜色偏离色表太远的像素（多在边缘）已剔除。两种方法受同样的影响，所以对比是公平的，但绝对数值要用原始 npz 重算。
- 措辞只用 "more temporally consistent / more stable"，不要说 "more accurate"。
- "edge instability" 这一项：两种方法的残差都集中在边缘，但这里混有光流在细杆件上的误差，不宜单独下强结论。
- 报告里写明：所需片段共 70 帧，全部使用（PDF 写的是前 100 帧）。
- 效率数字待在 GPU 空闲时重测，暂不写进段落。
