# Monocular Metric Depth Estimation & Video Consistency
AIAA 3201 · Project 3

从双目几何和逐帧单目基线出发，复现单图度量深度与视频深度模型，探索轻量改进，并完成深度应用 Demo。

## Part 1A / Part 2A：已完成实验

本分支 `part1a-2a` 发布双目几何与单图度量深度成果。视频分支为 [`part1b-2b`](https://github.com/cxiang359/aiaa3201-monocular-metric-depth/tree/part1b-2b)，尚未合并。

| 模块 | 说明 | 正式结果 |
| --- | --- | --- |
| [Part 1A：Classical Stereo](part1/classical_stereo/README.md) | 固定 StereoSGBM，Middlebury 15 对 trainingQ | MAE 1.9743 px；有效比例 75.98% |
| [Part 2A：Single-Image Metric Depth](part2/metric_depth/README.md) | UniDepthV2 ViT-S，NYU 654 张测试，RGB-only，无测试尺度对齐 | AbsRel 0.110923；RMSE 0.393103 m；δ1 89.4488% |

结果讨论：[双目分析](part1/classical_stereo/ANALYSIS.md)、[单图深度分析](part2/metric_depth/ANALYSIS.md)。上传范围、目录结构及未上传内容见 [PUBLICATION.md](PUBLICATION.md)。两部分均为 CPU 实验；与其他硬件结果比较时须报告设置。

## 模块目录与合并

```text
part1/
  classical_stereo/   # Part 1A，本分支：代码、文档、配置、结果
  mono_baseline/      # Part 1B，来自 part1b-2b 分支
part2/
  metric_depth/      # Part 2A，本分支：代码、文档、配置、结果
  video_depth/       # Part 2B，来自 part1b-2b 分支
```

上图为合并后的布局；本分支只包含两个 A 模块，B 模块仍在另一分支。运行命令前进入对应模块目录。A 模块的小型结果随模块保存，B 模块已有的根目录 results/ 布局保留。

## 成员与分工

| 成员 | Part 1 | Part 2 |
| --- | --- | --- |
| cxiang359@connect.hkuct-gz.edu.cn | Classical Stereo | Single-Image Metric Depth |
| zhaojinghan06@gmail.com | Monocular Frame Baseline | Video Depth |

Part 3、评价协议、报告和最终 Demo 共同完成；通过分支与 Pull Request 交叉审核。

## 项目任务

- **Part 1 — Baseline**：使用 StereoBM/SGBM 计算视差、转换度量深度并分析左右一致性；使用 MiDaS/DPT 或 Depth Anything V2-Small 建立逐帧单目基线，分析闪烁和尺度漂移。
- **Part 2 — Reproduction**：选择 UniDepthV2 或 Depth Pro 复现单图度量深度；使用 Video Depth Anything Small/Base，与逐帧 Depth Anything V2 比较时序稳定性。优先使用 Small 和 FP16。
- **Part 3 — Exploration**：选择尺度校准、时序融合、边界细化等一个可测量方向，实现轻量改进、受控消融和失败案例分析，并完成点云、测距或深度感知应用。

## 目录规划

| 目录 | 用途 |
| --- | --- |
| `part1/` | 双目与逐帧单目基线代码 |
| `part2/` | 单图度量深度与视频深度复现代码 |
| `part3/` | 改进方法、消融与应用 Demo |
| `configs/` | 运行参数、路径模板及方法配置 |
| `data/` | 本机数据集与模型权重，不上传 GitHub |
| `results/` | 小型指标表、精选图片与实验记录 |
| `report/` | 报告源码、参考文献和展示材料 |

空目录使用 `.gitkeep` 保留；加入实际文件后可移除占位文件。

## Timeline

| 日期 | 计划 |
| --- | --- |
| 11/3 19:00 | Literature Share |
| 11/8 | 尽量完成 Part 1 |
| 11/10 19:00 | 开会同步 Part 1 情况 |
| 11/13 20:00 | 开会讨论 presentation video |
| 11/15 | 完成 Part 2 |
| 11/17 | 提交 presentation video |
| 11/18、11/25 | 课堂进度展示，具体日期以课程排期为准 |
| 12/11 23:59 | 最终 Canvas 提交，截止时间以课程通知为准 |

## 数据与评价

- **Middlebury Stereo v3**：quarter-resolution 数据及左视图真值，使用 15 对训练图和标定文件进行双目实验。
- **NYU Depth V2**：使用 2.77 GB labeled MATLAB 数据、标准评估划分和 Depth Anything V2 metric-depth 评价协议。
- **指定视频**：使用课程要求的 davis_rollercoaster.mp4；不要求额外自录视频。保存浮点深度，采用整段固定可视化范围。
- **必需指标**：NYU 的 AbsRel、RMSE、δ1；双目的视差 MAE 与有效预测比例；推理时间和资源设置。视频定性比较为必需，时序误差、尺度漂移及边界精度为可选分析。

比较方法时统一数据划分、预处理、输入分辨率、评价代码及视频帧。视频相对深度采用整段固定归一化，颜色不代表共享米制尺度；单图 metric 模型不做测试集尺度对齐。展示实际数据中出现的困难案例。

## 环境与实验管理

Part 1A / Part 2A 已完成，安装、数据准备、运行和核验命令见各模块 README。每个模块完整放在自己的子目录中；模块自身使用 `part1/classical_stereo/data/`、`part2/metric_depth/data/`、`part2/metric_depth/weights/` 与各自 `results/`；脚本以所在模块为根目录定位文件。两位成员的模块可在各自 Part 目录下并列合并。

1. 克隆仓库：`git clone https://github.com/cxiang359/aiaa3201-monocular-metric-depth.git`。
2. 按所选模型官方实现准备环境，记录 Python、依赖版本、GPU/CUDA、上游 commit 和 checkpoint。
3. 复制 `configs/paths.example.yaml` 为 `configs/paths.local.yaml`，设置本机路径；数据放 `data/raw/`，权重放 `data/weights/`。
4. 每次实验用唯一 run_id 关联代码版本、配置、数据划分、权重来源、完整命令、评价协议、指标与耗时/显存，记录在 `results/<run_id>.md`。
5. 原始深度、视频与日志分别放 `results/raw/`、`results/videos/`、`results/logs/`，仅留本机；小型指标与精选图片可提交。

不上传大型数据集、模型权重、凭据或结果压缩包。

## 提交物

- **报告**：CVPR LaTeX 格式，6–8 页（不含参考文献），camera-ready；包含引言、相关工作、方法、定量/定性实验、消融、失败案例与结论。摘要末尾注明公开 GitHub 链接，引用课程推荐论文、数据集、模型及外部代码。
- **arXiv（可选）**：课程 PDF 将上传列为可选；若上传，在 Canvas 提交首页注明 arXiv ID，最终以课程通知为准。
- **代码**：公开 GitHub 仓库，README 提供依赖、数据准备、权重来源、运行与评价命令及代表性结果。
- **Depth Demo & Video**：提交必需片段的处理后视频及可运行或录制的应用 Demo，展示 RGB、预测深度和应用输出，打包为 `depth_results.zip`。
- **进度展示**：4 分钟录像，介绍方法、实验与下一步计划；至少完成 Part 1，争取完成 Part 2。
