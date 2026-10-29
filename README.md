# Monocular Metric Depth Estimation & Video Consistency
AIAA 3201 · Project 3

从双目几何和逐帧单目基线出发，复现单图度量深度与视频深度模型，探索轻量改进，并完成深度应用 Demo。

## 成员与分工

| 成员 | Part 1 | Part 2 |
| --- | --- | --- |
| cxiang359 · cxiang359@connect.hkuct-gz.edu.cn | Classical Stereo | Single-Image Metric Depth |
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
- **自录视频**：覆盖静态与移动相机场景；保存浮点深度，采用固定可视化范围。
- **指标**：AbsRel、SqRel、RMSE、RMSE-log、δ1/δ2/δ3；边界精度；光流对齐后的非遮挡区域时序误差与逐帧尺度漂移；FPS、延迟、峰值显存、模型大小与输入分辨率。

比较方法时统一数据划分、预处理、输入分辨率、评价代码及视频帧。相对深度仅在基准评价中按协议对齐；单图 metric 模型不做测试集尺度对齐。展示镜面、玻璃、天空、运动模糊及细物体等失败案例。

## 环境与实验管理

当前处于开发准备阶段，安装、推理与评价命令将在模块实现后补充。

1. 克隆仓库：`git clone https://github.com/cxiang359/aiaa3201-monocular-metric-depth.git`。
2. 按所选模型官方实现准备环境，记录 Python、依赖版本、GPU/CUDA、上游 commit 和 checkpoint。
3. 复制 `configs/paths.example.yaml` 为 `configs/paths.local.yaml`，设置本机路径；数据放 `data/raw/`，权重放 `data/weights/`。
4. 每次实验用唯一 run_id 关联代码版本、配置、数据划分、权重来源、完整命令、评价协议、指标与耗时/显存，记录在 `results/<run_id>.md`。
5. 原始深度、视频与日志分别放 `results/raw/`、`results/videos/`、`results/logs/`，仅留本机；小型指标与精选图片可提交。

不上传大型数据集、模型权重、凭据或结果压缩包。

## 提交物

- **报告**：CVPR LaTeX 格式，6–8 页（不含参考文献），camera-ready；包含引言、相关工作、方法、定量/定性实验、消融、失败案例与结论。摘要末尾注明公开 GitHub 链接，引用课程推荐论文、数据集、模型及外部代码。
- **arXiv**：按项目技术说明上传最终报告，在 Canvas 提交首页注明 arXiv ID；发布要求以课程最终通知为准。
- **代码**：公开 GitHub 仓库，README 提供依赖、数据准备、权重来源、运行与评价命令及代表性结果。
- **Depth Demo & Video**：提交必需片段的处理后视频及可运行或录制的应用 Demo，展示 RGB、预测深度和应用输出，打包为 `depth results.zip`。
- **进度展示**：4 分钟录像，介绍方法、实验与下一步计划；至少完成 Part 1，争取完成 Part 2。
