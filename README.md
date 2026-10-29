# aiaa3201-monocular-metric-depth
AIAA 3201 Project 3: Monocular Metric Depth Estimation and Video Consistency

两人课程项目：从双目几何与逐帧单目基线出发，复现度量深度和视频深度模型，再完成一个轻量改进与应用 Demo。当前仅有目录骨架，尚未实现算法或产生实验结果。

## 目录用途
保持扁平结构，先按文件名区分方法，需要时再增加子目录。
| 目录 | 放什么 |
| --- | --- |
| `part1/` | Classical Stereo 与逐帧单目基线代码，建议文件名 `stereo.py`、`monocular.py` |
| `part2/` | Single-Image Metric Depth 与 Video Depth 复现入口，建议 `metric_depth.py`、`video_depth.py` |
| `part3/` | 一个轻量改进、消融实验与应用 Demo 代码 |
| `configs/` | 可共享的运行参数与路径模板；本机配置不提交 |
| `data/` | 本机数据与权重；内容不入库，仅保留占位文件 |
| `results/` | 小型指标表、精选图片及实验记录；原始深度、视频和日志仅留本机 |
| `report/` | CVPR 报告源码、参考文献与展示材料；最终视频/ZIP 留本机或课程平台 |

建议文件名均是后续约定，当前没有对应可运行脚本。

## 分工
| 部分 | cxiang359（你） | 另一位组员 |
| --- | --- | --- |
| Part 1 | **Classical Stereo**：Middlebury、StereoBM/SGBM、视差转深度、左右一致性与误差分析 | **Monocular Frame Baseline**：逐帧 MiDaS/DPT 或 Depth Anything V2-Small、静态闪烁与移动尺度漂移 |
| Part 2 | **Single-Image Metric Depth**：UniDepthV2 或 Depth Pro，NYUv2 度量评估、内参/焦距与边界分析 | **Video Depth**：Video Depth Anything Small/Base，与逐帧 Depth Anything V2 比较 |
| Part 3 | 共同选定一个可测量问题，实现轻量改进、对照与消融，完成应用 Demo | 共同完成，并交叉复现对方模块 |

两人分别开发自己的模块，通过分支与 Pull Request 互审；报告、共享评价协议与最终提交共同负责。

## Timeline
以下按用户提供的 Notion 内容记录；`19.`、`20.` 按 19:00、20:00 理解，日期不额外推定年份。
| 日期 | 计划 |
| --- | --- |
| 11/3 19:00 | Literature Share |
| 11/8 | 尽量做完 Part 1 |
| 11/10 19:00 | 开会同步 Part 1 情况 |
| 11/13 20:00 | 开会讨论 presentation video |
| 11/15 | 做完 Part 2 |
| 11/17 | 提交 presentation video |

11/17 是小组视频提交节点；课程提醒另列 11/18、11/25 课堂展示和 12/11 23:59 最终 Canvas 提交，具体排期与截止时区以 Canvas 为准。Part 3、报告与最终 Demo 在 Part 2 后继续推进。
技术 PDF 标注 **Spring 2027**，与此前课程提醒的 11–12 月安排存在学期差异；技术要求按本次 PDF，日程按用户提供计划和课程实际通知核对。

## 数据与评价
- **Middlebury Stereo v3**：quarter resolution 输入与左视图真值，使用 15 对训练图和对应标定；计算视差、深度、无效区域及误差。
- **NYU Depth V2**：下载 2.77 GB labeled MATLAB 文件，使用标准评估划分及 Depth Anything V2 metric-depth 工具协议，不下载 428 GB raw 集。
- **自录视频**：至少覆盖静态相机与移动相机，保存浮点深度；视频模型优先 Small/FP16。帧基线与视频模型使用相同帧、分辨率和固定可视化范围。
- **必报指标**：AbsRel、SqRel、RMSE、RMSE-log、δ1/δ2/δ3；深度边界 precision/recall 或合理边界误差；有效非遮挡像素上的光流对齐时序误差、逐帧 median-scale drift；FPS、单帧延迟、峰值显存、模型大小与输入分辨率。
- 相对深度只在基准评价中按协议做 median 或 scale-and-shift 对齐，不称作 metric depth。Part 2 单图度量模型**不做测试集尺度对齐**。
- 公平对比固定划分、预处理、分辨率与评价代码；记录所有测试时修正。展示镜面、玻璃、天空、运动模糊、细物体等失败案例。
- Part 3 可选流引导融合、尺度校准、边界细化、不确定性平滑、部署或可测量应用；选一个即可，即使主指标未提升也必须提供受控消融与原因分析。

## 环境与运行
1. 克隆：`git clone https://github.com/cxiang359/aiaa3201-monocular-metric-depth.git`。
2. 按所选模型官方仓库分别准备环境，记录 Python/依赖、GPU/CUDA、上游 commit 与 checkpoint。当前未选定模型、未验证安装，不提供虚构的运行命令。
3. 复制 `configs/paths.example.yaml` 为 `configs/paths.local.yaml`，填写本机路径。数据建议放 `data/raw/`，权重放 `data/weights/`；本机配置与资源均被忽略。
4. 实现后在本节补齐经过验证的依赖安装、数据准备、推理、评估和训练/微调命令（若有；本项目不要求从头训练基础模型），以及代表性结果。
5. 每次运行保存 `results/<run_id>.md`：负责人、代码 commit、官方仓库与权重版本/来源/许可/校验值、依赖环境、配置、数据划分、单位/掩码/裁剪/尺度协议、完整命令、指标、耗时/显存和失败案例。小型指标/图同样带 run_id。
6. 原始浮点深度放 `results/raw/`，视频放 `results/videos/`，日志放 `results/logs/`，均不提交。提交前检查文件大小与隐私，不上传大型数据集、模型权重或打包文件。

## 课程提交清单
技术依据：用户提供的《Project 3: Monocular Metric Depth Estimation & Video Consistency — Spring 2027》，第 2–5 页；日程依据用户提供的 Notion 内容及此前课程提醒。
- [ ] Part 1、Part 2、Part 3 全部完成；有定量表、定性图、消融与分类失败案例。
- [ ] **报告**：CVPR LaTeX 模板，6–8 页（不含参考文献），camera-ready；摘要末尾放公开 GitHub 链接；包含引言、相关工作、方法/流程图、实验、结论/局限。
- [ ] **引用**：覆盖 Recommended Reading List，并正确引用数据、模型、外部仓库和借用组件；核对 PDF Section 4 与完整 References。
- [ ] **arXiv**：本次技术 PDF 明确要求上传最终报告，Canvas 提交首页注明 arXiv ID。此前提醒称预印本可选，需向 TA 确认差异，当前先按技术 PDF 准备。
- [ ] **公开代码**：README 包含依赖、数据准备、权重来源、真实运行/评价命令及代表性结果；不将原始代码文件直接交 Canvas。仓库此前检查为 private，最终需公开。
- [ ] **Depth Demo & Video（必交）**：所有必需片段的处理后视频，以及可运行或录制的应用 Demo；同时展示 RGB、预测深度和应用输出（点云、测距等）；打包为 `depth results.zip`。
- [ ] **课堂展示**：至少完成 Part 1，最好 Part 2；4 分钟录像介绍方法、实验与下一步计划。
- [ ] 最终按 Canvas 实际要求提交报告、结果包及 GitHub 链接，检查可访问性与回执。

[原 Notion 页面](https://app.notion.com/p/Part-1-2-Timeline-4086c22dbdb8805dbbf6eaeef73772aa?source=copy_link)的内容已由用户在聊天中补全；本 README 不依赖页面访问权限。
