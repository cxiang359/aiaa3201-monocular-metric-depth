# aiaa3201-monocular-metric-depth
AIAA 3201 Project 3: Monocular Metric Depth Estimation and Video Consistency

## 当前状态与依据
本次建立协作目录与文档骨架，算法、评估脚本及实验结果尚未实现。
- [Notion Part 1/2 Timeline](https://app.notion.com/p/Part-1-2-Timeline-4086c22dbdb8805dbbf6eaeef73772aa?source=copy_link)：2026-10-29 读取返回 404/无权限，**尚未核实页面计划**。
- 已核读课程附件《AIAA 3201 Project Release》（project_AIAA3201_reminder.pdf）第 3–7 页。
- Project 3 技术 PDF 原件未从原对话恢复。下列技术任务来自原对话梳理，**需与原件复核**，不是已逐项核对的课程要求。
- 仓库目前为 private；课程要求最终公开 GitHub 代码，提交前需完成公开性检查。

## 目录
```text
part1/
  stereo/README.md          # A：双目基线
  monocular/README.md       # B：逐帧单目基线
part2/
  metric_depth/README.md    # A：单图度量深度复现
  video_depth/README.md     # B：视频深度复现
part3/README.md             # 共同研究改进、消融
common/README.md            # 共享数据接口、指标、可视化
configs/                    # 配置模板与本机配置说明
data/README.md              # 下载说明、划分清单；数据仅留本机
weights/README.md           # 权重来源和版本；权重仅留本机
experiments/                 # 小型运行清单与记录模板
results/                    # 可提交的小型指标与精选图片
docs/                       # 分工、暂定时间线、来源与待确认项
submission/                 # 报告、展示、最终交付检查清单
```

## 两人分工与三部分
A/B 是工作角色，实际姓名和 GitHub 用户名待两人填写；不是课程规定的分工。
| 部分 | A | B | 共同验收 |
| --- | --- | --- | --- |
| Part 1 | Middlebury v3、StereoBM/SGBM、视差转深度 | MiDaS/DPT 或 Depth Anything V2-Small 逐帧预测 | 基线、尺度约定、误差与视频闪烁分析 |
| Part 2 | UniDepthV2 或 Depth Pro、NYU Depth V2 评估 | Video Depth Anything Small/Base，与逐帧基线比较 | 同一划分、输入、评价协议及失败案例 |
| Part 3 | 初步负责改进算法 | 初步负责评估和消融 | 根据结果选题，验证假设并分析局限 |
共同维护配置、报告和引用；交叉复现对方至少一次运行。详见 [协作计划](docs/plan.md)。

## 启动与运行配置
1. 克隆仓库：`git clone https://github.com/cxiang359/aiaa3201-monocular-metric-depth.git`。
2. 按各模块选定的官方实现分别建立环境；记录 Python、依赖锁定版本、GPU/CUDA、上游仓库 commit。当前没有已验证的统一安装或推理命令。
3. 复制 `configs/paths.example.yaml` 为 `configs/paths.local.yaml`，填写本机数据、权重及输出路径；本机文件不会入库。
4. 依照 data/ 与 weights/ 中的规范准备资源。提交数据清单及校验信息，不提交原始资源。
5. 模块实现时在对应 README 补全真实安装、推理、评估命令及一次可复现记录；配置模板目前仅作约定，尚无程序读取。
6. 每次运行使用 [实验模板](experiments/run-template.md)，以唯一 run_id 关联代码 commit、配置、数据划分、权重及结果。

## 数据、实验与结果
- 本机数据放 `data/raw/`、`data/processed/`，仅在 `data/manifests/` 提交无敏感信息的样本 ID 和划分。
- 本机模型放 `weights/`，记录上游链接、版本、许可和 SHA256。
- 原始预测、视频和日志放 `results/raw/`、`results/videos/`、`results/logs/`（忽略）。
- 小型指标表放 `results/metrics/`，精选图放 `results/figures/`，以 run_id 命名并关联实验记录。
- 所有方法记录深度单位、无效掩码、分辨率、裁剪、深度范围与尺度对齐方式；单目相对深度与原生 metric 输出分开比较，禁止使用测试真值进行未声明的校准。
- 不上传大型数据集、权重或打包文件；提交前检查暂存文件大小与内容，忽略规则不能替代人工检查。

## 课程里程碑与提交
课程提醒 PDF 确认：11/18、11/25 课堂进度展示（本组日期待课程安排），至少完成 Part 1，最好完成 Part 2；每组提交 4 分钟展示录像。最终截止 12/11 23:59，时区以 Canvas 为准。
报告包括摘要/引言、相关工作、方法、定量表与定性图、结论；摘要末尾放 GitHub 链接，相关工作覆盖技术说明参考文献列表。
最终公开代码必需；Demo/项目页、预印本在提醒 PDF 中为可选。原对话提及技术 PDF 对 arXiv 的要求可能不同，需向 TA 核实。
详见 [提交清单](submission/README.md) 与 [来源/待确认项](docs/sources.md)。

## 协作方式
分别使用 `feat/a-...`、`feat/b-...` 分支，经另一人检查后合并；避免同时修改同一模块。文档与真实实验同步更新，不将占位文件视作实现完成。
