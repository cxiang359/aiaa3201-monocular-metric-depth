# Part 2A：Single-Image Metric Depth

使用官方 UniDepthV2 ViT-S，仅输入 RGB，在 NYU Depth V2 官方 654 张测试图上评价未经尺度对齐的米制深度。没有训练或微调，不传真值相机内参或深度。视频深度属于另一成员的 `part1b-2b` 分支。

## 正式结果

2026-11-11 完成全部 654 张测试，独立核验通过：

| 指标 | 逐图等权结果 |
| --- | ---: |
| AbsRel ↓ | 0.110923 |
| RMSE ↓ | 0.393103 m |
| δ1 ↑ | 89.4488% |
| CPU 平均推理时间 | 1119.36 ms/张 |

- [中文分析及三个困难案例](ANALYSIS.md)
- [完整结果报告](results/nyu_test_vits/REPORT.md)
- [654 张逐图指标](results/nyu_test_vits/metrics.csv)
- [已有独立核验记录](results/nyu_test_vits/verification.json)
- [发布范围与原始结果说明](../../PUBLICATION.md)

## 新电脑准备环境

以下命令均在仓库的 `part2/metric_depth/` 内执行。建议独立 Python 3.11 环境；GPU 环境先按驱动安装适配的 PyTorch。上游主要支持 Linux，本机历史结果使用 Windows CPU，不代表上游全面支持 Windows。

```powershell
git clone https://github.com/lpiccinelli-eth/UniDepth.git vendor/UniDepth
git -C vendor/UniDepth checkout 8d8cfe4c7ee15297099983607febf0d4f32eb3d6
python -m pip install -r requirements-inference.txt
python prepare_assets.py --workers 8
python -m unittest -v test_part2
```

下载 labeled MAT 约 2.97 GB、官方 splits 和模型权重，支持续传。data/、weights/、vendor/ 为本机目录，不随仓库提交。

默认模型 revision 固定为本次 baseline 的 `038c238f06c87b6c2f5b3749fd51fbf442b1f218`。本分支只为下载器补充了固定 revision 参数，原实验的推理与评价源文件保持原字节；若主动改用其他 revision，需新建结果目录，不续接旧实验。

- [数据来源与校验值](provenance/asset_manifest.json)
- [权重来源、revision 与校验值](provenance/weights_manifest.json)
- [原实验依赖版本](logs/dependency_versions.json)

模型通过 runtime.py 引入 vendor/UniDepth；可选本机依赖目录 vendor/python 也被支持，但新机器可以直接在独立环境安装。只做本课程推理和三个深度指标，不需要编译训练用 CUDA 扩展或点云 KNN 扩展。

## 运行与独立核验

新电脑明确指定设备并使用新目录，避免与已发布实验混合：

```powershell
python -u run_part2.py --device cpu --output results/nyu_cpu_new
python verify_part2.py --output results/nyu_cpu_new
```

主推理程序会生成该目录的 REPORT.md 和 REPORT.html。summarize_results.py 是历史结果的中文整理工具，固定读取 results/nyu_test_vits，不接受 --output；自定义新目录可直接阅读其 REPORT.md。

中断后保持全部参数和目录一致，再加 `--resume`。先试三张可加 `--max-new 3`，这种结果是 partial，不能作为正式分数。

GPU 可使用 `--device cuda --output results/nyu_cuda_new`。Windows 一键脚本为 `./run_all.ps1`；CPU/GPU 或配置变化后始终新建目录。

## 模型与评价协议

- 官方 UniDepthV2 ViT-S，约 34.18M 参数；from_pretrained 本地加载，strict=True。
- 仅传 uint8 CHW RGB，归一化、padding、缩放由官方 infer 完成。
- 原实验 CPU FP32、8 线程、resolution_level=0；实际编码器输入 434×574，输出恢复 480×640。CUDA 使用官方 FP16 autocast。
- trainNdxs 795 张用于开发，testNdxs 654 张用于正式测试；MATLAB 索引只减一次 1。
- RGB 读取 images[i].transpose(2,1,0)，真值为 depths[i].T（filled depths，米），不用 rawDepths。
- 输出 480×640；必要时双线性插值，不对米制数值乘尺寸倍率。
- Python crop [45:471,41:601]；真值有限且 0.001<GT<10。
- 预测仅评价时 clip 到 [0.001,10]，原始 float32 米制预测保留不变。
- 每图计算 AbsRel、RMSE 和严格 ratio<1.25 的 δ1，再对 654 张图等权平均。
- 不做 median alignment、逐图尺度拟合或测试集校准。

本结果属于课程协议及当前分辨率下的 baseline，不宣称完全重现论文表格。

## 效率与证据范围

计时包括官方 infer 的预处理、前向计算与尺寸恢复，不包括读取、指标、保存和绘图；每次加载预热一次。CPU 没有峰值 GPU 显存，记 N/A。最大记录 RSS 832.8 MiB 是进程内存，不能当作显存。

原完整本地运行通过 9 项测试和 654 张原始预测的独立复算。历史核验记录随仓库发布，但 654 份 predictions/*.npy 和 samples/*.json 不上传。因此克隆后可读指标与图，完整复核需要重新生成预测。REPORT.html 由推理脚本在本地生成，核验器会检查它；图片内嵌 HTML 不上传。

GitHub 保留第一张、中位案例和三个困难图像的比较面板。困难图像为书店 88、书房 469、卧室 517；选取仅用于事后说明局限，不用于调参。

## 与 Part 3 衔接和引用

这是未经改进的固定 baseline。如果后续做全局尺度校准，只能在训练／开发集上拟合倍率，再固定用于测试；不得覆盖原始预测。边界细化也须在同一测试协议下比较。

引用见 [REFERENCES.bib](REFERENCES.bib)。官方源码和权重从上游获取，按其 CC BY-NC 4.0 许可使用，不在此重新分发。
