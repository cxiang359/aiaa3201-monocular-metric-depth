# Part 1A：Classical Stereo

固定 StereoSGBM 在 Middlebury Stereo v3 的 15 对 quarter-resolution trainingQ 上完成视差预测、米制深度转换、左右一致性诊断和失败分析。逐帧单目视频由另一成员负责，见 `part1b-2b` 分支。

## 正式结果

| 指标 | 每场景等权结果 |
| --- | ---: |
| 视差 MAE | 1.9743 px |
| 有效预测比例 | 75.98% |
| CPU 左匹配平均耗时 | 35.66 ms/对 |

完成于 2026-11-09。MAE 仅在有效预测与有效真值交集计算，覆盖率分母包含所有有限正真值像素，二者必须同时报告。不是 Middlebury 官方榜单分数。

- [中文分析及三组困难案例](ANALYSIS.md)
- [全场景报告](results/sgbm_final/REPORT.md)
- [15 场景指标](results/sgbm_final/metrics.csv)
- [已有独立核验记录](results/sgbm_final/verification.json)
- [发布范围与原始结果说明](../PUBLICATION.md)

## 新电脑运行

以下命令均在仓库的 `part1/` 内执行。在独立 Python 环境中安装依赖：

```powershell
python -m pip install -r requirements.txt
python -m unittest -v test_stereo
```

下载两个官方压缩包到本目录，保持原始文件名：

- [MiddEval3-data-Q.zip](https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-data-Q.zip)
- [MiddEval3-GT0-Q.zip](https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-GT0-Q.zip)

然后执行：

```powershell
python stereo.py prepare
python stereo.py run --lr-check
```

prepare 校验 ZIP CRC 并提取 15 个 trainingQ 场景。参考来源和 SHA-256 在 [provenance/data_manifest.json](provenance/data_manifest.json)；运行后本机 `data/manifest.json` 重新生成。

默认新结果位于 `results/sgbm_年月日_时分秒/`。对新结果生成失败图并核验（将 `<run>` 替换成实际目录名）：

```powershell
python make_failure_figures.py --output results/<run>
python verify_results.py --output results/<run>
```

Windows 也可运行 `./run_all.ps1`。只需测试一个场景时使用 `python stereo.py run --scenes Adirondack --lr-check`。

## 几何与协议

OpenCV 输出是放大 16 倍的定点视差，标定 baseline 为毫米：

```text
d_px = disparity_fixed16 / 16
Z_m = fx_px * baseline_mm / (d_px + doffs_px) / 1000
```

直接使用 quarter-resolution 标定，不再次缩放焦距。无效哨兵、搜索区间外预测及非正／非有限分母被排除，浮点无效值保存为 NaN。零视差在 doffs>0 时仍可能有效。

Middlebury PFM 的数据已是像素视差；scale 的符号只决定字节序，不应用其绝对值，且需要翻转行顺序。

固定配置见 [config.json](config.json)：SGBM_3WAY、blockSize=5、P1=200、P2=800、单线程；搜索范围仅由公开 calib.txt 的 ndisp 向上取整到 16 的倍数，不根据真值调参。未用非遮挡掩码排除遮挡。

额外 LR 诊断在 xR=round(xL-dL) 检查 abs(dL+dR)<=1 px，不改变主预测和评分掩码；黑色表示未检查。

## 效率、核验和原始输出

计时为一次预热后 3 次左匹配 compute 的中位数，再在场景间平均；不包括读写、灰度转换、反向匹配和绘图。CPU 的 GPU 显存为 N/A。硬件、版本、逐次计时见 run.json 和 scene_metrics.json。

原始完整运行的 6 项正确性测试和全场景独立复算通过，见 test_results.txt 和 verification.json。记录中的 110 张 PNG 指完整本地运行，本分支只保留总览、比较面板、失败图和元数据。

大批 arrays/*.npz、数据 ZIP、data/、生成 HTML 和无效旧实验不上传。克隆后可查看全部正式指标；若要重新核验原始浮点结果，需先重新生成该次实验。

可选本地 HTML：安装 `mistune>=3,<4` 后执行 `python export_html.py`。HTML 不纳入版本管理。

## 引用

- Hirschmüller, H. Stereo Processing by Semiglobal Matching and Mutual Information. TPAMI, 2008.
- Scharstein et al. High-Resolution Stereo Datasets with Subpixel-Accurate Ground Truth. GCPR, 2014.
- [Middlebury 标定与数据定义](https://vision.middlebury.edu/stereo/data/scenes2014/)
- [OpenCV StereoSGBM](https://docs.opencv.org/4.x/d2/d85/classcv_1_1StereoSGBM.html)
