"""Create a Chinese discussion brief after the full Part 2 test completes."""
from __future__ import annotations
import base64
import html
import json
from pathlib import Path
import runtime
from runtime import ROOT
import numpy as np
from nyu_data import NYULabeled
from protocol import course_mask


def main():
    out=ROOT/'results/nyu_test_vits'
    summary=json.loads((out/'summary.json').read_text(encoding='utf-8'))
    if summary['status']!='complete' or summary['images_evaluated']!=654:
        raise ValueError('This brief requires the complete 654-image test split')
    run=json.loads((out/'run.json').read_text(encoding='utf-8'))
    rows=[json.loads(p.read_text(encoding='utf-8')) for p in (out/'samples').glob('*.json')]
    ranked=sorted(rows,key=lambda r:r['absrel'])
    notes_path=ROOT/'case_notes.json'
    notes=json.loads(notes_path.read_text(encoding='utf-8')) if notes_path.exists() else {}
    cases=[]
    with NYULabeled(ROOT/'data/nyu_depth_v2_labeled.mat') as data:
        for row in ranked[-3:][::-1]:
            i=row['zero_based_index']
            def label(key):
                return ''.join(chr(int(c)) for c in data.file[data.file[key][0,i]][()].ravel())
            _,gt=data.read(i)
            pred=np.load(out/'predictions'/f'{i+1:04d}.npy')
            mask=course_mask(gt)
            ratio=float(np.median(pred[mask]/gt[mask]))
            case={**row,'scene':label('scenes'),'scene_type':label('sceneTypes'),
                  'median_raw_pred_gt_ratio':ratio,'note':notes.get(str(i+1),'')}
            cases.append(case)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for a,values,mean,color,title,xlabel in [
        (ax[0],[r['absrel'] for r in rows],summary['absrel'],'#3c82b4','Per-image relative depth error','AbsRel'),
        (ax[1],[100*r['delta1'] for r in rows],100*summary['delta1'],'#33987c','Per-image threshold accuracy','delta1 (%)')]:
        a.hist(values,bins=30,color=color,edgecolor='white')
        a.axvline(mean,color='#c34741',linestyle='--',label=f'Mean: {mean:.3f}')
        a.set_title(title);a.set_xlabel(xlabel);a.set_ylabel('Number of test images');a.legend()
    fig.savefig(out/'distribution.png',dpi=150);plt.close(fig)
    records={'summary':summary,'median_absrel':float(np.median([r['absrel'] for r in rows])),
             'worst_cases':cases,'case_selection':'Three largest AbsRel images; analysis only, no model/scale tuning'}
    (out/'analysis_details.json').write_text(json.dumps(records,indent=2,ensure_ascii=False),encoding='utf-8')
    md=['# Part 2 单图度量深度：结果与讨论','',
        '图片预览请打开 [ANALYSIS.html](ANALYSIS.html)。完整英文报告见 [REPORT.html](results/nyu_test_vits/REPORT.html)。','',
        '## 完成情况','',
        '**已完成你负责的单图度量深度 Part 2。** 官方 UniDepthV2 ViT-S 已在 NYU 官方 654 张测试图像上完整运行。'
        '仅输入 RGB，没有训练或微调，没有传入真值内参，也没有在测试集拟合尺度。视频深度属于组员负责的另一分支。','',
        '## 正式结果','',
        '| 指标 | 654 张逐图等权平均 | 含义 |','| --- | ---: | --- |',
        f"| AbsRel ↓ | **{summary['absrel']:.6f}** | 绝对深度误差除以真实深度，再求平均 |",
        f"| RMSE ↓ | **{summary['rmse_m']:.6f} m** | 每张图的深度均方根误差，再逐图平均 |",
        f"| δ1 ↑ | **{summary['delta1']:.4%}** | 预测/真值的双向比值小于 1.25 的像素比例，再逐图平均 |",'',
        f"AbsRel 的逐图中位数为 {records['median_absrel']:.6f}。平均值与中位数不同，体现困难图像对整体平均值的影响。"
        'AbsRel 并非米或厘米；δ1 也不是“所有像素的深度完全准确”。','',
        '![逐图指标分布](results/nyu_test_vits/distribution.png)','',
        '## 评价设置与实际资源','',
        '训练/开发集 795 张与测试集 654 张分离；使用 labeled 文件中的 filled depths（米），不是 rawDepths。'
        '输出 480×640，Python 裁剪 [45:471,41:601]，真值限制为 0.001 到 10 米之间。'
        '预测仅在评价时裁剪至 [0.001,10]；保存的 NPY 为未经尺度对齐的原始 float32 米制深度。'
        'AbsRel、RMSE、δ1 均逐图计算后等权平均，没有把像素拼接后进行加权平均。','',
        '官方模型 resolution_level=0，实测编码器输入 434×574。该分辨率档位是官方支持的推理选项，'
        '不是与论文所有分辨率/设置完全相同的复现，因此这些分数不能直接声称重现论文表格。','',
        f"本机 CPU FP32、8 线程；平均推理 **{summary['mean_inference_ms']:.2f} ms/张**，"
        f"中位数 **{summary['median_inference_ms']:.2f} ms/张**。计时包括官方 infer 的预处理、模型计算与尺寸恢复，"
        '不包括读文件、指标计算、保存和绘图，已排除一次预热。CPU 没有峰值 GPU 显存，记为 N/A。'
        f"运行中记录到的最大进程 RSS 为 {summary['max_process_rss_mb']:.1f} MiB，它不是 GPU 显存，也不是严格的全生命周期峰值内存。",'',
        '## 三个实际困难案例','',
        '以下为 AbsRel 最高的三张测试图。场景名来自 NYU 元数据。预测/真值的中位数比值只用于事后描述偏差，'
        '**没有拿这个比值调整预测**。同一场景的相邻视角可能同时进入前三，它们是三个困难图像案例，不应声称是三个独立场景类别。','',
        '| MATLAB ID | 场景 | AbsRel | RMSE (m) | δ1 | 原始预测/真值中位数 |',
        '| ---: | --- | ---: | ---: | ---: | ---: |']
    for c in cases:
        md.append(f"| {c['matlab_id']} | {c['scene']} | {c['absrel']:.4f} | {c['rmse_m']:.4f} | {c['delta1']:.2%} | {c['median_raw_pred_gt_ratio']:.3f} |")
    for c in cases:
        direction='偏大' if c['median_raw_pred_gt_ratio']>1.1 else '偏小' if c['median_raw_pred_gt_ratio']<.9 else '中位数接近真值，但仍可能有局部错误'
        md += ['',f"### 图像 {c['matlab_id']}：{c['scene']}",'',
               f"原始预测/真值的中位数为 {c['median_raw_pred_gt_ratio']:.3f}，说明该图像的距离估计{direction}。"
               '这只是对该图的描述，不能据此给全部测试图选择一个校准倍率。', '',
               c['note'] or '可对照下图中的 RGB、真值深度、预测深度和误差图观察模型局限；颜色只用于展示，数值结果以指标和 NPY 为准。','',
               f"![NYU {c['matlab_id']}](results/nyu_test_vits/figures/nyu_{c['matlab_id']:04d}.png)"]
    md += ['', '## 如何判断结果','',
           '已经建立了可复现、无测试尺度拟合的 metric-depth baseline，并量化了绝对距离误差。'
           '结果包含全量测试、逐图误差和真实困难案例，可以用于课程进度展示及后续改进对照。'
           '没有相同环境下的其他模型/设置对照，不能仅凭单一分数宣称达到 SOTA，也不能把 Part 1 的视差像素误差与这里的米制误差直接比较。','',
           '## 与 Part 3 的关系','',
           '原始预测已固定保存，之后可在相同测试协议下评估改进。若选择全局尺度校准，必须先在 795 张训练/开发图上拟合、固定，'
           '再用于测试集；不能用本次测试案例的真值拟合倍率，也不能挑一个让测试分数最好看的倍率。'
           '边界细化也需固定设置后与此 baseline 对比。当前没有提前混入任何 Part 3 改进。','',
           '## 文件入口','',
           '- [运行与复现说明](README.md)',
           '- [完整图片内嵌报告](results/nyu_test_vits/REPORT.html)',
           '- [654 张逐图指标](results/nyu_test_vits/metrics.csv)',
           '- [独立核验记录](results/nyu_test_vits/verification.json)',
           '- [9 项协议测试日志](logs/protocol_tests.txt)']
    text='\n'.join(md)
    (ROOT/'ANALYSIS.md').write_text(text,encoding='utf-8')
    # The installed Markdown renderer is only used for this presentation export.
    import mistune
    class Renderer(mistune.HTMLRenderer):
        def image(self,text,url,title=None):
            file=ROOT/url
            return '<figure><img alt="'+html.escape(text)+'" src="data:image/png;base64,'+base64.b64encode(file.read_bytes()).decode()+'"></figure>'
    body=mistune.create_markdown(renderer=Renderer(escape=True),plugins=['table'])(text)
    css='body{max-width:1120px;margin:30px auto;padding:0 22px;font:16px/1.85 Segoe UI,Microsoft YaHei,sans-serif;color:#173047}img{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #ccd8e1;padding:8px}h1,h2,h3{line-height:1.5}a{color:#176caa}'
    (ROOT/'ANALYSIS.html').write_text('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>Part 2 结果与讨论</title><style>'+css+'</style></head><body>'+body+'</body></html>',encoding='utf-8')
    print('Chinese analysis saved with embedded figures.')


if __name__=='__main__':main()
