"""RGB-only UniDepthV2 inference and exact NYU course evaluation, resumable."""
from __future__ import annotations
import argparse
import base64
import csv
import hashlib
import html
import json
import platform
import subprocess
import sys
import time
from datetime import datetime,timezone
from pathlib import Path

import runtime
from runtime import ROOT
import numpy as np
from protocol import evaluate_depth,aggregate,course_mask,resize_prediction,CROP
from nyu_data import NYULabeled,load_splits


def write_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    temp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def upstream_commit():
    return subprocess.check_output(['git','-C',str(ROOT/'vendor/UniDepth'),'rev-parse','HEAD'],text=True).strip()


def plot_comparison(path,rgb,gt,pred,row):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    y0,y1,x0,x1=CROP
    mask=course_mask(gt)
    error=np.full(gt.shape,np.nan,np.float32)
    error[mask]=np.abs(np.clip(pred,.001,10)[mask]-gt[mask])
    cm=plt.get_cmap('viridis').copy();cm.set_bad('black')
    em=plt.get_cmap('inferno').copy();em.set_bad('black')
    fig,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained')
    fig.suptitle(f"NYU MATLAB ID {row['matlab_id']} | AbsRel {row['absrel']:.4f} | RMSE {row['rmse_m']:.3f} m | delta1 {row['delta1']:.2%}")
    axes[0,0].imshow(rgb);axes[0,0].set_title('RGB only input (no GT camera)')
    from matplotlib.patches import Rectangle
    axes[0,0].add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,color='red',linewidth=1))
    im=axes[0,1].imshow(np.ma.masked_invalid(gt),vmin=0,vmax=10,cmap=cm)
    axes[0,1].set_title('Filled GT depth (m)');fig.colorbar(im,ax=axes[0,1],label='m',fraction=.046)
    im=axes[1,0].imshow(pred,vmin=0,vmax=10,cmap=cm)
    axes[1,0].set_title('Raw metric prediction (display range 0-10 m)');fig.colorbar(im,ax=axes[1,0],label='m',fraction=.046)
    im=axes[1,1].imshow(np.ma.masked_invalid(error),vmin=0,vmax=2,cmap=em)
    axes[1,1].set_title('Absolute error on course mask (display 0-2 m)');fig.colorbar(im,ax=axes[1,1],label='m',fraction=.046)
    for ax in axes.flat:ax.axis('off')
    path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,dpi=120);plt.close(fig)


def create_report(out,dataset,rows,meta):
    if not rows:return
    result=aggregate(rows)
    expected=meta['expected_images']
    complete=len(rows)==expected
    result.update({'status':'complete' if complete else 'partial', 'images_evaluated':len(rows),
                   'expected_images':expected,'split':meta['split'],
                   'mean_inference_ms':float(np.mean([r['inference_ms'] for r in rows])),
                   'median_inference_ms':float(np.median([r['inference_ms'] for r in rows])),
                   'gpu_peak_allocated_mb':max((r['gpu_peak_allocated_mb'] or 0) for r in rows) if meta['model']['device']=='cuda' else None,
                   'max_process_rss_mb':max(r.get('process_rss_mb',0) for r in rows),
                   'aggregation':'Equal per-image average; no scale fitting or alignment'})
    write_json(out/'summary.json',result)
    columns=['test_position','matlab_id','zero_based_index','absrel','rmse_m','delta1','valid_pixels',
             'prediction_clipped_pixels','inference_ms','gpu_peak_allocated_mb','process_rss_mb']
    with (out/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');w.writeheader();w.writerows(rows)
    # First, median and three most difficult test images; selection is analysis only.
    ranked=sorted(rows,key=lambda r:r['absrel'])
    selected={r['matlab_id']:r for r in [rows[0],ranked[len(rows)//2],*ranked[-3:]]}
    figures=[]
    for row in selected.values():
        rgb,gt=dataset.read(row['zero_based_index'])
        pred=np.load(out/'predictions'/f"{row['matlab_id']:04d}.npy")
        name=f"nyu_{row['matlab_id']:04d}.png";fig=out/'figures'/name
        plot_comparison(fig,rgb,gt,pred,row)
        figures.append((row,fig))
    write_json(out/'selected_cases.json',{'selection':'first, median AbsRel, and three highest AbsRel; descriptive only',
                                         'cases':[r for r,_ in figures]})
    title='Part 2: UniDepthV2 ViT-S / NYU course evaluation'
    md=[f'# {title}','',f"Status: **{result['status']}**, {len(rows)}/{expected} images ({meta['split']}).",'',
        '| AbsRel (lower) | RMSE, m (lower) | delta1 (higher) | Mean inference, ms |',
        '| ---: | ---: | ---: | ---: |',
        f"| {result['absrel']:.6f} | {result['rmse_m']:.6f} | {result['delta1']:.4%} | {result['mean_inference_ms']:.2f} |",'',
        '## Evaluation protocol','',
        'Official split: train/development 795 images; test 654 images. MATLAB IDs are converted to zero-based IDs once. '
        'RGB = images[i].transpose(2,1,0); filled depth in meters = depths[i].T. rawDepths is not used. '
        'No GT intrinsics, GT depths or scale calibration are passed to the model.', '',
        'Predictions are resized to 480x640 (bilinear if necessary). Python crop [45:471,41:601]. '
        'GT mask: finite, >0.001 and <10 m. Predictions clipped to [0.001,10] for metrics only; raw floating-point depth is saved unchanged. '
        'AbsRel, RMSE and delta1 (strict ratio <1.25) are computed per image and then averaged equally. '
        'No median alignment, scale fitting or per-image normalization. A partial run is not the final 654-image result.', '',
        '## Model and efficiency','',
        'Exact official architecture and checkpoint. infer(uint8 CHW RGB) performs official normalization, padding and resizing. '
        'Resolution level uses the upstream supported setting and is fixed for the run. '
        'Inference includes model preprocessing/forward/postprocessing, excludes data reading, metric calculation, disk writing and plotting. '
        'One warm-up on the first input is excluded. CPU peak GPU memory is N/A; CPU RSS is reported separately. '
        'GPU measurements, if used, synchronize CUDA and include peak allocated memory, not the caching allocator reserve.', '',
        '```json',json.dumps(meta['model'],indent=2),'```','',
        'The run records checkpoint revision/hash, upstream commit, input assets, source hashes and per-image timings. '
        'Course scores are not claimed to exactly reproduce a paper table because protocols may differ.','',
        '## Representative and difficult cases','',
        'The figures show the first input, median case and three largest AbsRel cases, without tuning the model on these cases. '
        'Depth color limits are fixed to 0-10 m for both GT and prediction; error is displayed at 0-2 m. '
        'Saturated color does not mean the underlying value has been truncated in the saved prediction.','']
    for r,fig in figures:
        md += [f"### MATLAB ID {r['matlab_id']}: AbsRel {r['absrel']:.4f}",'',f'![Comparison](figures/{fig.name})','']
    md += ['## Sources','',
           '- UniDepthV2: https://github.com/lpiccinelli-eth/UniDepth',
           '- Official ViT-S checkpoint: https://huggingface.co/lpiccinelli/unidepth-v2-vits14',
           '- NYU labeled dataset and official splits: URLs/hashes in data/asset_manifest.json.',
           '- Course Project 3 PDF, sections 5.2 and 6.2.']
    (out/'REPORT.md').write_text('\n'.join(md),encoding='utf-8')
    body=f'<h1>{html.escape(title)}</h1><p>Status: <b>{result["status"]}</b>; {len(rows)}/{expected} images.</p>'
    body+='<table><tr><th>AbsRel</th><th>RMSE (m)</th><th>delta1</th><th>Mean inference (ms)</th></tr>'
    body+=f'<tr><td>{result["absrel"]:.6f}</td><td>{result["rmse_m"]:.6f}</td><td>{result["delta1"]:.4%}</td><td>{result["mean_inference_ms"]:.2f}</td></tr></table>'
    body+='<p>RGB-only inference. Official NYU split, 480×640 output, crop [45:471,41:601], GT in (0.001,10) m, prediction clipped for evaluation only. No scale alignment. Equal per-image average.</p>'
    body+='<h2>Model and resources</h2><pre>'+html.escape(json.dumps(meta['model'],indent=2))+'</pre>'
    for row,fig in figures:
        body+=f'<h2>MATLAB ID {row["matlab_id"]} — AbsRel {row["absrel"]:.4f}</h2>'
        body+='<img alt="RGB, ground truth, metric prediction, error" src="data:image/png;base64,'+base64.b64encode(fig.read_bytes()).decode()+'">'
    css='body{max-width:1100px;margin:30px auto;padding:0 20px;font:16px/1.8 Segoe UI,Microsoft YaHei,sans-serif;color:#173047}img{width:100%;height:auto}table{border-collapse:collapse;width:100%}td,th{padding:10px;border:1px solid #ccd8e1}pre{white-space:pre-wrap;background:#f3f6f9;padding:15px}h2{margin-top:35px}'
    (out/'REPORT.html').write_text('<!doctype html><html><head><meta charset="utf-8"><title>Part 2 results</title><style>'+css+'</style></head><body>'+body+'</body></html>',encoding='utf-8')
    meta['status']=result['status'];meta['images_evaluated']=len(rows);meta['updated_utc']=datetime.now(timezone.utc).isoformat()
    write_json(out/'run.json',meta)
    print(json.dumps(result,indent=2),flush=True)


def run(args):
    dataset_path=args.data/'nyu_depth_v2_labeled.mat'
    splits=load_splits(args.data/'splits.mat')
    indices=[int(i) for i in splits[args.split]]
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    for name in ['predictions','samples','figures']: (out/name).mkdir(exist_ok=True)
    source_files=['runtime.py','protocol.py','nyu_data.py','model_adapter.py','run_part2.py']
    checkpoint=args.model/'model.safetensors'
    signature={'split':args.split,'indices':indices,'resolution_level':args.resolution_level,
               'requested_device':args.device,'threads':args.threads,
               'source_hashes':{f:sha(ROOT/f) for f in source_files},
               'checkpoint_sha256':sha(checkpoint),'config_sha256':sha(args.model/'config.json'),
               'upstream_commit':upstream_commit(), 'dataset_bytes':dataset_path.stat().st_size,
               'dataset_mtime_ns':dataset_path.stat().st_mtime_ns,
               'split_sha256':sha(args.data/'splits.mat')}
    existing=out/'run.json'
    if existing.exists():
        if not args.resume: raise FileExistsError('Run exists; use --resume or a new --output')
        meta=json.loads(existing.read_text(encoding='utf-8'))
        if signature!=meta['signature']: raise ValueError('Source/config/data signature changed; create a new run')
    else:
        meta={'status':'running','created_utc':datetime.now(timezone.utc).isoformat(),'signature':signature,
              'split':args.split,'expected_images':len(indices),'command':[sys.executable,*sys.argv],
              'platform':platform.platform(),'protocol':{'crop':CROP,'gt_depth_range_m':[.001,10],
              'scale_alignment':False,'aggregation':'equal image','output_hw':[480,640]}}
        for name,path in [('checkpoint_manifest',ROOT/'weights/manifest.json'),
                          ('data_manifest',args.data/'asset_manifest.json')]:
            if path.is_file():meta[name]=json.loads(path.read_text(encoding='utf-8'))
    with NYULabeled(dataset_path) as dataset:
        if args.report_only:
            rows=[json.loads((out/'samples'/f'{i+1:04d}.json').read_text(encoding='utf-8')) for i in indices if (out/'samples'/f'{i+1:04d}.json').is_file()]
            return create_report(out,dataset,rows,meta)
        from model_adapter import MetricModel
        model=MetricModel(args.model,args.device,args.threads,args.resolution_level)
        meta['model']=model.info;write_json(out/'run.json',meta)
        rgb,_=dataset.read(indices[0])
        print('Warm-up (excluded from timings)...',flush=True)
        model.infer(rgb)
        new=0;rows=[]
        for position,index in enumerate(indices):
            image_id=index+1
            record_path=out/'samples'/f'{image_id:04d}.json'
            prediction_path=out/'predictions'/f'{image_id:04d}.npy'
            if args.resume and record_path.is_file() and prediction_path.is_file():
                row=json.loads(record_path.read_text(encoding='utf-8'))
                if row['zero_based_index']!=index or sha(prediction_path)!=row['prediction_sha256']:
                    raise ValueError(f'Corrupt saved prediction: {image_id}')
                rows.append(row);continue
            rgb,gt=dataset.read(index)
            pred,resources=model.infer(rgb)
            pred=resize_prediction(pred)
            metrics=evaluate_depth(pred,gt)
            with prediction_path.with_suffix('.tmp').open('wb') as f:np.save(f,pred)
            prediction_path.with_suffix('.tmp').replace(prediction_path)
            row={'test_position':position,'zero_based_index':index,'matlab_id':image_id,
                 **metrics,**resources,'prediction_sha256':sha(prediction_path)}
            write_json(record_path,row);rows.append(row);new+=1
            message=f"[{position+1}/{len(indices)}] NYU {image_id}: AbsRel {row['absrel']:.4f} RMSE {row['rmse_m']:.3f}m delta1 {row['delta1']:.2%}; {row['inference_ms']/1000:.2f}s"
            print(message,flush=True)
            with (out/'progress.log').open('a',encoding='utf-8') as log:log.write(message+'\n')
            if args.max_new and new>=args.max_new:break
        # Include cached records even if a --max-new run stopped early.
        rows=[json.loads((out/'samples'/f'{i+1:04d}.json').read_text(encoding='utf-8')) for i in indices if (out/'samples'/f'{i+1:04d}.json').is_file()]
        meta['model']['encoder_shape']=model.encoder_shape
        create_report(out,dataset,rows,meta)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'data')
    p.add_argument('--model',type=Path,default=ROOT/'weights/unidepth-v2-vits14')
    p.add_argument('--output',type=Path,default=ROOT/'results/nyu_test_vits')
    p.add_argument('--split',choices=['train','test'],default='test')
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--threads',type=int,default=8)
    p.add_argument('--resolution-level',type=int,choices=range(10),default=0)
    p.add_argument('--max-new',type=int,default=0,help='Stop after N new predictions; status remains partial')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--report-only',action='store_true')
    a=p.parse_args()
    if a.threads<1 or a.max_new<0:p.error('Invalid thread count or max-new')
    run(a)


if __name__=='__main__':main()
