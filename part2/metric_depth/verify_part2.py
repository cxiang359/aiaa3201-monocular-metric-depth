"""Independently recompute all NYU metrics from saved raw meter predictions."""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
from pathlib import Path
from datetime import datetime,timezone
import runtime
from runtime import ROOT
import numpy as np


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def verify(out,data,allow_partial=False):
    import h5py
    from scipy.io import loadmat
    import cv2
    run=json.loads((out/'run.json').read_text(encoding='utf-8'))
    summary=json.loads((out/'summary.json').read_text(encoding='utf-8'))
    splits=loadmat(data/'splits.mat')
    ids=(splits['testNdxs' if run['split']=='test' else 'trainNdxs'].reshape(-1)-1).astype(int)
    rows=[]
    for index in ids:
        p=out/'samples'/f'{index+1:04d}.json'
        if p.exists():rows.append(json.loads(p.read_text(encoding='utf-8')))
    if not allow_partial:
        assert len(rows)==len(ids) and run['status']=='complete','Full split is not complete'
    assert len(rows)==summary['images_evaluated']
    for name,value in run['signature']['source_hashes'].items():assert digest(ROOT/name)==value, name
    scores=[]
    with h5py.File(data/'nyu_depth_v2_labeled.mat','r') as f:
        for row in rows:
            index=row['zero_based_index'];assert index in ids and row['matlab_id']==index+1
            p=out/'predictions'/f'{index+1:04d}.npy'
            assert digest(p)==row['prediction_sha256']
            pred=np.load(p);assert pred.dtype==np.float32 and pred.shape==(480,640) and np.isfinite(pred).all()
            gt=np.asarray(f['depths'][index]).T.astype(np.float64)
            crop=np.zeros((480,640),bool);crop[45:471,41:601]=True
            mask=crop & np.isfinite(gt) & (gt>.001) & (gt<10)
            pp=np.clip(pred,.001,10).astype(np.float64)[mask];gg=gt[mask]
            ar=float((np.abs(pp-gg)/gg).mean())
            rmse=float(np.sqrt(((pp-gg)**2).mean()))
            d1=float((np.maximum(pp/gg,gg/pp)<1.25).mean())
            for key,value in zip(['absrel','rmse_m','delta1'],[ar,rmse,d1]):
                np.testing.assert_allclose(row[key],value,rtol=1e-8,atol=1e-9)
            assert row['valid_pixels']==int(mask.sum())
            scores.append([ar,rmse,d1])
    means=np.mean(scores,axis=0)
    for key,value in zip(['absrel','rmse_m','delta1'],means):
        np.testing.assert_allclose(summary[key],value,rtol=1e-8,atol=1e-9)
    pngs=list((out/'figures').glob('*.png'))
    for p in pngs:
        assert cv2.imdecode(np.frombuffer(p.read_bytes(),np.uint8),cv2.IMREAD_UNCHANGED) is not None
    import re
    embedded=re.findall(r'data:image/png;base64,([A-Za-z0-9+/=]+)',(out/'REPORT.html').read_text(encoding='utf-8'))
    cases=json.loads((out/'selected_cases.json').read_text(encoding='utf-8'))['cases']
    assert len(embedded)==len(cases)
    for encoded in embedded:
        b=base64.b64decode(encoded,validate=True)
        assert cv2.imdecode(np.frombuffer(b,np.uint8),cv2.IMREAD_UNCHANGED) is not None
    result={'status':'pass','checked_utc':datetime.now(timezone.utc).isoformat(),'images_checked':len(rows),
            'full_split_complete':len(rows)==len(ids),'checks':['raw float32 depth and hashes',
            'official test/train IDs','GT crop and masks','independent AbsRel/RMSE/delta1',
            'equal image aggregation','source hashes','PNG and HTML embedded image decoding']}
    (out/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'results/nyu_test_vits')
    p.add_argument('--data',type=Path,default=ROOT/'data')
    p.add_argument('--allow-partial',action='store_true')
    a=p.parse_args();verify(a.output,a.data,a.allow_partial)
