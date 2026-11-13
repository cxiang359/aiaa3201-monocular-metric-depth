"""A real-image CPU compatibility test; not a NYU evaluation result."""
import argparse
import json
from pathlib import Path
import runtime
from runtime import ROOT
import numpy as np
from PIL import Image
from model_adapter import MetricModel


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',type=Path,required=True)
    p.add_argument('--threads',type=int,default=8)
    a=p.parse_args()
    rgb=np.array(Image.open(a.image).convert('RGB').resize((640,480)))
    model=MetricModel(ROOT/'weights/unidepth-v2-vits14',threads=a.threads)
    print('Loaded model',model.info,flush=True)
    pred,resources=model.infer(rgb)
    out=ROOT/'results/environment_smoke';out.mkdir(parents=True,exist_ok=True)
    np.save(out/'depth_m.npy',pred)
    Image.fromarray(rgb).save(out/'input.png')
    report={'kind':'real image compatibility test, NOT NYU benchmark','image':str(a.image),
            'model':model.info,'resources':resources,'depth_range_m':[float(pred.min()),float(pred.max())]}
    (out/'result.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
