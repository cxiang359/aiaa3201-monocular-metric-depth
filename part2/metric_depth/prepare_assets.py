"""Download the exact course NYU files and a pinned UniDepthV2 ViT-S checkpoint."""
from __future__ import annotations
import argparse
import hashlib
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_URLS = {
    "nyu_depth_v2_labeled.mat": "https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat",
    "splits.mat": "https://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat",
}
MODEL_ID = "lpiccinelli/unidepth-v2-vits14"


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''): h.update(b)
    return h.hexdigest()


def download(url, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        print('Already present:', target.name, target.stat().st_size, flush=True)
        return
    partial = target.with_name(target.name + '.partial')
    for attempt in range(1, 6):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            request = urllib.request.Request(url, headers={'Range': f'bytes={offset}-'} if offset else {})
            with urllib.request.urlopen(request, timeout=90) as response:
                append = offset > 0 and response.status == 206
                if not append: offset = 0
                length = response.headers.get('Content-Length')
                total = offset + int(length) if length else None
                last = time.monotonic()
                with partial.open('ab' if append else 'wb') as f:
                    current = offset
                    while True:
                        block = response.read(1024*1024)
                        if not block: break
                        f.write(block); current += len(block)
                        if time.monotonic() - last > 15:
                            print(f'{target.name}: {current/1e6:.1f} MB' + (f' / {total/1e6:.1f} MB' if total else ''), flush=True)
                            last = time.monotonic()
                if total is not None and current != total:
                    raise IOError(f'Incomplete transfer: {current}/{total}')
            partial.replace(target)
            print('Downloaded:', target.name, target.stat().st_size, flush=True)
            return
        except Exception as exc:
            print(f'Download attempt {attempt} failed: {exc}', flush=True)
            if attempt == 5: raise
            time.sleep(min(attempt * 2, 8))


def parallel_download(url, target, workers):
    """Resume the existing prefix, fetch disjoint HTTP ranges, then assemble."""
    if target.is_file(): return
    prefix = target.with_name(target.name+'.partial')
    offset = prefix.stat().st_size if prefix.exists() else 0
    req = urllib.request.Request(url, headers={'Range': 'bytes=0-0'})
    with urllib.request.urlopen(req, timeout=60) as r:
        if r.status != 206 or not r.headers.get('Content-Range'):
            return download(url,target)
        total = int(r.headers['Content-Range'].split('/')[-1]); r.read()
    parts_dir = target.parent / '.ranges'
    parts_dir.mkdir(parents=True,exist_ok=True)
    if offset > total: raise ValueError('Existing partial exceeds total size')
    # Preserve any successfully received prefix of old disjoint range files.
    cached=[]
    for part in parts_dir.glob('*.part'):
        start,end=map(int,part.stem.split('-'))
        size=part.stat().st_size
        if size > end-start+1: raise ValueError('Oversized cached range')
        if size: cached.append((start,start+size-1,part))
    cached.sort()
    cursor=offset; gaps=[]
    for start,end,part in cached:
        if start < cursor: raise ValueError('Overlapping cached data; inspect range files')
        if start > cursor: gaps.append((cursor,start-1))
        cursor=end+1
    if cursor<total: gaps.append((cursor,total-1))
    specs=[]
    chunk_bytes=16*1024*1024
    for start,end in gaps:
        for a in range(start,end+1,chunk_bytes): specs.append((a,min(end,a+chunk_bytes-1)))
    def worker(spec):
        start,end=spec
        part=parts_dir/f'{start}-{end}.part'
        for attempt in range(5):
            try:
                have=part.stat().st_size if part.exists() else 0
                if have==end-start+1: return part
                if have>end-start+1: raise ValueError('Oversized range part')
                req=urllib.request.Request(url,headers={'Range':f'bytes={start+have}-{end}'})
                with urllib.request.urlopen(req,timeout=90) as r:
                    expected=f'bytes {start+have}-{end}/{total}'
                    if r.status!=206 or r.headers.get('Content-Range')!=expected:
                        raise ValueError('Server did not honor requested range')
                    with part.open('ab') as f:
                        while b:=r.read(1024*1024): f.write(b)
                if part.stat().st_size!=end-start+1: raise IOError('Incomplete range')
                print(f'Range complete: {start}-{end}',flush=True)
                return part
            except Exception as exc:
                print(f'Range {start}-{end} attempt {attempt+1}: {exc}',flush=True)
                if attempt==4: raise
                time.sleep(2)
    print(f'Parallel download: {workers} connections, {len(specs)} missing chunks, total {total/1e6:.1f} MB',flush=True)
    pool=ThreadPoolExecutor(max_workers=workers)
    futures=[pool.submit(worker,s) for s in specs]
    try:
        while not all(f.done() for f in futures):
            for f in futures:
                if f.done(): f.result()
            downloaded=offset+sum(p.stat().st_size for p in parts_dir.glob('*.part'))
            print(f'NYU total: {downloaded/1e6:.1f}/{total/1e6:.1f} MB',flush=True)
            time.sleep(15)
        parts=sorted([p for _,_,p in cached]+[f.result() for f in futures],key=lambda p:int(p.stem.split('-')[0]))
    finally: pool.shutdown(wait=True)
    with prefix.open('ab') as out:
        import shutil
        for part in parts:
            with part.open('rb') as f: shutil.copyfileobj(f,out,4*1024*1024)
    if prefix.stat().st_size!=total: raise IOError('Assembly size mismatch')
    prefix.replace(target)
    for part in parts: part.unlink()
    print('Assembled:',target.name,total,flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--only', choices=['all', 'data', 'model'], default='all')
    p.add_argument('--workers', type=int, default=1, choices=range(1,17))
    p.add_argument('--revision', default='038c238f06c87b6c2f5b3749fd51fbf442b1f218',
                   help='Hugging Face checkpoint revision; defaults to the published baseline')
    args = p.parse_args()
    records = []
    if args.only in ('all', 'data'):
        # Small split file first; download the 2.8 GB labeled data with resume support.
        for name in ['splits.mat', 'nyu_depth_v2_labeled.mat']:
            destination = ROOT/'data'/name
            if name=='nyu_depth_v2_labeled.mat' and args.workers>1:
                parallel_download(DATA_URLS[name],destination,args.workers)
            else: download(DATA_URLS[name], destination)
            records.append({'path': str(destination.relative_to(ROOT)), 'url': DATA_URLS[name],
                            'bytes': destination.stat().st_size, 'sha256': sha256(destination)})
        (ROOT/'data/asset_manifest.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    if args.only in ('all', 'model'):
        with urllib.request.urlopen(f'https://huggingface.co/api/models/{MODEL_ID}/revision/{args.revision}', timeout=45) as r:
            info = json.load(r)
        revision = info['sha']
        available = {x['rfilename'] for x in info['siblings']}
        weight = 'model.safetensors' if 'model.safetensors' in available else 'pytorch_model.bin'
        if weight not in available: raise ValueError('No official checkpoint found')
        records = []
        for name in ['config.json', weight]:
            url = f'https://huggingface.co/{MODEL_ID}/resolve/{revision}/{name}'
            destination = ROOT/'weights/unidepth-v2-vits14'/name
            download(url, destination)
            records.append({'name': name, 'bytes': destination.stat().st_size, 'sha256': sha256(destination), 'url': url})
        (ROOT/'weights/manifest.json').write_text(json.dumps({'model_id': MODEL_ID, 'revision': revision, 'files': records}, indent=2), encoding='utf-8')


if __name__ == '__main__': main()
