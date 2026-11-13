"""Read the official labeled MATLAB file and its one-based official split."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import runtime


def load_splits(path):
    from scipy.io import loadmat
    data = loadmat(path)
    splits = {}
    for key, name, count in [('trainNdxs','train',795),('testNdxs','test',654)]:
        raw = np.asarray(data[key]).reshape(-1)
        if not np.isfinite(raw).all() or not (raw == np.floor(raw)).all():
            raise ValueError(f'Noninteger split IDs: {key}')
        ids = raw.astype(np.int64)-1
        if len(ids)!=count or len(np.unique(ids))!=count or ids.min()<0 or ids.max()>=1449:
            raise ValueError(f'Invalid official split: {key}')
        splits[name]=ids
    if set(splits['train']) & set(splits['test']): raise ValueError('Train/test overlap')
    if len(set(splits['train']) | set(splits['test'])) != 1449: raise ValueError('Incomplete split')
    return splits


def convert_sample(image, depth):
    # HDF5's reversed MATLAB layout: C,W,H -> H,W,C and W,H -> H,W.
    rgb = np.asarray(image).transpose(2,1,0).copy()
    gt = np.asarray(depth).T.copy()
    if rgb.shape != (480,640,3) or rgb.dtype != np.uint8: raise ValueError(f'Unexpected RGB {rgb.shape}, {rgb.dtype}')
    if gt.shape != (480,640): raise ValueError(f'Unexpected GT {gt.shape}')
    if not np.issubdtype(gt.dtype,np.floating): raise ValueError('GT depth must remain floating point')
    return rgb, gt


class NYULabeled:
    def __init__(self, path):
        import h5py
        self.file = h5py.File(path, 'r')
        if self.file['images'].shape != (1449,3,640,480) or self.file['depths'].shape != (1449,640,480):
            self.file.close(); raise ValueError('Unexpected NYU labeled HDF5 dataset layout')

    def read(self, zero_based_index):
        i=int(zero_based_index)
        return convert_sample(self.file['images'][i], self.file['depths'][i])

    def close(self): self.file.close()
    def __enter__(self): return self
    def __exit__(self, *args): self.close()
