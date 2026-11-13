"""Exact course NYU protocol, separate from upstream alignment evaluation."""
from __future__ import annotations
import numpy as np
import cv2

HEIGHT, WIDTH = 480, 640
CROP = (45, 471, 41, 601)  # Python half-open indexing, y0:y1, x0:x1
MIN_DEPTH, MAX_DEPTH = 0.001, 10.0


def resize_prediction(prediction):
    prediction = np.asarray(prediction, dtype=np.float32).squeeze()
    if prediction.ndim != 2: raise ValueError(f'Expected HxW depth, got {prediction.shape}')
    if not np.isfinite(prediction).all(): raise ValueError('Nonfinite model predictions; do not silently remove pixels')
    if prediction.shape != (HEIGHT, WIDTH):
        prediction = cv2.resize(prediction, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
    return prediction


def course_mask(gt):
    gt = np.asarray(gt)
    if gt.shape != (HEIGHT, WIDTH): raise ValueError(f'Expected 480x640 GT, got {gt.shape}')
    crop = np.zeros(gt.shape, bool)
    y0,y1,x0,x1 = CROP; crop[y0:y1, x0:x1] = True
    return crop & np.isfinite(gt) & (gt > MIN_DEPTH) & (gt < MAX_DEPTH)


def evaluate_depth(prediction, gt):
    """No scale alignment. Clip prediction, GT mask, then per-image metrics."""
    raw = resize_prediction(prediction)
    gt = np.asarray(gt, dtype=np.float64)
    mask = course_mask(gt)
    if not mask.any(): raise ValueError('No valid pixels within the course crop')
    pred = np.clip(raw, MIN_DEPTH, MAX_DEPTH).astype(np.float64)
    p, g = pred[mask], gt[mask]
    error = np.abs(p - g)
    ratio = np.maximum(g/p, p/g)
    return {'absrel': float(np.mean(error/g)),
            'rmse_m': float(np.sqrt(np.mean((p-g)**2))),
            'delta1': float(np.mean(ratio < 1.25)),
            'valid_pixels': int(mask.sum()),
            'prediction_clipped_pixels': int(((raw<MIN_DEPTH)|(raw>MAX_DEPTH))[mask].sum())}


def aggregate(rows):
    if not rows: raise ValueError('Cannot aggregate zero images')
    return {name: float(np.mean([r[name] for r in rows])) for name in ['absrel','rmse_m','delta1']}
