"""Render fixed, manually chosen diagnostic crops; never changes predictions."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import cv2

from stereo import ROOT, color_array, read_image, read_pfm

# Relative crop coordinates (x0,y0,x1,y1) are selected after visual inspection.
# Fixed ROIs, not a search for maximizing/minimizing an evaluation score.
CASES = [
    ("PianoL", (.60, .08, .99, .67), "Lighting difference: lamp and rear wall"),
    ("Jadeplant", (.38, .04, .92, .60), "Fine leaves, branches and depth discontinuities"),
    ("Pipes", (.24, .34, .92, .94), "Repeated pipes, junctions and boundaries"),
]


def render(output, data):
    folder = output / "failure_cases"
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    metrics = {r["scene"]: r for r in json.loads((output / "scene_metrics.json").read_text(encoding="utf-8"))}
    for scene, relative, title in CASES:
        with np.load(output / "arrays" / f"{scene}.npz") as z:
            pred, valid, error = z["disparity_px"], z["valid_prediction"], z["absolute_error_px"]
        gt = read_pfm(data / "trainingQ" / scene / "disp0GT.pfm")
        rgb = cv2.cvtColor(read_image(data / "trainingQ" / scene / "im0.png"), cv2.COLOR_BGR2RGB)
        right = cv2.cvtColor(read_image(data / "trainingQ" / scene / "im1.png"), cv2.COLOR_BGR2RGB)
        h, w = gt.shape
        x0, y0, x1, y1 = [int(v * size) for v, size in zip(relative, [w, h, w, h])]
        roi = np.s_[y0:y1, x0:x1]
        gt_valid = np.isfinite(gt) & (gt > 0)
        intersection = valid & gt_valid
        gt_count = int(gt_valid[roi].sum())
        n = int(intersection[roi].sum())
        roi_mae = float(error[roi][intersection[roi]].mean(dtype=np.float64)) if n else None
        coverage = n / gt_count if gt_count else None
        fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout="constrained")
        fig.suptitle(f"{scene}: {title}\nROI MAE {roi_mae:.2f} px | coverage {coverage:.1%} (descriptive crop only)", fontsize=13)
        axes[0, 0].imshow(rgb)
        axes[0, 0].add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False, edgecolor="#ff284a", linewidth=2))
        axes[0, 0].set_title("Left RGB: selected ROI")
        axes[0, 1].imshow(rgb[roi])
        axes[0, 1].set_title("Left RGB crop")
        axes[0, 2].imshow(right)
        axes[0, 2].set_title("Right RGB (full view; viewpoint differs)")
        maximum = metrics[scene]["num_disparities"]
        images = [color_array(gt, gt_valid, 0, maximum)[roi],
                  color_array(pred, valid, 0, maximum)[roi],
                  color_array(error, np.isfinite(error), 0, 10, "inferno")[roi]]
        titles = ["GT disparity crop (black = no GT)", "Prediction crop (black = invalid)", "Absolute error crop (black = unevaluated)"]
        for ax, image, label in zip(axes[1], images, titles):
            ax.imshow(image)
            ax.set_title(label, fontsize=10)
        for ax in axes.flat:
            ax.axis("off")
        for ax, cmap, high in [(axes[1, 0], "turbo", maximum), (axes[1, 1], "turbo", maximum),
                               (axes[1, 2], "inferno", 10)]:
            fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, high), cmap=cmap), ax=ax,
                         orientation="horizontal", fraction=.05, pad=.03, label="px")
        fig.savefig(folder / f"{scene}_crop.png", dpi=140)
        plt.close(fig)
        records.append({"scene": scene, "roi_xyxy": [x0,y0,x1,y1], "title": title,
                        "gt_valid_pixels": gt_count, "evaluated_pixels": n,
                        "roi_mae_px": roi_mae, "roi_valid_fraction": coverage,
                        "selection": "manual fixed visual diagnostic ROI, not a benchmark or causal ablation"})
    (folder / "roi_metrics.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "results/sgbm_final")
    p.add_argument("--data", type=Path, default=ROOT / "data")
    a = p.parse_args()
    render(a.output.resolve(), a.data.resolve())
