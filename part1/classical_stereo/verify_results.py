"""Check saved experiment artifacts independently of stereo.evaluate()."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def independent_pfm(path):
    # A separate struct-based reader is an oracle for row order and byte order.
    # Middlebury's payload is in px: header magnitude is deliberately unused.
    with path.open("rb") as f:
        assert f.readline().strip() == b"Pf"
        width, height = map(int, f.readline().split())
        signed_scale = float(f.readline())
        values = struct.unpack(("<" if signed_scale < 0 else ">") + str(width * height) + "f", f.read())
    return np.asarray(values, np.float32).reshape(height, width)[::-1].copy()


def verify(output, data):
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    rows = json.loads((output / "scene_metrics.json").read_text(encoding="utf-8"))
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert run["status"] == "complete"
    for name, expected in run["source_files"].items():
        assert digest(ROOT / name) == expected, f"Source changed since experiment: {name}"
    with (output / "metrics.csv").open(encoding="utf-8-sig", newline="") as f:
        csv_rows = list(csv.DictReader(f))
    assert len(csv_rows) == len(rows) == summary["scene_count"]
    assert [r["scene"] for r in rows] == run["scenes"]
    assert set(p.stem for p in (output / "arrays").glob("*.npz")) == set(run["scenes"])
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        assert digest(data / entry["path"]) == entry["sha256"]
    maes, coverages, scene_records = [], [], []
    sum_error, sum_gt, sum_evaluated = 0.0, 0, 0
    config = run["config"]
    for row, csv_row in zip(rows, csv_rows):
        scene = row["scene"]
        source = data / "trainingQ" / scene
        for name, expected in row["input_sha256"].items():
            assert digest(source / name) == expected
        gt = independent_pfm(source / "disp0GT.pfm")
        calibration = dict(line.split("=", 1) for line in (source / "calib.txt").read_text().splitlines() if "=" in line)
        fx = float(calibration["cam0"].lstrip("[").split()[0])
        baseline, doffs = float(calibration["baseline"]), float(calibration["doffs"])
        with np.load(output / "arrays" / f"{scene}.npz") as saved:
            raw = saved["raw_disparity_fixed16"]
            disparity = saved["disparity_px"]
            depth = saved["depth_m"]
            mask = saved["valid_prediction"]
            d = raw.astype(np.float64) / 16
            minimum = config["min_disparity"]
            expected_mask = ((raw > (minimum-1)*16) & (d >= minimum)
                             & (d < minimum+row["num_disparities"]) & (d+doffs > 0))
            np.testing.assert_array_equal(mask, expected_mask)
            assert disparity.shape == depth.shape == mask.shape == gt.shape
            assert disparity.dtype == depth.dtype == np.float32
            assert raw.dtype == np.int16
            np.testing.assert_array_equal(np.isnan(disparity), ~mask)
            np.testing.assert_array_equal(np.isnan(depth), ~mask)
            np.testing.assert_array_equal(disparity[mask], d[mask])
            np.testing.assert_allclose(depth[mask], fx*baseline/(d[mask]+doffs)/1000, rtol=2e-7)
            assert (depth[mask] > 0).all()
            gt_valid = np.isfinite(gt) & (gt > 0)
            np.testing.assert_array_equal(saved["gt_valid"], gt_valid)
            evaluate_mask = gt_valid & mask
            absolute = np.abs(disparity[evaluate_mask].astype(np.float64) - gt[evaluate_mask])
            np.testing.assert_allclose(saved["absolute_error_px"][evaluate_mask], absolute, rtol=1e-6, atol=2e-6)
            np.testing.assert_array_equal(np.isnan(saved["absolute_error_px"]), ~evaluate_mask)
            mae = float(absolute.mean())
            coverage = float(evaluate_mask.sum() / gt_valid.sum())
            np.testing.assert_allclose(mae, row["mae_px"], atol=2e-6)
            np.testing.assert_allclose(coverage, row["valid_fraction"], atol=1e-12)
            np.testing.assert_allclose(float(csv_row["mae_px"]), mae, atol=2e-6)
            np.testing.assert_allclose(float(csv_row["valid_fraction"]), coverage, atol=1e-12)
            if run["lr_diagnostic"]:
                status, residual = saved["lr_status"], saved["lr_residual_px"]
                assert set(np.unique(status)).issubset({0, 1, 2})
                np.testing.assert_array_equal(np.isfinite(residual), status > 0)
                assert (residual[status == 1] <= config["lr_threshold_px"]).all()
                assert (residual[status == 2] > config["lr_threshold_px"]).all()
                assert int((status > 0).sum()) == row["lr_checked_pixels"]
            maes.append(mae)
            coverages.append(coverage)
            sum_error += float(absolute.sum())
            sum_gt += int(gt_valid.sum())
            sum_evaluated += int(evaluate_mask.sum())
        scene_records.append({"scene": scene, "mae_px": mae, "valid_fraction": coverage, "status": "pass"})
    np.testing.assert_allclose(np.mean(maes), summary["macro_mae_px"], atol=2e-6)
    np.testing.assert_allclose(np.mean(coverages), summary["macro_valid_fraction"], atol=1e-12)
    np.testing.assert_allclose(sum_error/sum_evaluated, summary["pooled_mae_px"], atol=2e-6)
    np.testing.assert_allclose(sum_evaluated/sum_gt, summary["pooled_valid_fraction"], atol=1e-12)
    assert sum_gt == summary["gt_valid_pixels"] and sum_evaluated == summary["evaluated_pixels"]
    # Decode every emitted PNG, including all figures, instead of checking existence only.
    pngs = list(output.rglob("*.png"))
    assert len(list((output / "figures").glob("*/comparison.png"))) == len(rows)
    for png in pngs:
        image = cv2.imdecode(np.frombuffer(png.read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)
        assert image is not None and image.size, f"Unreadable PNG: {png}"
    assert (output / "REPORT.md").is_file()
    result = {"status": "pass", "checked_utc": datetime.now(timezone.utc).isoformat(),
              "scene_count": len(rows), "decoded_png_count": len(pngs),
              "source_hashes_match": True, "input_hashes_match": True,
              "checks": ["independent struct PFM reader", "fixed-point scaling", "valid masks and NaNs",
                         "metric depth geometry and units", "per-scene/CSV/summary metrics", "LR status invariants",
                         "PNG decodability", "input and source hashes"],
              "scenes": scene_records, "verifier_sha256": digest(Path(__file__))}
    (output / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/sgbm_final")
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    verify(args.output.resolve(), args.data.resolve())
