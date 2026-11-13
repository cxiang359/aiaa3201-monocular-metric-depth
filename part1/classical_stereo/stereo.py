"""Course Part 1: reproducible Middlebury quarter-resolution stereo exercise.

No learning or ground-truth-dependent prediction. All paths default relative to
this file, including on Windows with Chinese directory names.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import statistics
import sys
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
EXPECTED_SCENES = (
    "Adirondack", "ArtL", "Jadeplant", "Motorcycle", "MotorcycleE", "Piano",
    "PianoL", "Pipes", "Playroom", "Playtable", "PlaytableP", "Recycle",
    "Shelves", "Teddy", "Vintage",
)
URLS = {
    "MiddEval3-data-Q.zip": "https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-data-Q.zip",
    "MiddEval3-GT0-Q.zip": "https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-GT0-Q.zip",
}


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prepare_data(args):
    """Extract only 15 trainingQ scenes, never the hidden-ground-truth test set."""
    archives = args.archives.resolve()
    dest = args.data.resolve()
    manifest = {"source": URLS, "archives": [], "files": []}
    wanted_input = {"im0.png", "im1.png", "calib.txt"}
    for filename, url in URLS.items():
        archive = archives / filename
        if not archive.is_file():
            if not args.download:
                raise FileNotFoundError(f"Missing {archive}. Download {url} or use prepare --download.")
            archive.parent.mkdir(parents=True, exist_ok=True)
            temporary = archive.with_suffix(".zip.partial")
            print(f"Downloading {url}", flush=True)
            with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as out:
                shutil.copyfileobj(response, out)
            temporary.replace(archive)
        manifest["archives"].append({"name": filename, "bytes": archive.stat().st_size,
                                     "sha256": sha256(archive)})
        with zipfile.ZipFile(archive) as z:
            bad = z.testzip()
            if bad:
                raise ValueError(f"Corrupt ZIP member: {bad}")
            for info in z.infolist():
                parts = info.filename.split("/")
                # These public archives call the course's trainQ 'trainingQ'.
                if len(parts) != 4 or parts[:2] != ["MiddEval3", "trainingQ"]:
                    continue
                scene, basename = parts[2:]
                allowed = wanted_input if filename == "MiddEval3-data-Q.zip" else {"disp0GT.pfm", "mask0nocc.png"}
                if scene not in EXPECTED_SCENES or basename not in allowed:
                    continue
                target = dest / "trainingQ" / scene / basename
                target.parent.mkdir(parents=True, exist_ok=True)
                content = z.read(info)
                target.write_bytes(content)
                manifest["files"].append({"path": str(target.relative_to(dest)), "bytes": len(content),
                                          "sha256": hashlib.sha256(content).hexdigest()})
    for scene in EXPECTED_SCENES:
        for name in [*sorted(wanted_input), "disp0GT.pfm"]:
            if not (dest / "trainingQ" / scene / name).is_file():
                raise FileNotFoundError(f"Incomplete scene: {scene}/{name}")
    write_json(dest / "manifest.json", manifest)
    print(f"Prepared {len(EXPECTED_SCENES)} trainingQ scenes in {dest}", flush=True)


def read_pfm(path: Path) -> np.ndarray:
    """Read Middlebury disparity PFM: native px, signed scale for endian only.

    Middlebury/cvkit disparity maps do NOT use the header's scale magnitude.
    Applying the generic radiometric PFM scaling would corrupt GT units.
    https://github.com/roboception/cvkit/issues/5#issuecomment-481676813
    """
    with path.open("rb") as f:
        def header_line():
            while True:
                line = f.readline()
                if not line:
                    raise ValueError(f"Truncated PFM header: {path}")
                if line.strip() and not line.lstrip().startswith(b"#"):
                    return line.strip()
        magic = header_line()
        if magic not in (b"Pf", b"PF"):
            raise ValueError(f"Invalid PFM magic in {path}")
        width, height = map(int, header_line().split())
        scale = float(header_line())
        if width <= 0 or height <= 0 or not np.isfinite(scale) or scale == 0:
            raise ValueError(f"Invalid PFM dimensions/scale in {path}")
        channels = 1 if magic == b"Pf" else 3
        values = np.frombuffer(f.read(), dtype="<f4" if scale < 0 else ">f4")
        if values.size != width * height * channels:
            raise ValueError(f"PFM payload size mismatch: {path}")
        shape = (height, width) if channels == 1 else (height, width, channels)
        return np.flipud(values.reshape(shape)).astype(np.float32).copy()


def read_calib(path: Path) -> dict:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = map(str.strip, line.split("=", 1))
        if key in ("cam0", "cam1"):
            rows = [[float(x) for x in row.split()] for row in value.strip("[]").split(";")]
            result[key] = rows
        else:
            result[key] = float(value)
    for required in ("cam0", "cam1", "baseline", "doffs", "width", "height", "ndisp"):
        if required not in result:
            raise ValueError(f"Missing calibration key {required}: {path}")
    if np.shape(result["cam0"]) != (3, 3) or np.shape(result["cam1"]) != (3, 3):
        raise ValueError("Camera matrices must be 3x3")
    if result["cam0"][0][0] <= 0 or result["baseline"] <= 0 or result["ndisp"] <= 0:
        raise ValueError("Focal length, baseline and ndisp must be positive")
    return result


def read_image(path: Path) -> np.ndarray:
    # imread can fail on non-ASCII paths in some Windows OpenCV builds.
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def depth_from_disparity(disparity: np.ndarray, valid: np.ndarray, calib: dict) -> np.ndarray:
    denominator = disparity.astype(np.float64) + calib["doffs"]
    mask = valid & np.isfinite(disparity) & np.isfinite(denominator) & (denominator > 0)
    depth = np.full(disparity.shape, np.nan, dtype=np.float32)
    depth[mask] = calib["cam0"][0][0] * calib["baseline"] / denominator[mask] / 1000.0
    return depth


def create_matcher(config: dict, num_disparities: int, min_disparity: int | None = None):
    minimum = config["min_disparity"] if min_disparity is None else min_disparity
    return cv2.StereoSGBM_create(
        minDisparity=minimum, numDisparities=num_disparities,
        blockSize=config["block_size"], P1=config["p1"], P2=config["p2"],
        disp12MaxDiff=config["disp12_max_diff"], preFilterCap=config["pre_filter_cap"],
        uniquenessRatio=config["uniqueness_ratio"], speckleWindowSize=config["speckle_window_size"],
        speckleRange=config["speckle_range"], mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def prediction_valid(raw: np.ndarray, config: dict, num_disparities: int) -> np.ndarray:
    # OpenCV marks invalid values as (minDisparity - 1) * 16.
    # Zero disparity is legal when doffs > 0; do not discard it automatically.
    minimum = config["min_disparity"]
    disparity = raw.astype(np.float32) / 16.0
    return (raw > (minimum - 1) * 16) & (disparity >= minimum) & (disparity < minimum + num_disparities)


def evaluate(disparity: np.ndarray, valid: np.ndarray, gt: np.ndarray) -> tuple[dict, np.ndarray]:
    gt_valid = np.isfinite(gt) & (gt > 0)
    evaluation = gt_valid & valid & np.isfinite(disparity)
    error = np.full(gt.shape, np.nan, dtype=np.float32)
    error[evaluation] = np.abs(disparity[evaluation] - gt[evaluation])
    n_gt, n_pred = int(gt_valid.sum()), int(evaluation.sum())
    if n_gt == 0:
        raise ValueError("No finite positive ground-truth disparities")
    return {
        "gt_valid_pixels": n_gt,
        "evaluated_pixels": n_pred,
        "valid_fraction": n_pred / n_gt,
        "mae_px": float(error[evaluation].mean(dtype=np.float64)) if n_pred else None,
        "sum_absolute_error_px": float(error[evaluation].sum(dtype=np.float64)),
    }, error


def lr_diagnostic(left_disp, left_valid, right_raw, right_min, threshold):
    """Nearest-neighbor left-right residual; diagnostic only, no baseline filtering."""
    h, w = left_disp.shape
    yy, xx = np.indices((h, w))
    xr = np.rint(xx - np.nan_to_num(left_disp, nan=0)).astype(np.int32)
    in_bounds = (xr >= 0) & (xr < w)
    lookup = np.clip(xr, 0, w - 1)
    right_disp = right_raw.astype(np.float32) / 16.0
    sampled = right_disp[yy, lookup]
    right_valid = right_raw[yy, lookup] > (right_min - 1) * 16
    available = left_valid & in_bounds & right_valid
    residual = np.full((h, w), np.nan, dtype=np.float32)
    residual[available] = np.abs(left_disp[available] + sampled[available])
    # 0: unavailable/left invalid, 1: checked consistent, 2: checked inconsistent.
    status = np.zeros((h, w), np.uint8)
    status[available & (residual <= threshold)] = 1
    status[available & (residual > threshold)] = 2
    return residual, status


def color_array(values, mask, low, high, cmap="turbo"):
    scaled = np.clip((np.nan_to_num(values, nan=low) - low) / max(high - low, 1e-6), 0, 1)
    rgb = (plt.get_cmap(cmap)(scaled)[..., :3] * 255).astype(np.uint8)
    rgb[~mask] = 0
    return rgb


def save_rgb(path, rgb):
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError(f"Cannot encode {path}")
    path.write_bytes(encoded.tobytes())


def draw_scene(out, scene, left, right, disp, valid, depth, gt, error, calib, config, metrics, lr_status):
    out.mkdir(parents=True, exist_ok=True)
    gt_valid = np.isfinite(gt) & (gt > 0)
    lo, hi = 0.0, float(metrics["num_disparities"])
    finite_depth = depth[np.isfinite(depth)]
    zlo, zhi = (np.percentile(finite_depth, [2, 98]).tolist() if finite_depth.size else [0.0, 1.0])
    if zhi <= zlo:
        zhi = zlo + 1
    rgb_left, rgb_right = cv2.cvtColor(left, cv2.COLOR_BGR2RGB), cv2.cvtColor(right, cv2.COLOR_BGR2RGB)
    pred_color = color_array(disp, valid, lo, hi)
    gt_color = color_array(gt, gt_valid, lo, hi)
    depth_valid = np.isfinite(depth)
    depth_color = color_array(depth, depth_valid, zlo, zhi, "viridis")
    error_valid = np.isfinite(error)
    error_color = color_array(error, error_valid, 0, config["error_display_max_px"], "inferno")
    invalid_color = np.zeros((*valid.shape, 3), np.uint8)
    invalid_color[~valid] = [255, 60, 80]
    lr_color = np.zeros((*valid.shape, 3), np.uint8)
    if lr_status is not None:
        lr_color[lr_status == 1] = [60, 200, 100]
        lr_color[lr_status == 2] = [255, 60, 80]
    for name, img in [("disparity.png", pred_color), ("gt_disparity.png", gt_color),
                      ("depth_m.png", depth_color), ("invalid.png", invalid_color),
                      ("error_px.png", error_color)]:
        save_rgb(out / name, img)
    if lr_status is not None:
        save_rgb(out / "lr_diagnostic.png", lr_color)
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), layout="constrained")
    mae_label = f"{metrics['mae_px']:.3f}" if metrics["mae_px"] is not None else "N/A"
    fig.suptitle(f"{scene} | StereoSGBM | MAE {mae_label} px | valid {metrics['valid_fraction']:.1%}", fontsize=16)
    panels = [
        (rgb_left, "Left RGB", None), (rgb_right, "Right RGB", None),
        (gt_color, "GT disparity (black = no GT)", ("turbo", lo, hi, "px")),
        (pred_color, "Predicted disparity (black = invalid)", ("turbo", lo, hi, "px")),
        (depth_color, "Predicted metric depth (clipped for display)", ("viridis", zlo, zhi, "m")),
        (invalid_color, "Invalid predictions (red), valid (black)", None),
        (error_color, "Absolute disparity error (black = unevaluated)", ("inferno", 0, config["error_display_max_px"], "px")),
        (lr_color, "LR: green = pass; red = fail; black = unchecked" if lr_status is not None else "LR diagnostic not requested", None),
    ]
    for ax, (img, title, colorbar) in zip(axes.flat, panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        if colorbar:
            cmap, vmin, vmax, label = colorbar
            fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin, vmax), cmap=cmap),
                         ax=ax, orientation="horizontal", fraction=.04, pad=.025, label=label)
    fig.savefig(out / "comparison.png", dpi=125)
    plt.close(fig)
    return {"disparity_px": [lo, hi], "depth_m": [zlo, zhi],
            "absolute_error_px": [0, config["error_display_max_px"]],
            "depth_range_source": "2nd/98th percentiles of valid predicted depth, visualization only"}


def hardware_info():
    cpu = platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown")
    return {"platform": platform.platform(), "cpu": cpu, "logical_cpus": os.cpu_count(),
            "python": platform.python_version(), "opencv": cv2.__version__,
            "numpy": np.__version__, "matplotlib": matplotlib.__version__,
            "backend": "OpenCV CPU", "precision": "uint8 grayscale input; int16 fixed-point output; float32 stored maps",
            "gpu_peak_memory": "N/A: CPU-only stereo, no GPU inference",
            "opencv_threads": cv2.getNumThreads()}


def validate_config(config):
    if config["method"] != "StereoSGBM" or config["mode"] != "SGBM_3WAY":
        raise ValueError("This implementation supports StereoSGBM / SGBM_3WAY")
    block = config["block_size"]
    if not isinstance(block, int) or block <= 0 or block % 2 != 1:
        raise ValueError("block_size must be a positive odd integer")
    if config["p2"] <= config["p1"] or config["p1"] <= 0:
        raise ValueError("Require 0 < P1 < P2")
    if config["timing_repeats"] < 1 or config["opencv_threads"] < 1:
        raise ValueError("timing_repeats and opencv_threads must be positive")


def summarize(rows):
    maes = [r["mae_px"] for r in rows if r["mae_px"] is not None]
    n_gt = sum(r["gt_valid_pixels"] for r in rows)
    n_pred = sum(r["evaluated_pixels"] for r in rows)
    return {
        "scene_count": len(rows), "scenes_with_valid_predictions": len(maes),
        "macro_mae_px": statistics.mean(maes) if maes else None,
        "macro_valid_fraction": statistics.mean(r["valid_fraction"] for r in rows),
        "pooled_mae_px": sum(r["sum_absolute_error_px"] for r in rows) / n_pred if n_pred else None,
        "pooled_valid_fraction": n_pred / n_gt,
        "gt_valid_pixels": n_gt, "evaluated_pixels": n_pred,
        "mean_inference_ms": statistics.mean(r["inference_ms"] for r in rows),
        "mean_pipeline_ms": statistics.mean(r["pipeline_ms"] for r in rows),
        "aggregation": "Macro: equal weight per scene; pooled: pixel-weighted. MAE uses valid predictions only.",
    }


def draw_overview(data, out, rows):
    fig, axes = plt.subplots(5, 3, figsize=(15, 19), layout="constrained")
    for ax, row in zip(axes.flat, rows):
        scene = row["scene"]
        image = read_image(data / scene / "im0.png")
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{scene}: {row['mae_px']:.2f} px, {row['valid_fraction']:.1%} valid", fontsize=11)
        ax.axis("off")
    for ax in axes.flat[len(rows):]:
        ax.axis("off")
    fig.suptitle("Middlebury trainingQ: all scenes (MAE on valid predictions)", fontsize=16)
    fig.savefig(out / "overview.png", dpi=100)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), layout="constrained")
    labels = [r["scene"] for r in rows]
    axes[0].barh(labels, [r["mae_px"] or 0 for r in rows], color="#3674a3")
    axes[0].set_xlabel("Mean absolute disparity error (px)")
    axes[1].barh(labels, [100*r["valid_fraction"] for r in rows], color="#369579")
    axes[1].set_xlabel("Valid prediction fraction on GT-valid pixels (%)")
    axes[1].set_xlim(0, 100)
    for ax in axes:
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=.2)
    fig.savefig(out / "metrics.png", dpi=140)
    plt.close(fig)


def write_report(out, rows, summary, metadata):
    lines = ["# Part 1 - Classical Stereo: experiment results", "",
             f"UTC timestamp: {metadata['created_utc']}", "",
             "This is the stereo member's Part 1 deliverable. The frame-wise monocular/video baseline belongs to the other member.", "",
             "## Protocol", "",
             "15 Middlebury trainingQ pairs at native quarter resolution; left-view disp0GT.pfm. "
             "No rescaling, no training and no GT-based parameter tuning. Each scene uses the same SGBM settings; "
             "search range is the public calib.txt ndisp rounded up to a multiple of 16.", "",
             "Middlebury PFM float payloads are already pixel disparities: the header scale sign selects "
             "endianness; its magnitude is not applied. Rows are read bottom-up.", "",
             "OpenCV fixed-point disparity is divided by 16. An invalid prediction is a sentinel value, "
             "a disparity outside the configured search interval, or a nonpositive/nonfinite (d + doffs). "
             "Zero disparity may remain valid when doffs > 0. Invalid floating-point outputs are NaN.", "",
             "Ground-truth mask: isfinite(GT) & (GT > 0). Coverage denominator is all GT-valid pixels. "
             "MAE is measured on their intersection with valid predictions; no-GT or invalid-prediction pixels "
             "do not contribute to MAE. Therefore MAE must be read together with coverage. "
             "The non-occlusion mask is not used, so this is an all-GT-valid-pixel course exercise, not an official benchmark score.", "",
             "Depth in meters: Z = fx * baseline_mm / (disparity_px + doffs_px) / 1000. "
             "Use each scene's quarter-resolution calibration without scaling focal length again.", "",
             "Optional LR diagnostics do not change the primary predictions or metric mask. "
             "The reverse matcher has a negative disparity interval; residual is |dL + dR| "
             "at nearest-neighbor xR = round(xL - dL). Black in the LR map means unchecked, not consistent.", "",
             "## Results", "",
             f"- Equal-scene (macro) MAE: **{summary['macro_mae_px']:.4f} px**.",
             f"- Equal-scene valid fraction: **{summary['macro_valid_fraction']:.2%}**.",
             f"- Pixel-pooled MAE: **{summary['pooled_mae_px']:.4f} px**.",
             f"- Pixel-pooled valid fraction: **{summary['pooled_valid_fraction']:.2%}**.",
             f"- Mean left-matcher inference time: **{summary['mean_inference_ms']:.2f} ms/pair**.", "",
             "| Scene | Size (W x H) | Search disparities | MAE (px) | Valid (%) | Inference (ms) |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for r in rows:
        mae = f"{r['mae_px']:.4f}" if r["mae_px"] is not None else "N/A"
        lines.append(f"| {r['scene']} | {r['width']} x {r['height']} | {r['num_disparities']} | "
                     f"{mae} | {100*r['valid_fraction']:.2f} | {r['inference_ms']:.2f} |")
    lines += ["", "![Metrics](metrics.png)", "", "![All scenes](overview.png)", "",
              "## Efficiency and reproducibility", "",
              "Inference time is the median of timed left matcher.compute calls after one untimed warm-up "
              "per scene. Input decoding, grayscale conversion, allocation, right matching, visualization and saving "
              "are excluded. pipeline_ms is one warm-run of left matching plus validity and depth conversion "
              "on predecoded grayscale input; it also excludes I/O and optional LR. These are CPU measurements, not GPU FPS.", "",
              "```json", json.dumps(metadata["hardware"], indent=2, ensure_ascii=False), "```", "",
              "Exact settings, source hashes, command, input-file hashes, timing samples and units are in run.json "
              "and scene_metrics.json. Dataset archive hashes are in data/manifest.json. "
              "No claims of metric-depth accuracy are made from the disparity exercise alone.", "",
              "## Per-scene comparisons", ""]
    for r in rows:
        lines += [f"### {r['scene']}", "", f"![{r['scene']}](figures/{r['scene']}/comparison.png)", ""]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run(args):
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    validate_config(config)
    cv2.setNumThreads(config["opencv_threads"])
    data = args.data.resolve() / "trainingQ"
    selected = args.scenes or list(EXPECTED_SCENES)
    if len(set(selected)) != len(selected) or any(s not in EXPECTED_SCENES for s in selected):
        raise ValueError("Scenes must be unique official trainingQ scene names")
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {out}. Use a new --output path.")
    for folder in (out, out / "arrays", out / "figures"):
        folder.mkdir(parents=True, exist_ok=True)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running", "command": [sys.executable, *sys.argv],
        "source_files": {p.name: sha256(p) for p in [Path(__file__), args.config]},
        "config": config, "hardware": hardware_info(), "scenes": selected,
        "lr_diagnostic": args.lr_check, "dataset": str(data),
        "course_scope": "Stereo only; partner owns monocular frame baseline",
        "units": {"disparity": "pixels", "depth": "meters", "baseline": "millimeters"},
        "pfm_convention": "Middlebury native pixel float payload; scale sign is endian, magnitude ignored",
    }
    write_json(out / "run.json", metadata)
    rows = []
    for index, scene in enumerate(selected, 1):
        folder = data / scene
        calib = read_calib(folder / "calib.txt")
        left, right = read_image(folder / "im0.png"), read_image(folder / "im1.png")
        shape = (int(calib["height"]), int(calib["width"]))
        if left.shape[:2] != shape or right.shape != left.shape:
            raise ValueError(f"Image/calibration shape mismatch: {scene}")
        num_disp = int(np.ceil(calib["ndisp"] / 16.0) * 16)
        if num_disp + max(0, config["min_disparity"]) >= shape[1]:
            raise ValueError(f"Disparity interval exceeds image width: {scene}")
        gray_left, gray_right = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY), cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        matcher = create_matcher(config, num_disp)
        matcher.compute(gray_left, gray_right)  # Untimed warm-up.
        timings = []
        for _ in range(config["timing_repeats"]):
            start = time.perf_counter()
            raw = matcher.compute(gray_left, gray_right)
            timings.append((time.perf_counter() - start) * 1000)
        # One separately timed forward pipeline, with the exact result saved below.
        start = time.perf_counter()
        raw = matcher.compute(gray_left, gray_right)
        disparity = raw.astype(np.float32) / 16.0
        valid = prediction_valid(raw, config, num_disp)
        denominator = disparity + calib["doffs"]
        valid &= np.isfinite(denominator) & (denominator > 0)
        disparity[~valid] = np.nan
        depth = depth_from_disparity(disparity, valid, calib)
        pipeline_ms = (time.perf_counter() - start) * 1000
        # GT is deliberately loaded only after all predictions are computed.
        gt = read_pfm(folder / "disp0GT.pfm")
        if gt.shape != shape:
            raise ValueError(f"GT shape mismatch: {scene}")
        metrics, error = evaluate(disparity, valid, gt)
        metrics.update({"scene": scene, "width": shape[1], "height": shape[0],
                        "num_disparities": num_disp, "calib_ndisp": calib["ndisp"],
                        "inference_ms": statistics.median(timings), "timing_samples_ms": timings,
                        "pipeline_ms": pipeline_ms,
                        "input_sha256": {n: sha256(folder / n) for n in ["im0.png", "im1.png", "calib.txt", "disp0GT.pfm"]}})
        payload = {"raw_disparity_fixed16": raw, "disparity_px": disparity, "depth_m": depth,
                   "valid_prediction": valid, "gt_valid": np.isfinite(gt) & (gt > 0), "absolute_error_px": error}
        lr_status = None
        if args.lr_check:
            right_min = -config["min_disparity"] - num_disp + 1
            right_matcher = create_matcher(config, num_disp, right_min)
            start = time.perf_counter()
            right_raw = right_matcher.compute(gray_right, gray_left)
            metrics["lr_reverse_inference_ms"] = (time.perf_counter() - start) * 1000
            residual, lr_status = lr_diagnostic(disparity, valid, right_raw, right_min, config["lr_threshold_px"])
            available = int((lr_status > 0).sum())
            metrics["lr_checked_pixels"] = available
            metrics["lr_failed_fraction_of_checked"] = float((lr_status == 2).sum() / available) if available else None
            payload.update({"lr_residual_px": residual, "lr_status": lr_status})
        np.savez_compressed(out / "arrays" / f"{scene}.npz", **payload)
        metrics["visualization_ranges"] = draw_scene(out / "figures" / scene, scene, left, right,
                                                      disparity, valid, depth, gt, error, calib, config, metrics, lr_status)
        write_json(out / "figures" / scene / "metadata.json", metrics)
        rows.append(metrics)
        write_json(out / "scene_metrics.json", rows)
        print(f"[{index}/{len(selected)}] {scene}: MAE={metrics['mae_px']:.4f}px "
              f"coverage={metrics['valid_fraction']:.2%} time={metrics['inference_ms']:.1f}ms", flush=True)
    summary = summarize(rows)
    write_json(out / "summary.json", summary)
    columns = ["scene", "width", "height", "num_disparities", "mae_px", "valid_fraction",
               "gt_valid_pixels", "evaluated_pixels", "inference_ms", "pipeline_ms"]
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    draw_overview(data, out, rows)
    metadata["status"] = "complete"
    metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["summary"] = summary
    write_json(out / "run.json", metadata)
    write_report(out, rows, summary, metadata)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Results: {out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    p = subs.add_parser("prepare", help="Validate local ZIPs and extract only trainingQ")
    p.add_argument("--archives", type=Path, default=ROOT)
    p.add_argument("--data", type=Path, default=ROOT / "data")
    p.add_argument("--download", action="store_true", help="Download missing official archives (requires network)")
    p.set_defaults(func=prepare_data)
    p = subs.add_parser("run", help="Predict, evaluate and render stereo results")
    p.add_argument("--data", type=Path, default=ROOT / "data")
    p.add_argument("--config", type=Path, default=ROOT / "config.json")
    p.add_argument("--output", type=Path, default=ROOT / "results" / datetime.now().strftime("sgbm_%Y%m%d_%H%M%S"))
    p.add_argument("--scenes", nargs="+", choices=EXPECTED_SCENES)
    p.add_argument("--lr-check", action="store_true", help="Extra reverse-match diagnostic; never changes main metrics")
    p.set_defaults(func=run)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
