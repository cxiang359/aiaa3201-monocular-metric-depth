# Part 1A / Part 2A publication

## Layout and ownership

This branch starts from `main` and adds the Classical Stereo and Single-Image Metric Depth deliverables. It does not include the separate `part1b-2b` video branch or claim Part 3 is complete.

```text
part1/classical_stereo/
  README.md, ANALYSIS.md
  stereo.py, config.json, requirements.txt, test_stereo.py
  run_all.ps1, verify_results.py, make_failure_figures.py, export_html.py
  provenance/data_manifest.json
  results/sgbm_final/
    REPORT.md, metrics.csv, summary.json, run.json, scene_metrics.json
    verification.json, test_results.txt, metrics.png, overview.png
    figures/<scene>/comparison.png, metadata.json
    failure_cases/*.png, roi_metrics.json
part2/metric_depth/
  README.md, ANALYSIS.md, REFERENCES.bib
  *.py, requirements-inference.txt, run_all.ps1, case_notes.json
  provenance/asset_manifest.json, weights_manifest.json
  logs/upstream_commit.txt, dependency_versions.json, protocol_tests.txt
  results/nyu_test_vits/
    REPORT.md, metrics.csv, summary.json, run.json, verification.json
    selected_cases.json, analysis_details.json, distribution.png
    figures/nyu_*.png
```

Each A deliverable is self-contained under `part1/classical_stereo/` or `part2/metric_depth/`. Code resolves paths relative to its own file, so moving the complete module preserves internal paths and experiment source hashes. After merging the separate video branch, these sit beside `part1/mono_baseline/` and `part2/video_depth/`. Root README and ignore-rule changes should still be reviewed when merging both branches.

## Included evidence

- Complete 15-scene stereo metric table, calibration/protocol metadata, overview, per-scene comparison panels, and three failure crops.
- Complete 654-image NYU metric CSV, fixed model/settings, five representative/difficult comparison panels, and distribution plot.
- Original inference/evaluation source files and existing correctness tests.
- Recorded validation results from the complete local runs, dated in each verification JSON. These historical records describe the full local artifacts, including files intentionally not published.
- Data/checkpoint source URLs, hashes and pinned versions, without redistributing those assets.

Absolute workstation paths in published run metadata are replaced with `<project>` or `python`. Measured numbers, source hashes and checkpoint/data hashes are preserved. Documentation is adapted for GitHub Markdown; generated HTML is omitted.

## Intentionally local

Datasets, ZIP archives, model weights, vendored UniDepth/dependencies, raw NPY/NPZ predictions, 654 individual sample records, generated embedded HTML, temporary/cache files, invalid early experiments, course PDF and downloaded papers are not uploaded.

The tracked verification JSON files are evidence of earlier complete local checks, not a claim that raw predictions are bundled. To independently recompute metrics on a new machine, download the data and weights, regenerate predictions and sample records, then run the verifier. Simply cloning this branch is enough to read the results, but not enough to rerun full raw-output verification.

## Reproduction entry points

- Stereo setup and execution: [part1/classical_stereo/README.md](part1/classical_stereo/README.md).
- UniDepth setup and execution: [part2/metric_depth/README.md](part2/metric_depth/README.md).
- Run scripts produce local datasets, floating-point outputs and HTML previews, all excluded from Git.
- UniDepth upstream source is cloned at the recorded commit, rather than copied into this repository. Respect the upstream model license (CC BY-NC 4.0); attribution is in `part2/metric_depth/REFERENCES.bib`.

## Scope of results

Stereo MAE and valid coverage are reported together. NYU metrics use the course crop and equal-image averaging, without test-set scale fitting. Timings are CPU measurements with the documented scope; they are not comparable to video GPU timings as a model speed ranking. Part 3 improvements, application Demo and the final combined report remain separate work.
