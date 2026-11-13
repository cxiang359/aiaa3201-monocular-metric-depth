$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONIOENCODING = 'utf-8'
python -m unittest -v test_part2
if ($LASTEXITCODE -ne 0) { throw 'Course protocol tests failed.' }
python prepare_assets.py --workers 8
if ($LASTEXITCODE -ne 0) { throw 'Assets are incomplete.' }
if (Test-Path -LiteralPath 'results/nyu_test_vits/run.json') {
    python -u run_part2.py --resume
} else {
    python -u run_part2.py
}
if ($LASTEXITCODE -ne 0) { throw 'Metric-depth evaluation failed; predictions can be resumed.' }
python verify_part2.py
if ($LASTEXITCODE -ne 0) { throw 'Result verification failed.' }
python summarize_results.py
if ($LASTEXITCODE -ne 0) { throw 'Chinese summary generation failed.' }
