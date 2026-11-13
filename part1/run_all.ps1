$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
python -m unittest -v test_stereo
if ($LASTEXITCODE -ne 0) { throw 'Correctness checks failed.' }
python stereo.py prepare
if ($LASTEXITCODE -ne 0) { throw 'Dataset preparation failed.' }
$runOutput = Join-Path 'results' ('sgbm_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
python stereo.py run --lr-check --output $runOutput
if ($LASTEXITCODE -ne 0) { throw 'Stereo experiment failed.' }
python make_failure_figures.py --output $runOutput
if ($LASTEXITCODE -ne 0) { throw 'Diagnostic figures failed.' }
python verify_results.py --output $runOutput
if ($LASTEXITCODE -ne 0) { throw 'Artifact verification failed.' }
