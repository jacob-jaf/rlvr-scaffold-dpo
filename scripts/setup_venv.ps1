# Create or refresh the repo .venv. Use python.org Python, not Anaconda.
# Usage: powershell -ExecutionPolicy Bypass -File scripts/setup_venv.ps1 [-Train] [-OllamaOnly]

param(
    [switch]$Train,
    [switch]$OllamaOnly
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path $PSScriptRoot -Parent
Set-Location $RepoRoot

$Py312 = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
if (-not (Test-Path $Py312)) {
    throw "Python 3.12 not found at $Py312. Install: winget install Python.Python.3.12"
}

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Creating .venv with $Py312 ..." -ForegroundColor Cyan
    & $Py312 -m venv .venv
}

$VenvPy = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$cfg = Get-Content ".venv\pyvenv.cfg" -Raw
if ($cfg -match "anaconda") {
    Write-Host "Removing Anaconda-based .venv and recreating ..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force ".venv"
    & $Py312 -m venv .venv
    $VenvPy = Join-Path $RepoRoot ".venv\Scripts\python.exe"
}

Write-Host "Venv Python:" -ForegroundColor Cyan
& $VenvPy -c "import sys; print(sys.executable)"

& $VenvPy -m pip install --upgrade pip wheel

if ($OllamaOnly -and -not $Train) {
    & $VenvPy -m pip install -r requirements-ollama.txt
    Write-Host "Ollama-only venv ready." -ForegroundColor Green
    exit 0
}

Write-Host "Installing CUDA PyTorch (cu124) ..." -ForegroundColor Cyan
& $VenvPy -m pip install torch --index-url https://download.pytorch.org/whl/cu124

Write-Host "Installing training stack ..." -ForegroundColor Cyan
& $VenvPy -m pip install -r requirements-train.txt

Write-Host "`n=== smoke test ===" -ForegroundColor Cyan
& $VenvPy -c @"
import torch
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu', torch.cuda.get_device_name(0))
import pyarrow; print('pyarrow', pyarrow.__version__)
from datasets import Dataset
from trl import DPOTrainer, DPOConfig
print('DPOTrainer import OK')
"@

Write-Host "`nVenv ready: $VenvPy" -ForegroundColor Green
Write-Host "Activate:  .\.venv\Scripts\Activate.ps1"
Write-Host "Or set:    `$env:RLVR_PYTHON = '$VenvPy'"
