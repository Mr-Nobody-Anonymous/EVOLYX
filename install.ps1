# EVOLYX One-Click Windows Installer
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "         EVOLYX Windows Automated Installer               " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Check Python
$pyCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pyCmd) {
    Write-Error "Python 3 is not found on your PATH. Please install Python 3.10+ from python.org or the Microsoft Store."
    exit 1
}

$pyVersion = python --version 2>&1
Write-Host "[+] Found Python: $pyVersion" -ForegroundColor Green

# 2. Check or create Virtual Environment
$venvPath = Join-Path $PSScriptRoot ".venv"
if (-not (Test-Path $venvPath)) {
    Write-Host "[*] Creating virtual environment at .venv..." -ForegroundColor Yellow
    python -m venv $venvPath
}

# 3. Activate Virtual Environment
$activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    Write-Host "[*] Activating virtual environment..." -ForegroundColor Yellow
    & $activateScript
}

# 4. Run the Python Universal Installer
$env:PYTHONPATH = $PSScriptRoot
python "$PSScriptRoot\install.py" @args
