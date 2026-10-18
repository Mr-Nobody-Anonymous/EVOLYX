# EVOLYX Environment Bootstrap Script (PowerShell)
Write-Host "=========================================="
Write-Host "  Bootstrapping EVOLYX AI Research Platform"
Write-Host "=========================================="

# Check Python
$pyVer = python --version 2>&1
Write-Host "Python detected: $pyVer"

# Set PYTHONPATH
$env:PYTHONPATH = "C:\Users\hp\Desktop\EVOLYX"

# Ensure directories exist
$dirs = @("artifacts\models", "artifacts\datasets", "artifacts\evaluations", "experiments\results")
foreach ($d in $dirs) {
    if (-not (Test-Path $d)) {
        New-Item -ItemType Directory -Path $d -Force | Out-Null
    }
}

# Run test battery
Write-Host "`nRunning initial verification tests..."
python -m pytest tests/

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n[SUCCESS] EVOLYX Platform verified and ready."
    Write-Host "To start the CLI:        .\scripts\run_cli.ps1"
    Write-Host "To start the REST API:   .\scripts\run_api.ps1"
    Write-Host "To run an experiment:    .\scripts\run_experiment.ps1"
} else {
    Write-Error "Verification tests failed."
}
