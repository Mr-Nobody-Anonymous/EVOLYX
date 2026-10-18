# PowerShell script to run EVOLYX test suite
$env:PYTHONPATH = "C:\Users\hp\Desktop\EVOLYX"
Write-Host "Running EVOLYX test suite..."
python -m pytest tests/ -v
