# Run EVOLYX FastAPI Server
$env:PYTHONPATH = "C:\Users\hp\Desktop\EVOLYX"
Write-Host "Starting EVOLYX REST API on http://127.0.0.1:8000..."
python -m uvicorn apps.api.main:app --host 127.0.0.1 --port 8000 --reload
