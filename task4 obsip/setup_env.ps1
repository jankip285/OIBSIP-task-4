$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ScriptRoot

$pythonLauncher = "py -3.11"
try {
    & $pythonLauncher --version | Out-Null
} catch {
    Write-Error "Python 3.11 is required. Install Python 3.11 and retry."
    Exit 1
}

if (-Not (Test-Path -Path ".venv")) {
    Write-Host "Creating virtual environment..."
    & $pythonLauncher -m venv .venv
}

Write-Host "Activating virtual environment..."
$activate = Join-Path $ScriptRoot ".venv\Scripts\Activate.ps1"
if (-Not (Test-Path $activate)) {
    Write-Error "Virtual environment activation script not found."
    Exit 1
}

Write-Host "Installing dependencies..."
& $pythonLauncher -m pip install --upgrade pip setuptools wheel
& $pythonLauncher -m pip install -r requirements.txt

Write-Host "Setup complete. Run './run_app.ps1' to launch the app."