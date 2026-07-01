$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ScriptRoot

if (-Not (Test-Path -Path ".venv")) {
    Write-Error "Virtual environment not found. Run './setup_env.ps1' first."
    Exit 1
}

& ".venv\Scripts\Activate.ps1"
python app.py