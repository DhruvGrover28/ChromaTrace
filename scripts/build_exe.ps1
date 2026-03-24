$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "Creating venv..."
    python -m venv .venv
}

& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r requirements.txt
& $venvPython -m pip install pyinstaller

$distDir = Join-Path $projectRoot "dist"
if (Test-Path $distDir) {
    Remove-Item $distDir -Recurse -Force
}

& $venvPython -m PyInstaller --noconsole --onefile --name ChromaTrace src\vision_lab.py

Write-Host "Build complete: dist\ChromaTrace.exe"
