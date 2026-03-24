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

& $venvPython -m PyInstaller --noconsole --onedir --name ChromaTrace src\vision_lab.py

$zipPath = Join-Path $distDir "ChromaTrace-win64.zip"
if (Test-Path $zipPath) {
    Remove-Item $zipPath -Force
}

Compress-Archive -Path (Join-Path $distDir "ChromaTrace") -DestinationPath $zipPath

Write-Host "Build complete: dist\ChromaTrace\ (folder)"
Write-Host "Release zip: dist\ChromaTrace-win64.zip"
