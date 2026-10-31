# Run from extracted VHE/ directory. Initializes LOCAL git only. No remote, no push.
$ErrorActionPreference = "Stop"
if (-not (Test-Path "pyproject.toml")) { throw "Run this from the VHE project root." }
if (-not (Test-Path ".git")) { git init -b main }
py -3 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -e ".[dev]"
& .\.venv\Scripts\python.exe -m pytest -q
Write-Host "Review git status, then: git add .; git commit -m 'chore: bootstrap VHE research'"
