#!/usr/bin/env bash
set -euo pipefail
[[ -f pyproject.toml ]] || { echo "Run from VHE project root" >&2; exit 1; }
[[ -d .git ]] || git init -b main
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest -q
echo "Review git status, then: git add . && git commit -m 'chore: bootstrap VHE research'"
