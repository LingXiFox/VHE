"""Create an append-only experiment record. Never reuses experiment IDs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys


def git_info(root: Path) -> dict:
    def query(*args: str) -> str:
        result = subprocess.run(["git", *args], cwd=root, text=True,
                                capture_output=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else "unavailable"
    return {"commit": query("rev-parse", "HEAD"),
            "status": query("status", "--porcelain")}


def create_record(root: Path, exp_id: str, title: str, hypothesis: str) -> Path:
    if not exp_id.startswith("EXP-") or len(exp_id) != 8 or not exp_id[4:].isdigit():
        raise ValueError("Experiment IDs must match EXP-0001, EXP-0002, etc.")
    target = root / "experiments" / exp_id
    target.mkdir(parents=True, exist_ok=False)  # Fail rather than overwrite.
    (target / "artifacts").mkdir()
    meta = {"experiment_id": exp_id, "title": title, "hypothesis": hypothesis,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version, "platform": platform.platform(),
            "git": git_info(root), "seed": None, "dataset_manifest_sha256": None,
            "command": None, "torch_version": None, "cuda_version": None,
            "gpu": None, "config_sha256": None, "notes": "Fill before running."}
    (target / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
                                          encoding="utf-8")
    content = f"""# {exp_id} — {title}

## 假设
{hypothesis}

## 变量 / 对照组 / 数据划分
TODO

## 完整复现命令 / 环境与哈希
TODO (metadata.json)

## 预注册通过/失败条件
TODO

## 实际结果与主观听音
未运行。

## 失败、负面发现、错误日志
TODO（即便未成功也必须记录）

## 结论 / 下次实验
TODO
"""
    (target / "README.md").write_text(content, encoding="utf-8")
    return target


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--id", required=True)
    p.add_argument("--title", required=True)
    p.add_argument("--hypothesis", required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parent.parent
    print(create_record(root, args.id, args.title, args.hypothesis))


if __name__ == "__main__":
    main()
