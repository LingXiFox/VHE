# 实验记录

使用 `python tools/new_experiment.py --id EXP-0001 --title "..." --hypothesis "..."` 创建实验目录。编号单调递增；禁止覆盖已有 ID。**失败和中止也必须保留。**

每个实验包含：`README.md`（假设/变量/对照/结果/失败）、`metadata.json`（环境/时间/git）、`metrics.jsonl`（运行产生；不进 Git）、`artifacts/`（音频/权重；不进 Git）、`checksums.sha256`（有需要时手动补充）。
