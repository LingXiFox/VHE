# EXP-0025 — Can a small VHE model overfit cached pairs and improve dev loss in a short run?

## 假设
16 通道 ResidualUNet 能把 humanization loss 在固定 8 对缓存上明显压低；在 369 条对齐后的 train 缓存上做 400 步随机裁切训练，能把它自己的 dev loss 压到低于初始值，且输出长度/采样率与输入一致。

## 变量 / 对照组 / 数据划分
- 臂 A（overfit）：8 对固定 batch，60 步，lr 3e-4，channels 16，crop 1.6 s。目的是证明「能过拟合小批次且没有长度/形状错误」（MILESTONES P2 判定项）。
- 臂 B（短跑）：train 缓存（40 说话人 × 至多 10 句）随机裁切 400 步，batch 4，每 50 步在 dev 缓存（12 说话人 × 5 句，居中裁切）上评估。
- 两臂共用同一份 train/dev 缓存；说话人不交叉（train 40 人 / dev 12 人，来自 v4 manifest 的 train/dev split）。
- 对照组即「第 1 次评估」与「第 400 步评估」，不是外部 baseline；本实验只验证训练管线可用，不宣称自然度提升。

## 数据与中间产物（不入 Git）
- manifest: `~/datasets/VHE-Datasets/derived/aishell3-scan/v4/manifest.jsonl`（sha256 1a8138c4d13e547d1d266a7aaf50d3a002a8e32dad78b0555f1b417c77074f31）
- 对齐索引: `derived/aishell3-align/train40x10.jsonl`（369 句）、`derived/aishell3-align/dev12x5.jsonl`（60 句），对齐模型 v3。
- 配对缓存: `derived/aishell3-pairs/p2-train40x10`、`derived/aishell3-pairs/p2-dev12x5`。

## 预注册通过/失败条件
1. 臂 A：第 60 步 total ≤ 0.6 × 第 1 步 total，且全程无 NaN、输出长度 == 输入长度（24000 Hz，1.6 s = 38400 样本）。
2. 臂 B：最后一次 dev total < 第一次 dev total（即训练在 dev 上不是纯噪声）。
3. 两臂都不得出现长度或形状错误；出现即判失败并记录日志。

以上任一不满足 → 结论为「管线可用性未达成」，不得解释为模型无效，也不得宣称自然度。

## 已知局限（运行前写下）
- dev 只有 60 对、且都是 forced 对齐通过的句子，训练分布可能偏易。
- humanization loss 下降不等于听感变好（AGENTS.md）；本实验不做听音。
- 单机单卡 RTX 4090 Laptop 16 GB，FP32，无混合精度。
- 配对缓存的产率由 EXP-0023/0024 的就绪策略决定（约 56.7%），未通过的句子不进入训练。

## 实际结果与主观听音
未运行（运行前写入本节）。

## 失败、负面发现、错误日志
待运行。

## 结论 / 下次实验
待运行。
