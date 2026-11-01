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
### 数据（v2 缓存，mandarin_formant 后端）
| 缓存 | 请求 | 写入 | 拒绝 | 主要拒绝原因 |
|---|---|---|---|---|
| train40x10-v2 | 369 | **217（58.8%）** | 152 | pitch_gap_interpolated 146、status_rejected/error_issues 8、low_pitch_coverage 4、no_alignment/no_pitch_frames 2、low_boundary_confidence 1 |
| dev12x5-v2 | 60 | **34（56.7%）** | 26 | pitch_gap_interpolated 26 |

产率与 EXP-0023 就绪率探针（56.7%）一致，说明配对管线可复现；瓶颈仍是 F0 内部间隙，不是对齐。

### 臂 A（overfit 8 对，60 步，lr 3e-4，channels 16，crop 1.6 s，batch 8）
- 实跑 5.19 s（RTX 4090 Laptop，CUDA，FP32）。
- total：step 1 **3.7401** → step 60 **3.4169**（0.914×），best 3.4148；期间 step 45 反弹到 3.9523。
- 分项（step 60）：spectral_convergence 0.8087、log_magnitude 2.0023、periodicity 0.2261。
- **预注册判据 1 失败**（要求 ≤0.6×，实测 0.914×）；无 NaN、无长度/形状错误（判据 3 通过）。
- 日志：`logs/arm_a_overfit8.log.jsonl`、`logs/arm_a_overfit8.summary.json`。
- 检查点：`checkpoints/step-000020/000040/000060.{pt,json}`（远端保留）。

### 臂 B（随机裁切 400 步，lr 3e-4，batch 4，dev 每 50 步评估）
- 实跑 10.73 s。
- train total 首值 4.0874 → 末值 3.3406；summary 记 best_total **2.8744**（落在未记录日志的步），日志内最低点 2.9711@step 240；波动大（step 20 4.0567、step 100 3.9275）。
- **dev total 单调下降：3.4554（100 步）→ 3.3862（200）→ 3.3125（300）→ 3.2451（400）**，末值比首次评估低 6.1%，判据 2 通过。
- 分项（step 400）：spectral_convergence 0.8582、log_magnitude 1.8448、periodicity 0.2640。
- 日志：`logs/arm_b_400.log.jsonl`、`logs/arm_b_400.summary.json`。
- 检查点：`checkpoints/step-000100/000200/000300/000400.{pt,json}`（远端保留）。

### 判据汇总
| 判据 | 结果 |
|---|---|
| 1. 臂 A 第 60 步 total ≤ 0.6×首值 | **失败**（0.914×） |
| 2. 臂 B 末次 dev < 首次 dev | 通过（3.4554 → 3.2451） |
| 3. 无 NaN / 长度形状错误 | 通过 |

### 主观听音
未做。本实验不产出音频，也不能用 loss 下降推断自然度。

## 失败、负面发现、错误日志
- **失败 1（配对缓存 run 1，2026-11-01）**：`vhe pair-cache` 对 369 条 train / 60 条 dev 全部拒绝（`written: 0`），原因计数 `status_blocked 367`、`error_issues 369`、`no_alignment 369`、`no_pitch_frames 369`、`status_rejected 2`。单句探针显示真实原因是 `backend_inventory_insufficient`：`build_pair` 未收到 backend，退回 `get_backend()` 的默认 formant_proxy（inventory `vowels`），6/6 音节无法渲染（`onset 'j' is not in the 'vowels' inventory`）。EXP-0023 的探针显式传了 `MandarinFormantSynth()`，而 `build_pair_cache` 没传，属管线接线遗漏。
- 修复：`build_pair_cache(..., backend_name="mandarin")`，CLI 增加 `--backend`；commit `95c9cf9`。失败的缓存目录 `derived/aishell3-pairs/p2-train40x10` 与 `p2-dev12x5` 原样保留（0 条 pair），重建写入 `-v2` 后缀目录，不覆盖。
- 教训：manifest 的 `renderable` 是用 mandarin 36 音素表算的，任何下游渲染器都必须用同一 inventory，否则就绪率会以「全拒」而不是报错的形式暴露。
- **失败 2（配对缓存 run 2，2026-11-01）**：`--backend` 默认值写成 `"mandarin"`，而 `get_backend()` 以 backend 自身名字为键（`('formant_proxy', 'mandarin_formant')`），启动即 `KeyError: "Unknown synth backend 'mandarin'"`，未写出任何目录。修复：默认改为 `mandarin_formant`，commit `cf1f428`。
- **失败 3（流程）**：失败 1 的修复先提交、后才把失败原因补进本文件；失败 2 同理。记录晚于提交，属流程违规，保留不删。

## 结论 / 下次实验
- 配对缓存与训练管线在 4090 上跑通，端到端 400 步只需 11 s；dev loss 下降说明训练确实在学习，不是纯噪声。
- 但「8 对过拟合」在 60 步、lr 3e-4 下没有达成（0.914×），**不能声称 P2 的 overfit 判定项已通过**。是否只是步数预算问题 → EXP-0026（600 步、lr 3e-3）。
- 缓存产率 58.8%，缺失的 41% 全部由 F0 间隙政策挡掉；要扩大训练集必须先解决浊音判决（EXP-0025 未做，见 EXP-0024 结论）。
- 未做：GPU 峰值显存（未插桩）、语音听辨、dev 音频导出、多 seed 复现。检查点写了但未在真实训练上做 resume 演练（只有单测覆盖）。