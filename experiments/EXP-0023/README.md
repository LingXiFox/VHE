# EXP-0023 — Forced-alignment boundaries in the pairing pipeline and the listening package

## 假设
Replacing g2p nominal onset/nucleus durations with chain-HMM phone spans keeps the mechanical render aligned to the real utterance better than the energy-valley DP, and the resulting 20-pair package lets a human judge content and timing preservation.

## 变量 / 对照组 / 数据划分
- 唯一变量：机械音频的**音素时长来源**。
  - 对照 A（`dp`）：能量谷 DP 给音节窗口，音节内部用 `g2p.syllable_plan` 的名义时长（声母固定名义值，韵母吃剩余）。
  - 实验 B（`forced`）：链式 HMM 强制对齐给出每个音素的 span，机械合成直接用这些实测时长。
- 其余全部固定：同一批 wav、同一 `PRESETS["yang"]` 音色模板、同一 F0 跟踪器（viterbi）、同一增益匹配、同一 24 kHz 输出、同一 seed。
- 数据：v4 manifest 的 **dev** 划分（对齐器只用 train 训练，说话人零重叠），4 个说话人 x 5 句 = 20 句，seed 42。
- 交付物：每句三个 WAV（`real`/`mech_forced`/`mech_dp`）+ 一张逐句表 + 逐音节时长表。音频**不进 Git**，落在数据盘 `derived/aishell3-listen/EXP-0023/`。

## 完整复现命令 / 环境与哈希
见 `metadata.json` 的 `command`。执行机：4090 WSL2（`ssh -p 2222 hands@10.0.0.190`），模型与索引都落在 `~/datasets/VHE-Datasets/derived/`。

## 预注册通过/失败条件
主要判据是人工听音（本实验存在的理由就是它），客观指标只作记录、不作 gating：

1. **人工听音（主判据）**：主人对 20 对机械音频按 1–5 打分，两个维度——内容可辨识度（能否听出原句的字）与时间对齐（音节是否落在原句位置）。
   通过 = 两个维度中位数均 ≥ 4，且**没有任何一对出现音节数错误、漏字或明显错位**。
2. **客观记录（不判定成败）**：逐句 log-mel 距离（real vs mech，两个臂各一列）、逐音节时长比（机械/真人）及其分布、两臂的声母时长占比差异、被阻塞的句子数。
3. 若 20 对里出现阻塞（音素不在 36 音素表、对齐序列与转写不一致），必须逐条列出原因，不能静默换句。

失败判据同样预注册：只要出现 1 对音节数错误或漏字，本实验即判负，需要先修对齐/配对再谈听感。

## 实际结果与主观听音
未运行。

## 失败、负面发现、错误日志
TODO（即便未成功也必须记录）

## 结论 / 下次实验
TODO
