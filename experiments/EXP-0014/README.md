# EXP-0014 — F0 tracking on real speech (small sample) and synthetic ground truth

## 假设

The YIN/CMND estimator with the EXP-0011 octave rule tracks synthetic vowels within 2% of ground truth from 80 to 400 Hz, and on a small real subset (20 AISHELL-3 utterances, 44.1 kHz) shows no systematic octave error: median F0 per utterance is coherent with speaker gender and re-estimating at 24 kHz changes the median by less than 3%.

## 变量 / 对照组 / 数据划分

- 变量 A（合成真值）：音高轮廓，5 条常数（110/165/210/300/350 Hz）+ 上行滑音 110→220 Hz + 下行滑音 300→130 Hz + 颤音 200±12 Hz@5 Hz；每个轮廓用 `MandarinFormantSynth` 渲染 /a/，F0 真值就是传入的 `PitchCurve`。
- 变量 B（估计器设置）：默认 `fmax_hz=900` vs `fmax_hz=500`。其余参数固定 `frame_ms=40`、`hop_ms=10`、`fmin_hz=65`、`voicing_threshold=0.45`。
- 变量 C（真实语音）：test split 中每位说话人取 1 条、时长 ≤ 3 s、共 12 人（来源 manifest `651bbbac…`），44.1 kHz 重采样到 24 kHz。
- 对照组：无真实语音标注，因此真实部分只做描述性统计（发声比例、F0 区间、相邻帧八度跳变率），不做精度声明。

## 完整复现命令 / 环境与哈希

```
python experiments/EXP-0014/probe_f0.py \
  --manifest /Volumes/LingXiWorkspace/VHE-Datasets/derived/aishell3-scan/v2/manifest.jsonl \
  --output experiments/EXP-0014/logs/f0_report.json
```

- 输出：`logs/f0_report.json`（有效）、`logs/f0_report_v1_unfiltered.json`（有缺陷的第一版，保留）。
- Git `43d3218` + 未提交工作树；macOS arm64 / Python 3.14.6 / CPU，纯 numpy，无 torch。
- 只读 12 个 WAV，未批量生成音频、未训练。

## 预注册通过/失败条件

1. 合成常数音：默认设置在 80–400 Hz 全部 |误差| ≤ 2%（≈34 cents）→ **失败**（见结果）。
2. 合成滑音/颤音：相邻帧八度跳变率 ≤ 5% → **失败**。
3. 真实语音：每位说话人的中位 F0 落在 65–500 Hz 且同一说话人内八度跳变率 ≤ 10% → 通过。
4. 判据失败即视为"估计器不可用于成对数据构造"，不得用调参掩盖。

## 实际结果与主观听音

合成（`hit50` = 落在 ±50 cents 的帧比例，`oct` = |误差| > 550 cents 的帧比例，`med` = 中位绝对误差 cents）：

| 轮廓 | 默认 hit50 | 默认 oct | 默认 med | fmax=500 hit50 | fmax=500 oct | fmax=500 med |
|---|---|---|---|---|---|---|
| constant 110 | 1.00 | 0.00 | 5 | 1.00 | 0.00 | 5 |
| constant 165 | 1.00 | 0.00 | 5 | 1.00 | 0.00 | 5 |
| constant 210 | **0.02** | **0.98** | **2404** | 1.00 | 0.00 | 5 |
| constant 300 | **0.02** | **0.98** | **1903** | 1.00 | 0.00 | 5 |
| constant 350 | 1.00 | 0.00 | 5 | 1.00 | 0.00 | 5 |
| glide 110→220 | 0.89 | 0.11 | 5 | 1.00 | 0.00 | 5 |
| glide 300→130 | 0.67 | 0.33 | 7 | **0.88** | **0.12** | 5 |
| vibrato 200±12 | 0.64 | 0.36 | 1 | **0.94** | **0.06** | 1 |

- 210 / 300 Hz 的默认误差约 2 个八度（2404 / 1903 cents），与 EXP-0011 附录 E5 的 842 / 902 Hz 一致：F1≈800 Hz 落在 3f0/4f0 上，CMND 首个谷抓到谐波。350 Hz 反而正常（F1 不在整数倍上），说明缺陷随 f0 与共振峰的关系离散出现，不是单调的"高音不准"。
- 即使 `fmax_hz=500` 把常数音修好，**下行滑音仍有 12% 八度帧、颤音 6%**：滑音跨越 `fmax` 以下但经过谐波密集区时仍会锁错，估计器缺少帧间连续性约束。

真实语音（12 位不同说话人，每位 1 条，24 kHz，fmax=500）：

| 指标 | 范围 | 汇总 |
|---|---|---|
| 发声帧比例 | 0.415 – 0.702 | 中位 ≈ 0.52 |
| 每条中位 F0 | 120.9 – 289.9 Hz | 12 条中位值的中位数 225.9 Hz |
| 相邻帧八度跳变率 | 0.000 – 0.056 | 均值 0.0248 |
| 越界帧（<65 或 >500 Hz） | 0 – 1 帧/条 | 2 条各有 1 帧 |

- 中位 F0 与说话人性别相容（`SSB1328` 120.9 Hz、`SSB0710` 133.3 Hz 为男声；其余 200–290 Hz 为女声），未见系统性上偏；但 `p05` 出现 69.8–95.9 Hz 这类"半频"离群值，与 2–5% 的相邻帧八度跳变一致——是逐帧抖动，不是整条偏移。
- 未做听音；本实验不评价自然度。

## 失败、负面发现、错误日志

- **假设 1 与 2 均失败**：估计器在 210/300 Hz 常数音上默认输出约 4f0/3f0，且滑音/颤音即使限定搜索范围也仍有 6–12% 八度帧。上一轮 EXP-0011 的结论只在 65–900 Hz 的部分频段成立，不能外推为"可用"。
- **第一版真实统计有实现缺陷（保留证据）**：`logs/f0_report_v1_unfiltered.json`。缺陷是把 `estimate_f0` 的无声帧（`f0_hz = 0`、`voicing = 0`）当成测量值，导致 `voiced_ratio = 1.0`、`median_f0 = 0.0`、约 90 帧"越界"。修正为按 `voicing >= 0.45 且 f0 > 0` 过滤后重跑，才得到上表。旧文件未覆盖，用于记录这次口径错误。
- **第一版取样还有第二个缺陷**：按 manifest 顺序取前 12 条，全部来自同一说话人 `SSB0997`，统计被单人主导。已改为每位说话人取 1 条。两处都写在这里，不悄悄改。
- **假设中"44.1 kHz 与 24 kHz 重估中位数差 < 3%"未测**：脚本只在 24 kHz 上估计一次，该子假设本轮无数据，不能声称通过。
- 未解决：估计器无置信度输出、无帧间连续约束；`fmax_hz` 需由调用方按说话人音域设定，而真实场景没有该先验。

## 结论 / 下次实验

- 结论：**当前 F0 估计器不能直接用于成对数据构造**。若必须提取，只能 `fmax_hz ≤ 500` 并在成对上记录"F0 可能含八度错误"的标志，不能默认可信。
- 下次（EXP-0015）：加多候选（lag, 2·lag, 3·lag）子谐波校验 + 帧间连续性（相邻帧比值约束/维特比平滑），用本实验同一套合成轮廓与 12 条真实语音做回归；判据沿用本文件的 1–3。修复后再做 P1.4 的成对构造，避免把八度错误固化进训练数据。
