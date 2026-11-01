# EXP-0015 — Reliable F0 estimation with multi-candidate CMND and Viterbi continuity

## 假设

Multi-candidate CMND with harmonic-support scoring and Viterbi continuity reads 110–350 Hz sustained vowels and the 110→220 Hz, 300→130 Hz and 200±12 Hz@5 Hz contours within 50 cents **at the default 65–900 Hz range (no fmax narrowing)**, and on 12 real AISHELL-3 utterances it removes most of the octave jumps of the single-dip estimator while explicitly rejecting frames it cannot resolve instead of interpolating.

来源：EXP-0014 记录的单 dip 估计器在 210/300 Hz 合成 /a/ 上 98% 帧读成 2×–4× 高（842/902 Hz），只有把 `fmax_hz` 压到 500 Hz 才恢复。用户要求修复根因，**不得用缩窄搜索范围掩盖**。

## 变量 / 对照组 / 数据划分

- 变量 A（候选与决策）：新跟踪器 `vhe.data.pitch.track_pitch`（多候选 CMND + 谐波支撑 + Viterbi 连续性 + 显式拒绝）。
- 对照组 1：`legacy_default` — `vhe.data.f0.estimate_f0`，默认 `fmax_hz=900`（EXP-0014 的失败配置）。
- 对照组 2：`legacy_fmax500` — 同一估计器但 `fmax_hz=500`（EXP-0014 中"能用但靠掩盖"的配置）。
- 数据 A（合成真值）：与 EXP-0014 完全相同的 8 条轮廓 — 常数 110/165/210/300/350 Hz，上行 110→220 Hz，下行 300→130 Hz，颤音 200±12 Hz@5 Hz；均由 `MandarinFormantSynth` 渲染 /a/（24 kHz），F0 真值即传入的 `PitchCurve`。
- 数据 B（真实语音）：与 EXP-0014 相同的 12 条 — test split、每位说话人 1 条、时长 ≤ 3 s、44.1 kHz→24 kHz，来自 manifest `10539f61…`（v3）。
- 唯一变量（v1→v2）：分析窗 `frame_ms` 40 → 30（见"失败与勘误"），其余参数不变。

## 完整复现命令 / 环境与哈希

```
python experiments/EXP-0015/probe_pitch.py \
  --manifest /Volumes/LingXiWorkspace/VHE-Datasets/derived/aishell3-scan/v3/manifest.jsonl \
  --output experiments/EXP-0015/logs/pitch_report_v2.json
```

- 环境：macOS-27.2-arm64 / Python 3.14.6 / CPU（纯 NumPy，无 torch、无 GPU）；Git 基线 `b9047f5` + 未提交的 pitch 工作树。
- 数据 manifest：`10539f61d015bc81dff88cb0ee5ba72b640ab703ee57373324f7c446a624c80f`（88035 句，218 说话人）；12 条样本与 EXP-0014 逐条相同，便于对照。
- 输出：`logs/pitch_report_v2.json`（定稿）、`logs/pitch_report_v1.json`（frame_ms=40 的失败版，保留作为窗口消融证据）。
- 只读 12 个 WAV，未批量生成音频、未训练、未改原始语料。

## 预注册通过/失败条件

1. 合成常数音：**默认** `fmax_hz=900` 下 8 条轮廓的 ±50 cents 命中率 ≥ 0.95，且八度错误率（|误差| > 550 cents）≤ 0.02。
2. 合成滑音/颤音：相邻帧八度跳变率 ≤ 5%，中位帧间跳变 ≤ 60 cents。
3. 真实语音：每人中位 F0 落在 65–500 Hz；相邻帧八度跳变率均值 ≤ 0.5%（EXP-0014 legacy 为 2.48%）。
4. 拒绝必须是显式的：被拒绝帧 `f0_hz == 0` 且有 reason，禁止插值；拒绝率要报告，不能靠降低覆盖率换取干净曲线。
5. 判据失败即记录为失败，不允许通过缩窄 `fmax` 或放大置信度阈值"修好"。

## 实际结果

### 合成真值（`hit50` / 八度错误率；`step` = 中位帧间跳变 cents）

| 轮廓 | legacy 默认 | legacy fmax=500 | 新跟踪器 | 新拒绝率 | 新 step |
|---|---|---|---|---|---|
| constant 110 | 1.00 / 0.00 | 1.00 / 0.00 | **1.00 / 0.00** | 0.00 | 1.3 |
| constant 165 | 1.00 / 0.00 | 1.00 / 0.00 | **1.00 / 0.00** | 0.00 | 1.4 |
| constant 210 | 0.02 / 0.98 | 1.00 / 0.00 | **1.00 / 0.00** | 0.00 | 1.4 |
| constant 300 | 0.02 / 0.98 | 1.00 / 0.00 | **1.00 / 0.00** | 0.00 | 1.4 |
| constant 350 | 1.00 / 0.00 | 1.00 / 0.00 | **1.00 / 0.00** | 0.00 | 1.3 |
| glide 110→220 | 0.89 / 0.11 | 1.00 / 0.00 | **1.00 / 0.00** | 0.00 | 18.9 |
| glide 300→130 | 0.67 / 0.33 | 0.88 / 0.12 | **1.00 / 0.00** | 0.00 | 24.0 |
| vibrato 200±12@5 Hz | 0.64 / 0.36 | 0.94 / 0.06 | **1.00 / 0.00** | 0.00 | 21.4 |

8/8 轮廓达标：默认 900 Hz 搜索范围内全部命中，零八度错误。判据 1、2 通过。

### 真实语音（12 条 / 12 位说话人，v2）

- 总帧 2052：`ok` 1077、`silence` 577、`no_candidate` 281、`octave_ambiguous` 69、`continuity_gated` 24、`low_confidence` 24。
- 平均拒绝率 **0.4835**；剔除静音帧后"本可发声但被拒"的比例 **0.2698**（398/1475）。
- 12 个说话人中位 F0 的中位数 **226.6 Hz**（legacy fmax=500 为 225.9 Hz，差 0.3%）。
- 相邻帧八度跳变率均值 **0.0027**（EXP-0014 legacy 0.0248，约 9× 降低）→ 判据 3 通过。
- 中位帧间跳变 6.76–31.16 cents（12 条中位数 15.0），无插值（判据 4 通过：拒绝帧全部 `f0_hz=0` + reason）。

| utterance | speaker | frames | legacy 中位 | 新中位 | 新发声比 | 拒绝率 | 非 ok reason | step | 八度跳变 |
|---|---|---|---|---|---|---|---|---|---|
| SSB09970003 | SSB0997 | 210 | 289.9 | 285.5 | 0.443 | 0.557 | silence 80 / no_candidate 12 / low_conf 6 / ambiguous 11 / gated 8 | 14.33 | 0.0000 |
| SSB12740001 | SSB1274 | 237 | 225.9 | 226.3 | 0.620 | 0.380 | silence 75 / no_candidate 4 / low_conf 1 / ambiguous 9 / gated 1 | 15.53 | 0.0000 |
| SSB13280001 | SSB1328 | 228 | 120.9 | 121.5 | 0.618 | 0.382 | silence 69 / no_candidate 11 / ambiguous 6 / gated 1 | 16.05 | 0.0000 |
| SSB14520002 | SSB1452 | 154 | 202.8 | 202.0 | 0.494 | 0.506 | silence 4 / no_candidate 64 / low_conf 1 / ambiguous 6 / gated 3 | 17.85 | 0.0000 |
| SSB00180248 | SSB0018 | 132 | 274.1 | 273.9 | 0.697 | 0.303 | silence 13 / no_candidate 23 / low_conf 3 / gated 1 | 17.08 | 0.0000 |
| SSB02880036 | SSB0288 | 143 | 216.0 | 217.0 | 0.385 | 0.615 | silence 22 / no_candidate 60 / low_conf 2 / ambiguous 4 | 12.98 | 0.0000 |
| SSB02990037 | SSB0299 | 157 | 221.0 | 220.6 | 0.516 | 0.484 | silence 37 / no_candidate 39 | 14.26 | 0.0000 |
| SSB04350040 | SSB0435 | 134 | 280.3 | 250.0 | 0.418 | 0.582 | silence 73 / no_candidate 3 / gated 2 | 15.02 | 0.0182 |
| SSB07100231 | SSB0710 | 149 | 133.3 | 133.1 | 0.477 | 0.523 | silence 70 / ambiguous 8 | 20.15 | 0.0143 |
| SSB07580064 | SSB0758 | 146 | 226.0 | 226.8 | 0.493 | 0.507 | silence 39 / no_candidate 26 / ambiguous 9 | 10.18 | 0.0000 |
| SSB09130001 | SSB0913 | 134 | 246.1 | 246.5 | 0.463 | 0.537 | silence 62 / no_candidate 2 / low_conf 1 / ambiguous 6 / gated 1 | 6.76 | 0.0000 |
| SSB10080006 | SSB1008 | 228 | 268.6 | 280.7 | 0.575 | 0.425 | silence 33 / no_candidate 37 / low_conf 10 / ambiguous 10 / gated 7 | 31.16 | 0.0000 |

未做主观听音（本阶段无听音评分流程），因此上表只是客观统计，不构成"F0 已可用于成对数据构造"的结论。

## 失败、负面发现、错误日志

1. **v1（`frame_ms=40`）失败**：下行滑音 300→130 Hz 命中率 0.84、八度错误 0.16，其余轮廓全对。根因：40 ms 窗内音高变化约 6%，真周期不再是 CMND 谷，而 /a/ 的 ~880 Hz 共振峰是静止的，于是 2×（442 Hz）获胜。改 30 ms 后 1.00/0.00，其余 7 条不变 → 作为"窗长"单变量消融保留（`logs/pitch_report_v1.json` 未删除）。
2. **整数倍候选不能当歧义**：纯 210 Hz 正弦在 2T/3T 处同样高度周期，最初把全部帧判 `octave_ambiguous`（拒绝率 1.0）。修正为"先按整数倍家族折叠，只保留最短周期，再判非整数比歧义"。
3. **基础 dip 选取两次踩坑**：按"最深 4 个"选会在 350 Hz 上丢掉基频（342/273/205/136，68 被挤掉）；按"最短 4 个"选会在 /a/ 165 Hz 上被共振峰浅谷（29/58/87/116）挤掉真正的 145。最终两段式：先取 `cmnd ≤ 0.15` 的强谷，再在其内取最短 4 个。
4. **`cmnd_curve` 的 off-by-one**：`lags[i] = i+1` 而代码用 `cmnd[lag]`，造成 +5…+40 cents 系统性偏高（65→65.18、350→355.18 Hz）。改 `cmnd[lag-1]` 后 65–800 Hz 纯音误差 ≤ 0.1 cents。
5. **家族折叠排序写反**：`sorted(key=lambda c: c.f0_hz)` 实际按"周期从长到短"排序，导致保留半频；且整数倍范围 2..4 太窄（500 Hz 被 6× 的 83.77 Hz 打败）。改为 `-f0_hz` + 2..16 整数倍。
6. **谐波支撑权重**：等权时颤音帧上 826 Hz（共振峰周期）与 206 Hz（真周期）只差 0.006 分；改为 `SUPPORT_WEIGHT=0.3`，支撑能否决明显半频，但不能压倒深度证据。
7. **歧义判据过宽**：最初用评分窗 0.10，干净 300 Hz /a/ 有 18% 帧因 ~900 Hz 共振峰的 90 Hz 伪候选被判歧义（发声率掉到 0.82）。改为比较 aperiodicity（比值 ≤ 1.5×，含 0.010 绝对下限）后恢复 1.00/0.00。
8. **流程勘误**：本实验目录最初由手工创建（先写 `probe_pitch.py` 后有记录），违反"先调用 `tools/new_experiment.py`"的规范。已用该工具重建骨架并把 `probe_pitch.py`、两份日志原样放回（md5 校验 v2 报告一致），未丢失任何失败结果。

## 结论 / 下次实验

- 结论：在**不缩窄 fmax** 的前提下，显式多候选 + 谐波支撑 + Viterbi 连续性把 EXP-0014 的两个致命失败（210/300 Hz 倍频、滑音/颤音）修复到合成真值 8/8 全对；真实语音的八度跳变率下降约 9×，代价是 27% 的"本可发声帧"被显式拒绝。
- 不宣称"F0 已可直接用于成对数据构造"：真实语音无真值标注，且拒绝率偏高、个别句子中位 F0 与 legacy 相差 0.18 倍频程（SSB04350040 280.3→250.0 Hz），原因未查明。
- 下次（仍属 P1，不训练）：①给 12 条样本做人工听音复核，确认 250.0 vs 280.3 Hz 谁对；②把拒绝帧在配对构造里显式标为"无 F0"，禁止回填；③评估 `no_candidate` 偏高的原因（清音/气声 vs max_cmnd 过严）；④再决定是否把 `track_pitch` 设为 `pairing` 默认，而不是现在的 `estimate_f0`。
