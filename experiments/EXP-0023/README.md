# EXP-0023 — Forced-alignment boundaries in the pairing pipeline and the listening package

## 假设
Replacing g2p nominal onset/nucleus durations with chain-HMM phone spans keeps the mechanical render aligned to the real utterance better than the energy-valley DP, and the resulting 20-pair package lets a human judge content and timing preservation.

## 变量 / 对照组 / 数据划分
- 唯一变量：机械音频的**音素时长来源**。
  - 对照 A（`dp`）：能量谷 DP 给音节窗口，音节内部用 `g2p.syllable_plan` 的名义时长（声母固定名义值，韵母吃剩余）。
  - 实验 B（`forced`）：链式 HMM 强制对齐给出每个音素的 span，机械合成直接用这些实测时长。
- 其余全部固定：同一批 wav、同一 `PRESETS["yang"]` 音色模板、同一 F0 跟踪器（viterbi）、同一增益匹配、同一 24 kHz 输出、同一 seed。
- 数据：v4 manifest 的 **dev** 划分（对齐器只用 train 训练，说话人零重叠），8 个说话人 x 3 个 training-ready 句 = 24 句（正式包 run5），seed 42。
- 交付物：每句三个 WAV（`real`/`mech_forced`/`mech_dp`）+ 一张逐句表 + 逐音节时长表 + 听音表。音频**不进 Git**，落在数据盘 `derived/aishell3-listen/EXP-0023-run5/`。

## 完整复现命令 / 环境与哈希
见 `metadata.json` 的 `command` 与 `commands`。执行机：4090 WSL2（`ssh -p 2222 hands@10.0.0.190`），模型 `derived/aishell3-aligner/v3/model.npz`（73 KB，sha256 见 metadata），只训练 train 划分 60 说话人 x 40 句 = 1334 句 / 0.8837 h。

## 预注册通过/失败条件
主要判据是人工听音（本实验存在的理由就是它），客观指标只作记录、不作 gating：

1. **人工听音（主判据）**：主人对 20 对机械音频按 1–5 打分，两个维度——内容可辨识度（能否听出原句的字）与时间对齐（音节是否落在原句位置）。
   通过 = 两个维度中位数均 ≥ 4，且**没有任何一对出现音节数错误、漏字或明显错位**。
2. **客观记录（不判定成败）**：逐句 log-mel 距离（real vs mech，两个臂各一列）、逐音节时长比（机械/真人）及其分布、两臂的声母时长占比差异、被阻塞的句子数。
3. 若 20 对里出现阻塞（音素不在 36 音素表、对齐序列与转写不一致），必须逐条列出原因，不能静默换句。

失败判据同样预注册：只要出现 1 对音节数错误或漏字，本实验即判负，需要先修对齐/配对再谈听感。

## 实际结果

### 数据构建（5 次运行，全部保留）
| 运行 | 说话人 x 候选 | 每说话人配额 | candidates | selected | rendered | forced_ok | forced_ready | dp_ok | dp_ready | logmel forced / dp |
|---|---|---|---|---|---|---|---|---|---|---|
| run1 | 4 x 5 | 5 | 20 | 20 | 2 | 2 | 2 | — | — | 中断（见下） |
| run2 | 4 x 5 | 5 | 20 | 20 | 8 | 8 | 2 | 10 | 0 | 7.27482 / 6.03388 |
| run3 | 4 x 25 | 5 | 64 | 27 | 26 | 26 | 18 | 15 | 0 | 6.28423 / 5.98200 |
| run4 | 5 x 30 | 4 | 77 | 37 | 34 | 34 | 16 | 15 | 0 | 6.02491 / 6.17024 |
| **run5（正式）** | 8 x 20 | 3 | 146 | 29 | 29 | 29 | **24** | 16 | **0** | **5.58772 / 6.01663** |

run5 的 8 个说话人各 3 句：SSB0145 / SSB0601 / SSB0822 / SSB1096 / SSB1585 / SSB1593 / SSB1806 / SSB1863。
24 对 A/B 文件与评分表：`EXP-0023-run5/ab/<utt>.wav`（真人 + 1.0 s 静音 + 机械，24 kHz）与 `EXP-0023-run5/sheet.md`。文件名与顺序由 `make_listening_sheet.py` 按 utt_id 排序生成，可复现。

### 解析出的关键 bug：span 时间基被加了两次（commit 57ed0ef）
`PhoneAligner.align()` 返回的 span 已经是**文件时间基**（内部 `offset_s = lead * hop_s` 已折进 `start_s`）。两个消费者又各自加了一次 offset：

- `src/vhe/data/align_index.py`（旧的 align_rows）
- `experiments/EXP-0023/build_listening_package.py`（`spoke_spans`）

后果：每个 span 晚 0.2–0.6 s，音高窗口整体落到下一个音节，末音节落进静音，于是 `insufficient_pitch_points`（error 级）几乎总出现在最后一个音节。**这才是 run1/run2 只有 2/20 ready 的真实原因**，与 F0 估计器本身无关（同期 F0 探针的次八度偏差被误判为主因，见 EXP-0024 的动机）。

修复后同一批 20 句：ready 2/20 → 10/20，rejected 12 → 1（SSB07510300 `mostly_unvoiced`）。回归测试 `tests/test_data_aligner.py::test_align_reports_spans_in_file_time`（lead=0.20 时断言 `spans[0].start_s == offset_s`）与 `tests/test_data_align_index.py` 的 StubAligner 同步改为文件时间基。

### 语料级就绪率（`probe_readiness.py`，`logs/readiness_dev60.json`）
固定样本 12 个 dev 说话人 x 5 句（seed 42，renderable only），不做提前停止：

- attempted 60、aligned 60、align_skips 0、**ready 34 = 56.7%**
- 26 个 blocked 的 readiness reasons **全部且仅**为 `pitch_gap_interpolated`；其中 3 句同时有 `low_pitch_coverage`、16 句有 `pitch_edge_held`（后者不致命）
- 对齐置信度 min 0.618、mean 0.854；v3 模型 0 例 `bad_phone_span`、0 例音素序列不一致

含义：4333 条 renderable train 句按此产率约可产 2400 对；单一瓶颈是 F0 内部间隙政策（`max_pitch_gap_s = 0.02`），不是对齐。

### 两臂对比（客观记录，不判成败）
- `dp` 臂的 training_ready **始终为 0**：15–16 句里稳定出现 `segmentation_count_disagrees`（能量谷音节数对不上）与 `syllable_plan_blocked`，这是 EXP-0011 已知的 36 音素表覆盖问题，不是本轮新问题。
- 平均 log-mel 距离：forced 5.59 vs dp 6.02（run5）。**这一项不能用来判优劣**：它衡量的是「机械 vs 真人」的整体谱差异，与音素边界是否准确并非单调关系（run3 里 dp 反而更低）。真正的判据是听音。

### 人工听音（2026-11-01，主人，24 对）
**不通过。** 主人原话：「后面的机械声效果不咋地，听不清还有类似电子杂音一样」。两个缺陷独立记：
1. **可懂度**：听不清（不是音量小，见下）。
2. **伪影**：类似电子杂音。

客观旁证（本轮在 29 个渲染目录上实测，含 24 对 + 5 个未 ready）：

| 量 | real | mech | 判读 |
|---|---|---|---|
| RMS (dBFS) | −29.3（均值） | −29.3 | 响度已对齐，不是音量问题 |
| 峰值 | 0.171 | **0.132** | 机械动态被压平；3 句差 >0.1（SSB15850030 0.35→0.25、SSB15930049 0.25→0.12、SSB15930056 0.23→0.12） |
| 周期性（归一化自相关峰） | 0.570 | 0.578 | 浊音结构没丢，不是嗡嗡噪声当元音 |
| 中位 F0 | 213.6 Hz | 218.8 Hz | 0/29 出现半八度以上偏差，音高没算错 |
| 谱平坦度（300–4000 Hz） | 0.051 | **0.023** | 机械更「纯音」，比真人更像蜂鸣器 |
| 6 kHz 以上能量占比 | 0.008 | 0.011（个别 0.029） | SSB06010128 0.0293 vs real 0.0024（12×）、SSB18630040 0.0254 vs 0.0034、SSB18060038 0.0265 vs 0.0050 |
| 相邻样本跳变 >0.3 | 0 处 | SSB18630018 有 **16 处**（max 0.467） | 拼接处爆音/咔哒 |

机制判断：**主体是共振峰合成器的音质与音段过渡，不是配对管线**（响度、F0、浊音都对得上）。已知的合成器缺陷正好对上听感：
- /f/ 质心 6156 Hz 高于 /sh/ 的 5051 Hz（EXP-0010 已记录），部位线索反了 → 嘶声听起来像宽带电子噪声；
- 无协同发音，音段拼接无交叉淡化 → 咔哒与「数字感」；
- 元音谐波在 3 个共振峰之外没有谱倾斜塑形，听起来像蜂鸣器（平坦度 0.023 佐证）。

因此 P2 的「听音验收」从**未完成**改判为 **失败**，且失败点在 x_mechanical 的质量本身。继续在现有机械声上训练没有意义：模型学到的是「把蜂鸣器变真人」，目标不可识别。

## 失败、负面发现、错误日志
1. **run1 中断**：脚本两臂都写 `real.wav`，第二次触发 `FileExistsError: Refusing to overwrite`（`write_wav` 拒绝覆盖是刻意设计）。改为先写一次、后写跳过。残留的 `EXP-0023/`（2 句）保留不删。
2. **v3 对齐器第一次训练崩溃**：`--speakers 60` 时 `AlignmentError: 459 frames cannot hold 463 phone frames`。机制是 flat start 里每个音素块独立 `round()`，块数多时累计向上漂移超过总帧数，旧代码把漂移全塞给最后一块并 `max(1, ...)` 截断。修复：改为累积取整（Bresenham），并断言 `sum(assigned) == frames`。崩溃运行目录 `derived/aishell3-aligner/v3-crash/`（空）保留。回归测试 `test_flat_start_splits_every_frame_when_rounding_would_overflow`。
3. **时间基双重偏移**：见上，这是本实验最大的负面发现——它把 40% 的数据打成 rejected，且一开始被误归因给 F0 估计器。
4. **dp 臂永远不 ready**：不是 bug，是 EXP-0011 已知的覆盖限制；记录在此以免下次再当成回归。
5. **`probe_readiness` 初版把 `PitchTrack.voiced_f0` 当方法调用**（它是 property），报 `TypeError: 'tuple' object is not callable`。

## 结论 / 下次实验
- 强制对齐可以作为配对管线的音素时长来源：0 例序列不一致、0 例非法 span，置信度 ≥0.618；dp 臂在 readiness 上完败（0/16）。
- 语料级产率 56.7%，瓶颈单一且已定位到 F0 内部间隙政策，交由 EXP-0024 定量。
- 听音验收**已做，结论失败**（2026-11-01）：听不清 + 电子杂音；客观旁证指向合成器音质与音段过渡，不是配对管线。P2 的「听音」前置项保持开放，且前置条件变为「先换/修机械声源」。
- 下次：EXP-0024 用自校准 + 音素上下文两臂判断间隙政策能否放宽；若不放宽，则按 56.7% 产率直接进入 P2 训练基础设施。
