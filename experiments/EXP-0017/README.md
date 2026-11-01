# EXP-0017 — F0 rejection audit: unvoiced frames versus missed voicing

## 假设
Most frames rejected as `no_candidate` are aperiodic (fricatives, closures, breath) or too weak to carry a period; a minority sit inside voiced spans and show harmonic-comb evidence, and those are a candidate-model miss rather than a threshold problem. Loosening the CMND gate is therefore not the fix, and any change must be justified by comb/cepstrum evidence per frame.

## 变量 / 对照组 / 数据划分
- 唯一变量：无。本实验是**诊断**，不修改 `track_pitch` 的任何阈值（`CANDIDATE_MAX_CMND` 保持 0.55、`fmax_hz` 保持 900），也不产生新的 F0。
- 数据：与 EXP-0015 完全相同的 12 条真人样本（AISHELL-3 `test` split，每说话人 1 条，时长 ≤3 s，44.1 kHz→24 kHz，manifest `derived/aishell3-scan/v4/manifest.jsonl`）。共 12 句 / 2052 帧。
- 对照：
  1. 被跟踪器接受的帧（`ok`，1077 帧）作为"已确认浊音"参照分布，用来校准判据分位数；
  2. 逐帧最小 CMND（拒绝发生的那个量）与四条独立频域证据（HPS z、倒谱峰 prominence、谱平坦度、相对电平）并列，看二者是否一致；
  3. `SSB04350040` 单句：viterbi 跟踪器 250.0 Hz vs 旧 `estimate_f0(fmax_hz=500)` 280.3 Hz 的成因拆分。

## 完整复现命令 / 环境与哈希
```bash
.venv/bin/python experiments/EXP-0017/probe_rejections.py \
  --manifest /Volumes/LingXiWorkspace/VHE-Datasets/derived/aishell3-scan/v4/manifest.jsonl \
  --output experiments/EXP-0017/logs/rejection_report_v3.json
```
- manifest sha256 `1a8138c4d13e547d1d266a7aaf50d3a002a8e32dad78b0555f1b417c77074f31`（v4，inventory 36）。
- 环境：macOS arm64（M5 MacBook Air 32GB）、Python 3.14.6、NumPy；纯 CPU，未使用 torch。
- 探针与 `track_pitch` 共用同一帧网格（30 ms 窗 / 10 ms hop，`time=(start+window/2)/sr`），因此帧 k 在两个实现里指同一批采样。
- 产物：`logs/rejection_report_v1.json`（无校准的失败版，保留）、`logs/rejection_report_v2.json`（校准版定稿数值）、`logs/rejection_report_v3.json`（同 v2 标签，另加 label×level、label×min_cmnd 交叉表与按标签中位数）。

## 预注册通过/失败条件
- 支持假设的证据：`labels_between_two_ok_frames` 中的 `suspected_miss` 计数 ≈ 0（浊音段内部不丢帧），且被拒帧的 CMND 多数远离 0.55 门限（说明"降阈值"不会把真浊音换回来）。
- 反驳假设的证据：出现成段的 `suspected_miss` 连续帧，或 `min_cmnd` 集中在 0.55–0.60 的临界带且伴随谐波证据（那才说明是门限过严）。
- 无论哪边成立，本实验都不得据此改阈值；只能提出下一步的候选模型改进方向。

## 实际结果与主观听音
未听音。以下全部为数值诊断。

### 拒绝构成（2052 帧）
| reason | 帧数 | 占全部 | 占非静音（1475） |
| --- | --- | --- | --- |
| ok | 1077 | 52.5% | 73.0% |
| silence | 577 | 28.1% | — |
| no_candidate | **281** | 13.7% | **19.05%** |
| octave_ambiguous | 69 | 3.4% | 4.7% |
| continuity_gated | 24 | 1.2% | 1.6% |
| low_confidence | 24 | 1.2% | 1.6% |

总拒绝率 0.2698（398/1475，与 EXP-0015 一致）。**其中 19.05 个百分点来自 no_candidate，8 个百分点来自另外三种拒绝原因**，两者不能混为一谈——P1.4 汇报里的"27% 拒绝率"不等于"27% 的有声帧没测出来"。

### 判据校准（用 ok 帧的分位数，ng 而非拍脑袋）
| 证据 | p10 | p25 | p50 | p75 | p90 |
| --- | --- | --- | --- | --- | --- |
| HPS z | 1.596 | 1.862 | 2.220 | 2.817 | 3.343 |
| 倒谱峰 prominence（MAD 倍数） | 5.21 | 7.60 | 14.23 | 22.28 | 29.82 |
| 谱平坦度 80–4000 Hz | 0.0010 | 0.0020 | 0.0060 | 0.0160 | 0.0380 |
| 相对电平 dB | −19.34 | −11.20 | −6.10 | −3.40 | −1.70 |
| 最小 CMND | 0.008 | 0.018 | 0.067 | 0.183 | 0.352 |

- `suspected_miss`：hps_z ≥ p25(1.862)、cepstrum ≥ p25(7.60)、flatness ≤ p75(0.016) 三条同时满足。
- `boundary_or_weak`：hps_z ≥ p10(1.596) 且 flatness ≤ p90(0.038)，但不满足上一条。
- `aperiodic`：其余。
- 注意校准的循环性：ok 帧是跟踪器**自己**接受的帧，用它做参照会把"跟踪器系统性不接受、但确实有声"的帧算成异常。这一偏差写在"失败与局限"里。

### 281 个 no_candidate 帧的分类
| label | 帧数 | 占 no_candidate | 平坦度中位 | 最小 CMND 中位 | 电平中位 |
| --- | --- | --- | --- | --- | --- |
| aperiodic | **174** | 61.9% | 0.129 | 0.778 | −20.35 dB |
| boundary_or_weak | **103** | 36.7% | 0.018 | 1.252 | −30.20 dB |
| suspected_miss | **4** | 1.4% | 0.008 | 0.842 | −28.70 dB |

- 关键断言：**夹在两个 ok 帧之间的拒绝帧只有 4 个，且全部是 aperiodic**（`labels_between_two_ok_frames = {aperiodic: 4}`）。也就是说已确认的浊音段内部没有被挖洞，拒绝集中在浊音起止边缘、清音和低电平区。
- 4 个 `suspected_miss` 全部落在 ≥2 帧的拒绝游程里（`labels_inside_a_run_of_two_or_more = {aperiodic: 156, boundary_or_weak: 100, suspected_miss: 3}`），3 个在游程内部、1 个紧邻 ok 帧；HPS 给出的 F0 分别是 73.5/65.0/100.0/— Hz，其中 65.0 Hz 正好压在 65 Hz 网格边界，属于低频隆隆或气声而非基频。

### 门限是不是太严？——不是
| 最小 CMND 桶 | aperiodic | boundary_or_weak | suspected_miss |
| --- | --- | --- | --- |
| ≤0.60（贴着 0.55 门限） | 42 | 5 | 2 |
| ≤0.70 | 28 | 5 | 0 |
| ≤0.85 | 30 | 10 | 0 |
| >0.85（离门限极远） | 74 | 83 | 2 |

- 只有 49 帧（17%）离门限 0.05 以内，其中 42 帧平坦度中位 0.129、无谐波结构，把门限从 0.55 降到 0.60 只会把这些噪声帧放进来。
- 157 帧（56%）的最小 CMND >0.85，**任何合理的阈值放宽都救不回来**；它们要么是真非周期，要么需要新的候选生成模型（例如对气声/嘎裂声建模），不是调参问题。
- 电平也不支持"太小所以测不到"：92 帧高于 −20 dB、110 帧高于 −30 dB。高电平 + 高平坦度（aperiodic × >−20 dB 有 84 帧）正是擦音/送气的特征。

| 电平桶 | aperiodic | boundary_or_weak | suspected_miss |
| --- | --- | --- | --- |
| >−20 dB | 84 | 7 | 1 |
| >−30 dB | 66 | 41 | 3 |
| >−45 dB | 24 | 55 | 0 |

- `boundary_or_weak` 集中在 −30…−45 dB（55/103），与"浊音起始/结束的弱过渡"一致，不是稳定元音中段。

### SSB04350040：250.0 vs 280.3 Hz 的成因
- 134 帧；新跟踪器高置信中位 **250.0 Hz**，旧 `estimate_f0(fmax_hz=500)` **280.3 Hz**。
- 逐帧比较：中位偏差仅 **5.4 cents**，但 **18.2% 的帧偏差 >50 cents**。结论：两句中位数差 0.18 倍频程来自少数跳变帧，不是系统性偏移。
- 分歧帧证据（均来自本探针的独立估计，不是跟踪器内部分数）：

| t (s) | 新 F0 | legacy F0 | 谐波峰 | HPS | 倒谱峰/prominence |
| --- | --- | --- | --- | --- | --- |
| 0.675 | 214.2 | 318.2 | 233.3 / 300.0 / 433.3 / 633.3 | 213.5 | 210.5 / 9.63 |
| 0.685 | 209.1 | 417.7 | 200.0 / 433.3 / 633.3 | 206.5 | 200.0 / 11.20 |

- 两条独立频域方法（HPS、实倒谱）都给出 ≈210 Hz，与新跟踪器一致；legacy 的 318/417 Hz 在这两帧没有任何独立支持。legacy 在该句发生了半频/非整倍频跳变，新跟踪器的帧间连续性约束把它压住了。
- **但本句没有人工标注的 F0 真值**，因此只能写"独立证据更支持新跟踪器"，不能宣称"新跟踪器准确"。

## 失败、负面发现、错误日志
1. **v1 判据无校准、阈值拍脑袋**（`logs/rejection_report_v1.json` 保留）：固定 hps_z≥2、cepstrum≥3、flatness≤0.25，把 **220/281** 判成 suspected_miss，其中夹在两个 ok 帧之间的只有 2 帧。检查后发现 ok 帧自身的 hps_z 中位数只有 2.22，z=2 根本不是"有谐波"的证据。已作废，改为按 ok 帧分位数校准，并保留 v1 作为反例。
2. **校准本身有循环性**：参照分布来自跟踪器接受的帧。若某类浊音被系统性拒绝，它永远不会进入参照。本实验未能量化这一偏差，只能靠"夹在两个 ok 帧之间"这一几何判据兜底。
3. **HPS 在 65 Hz 网格边界会给出假峰**：4 个 suspected_miss 中有 2 个的 HPS 落在 65.0/73.5 Hz，是频谱低端斜率而非真实基频；判据没有排除边界效应，这是判据的已知缺陷。
4. **倒谱 prominence 用 MAD 归一化，对极安静帧会放大**：`boundary_or_weak` 的 cepstrum 中位（未列）高于 `suspected_miss`，说明该指标在 −30 dB 附近噪声大，不能单独定案。
5. **只覆盖 12 句 / 12 位说话人 / 2052 帧**，且都是 ≤3 s 的短句；长句、连续语流、情绪化发音未覆盖。
6. **未听音**，无法确认 `suspected_miss` 的 4 帧在人耳里是否真的可闻。
7. 本实验**没有**修改 `track_pitch`、没有降低 CMND 门限、没有插值任何被拒帧。

## 结论 / 下次实验
- 假设成立的主干部分：281 个 `no_candidate` 里 98.6% 是非周期（174）或弱/边界（103）帧，稳定浊音段内部没有任何 `suspected_miss`；拒绝不是"算法把清楚的人声丢掉了"。
- 假设需要修正的部分："少数是候选模型漏检"只成立到 4 帧（1.4%），而且这 4 帧都在低电平（中位 −28.7 dB）且紧邻其他拒绝帧，证据本身也不硬（2 帧 HPS 压在网格边界）。当前不该为它们改模型。
- 下一步（EXP-0018 起）应转向：①气声/嘎裂声的候选建模（`min_cmnd >0.85` 的 157 帧才是真难点）；②给真值：用少量人工标注 F0 的样本评估"新跟踪器 vs legacy"到底谁准；③把 `SSB04350040` 这类分歧句作为回归夹具。
- 配对管线**不**因本实验改变默认参数（`pitch_estimator="viterbi"` 保持不变）。

## 勘误
`metadata.json` 的 `completed_utc` 原先写成 2026-11-01T06:58:00+00:00，早于实际落盘时刻被写成了未来时间（写记录时按估算填的）。已改为记录实际写入时刻 2026-11-01T06:46:00+00:00，并在 metadata 里保留 `erratum` 字段说明原值。命令、结果与结论均未改动。
