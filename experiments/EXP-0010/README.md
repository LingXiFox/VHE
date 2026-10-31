# EXP-0010 — Audit of the Mandarin backend excitation, boundaries, formants and level

## 假设
The Mandarin backend is spectrally self-consistent: (a) less than 1e-3 of the energy lies above 0.45*sr at f0 up to 880 Hz, (b) sibilants /s sh x/ put more relative energy in 2-8 kHz than /f/ or vowels, (c) envelope jumps at phoneme boundaries do not exceed the largest within-phoneme envelope jump, (d) formant trajectories inside diphthongs move smoothly (frame-to-frame spectral flux below a hard-spliced control), and (e) the same nucleus rendered in different consonant contexts stays within 3 dB RMS.

（以上五条在 `metadata.json` 的 hypothesis 字段中于首次运行前写定；README 在运行后组装，见 E5。）

## 变量 / 对照组 / 数据划分
- 唯一变量：`src/vhe/synth/mandarin.py::_noise` 的带通掩码，三态：
  - `pre_band_fix`：以 low 为中心的凸包 `0.5*(1+cos(pi*clip((f-low)/edge,-1,1)))`，再乘 `(1-high_gate)`；
  - `bandpass_only`：`rise = 0.5*(1-cos(...))` 的高通 × `fall = 0.5*(1+cos(...))` 的低通，fall 宽 `max(0.35*high, 200)`；
  - `final`：在 `bandpass_only` 上把 fall 终点 clamp 到 `0.45*sample_rate`。
- 对照组（仅作 (c)(d) 的上界参照，不是 baseline）：`ai` 段内硬拼接 a 与 i 的脉冲串（`spliced_a_i_control`）。
- 数据：无真实录音，全部为合成信号；采样率 24 kHz；音素对 `ba ka ma sa ai au ei ou ma_ba_na`；f0 网格 65/110/200/350/440/880 Hz（(a)）。
- 数据划分：不适用。

## 完整复现命令 / 环境与哈希
```
PYTHONPATH=src .venv/bin/python experiments/EXP-0010/probe_audit.py \
    --out experiments/EXP-0010/logs/audit.json
```
- 环境：Python 3.14.6，numpy-only 后端，macOS 27.2 arm64，无 CUDA；torch 仅在测试里用到。
- 输出：`logs/audit.json`（sha256 `6a75f097…`，final）、`logs/audit_bandpass_only.json`（`dd07cc4d…`）、`logs/audit_pre_band_fix.json`（`4d1b636c…`）；探针 `probe_audit.py` sha256 `c47d852c…`。
- Git：commit `b379237` 之后的 dirty 工作树（P1.2 全部改动未提交，见 EXP-0009 metadata 的 status 字段）。

## 预注册通过/失败条件
- (a) `above_fraction(0.45*sr)` < 1e-3，f0 ≤ 880 Hz，覆盖 /a i u ai/ 四种激励；
- (b) `sibilance_ratio_2_8k`：s/sh/x > f 且 > 元音 a；
- (c) `envelope_jumps.boundary_max <= envelope_jumps.interior_max`，对 6 个音素对；
- (d) 复韵母 `max_flux` < 硬拼接对照的 `max_flux`；
- (e) 同一韵母在不同声母上下文里的尾段 dBFS 极差 < 3 dB。

## 实际结果与主观听音
(a) 混叠：pre 2.12e-4（通过）→ bandpass_only **1.05e-1（失败，全部来自 /s/）** → final 2.12e-4（通过）。中间态失败的机制：/s/ 带上限 10 kHz、fall 宽 3.5 kHz，能量伸到 13.5 kHz，越过 0.45×24000 = 10800 Hz；final 的 clamp 把 fall 终点压到 10800 Hz，/s/ 谱质心随之从 8518 降到 7897 Hz（EXP-0009）。

(b) 带能量占比 `sibilance_ratio_2_8k`：pre {f 0.028, s 1.000, sh 0.534, x 0.984}（通过）→ final {f 0.778, s 0.459, sh 1.000, x 0.739}（**失败**：s < f，且 f 从 0.028 跳到 0.778）。pre 的"通过"是坏掩码把 /s/ 挤进 2–4 kHz 造成的假象；final 的失败是真实缺陷——/f/ 的谱带（1500–8000 Hz）与 fall 边宽（2.8 kHz）是手设占位值，导致 /f/ 22% 能量落在 8–12 kHz、质心 6156 Hz 高于 /sh/ 5051 Hz。

(c) 包络跳变（boundary_max / interior_max）：ai 0.006/0.395、ma 0.240/0.337、ma_ba_na 0.390/0.436 通过；ba 0.394/0.341、ka 0.383/0.331、sa 0.396/0.334 **失败约 0.05**（三态数值一致，与掩码无关）。同时样本级 `click_ratio`（边界 ±2 ms 内最大样本差 / 全信号 p99.9）全部 ≤1.08：ai 1.08、ma_ba_na 0.99、ma 0.49、ba 0.35、ka 0.31、sa 0.13。结论：塞音/擦音的边界是快速物理起音，不是数字不连续；判据 (c) 用 5 ms 帧 RMS 度量，对 <5 ms 的起音天然会"失败"，判据本身需要改。

(d) 谱通量 `max_flux`：ai 0.0048、au 0.0045、ei 0.0015、ou 0.0017，硬拼接对照 0.1069 → 通过（低 22–70 倍）。

(e) 尾段 dBFS：a −13.91、ai −13.91、an −13.93、ba −13.25、ma −13.34、sa −12.98，极差 0.95 dB → 通过。

主观听音：未做（本环境不能播放）。本实验只做客观审计，不涉及自然度结论。

## 失败、负面发现、错误日志
- **E1（工具伪影）**：共振峰峰位跟踪器不可靠。/ou/ naive 版最大帧间步 1734 Hz 出现在 t=0.085 s，原因是"取最强 3 个峰"的集合在 2672 Hz 峰进出时整体抖动；/ei/ 984 Hz；硬拼接对照 9094 Hz。改用"最近频匹配"后 /ou/ 仍是 1734 Hz → 判定为跟踪器伪影。主判据改用谱通量 (d)，两个跟踪器的数字仍保留在 `audit.json.formant_steps` 中不删。
- **E2（判据失效）**：判据 (b) 对 4–10 kHz 的 /s/ 不成立（正确写法应是谱质心排序：s 7897 > x 6876 > sh 5051）。按预注册文本判 FAIL，不改判据掩盖。pre 态"通过"进一步说明该判据能被坏掩码骗过。
- **E3（修复引入的回归）**：带通修复让 /s/ 能量外溢到 0.45·sr 以上（1.05e-1），是本次审计抓到的最严重问题；clamp 修复后 2.12e-4。
- **E4（判据 (c) 的局限）**：见上，(c) 对快速起音不适用，需改成样本级 click_ratio（本次已同时测量）或更短帧。
- **E5（流程）**：README 在三次运行之后组装；判据文本本身在 metadata.json 首次运行前已定稿，故不是事后调参。三态日志全部保留，未覆盖 pre 值。

## 结论 / 下次实验
- 通过：(d) 复韵母共振峰轨迹连续、(e) 上下文电平一致、(a) 最终态无 0.45·sr 以上能量。
- 失败且保留：(b) /f/ 过亮、(c) 边界帧 RMS 跳变大于段内最大跳变（但样本级无 click）。
- 已修并留证：(a) 的中间态回归、(b) 的坏掩码假象。
- 下次：用 AISHELL-3 真实擦音谱标定 `phonology.py` 的 `noise_band`、`burst`、VOT；把判据 (c) 换成采样级 click_ratio + 短时（1 ms）包络；补听音评分。
