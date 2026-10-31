# EXP-0009 — Mandarin phoneme backend capability acceptance

## 假设
A 26-phoneme Mandarin source-filter backend driven only by phone ids, F0 curve, timing and a voice template renders every advertised phoneme plus multi-phoneme sequences with <=1% median F0 error, a measurable voiced/unvoiced contrast (autocorrelation peak >0.85 for vowels/nasals/laterals, <0.30 for fricatives), template-dependent formant positions, and bit-identical CPU/MPS tensor hand-off, without copying any real-recording waveform.

## 变量 / 对照组 / 数据划分
- 唯一变量：`src/vhe/synth/mandarin.py` 的噪声带通实现。三个状态各跑一次同一探针：
  - `pre_band_fix`：`low_gate = 0.5*(1+cos(pi*clip((f-low)/edge,-1,1)))`（以 low 为中心的凸包）+ `mask = low_gate*(1-high_gate)`；
  - `bandpass_only`：改为 rise×fall 真带通，但 fall 宽度仍为 `max(0.35*high, 200)`；
  - `final`（= 指标表默认值）：带通 + 把 fall 终点 clamp 到 `0.45*sample_rate`（`/s/` 上限 10 kHz 时原 fall 会伸到 13.5 kHz）。
- 对照组/基准：无外部 baseline。`formant_proxy` v0.1.0（vowels inventory）不参与本次对比，只在 `tests/test_synth_contract.py` 中保持回归。
- 数据：无真实录音。全部输入是 `SynthesisRequest`（音素序列 + 常数或轮廓 F0 + Timing + `PRESETS["yang"]/["yin"]` 模板），采样率 24 kHz，音素时长 0.15–0.25 s。
- 数据划分：不适用（合成探针，无训练）。

## 完整复现命令 / 环境与哈希
```
PYTHONPATH=src .venv/bin/python experiments/EXP-0009/probe_capability.py \
    --out experiments/EXP-0009/logs/report.json \
    --audio-dir experiments/EXP-0009/artifacts
```
- 环境：Python 3.14.6，torch 2.15.0，macOS 27.2 arm64（Apple Silicon），无 CUDA；numpy-only 后端。
- 输出：`logs/report.json`（sha256 `ac8822cb…`）、`logs/report_bandpass_only.json`（`ac56471b…`）、`logs/report_pre_band_fix.json`（`b58c03b6…`）；探针 `probe_capability.py` sha256 `fc6e1356…`。
- WAV：`artifacts/`、`artifacts_bandpass_only/`、`artifacts_pre_band_fix/` 各 34 个（26 音素 + 6 序列 + 2 模板），`*.wav` 与 `experiments/*/artifacts*/` 均在 .gitignore 内，未提交。
- Git：探头首次运行在 commit `b379237` 之后的工作树（dirty，含 P1.2 全部未提交改动）。

## 预注册通过/失败条件
探针写出这些数字之前，判据写在探针代码与本节（判定在运行后回填，见「失败与勘误」E1）：
1. **C1 渲染完整性**：26 个音素全部 finite、`peak <= 1.0`、样本数精确等于 `round(duration*sr)`、`rms > 1e-3`。
2. **C2 有声/无声对比**：元音/鼻音/边音 `autocorrelation_peak > 0.85`；`f/s/sh/x` `< 0.30`。
3. **C3 F0 保持**：`/a/` 在 65/110/200/350/440/880 Hz 的中位数误差 `< 1%`（`fmax_hz = min(900, 1.5*f0)`）。
4. **C4 模板差异**：yang 与 yin 的前三个谱包络峰，对应峰位差 `> 50 Hz`。
5. **C5 序列完整性**：6 条多音素序列长度精确、中位数 F0 误差 `< 1%`。
6. **C6 张量交接**：CPU 与 MPS 往返后与 numpy 原数组 bit-equal，dtype 保持 float32。
- 未预注册但记录：`/s/` 的 0.45·sr 以上能量（由 EXP-0010 判定）。

## 实际结果与主观听音
C1（final）：failures = []；rms 范围 0.0345–0.2053；26 音素样本数全部精确。

C2（final）：a/ai/an/ang/au/ei/i/l/m/n/ou/u 0.937–0.956，b/d/g 0.956–0.957，c/j/q/z 0.819–0.842，k 0.858，p 0.874，t 0.876，f 0.067，s 0.086，sh 0.082，x 0.081 → 通过。
（pre_band_fix：f 0.189 / s 0.103 / sh 0.134 / x 0.130，也已 <0.30。）

C3（final）：65→65.14（0.209%）、110→110.25、200→200.44、350→350.78、440→440.98、880→881.94（0.209–0.227%）→ 通过。误差来自 `estimate_f0` 的抛物线亚样本细化，不是合成器。

C4（final）：yang 峰 [820, 1078, 1312] Hz，yin 峰 [586, 680, 867] Hz，逐峰差 234/398/445 Hz → 通过。

C5（final）：a_ai 0.002%、ai_au_ei_ou 0.010%、ma 0.119%、ba 0.166%、ma_ba_na 0.121%、sa 0.025%，长度全部精确 → 通过。

C6（final）：torch 2.15.0，mps_available True，CPU/MPS 往返均 bit-equal，MPS dtype float32 → 通过。

擦音谱质心（Hz）三态对照：

| phoneme | pre_band_fix | bandpass_only | final |
| --- | --- | --- | --- |
| f | 1548 | 6156 | 6156 |
| s | 4077 | 8518 | 7897 |
| sh | 2038 | 5051 | 5051 |
| x | 3073 | 6876 | 6876 |

主观听音：本机无法在此环境中播放，未做听音评分。试听文件在 `artifacts/`，听感缺陷只能由下一次带听音记录的实验给出，本实验不宣称自然度。

## 失败、负面发现、错误日志
- **E1（流程偏差）**：探针先写、数字先跑，README 的判据段落是运行后回填的。C1–C6 的阈值当时只存在于探针/测试代码里，没有在运行前落盘。后续实验（EXP-0010）改为先写 README 再跑。
- **E2（真实缺陷，已修）**：`pre_band_fix` 的掩码是「以 low 为中心的凸包」，`/s/` 只到 6.4 kHz、`/sh/` 2.7 kHz 以上全丢、`/f/` 96% 能量挤在 1–2 kHz。由 EXP-0010 审计发现，本实验的三态对照记录修复前后数值。修复后 `/f/` 质心 6156 Hz 高于 `/sh/` 5051 Hz，物理上不合理，仍未解决（见下条）。
- **E3（未解决）**：`/f/` 22% 能量落在 8–12 kHz、质心 6156 Hz，比 `/sh/` 更亮。唇齿擦音的谱带（1500–8000 Hz）与 fall 边宽（`max(0.35*high, 200)` = 2.8 kHz）都还是手设占位值，需要真实 AISHELL-3 擦音谱标定；本实验不调参掩盖。
- **E4（限制，非缺陷）**：`estimate_f0` 在默认 65–900 Hz 搜索范围下，`/a/` 210 Hz 读 842 Hz、300 Hz 读 902 Hz（F1≈800 Hz 落在 4f0/3f0 上）。探针用 `fmax_hz=1.5*f0` 规避，真实数据管线不能这样预先知道 f0 → 已在 tests/test_synth_mandarin.py 写成文档性断言，交叉记录于 EXP-0011。

## 结论 / 下次实验
- C1–C6 在 final 状态全部通过；`/s/` 带外泄漏由 EXP-0010 的 clamp 修复（0.1047 → 2.12e-4），本实验只记录质心从 8518 降到 7897 的副作用。
- 能力验收成立的范围仅限「26 个音素能按请求渲染、F0/时长/模板按请求变化、CPU/MPS 数值一致」。**不代表发音正确、不代表自然度**：擦音谱带是手设的，塞音 VOT 未与真实语音对齐，复韵母轨迹是线性插值。
- 下次：接到 AISHELL-3 后，用真实擦音/塞音谱统计标定 `phonology.py` 的 `noise_band`/`burst`/VOT，并补听音评分（EXP-0012 起）。
