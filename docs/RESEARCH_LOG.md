# VHE 实验日志（仅追加，不覆盖历史）

## 2026-10-31 | 项目创立 / 科学设计

- 提出：上游物理/参数合成器创造音色和语音内容；独立 VHE 只接收机械音频张量，自然化后输出同形状张量。
- 假设：通用 VHE 能改进自然度且保留未见音色的身份。
- 潜在风险：自然度与音色参数耦合；真人配对训练会鼓励模仿真人；缺少音素/音高信息会导致不可逆信息丢失。
- 当前状态：**未训练 VHE，没有自然化成果。** 仅有机械声音 CPU 最小代理合成器和测试，质量不能代表物理/神经合成终点。
- 研究优先顺序：复现与对齐 → 小模型 → 保留身份 → 主观评价与消融 → 扩充规模。

### 勘误记录格式（追加）

日期 / 发现问题 / 原结论 / 修正结论 / 证据路径 / 相关 commit。

## 2026-10-31 | EXP-0000 首个基线实测

- 运行：`PYTHONPATH=src:. python -m pytest -q` → 7 passed。
- 同一参数连续生成两份 WAV，SHA256 完全相同：`cefbcaec99a276683989f05abc537c6dfb8edfeef164be7728ee22775ada7985`。
- 第一次安装 dev 依赖失败：环境网络不可达导致 ruff 不可用；改用已安装的 numpy/pytest 进行离线验证。保留失败，未声称环境完整安装成功。
- 当前没有神经训练、没有真人评估、没有原创音色泛化证据。完整记录：`experiments/EXP-0000/`。

## 2026-10-31 | EXP-0001 跨平台后端自检

- 扩充 CUDA/MPS/CPU 调度、可移植残差 Conv1d 原型、跨平台 `state_dict` 存取、`doctor`/`smoke` 命令和单元测试。
- 当前 Linux CPU 环境：**17 passed, 2 skipped**（真实 CUDA/MPS 不可用）；CPU 3 步梯度更新有限且损失下降。
- 显式请求不可用的 `cuda` / `mps` 正常报错（退出码 2）；模拟测试验证自动优先级，但不构成真机硬件验证。
- 未训练真实 VHE，没有真人自然化结果；下一步在 Windows RTX 4090 与 macOS Apple MPS 分别执行真机测试。详细见 `experiments/EXP-0001/`。

## 2026-10-31 | EXP-0002 macOS Apple M5 真机 MPS 验收

- 环境：macOS 27.2 / Apple M5 / Python 3.14.6（uv venv）/ PyTorch 2.15.0 / MPS available=true。
- `pytest -q`：**18 passed, 1 skipped**（跳过项为无 NVIDIA CUDA 硬件）。
- `doctor --device mps` 与 `smoke --device mps --samples 1024 --steps 3` 真机通过；losses 0.0010205598 → 0.0009619559 → 0.0008714072，与本机 CPU 逐位相同。
- 机械声同 seed 两次生成 SHA256 均为 `cefbcaec99a276683989f05abc537c6dfb8edfeef164be7728ee22775ada7985`，与 EXP-0000 的 Linux 结果一致。
- 显式 `--device cuda` 在本机退出码 2，未静默回退。
- **仍未训练 VHE**，无自然度/音色保持证据。Windows RTX 4090 真机仍未验证。
- 证据：`experiments/EXP-0002/`。

## 2026-10-31 | EXP-0003 P0-B 合成器契约重构（字节级等价）

- 变更：`src/vhe/synth.py`（126 行单体）→ `src/vhe/synth/{contract,formant_proxy,registry,__init__}.py` + `src/vhe/audio.py`。
  新增 `SynthesisRequest`（template/phonemes/pitch/timing/sample_rate/seed）、`SynthesisResult`（含 backend 版本与 request fingerprint）、
  `PitchCurve`（线性插值、区间外 clamp）、`Timing`（前后静音 ≤5 s）、`SynthBackend` 协议与注册表。
- 结果：`pytest -q` **35 passed, 1 skipped**（跳过项为无 NVIDIA CUDA）；重构前 18 passed。
- 字节等价：同一 CLI 命令连续两次输出 SHA256 均为 `cefbcaec99a276683989f05abc537c6dfb8edfeef164be7728ee22775ada7985`，与 EXP-0000/EXP-0002 完全一致。
- 失败记录（保留）：重构中把 `write_wav` 的量化改成 float64，导致 13/28800 个样本相差 1 LSB；已改回 float32 量化并以测试固定。
  契约测试自身也写错两次（过零率冒充音高、未归一化自相关偏向长滞后；WAV 往返 1 LSB 容差不成立），改法与原因见实验记录。
- 流程偏差（不隐去）：实验记录在实现完成后、验证运行前创建，晚于 AGENTS.md 要求的"运行前"。
- **VHE 仍未训练**，无任何自然度或音色保持证据。证据：`experiments/EXP-0003/`。

## 2026-10-31 | EXP-0004 AISHELL-3 适配器（合成夹具验证，真实语料阻塞）

- 新增 `src/vhe/data/aishell3.py`：目录发现、`content.txt`/`spk-info.txt`/`pinyin.txt` 解析、只读 WAV 头的校验、
  SHA256 溯源、按说话人分组切 dev（官方 test 不动）、JSONL manifest 读写；新增 `python -m vhe manifest`。
- 失败可见：拒绝项一律落成 `ParseIssue`（missing_wav / not_mono / not_16bit / empty_text / too_short / too_long /
  unexpected_sample_rate(warn) / pinyin_mismatch(warn) / bad_line / unreadable_wav），不静默过滤。
- 结果：适配器 **14 passed**，全库 **49 passed, 1 skipped**（CUDA）；manifest SHA256 `d49953ec…`（10 行），
  同一输入可复现，重复写入被拒绝（退出码 2）。
- 真实 AISHELL-3 不在本机，且未获授权下载 19 GB 语料 → **真实解析率、听音抽检、P1 真实验收均未完成**。
- 失败记录（保留）：`empty_text` 曾被 `strip` 吃掉 TAB 退化成 bad_line；测试路径漏了说话人目录导致假失败；
  夹具重写 WAV 触发拒绝覆盖。记录创建晚于实现（流程偏差，同 EXP-0003）。
- **VHE 仍未训练**，无自然度或音色保持证据。证据：`experiments/EXP-0004/`。

## 2026-10-31 | EXP-0005 信号前端：有理数重采样、自相关 F0、能量分段（仅合成信号）

- 新增 `src/vhe/data/resample.py`（多相 windowed-sinc，Kaiser β=8.6，按 `Fraction(src,dst)` 精确映射相位）、
  `src/vhe/data/f0.py`（FFT 自相关 + 无偏修正 + 抛物线插值 + 三倍频保护）、`src/vhe/data/segment.py`（能量谷分段 + 质量 issue）。
- 结果：通带 0.5/1/2/4 kHz 偏差 ≤0.0001 dB，15 kHz 残留 **-87.28 dB**（线性插值对照 **-0.34 dB**，别名原样通过）；
  稳态 F0 100–400 Hz 误差 0.02%–0.84%，100→400 Hz 扫频平均误差 0.14%；4 突发精确分为 4 段且全 voiced、0 issue；
  10 ms 点击被丢弃并记 `segment_too_short`；纯静音记 `no_speech`(error)。`pytest -q` **71 passed, 1 skipped**。
- 失败记录（保留）：多相索引把 `Fraction(dst,src)` 的分子/分母当 (step, phases)，上采样直接 `IndexError`；
  F0 无偏修正偏袒长滞后，220 Hz 与 400 Hz 分别读成 73.33 Hz、133.25 Hz（=f0/3），加"最短强局部峰"规则后修正；
  通带用峰值测量在 4 kHz 得到 -1.25 dB（6 样本/周期、相位误差 `20log10(cos30°)`），改整周期 RMS 后为 0 dB；
  分段时长曾加满窗导致 20 ms 点击算成 65 ms，改为 `frames*hop`；15 ms 点击仍卡在 60 ms 边界，记为已知限制。
- **仅合成信号**：本机无 AISHELL-3，真实语音上的 F0/分段精度与人工听音均未验证。证据：`experiments/EXP-0005/`。
