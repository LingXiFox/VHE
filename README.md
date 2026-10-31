# VHE — 基于说话人无关神经声学修正的人声自然化模型

**Research status: EXP-0005 / 未训练 VHE / macOS Apple M5 MPS 真机已验收（EXP-0002），Windows RTX 4090 CUDA 真机仍未验证；P1 数据侧因本机无 AISHELL-3 而阻塞（适配器与信号前端仅在合成信号上验证）。**

目标：输入仅为物理/参数合成器生成的**机械人声音频张量** `x [B,1,N]`，输出自然化人声音频 `y [B,1,N]`。VHE **不接收**文本、拼音、音素、MIDI 或说话人 ID，不生成目标内容/音色/音高；这些均由上游合成器决定。

```text
Voice Template + Phonemes + MIDI/Expression
             ↓
   Physical Synthesizer (upstream)
             ↓
     x_mech: Float Tensor [B,1,N]
             ↓
       VHE: x_mech → y_voice
             ↓
    Naturalized audio tensor [B,1,N]
```

> 这是一项尚待实验验证的设计。**说话人无关（speaker-independent）是目标，不是已证事实。** 自然感、音色和声门物理参数并不严格可分。机械声没有提供的信息，模型无法可靠恢复，只能推测，可能改变身份。研究结论必须由盲测与未见模板泛化支持。

## 已有内容（非神经模型）

- `src/vhe/synth/`：**机械声**上游契约与后端。`contract.py` 定义 `SynthesisRequest`/`SynthesisResult`/`SynthBackend`（分段时长、线性 F0 曲线、前后静音、请求指纹），`formant_proxy.py` 是谐波＋共振峰**代理**实现，`registry.py` 负责后端注册。当前只支持 `a/i/u`，不是声道物理引擎，没有辅音、声调、韵律，也不支持中文整句或 `yuan`。契约细节见 **[docs/SYNTH_CONTRACT.md](docs/SYNTH_CONTRACT.md)**。
- `src/vhe/audio.py`：16-bit PCM 读写与峰值归一；量化固定在 float32 下进行（字节级可复现）。
- `src/vhe/data/`：数据侧（未接入真实数据）。
  - `aishell3.py`：**AISHELL-3 适配器**。只读目录结构与 WAV 头，把每一条被拒文件记成显式 issue code（missing_wav / not_mono / empty_text / too_short …），按说话人切 dev 且不动官方 test，输出只含相对路径与 SHA256 的 JSONL manifest。
  - `resample.py` / `f0.py` / `segment.py`：44.1 kHz → 契约采样率的多相 windowed-sinc 重采样（15 kHz 残留 -87 dB）、F0 自相关跟踪（100–400 Hz 合成信号误差 <1%）、能量谷音节分段（丢弃的短突发记 `segment_too_short`，静音记 `no_speech`）。
- `tests/test_synth.py` / `tests/test_synth_contract.py`：尺寸、确定性、参数差异、不支持音素报错、禁止覆盖；以及请求字段白名单、静音不改浊音样本、指纹追踪、**历史 WAV 基线的 SHA256 字节级回归**。
- `src/vhe/backend.py`：CUDA/MPS/CPU 自动检测，设备指定失败时不偷偷回退。
- `src/vhe/model.py`：可移植的残差 Conv1d **未训练原型**（初始化为恒等映射）。
- `src/vhe/smoke.py`：合成张量的前/反向传播和优化器自检，**不是语音自然化训练**。
- `src/vhe/checkpoint.py`：跨平台 `state_dict` 存取工具。
- `docs/`：实验协议、数据授权、研究日志、文献、里程碑、合成器契约、平台验收规范。
- `agent/START_HERE.md`：交给本地 Agent 的可执行开发任务。
- `tools/new_experiment.py`：创建只增不改的实验目录和模板记录。

## Windows CUDA / macOS MPS 两端运行

统一使用 PyTorch，设备选择：`--device auto|cuda|mps|cpu`。

```bash
python -m vhe doctor --device auto    # 检查实际 PyTorch GPU 可用性
python -m vhe smoke --device auto     # 小型前向+反向传播自检，非真人化训练
```

- Windows CUDA：使用 NVIDIA 驱动兼容的 **CUDA 版 PyTorch wheel**。
- macOS MPS：Apple Silicon，原生 arm64 Python，支持 MPS 的 PyTorch。
- CPU：无 GPU 也可进行开发、自检；不强求 GPU。
- 未支持设备被明确指定时直接失败，不会静默 CPU 回退；原型模型统一 FP32。
- 权重可以通过 `src/vhe/checkpoint.py` 的 `save_weights` / `load_weights` 在两端交换；完整训练恢复尚未实现。
- `python -m vhe synth` 目前仍是 NumPy CPU 机械声基线，**不会因启用 CUDA/MPS 变快**。

安装命令、后端局限和真机验收记录见 **[docs/PLATFORMS.md](docs/PLATFORMS.md)**。

## 在本机开始（Mac / Windows）

机械声基线要求 Python 3.10+ 与 NumPy；神经模型运行推荐 Python 3.11/3.12 与 PyTorch。推荐在项目解压目录执行：

```bash
# 如未初始化仓库：git init -b main
python -m venv .venv
# 激活虚拟环境后：
python -m pip install -e '.[dev]'
pytest -q
python -m vhe synth --phonemes a,i,u --preset yang --f0 220 --duration 1.2 --sr 24000 --seed 42 --output data/demo/new_yang_aiu.wav
python -m vhe manifest --root data/raw/aishell3 --output data/manifests/aishell3.jsonl  # 需要本地 AISHELL-3；拒绝覆盖
python tools/new_experiment.py --id EXP-00NN --title "<实验标题>" --hypothesis "<预注册假设>"  # 编号只增不改，已存在会直接失败
```

Windows PowerShell 激活：`\.venv\Scripts\Activate.ps1`；macOS/Linux：`source .venv/bin/activate`。
PowerShell 也可直接运行 `py -3.11 -m pip install -e '.[dev]'`（需要先选定解释器，建议虚拟环境）。

**数据集不随代码分发，也不要自动下载 19GB AISHELL-3。** 真实数据必须按 `docs/DATA_POLICY.md` 管理，手动审查许可/本地路径。禁止把录音、个人信息、大模型权重上传到 Git。

AISHELL-3 的格式与许可已在 [docs/DATA_POLICY.md](docs/DATA_POLICY.md) 中核实并记录（Apache-2.0、44.1 kHz mono 16-bit、`content.txt` 为「汉字+带声调拼音」交替）。适配器在合成夹具上通过（EXP-0004），**真实语料尚未解析、尚未人工听音**。

## 科学实验门槛

任何「变真人了」的说法必须至少同时有：

1. 人工盲听：自然度（MOS 或成对偏好），提供样本数、置信区间、协议。
2. 保真：内容、F0、时长、响度偏差；客观指标及失败例。
3. 音色保持：原始与输出在跨内容片段上的一致性；识别模型存在**机械声域偏差**，不能仅凭一个 embedding 指标。
4. 泛化：训练、验证、测试**按说话人分组**；最终评价未见音色模板、未见文本，且说明物理合成器是否与训练一致。
5. 重现：记录代码提交、参数配置、随机种子、数据版本、环境/驱动/CUDA、训练曲线、音频样本及哈希。

训练目标与细节见 `docs/PROTOCOL.md`，研究证据见 `docs/LITERATURE.md`。

## 开发过程与资料归属

- 研究日志只追加：`docs/RESEARCH_LOG.md`。
- 每个运行单独建 `experiments/EXP-XXXX/`，保留失败实验，不能覆盖旧实验。
- 修改结论必须留下修订条目，不能擦掉最初错误猜想。
- 每个阶段提交 Git；远程仓库为 <https://github.com/LingXiFox/VHE>（public，不含任何真实录音）。
- **未选择开源许可证**：许可证由项目所有者决定，仓库当前无 LICENSE 文件。
- 真实录音、数据集与训练产物不得进入 Git（见 `docs/DATA_POLICY.md` 与 `.gitignore`）。
