# VHE 跨平台运行：Windows CUDA / macOS MPS / CPU

## 范围和当前验证状态

- **同一份 PyTorch 源代码、同一个 `state_dict` 权重文件**，通过设备调度层在 Windows+NVIDIA CUDA、macOS+Apple MPS、CPU 上运行。
- `python -m vhe doctor --device auto`：只检查环境，不执行训练。
- `python -m vhe smoke --device auto`：在选定设备执行 **Conv1d 原型的前向、反向和优化器更新**，用合成正弦波和缩放目标自检；**绝不是语音自然化训练或质量证明**。
- `python -m vhe synth ...`：现有 NumPy 机械声基线仍在 CPU 上执行，设备参数目前仅用于神经网络原型。AISHELL-3/真正的 VHE 训练尚未实现。
- 已经在当前 Linux/CPU 环境验证单元测试和 smoke；**尚未在用户的 Windows NVIDIA 和 macOS Apple GPU 上实际执行**。必须在两台真机上分别运行并归档终端结果，不得把模拟设备分发测试当作硬件通过。

## Windows / CUDA

要求：Python 3.11 或 3.12（推荐 64 位）、支持的 NVIDIA 驱动。不要直接把 Python 的普通 CPU 版 PyTorch 当作 CUDA 版。

PowerShell，**在仓库根目录**：

```powershell
py -3.11 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -e ".[dev]"
# 从 https://pytorch.org/get-started/locally/ 选择 Windows / Pip / Python / CUDA，
# 按官网显示的命令安装对应的 CUDA wheel（示意：替换为官网当时给出的 index URL）：
& .\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
& .\.venv\Scripts\python.exe -m vhe doctor --device cuda
& .\.venv\Scripts\python.exe -m vhe smoke --device cuda
& .\.venv\Scripts\python.exe -m pytest -q
```

如果 `py -3.11` 不存在，安装合适的 Python，或明确使用已安装的兼容版本。PyTorch CUDA wheel 与 NVIDIA 驱动需要兼容；如果 CUDA wheel 安装失败，以官方选择器为准，不要盲目升级系统 CUDA Toolkit 或驱动。

## macOS / MPS

要求：Apple Silicon + 支持 MPS 的 macOS/PyTorch；推荐 Python 3.11 或 3.12，arm64 原生终端，**不要使用 Rosetta x86_64 Python**。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dev,train]'
.venv/bin/python -m vhe doctor --device mps
.venv/bin/python -m vhe smoke --device mps
.venv/bin/python -m pytest -q
```

如 `doctor` 报 `mps_available=false`，先检查 Python 架构、系统版本、PyTorch 安装，而不是强行用 MPS 或声称已加速。

## 通用 CPU / 无 GPU

```bash
python -m vhe doctor --device cpu
python -m vhe smoke --device cpu
python -m pytest -q
```

## 后端选择 / 精度 / 实验约束

- `--device auto`：优先 `cuda`，然后 `mps`，否则 `cpu`。`--device cuda|mps` **不可用时必须报错**，不会悄悄 CPU fallback。
- 第一版模型统一 `float32`，不启用 AMP、`torch.compile`、自定义 CUDA kernel、FlashAttention、第三方 fused ops。MPS 原生不支持 float64：训练/特征处理显式统一为 float32；如个别离线声学分析必须 float64，放在 CPU 预处理管道。
- 如果以后启用 `PYTORCH_ENABLE_MPS_FALLBACK=1`，必须记录具体 fallback 算子和性能，不能把 CPU 执行伪装成纯 MPS 训练；默认不开启。
- 保存跨平台权重使用 `vhe.checkpoint.save_weights`（仅 `state_dict`，CPU tensor，不覆盖旧文件），加载使用 `load_weights(..., device='auto')`（`weights_only=True`）。**尚未实现完整 optimizer/checkpoint/resume 训练恢复**。
- CUDA/MPS 数值可能因不同 kernel 略有差异，不能要求不同设备输出 WAV 字节完全相同。同设备同版本可复现与跨设备指标一致性应分别报告。
- RTX 4090 Laptop 16GB 优先真实训练；MPS 用于小规模训练、调试和推理。设备区别通过性能配置管理，不写死在模型层。
- 记录：`torch.__version__`、torch CUDA runtime、CUDA driver（若有）、GPU 名称、设备类型、OS/Python、精度、seed、批大小、音频长度、测试命令及错误日志。

资料：[PyTorch 官方安装](https://pytorch.org/get-started/locally/)；[PyTorch MPS](https://docs.pytorch.org/docs/stable/notes/mps.html)；[MPS 环境变量](https://docs.pytorch.org/docs/stable/mps_environment_variables.html)。
