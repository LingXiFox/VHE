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
## Windows + WSL2 + NVIDIA CUDA（2026-11-01 真机验证通过）

实验机通过 WSL2 提供 Linux 环境，**不使用原生 PowerShell 路径**：`ssh -p 2222 hands@10.0.0.190`。

实测环境：

| 项 | 值 |
|---|---|
| OS / kernel | Ubuntu 26.04.1 LTS，`6.18.33.2-microsoft-standard-WSL2`，x86_64，28 核 |
| GPU | NVIDIA GeForce RTX 4090 Laptop GPU，16376 MiB，driver 621.01 |
| GPU 直通 | `/dev/dxg` 存在，`libcuda.so.1` 在 `/usr/lib/wsl/lib`（**不需要**在 WSL 里装驱动或 CUDA Toolkit） |
| `nvidia-smi` | **不在 PATH**，只能用 `/usr/lib/wsl/lib/nvidia-smi` |
| Python | 仓库 venv 用 uv 装 CPython 3.12.14（系统 Python 是 3.14.4，torch 的 cp314 轮子当时未取） |
| torch | `2.11.0+cu128`（CUDA runtime 12.8），`torch.cuda.is_available() == True` |
| 仓库 | `/home/hands/VHE`，`git clone` 自 GitHub，HEAD 与远端一致 |
| 数据 | `/home/hands/datasets/VHE-Datasets`（ext4，**不要放 `/mnt/c`**，9p 会让 8.8 万个小文件极慢） |

安装与验证（实测命令）：

```bash
cd ~/VHE
~/.local/bin/uv venv --python 3.12 .venv
~/.local/bin/uv pip install --python .venv/bin/python \
    --index-url https://download.pytorch.org/whl/cu128 torch
~/.local/bin/uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m vhe doctor --device cuda     # selected_device=cuda, cuda_available=true
.venv/bin/python -m vhe smoke --device cuda      # forward/backward/optimizer step on GPU
.venv/bin/python -m pytest -q                    # 404 passed / 2 skipped（skip 均为 MPS 专用用例）
```

数据镜像与路径重定位（manifest 里的 `root` 记的是 mac 路径，不可直接复用）：

```bash
# 本机（mac）→ 实验机
rsync -rt -e "ssh -p 2222" /Volumes/LingXiWorkspace/VHE-Datasets/AISHELL-3/extracted/ \
    hands@10.0.0.190:/home/hands/datasets/VHE-Datasets/AISHELL-3/extracted/
rsync -rt -e "ssh -p 2222" /Volumes/LingXiWorkspace/VHE-Datasets/derived/ \
    hands@10.0.0.190:/home/hands/datasets/VHE-Datasets/derived/
```

```python
from vhe.data.dataset import CorpusPairs
pairs = CorpusPairs.from_manifest(
    "/home/hands/datasets/VHE-Datasets/derived/aishell3-scan/v4/manifest.jsonl",
    root="/home/hands/datasets/VHE-Datasets/AISHELL-3/extracted")   # 5314 句
```

传输与镜像校验（2026-11-01）：88035 个 wav / 88047 个文件 / 27,208,419,174 B；5314 条可渲染文件逐文件 sha256 聚合摘要
`d26b7f686604c6c243f5fa9d3b8836834267720a04f53b4dbd6080d15c9db0da`、字节数 1,069,666,944，与本机完全一致；
v4 manifest sha256 `1a8138c4…` 两端一致；`CorpusPairs` 在实验机加载得到 5314（train 4333 / dev 528 / test 453），
`staleness.changed = 0`。

**仍未验证**：这台机器是 WSL2，不是原生 Windows 图形会话；如果以后必须在纯 PowerShell 下运行，仍需按上一节单独验收。
`/usr/lib/wsl/lib` 未加入 PATH 是有意为之，避免与未来系统 CUDA 冲突——需要 `nvidia-smi` 时写全路径。

