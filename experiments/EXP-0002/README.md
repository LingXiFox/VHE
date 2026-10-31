# EXP-0002 — macOS Apple M5 真机 MPS 后端验收与确定性复测

## 预注册假设

1. 同一 FP32 残差 Conv1d 原型可在 **Apple M5 真机 MPS** 完成前向、反向与优化器更新，损失有限且下降。
2. 同 seed、同参数的机械声 WAV 在本机连续两次运行逐字节一致，并且与 EXP-0000 在 Linux/CPU 上得到的历史哈希一致（跨平台字节级一致性是观察结果，不是承诺；见 docs/PLATFORMS.md）。
3. 显式请求本机不存在的 cuda 必须报错退出，不得静默回退 CPU。

## 唯一变量与对照

- 变量：执行设备（MPS 真机 vs CPU）。
- 对照：EXP-0001 的 Linux/CPU 结果（17 passed / smoke losses）。
- 不涉及训练数据、不涉及真人录音、不涉及 VHE 自然化质量。

## 环境（实测，非假设）

| 项 | 值 |
|---|---|
| OS | macOS 27.2 (Darwin 27.2.0, arm64) |
| CPU | Apple M5, 32 GB |
| Python | 3.14.6（uv 管理，仓库内 .venv） |
| PyTorch | 2.15.0（PyPI macOS arm64 wheel） |
| MPS built / available | true / true |
| CUDA | 不可用（无 NVIDIA 硬件） |
| 精度 | float32 |
| git commit | 见 metadata.json（运行时 HEAD = c7e102a，工作区干净） |

## 完整复现命令

```bash
uv venv --python 3.14 .venv
uv pip install --python .venv/bin/python -e . pytest torch
.venv/bin/python -m vhe doctor --device auto
.venv/bin/python -m vhe doctor --device mps
.venv/bin/python -m vhe smoke --device mps --samples 1024 --steps 3
.venv/bin/python -m vhe smoke --device cpu --samples 1024 --steps 3
.venv/bin/python -m pytest -q -v
```

原始终端输出保存在本目录 logs/（doctor-auto.json / doctor-mps.json / smoke-mps.json / smoke-cpu.json / pytest.txt）。

## 预注册通过条件

- pytest -q 全绿（仅允许「无 CUDA 硬件」的条件跳过）。
- MPS smoke 返回 selected_device == mps，损失有限且单调下降。
- doctor --device cuda 退出码 2。
- 两次同 seed WAV 的 SHA256 相同。

## 实际结果

- doctor --device auto → selected_device = mps，mps_available = true，cuda_available = false。
- smoke --device mps → 通过，model_parameters = 6265，output_shape = [1,1,1024]，losses:
  0.0010205598082393408 → 0.0009619559277780354 → 0.0008714072173461318。
- smoke --device cpu → 通过，losses 与 MPS **逐位相同**。本机上 MPS kernel 与 CPU 结果一致；这不是「所有 MPS 训练都与 CPU 位一致」的证据，跨设备数值差异仍按 docs/PLATFORMS.md 单独报告。
- doctor --device cuda → 退出码 2，报 CUDA requested but unavailable.（预期错误路径，不是缺陷）。
- pytest -q → **18 passed, 1 skipped**（跳过项为「需要真实 NVIDIA CUDA」）。
- 机械声确定性：同命令生成 /tmp/vhe_check/run_a.wav 与 run_b.wav，SHA256 均为
  cefbcaec99a276683989f05abc537c6dfb8edfeef164be7728ee22775ada7985，与 EXP-0000 在 Linux 上的历史值**完全相同**
  （28800 samples, 24 kHz, 1.2 s）。

## 失败、负面发现、错误日志

- 无测试失败。运行前本机系统 Python 3.14.6 是裸解释器：import numpy / import torch 均 ModuleNotFoundError。
  已按 pyproject 已声明的依赖（numpy、pytest、torch 属于 [train] extra 已声明项）建立仓库内 .venv；
  ruff / soundfile / scipy / tensorboard 本轮**未安装**（当前代码路径不需要）。
- **未验证 Windows RTX 4090 真机**。docs/MILESTONES.md 的「双端验收」因此仍为部分完成：
  MPS 侧已真机通过，CUDA 侧仍为未验证，不得把本实验当作 CUDA 通过。

## 结论 / 下次实验

P0 的本机（macOS MPS）验收条件满足：单测全绿、显式设备拒绝正确、MPS 真机前向/反向往返成功、机械声 WAV 同 seed 哈希稳定。
**VHE 仍未训练**，本实验不提供任何自然度或音色保持证据。
下一实验：把 synth.py 重构为可插拔的上游合成器契约，并用字节级回归测试证明重构不改变现有输出。
