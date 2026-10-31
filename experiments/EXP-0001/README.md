# EXP-0001 — 双平台后端兼容性与神经网络梯度自检

## 预注册假设
同一 PyTorch FP32 原型在 CUDA/MPS/CPU 上应满足 `[B,1,N]` 输入/输出、可执行前向/反向传播、有限损失、可更新权重；明确要求 GPU 而硬件不存在时必须失败，不得静默回退。

## 实验设计及对照
- CPU：真实运行 PyTorch Conv1d + AdamW + 合成正弦输入（不是 VHE 自然化训练）。
- CUDA/MPS：当前容器无 GPU，验证显式拒绝；另外以仿真 backend 检查 `auto` 分支选择优先级，**不等于 GPU 实测**。
- 目标：`fake_target = x * 0.85`，仅用于证明优化器在设备上能够执行；**不含真人数据**。
- 固定随机种子 42，FP32，批量 1，音频采样点 1024，3 个更新步骤。

## 验收标准
- CPU 测试无失败；至少一次有效梯度、设备及输出形状一致；损失有限。
- 无硬件时不宣称 CUDA/MPS 已实际通过；两端真机验证作为未完成事项。

## 实际结果（2026-10-31）

```text
PYTHONPATH=src:. python -m pytest -q
17 passed, 2 skipped (CUDA/MPS hardware missing)

PYTHONPATH=src:. python -m vhe smoke --device cpu --samples 1024 --steps 3
selected_device: cpu
PyTorch: 2.10.0+cpu
model_parameters: 6265
losses: 0.0010205598 → 0.0009619559 → 0.0008714072
output_shape: [1,1,1024]
status: backend-smoke-passed-not-humanization
```

`doctor --device cuda` 和 `doctor --device mps` 均以退出码 2 报设备不可用，这是预期的错误路径测试。纯软件模拟的 `auto` 选择优先级（CUDA→MPS→CPU）单测已通过。

## 局限、失败与后续
- **未在 Windows CUDA / macOS MPS 真机上验证**，不能把 CPU 结果当成双 GPU 支持已验证。
- 尚无真实语音数据、声学损失、训练模型、自然度提升或音色保持结论。
- 后续在真实两端分别运行 `doctor --device cuda|mps`、`smoke --device cuda|mps`、`pytest -q`，保留逐机实验记录和环境详情。
