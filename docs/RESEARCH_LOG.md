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
