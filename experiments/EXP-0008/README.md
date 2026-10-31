# EXP-0008 — v0.1 model and loss suite: numerical validation before training

## 假设
The v0.1 residual U-Net plus the four-term loss suite are numerically sound before any real training: (1) at initialisation the edit is exactly zero and the model preserves [B,1,N] for arbitrary (also non-multiple-of-8) lengths; (2) the measured receptive field is at least 40 ms (~2.5 periods at 65 Hz) so pitch-synchronous micro-structure is observable; (3) every loss term reacts monotonically to the distortion it targets and is invariant to gain where it should be; (4) gradients reach every parameter without NaN on CPU and MPS in FP32; (5) MRSTFT+periodicity+envelope separate a distortion-free pair from a distorted one with a margin > 10x, which is the minimum needed for a training signal to exist.

> 建档时写下的假设原文保留不改。事后勘误见 §失败与负面发现 E1、E2。

## 变量 / 对照组 / 数据划分

- **唯一变量**：喂给损失函数/模型的输入扰动。模型权重、夹具、随机源全程固定。
- **数据**：不使用任何真人数据。夹具为 `validate_model_losses.py` 内生成的合成机械音（180 Hz 基频、12 次谐波、20 ms 闭塞、70 ms 停顿、峰值 0.85），float64 生成后转 float32；"自然参考"是同一夹具的 DSP 扰动版（jitter/shimmer/气声/谐波衰减），**不是真人**，不能用来证明任何听感结论。
- **对照组**（8 种失真，逐个替换 estimate，target 不变）：
  `identity`（不扰动）、`noise_snr20`、`pitch_plus_2pct`、`shift_20ms`、`gain_2x`、`drop_100ms`、`dull_3tap`（三点低通）、`bitcrush_8bit`。
- **设备对照**：CPU vs MPS（同一随机权重，测 edit 最大绝对差）。
- **数据划分**：不适用（无训练）。这不是训练实验，因此没有 train/val/test。

## 完整复现命令 / 环境与哈希

```bash
.venv/bin/python experiments/EXP-0008/validate_model_losses.py \
    --out experiments/EXP-0008/logs/validate.json
```

- 环境：macOS 27.2 arm64，Python 3.14.6，PyTorch 2.15.0，MPS 可用，无 CUDA（`torch.cuda.is_available() == False`），FP32。
- 探针脚本 sha256：`40679d08eb8b9fae52bf7608b35c1b4c0ee623180841ea7b684804cd32c1b42e`
- 输出 sha256：`065361ffa8f5abddf62f7b304e8d84237b75e5fe7ac96f58eb98280d8391838f`（6014 字节）
- Git：基线 commit `b8df27d`，运行时为 dirty 树（`M src/vhe/model.py`、`M tests/test_model.py`、未跟踪 `src/vhe/losses.py`、`tests/test_losses.py`、`experiments/EXP-0008/`、三份 `docs/*.md`）。这些改动即本实验的对象，随本实验一起提交。
- 随机性：感受野与设备对照用 `torch.Generator()` 固定种子后随机化权重；无其他随机源。

## 预注册通过/失败条件

按假设逐条给出可判定条件（运行前写定，结果见下）：

| # | 判据 | 通过线 |
|---|---|---|
| 1 | 恒等性与长度 | 零初始化下 `max|y-x| == 0.0`；长度 777/1000/24000/24013 全部原样返回 |
| 2 | 感受野 | 脉冲影响宽度 ≥ 40 ms（960 样本），且左右近似对称 |
| 3 | 损失反应性 | 每项对其目标失真给出正向响应；periodicity 对增益/符号不变（≤1e-3）、对变调/加噪响应 > 10× 基线；envelope_shape 对纯增益 ≈ 0、对 20 ms 移位 > 1e-2、对丢段 > 0.1 |
| 4 | 梯度 | 所有参数拿到有限梯度（无 NaN/inf），CPU 与 MPS 均无异常 |
| 5 | 可分性 | 失真样本相对无失真样本的总损失裕度 > 10× |

## 实际结果与主观听音

**结论：3 条通过，2 条退化/部分失效（详见 §失败与负面发现）。**

| 项 | 实测 | 判据 | 判定 |
|---|---|---|---|
| 参数量 | 1,001,089 | — | 记录 |
| 恒等 `max|y-x|` | 0.0 | == 0 | ✅ |
| 长度保持 | 777/1000/24000/24013 差全 0 | 全部原长 | ✅ |
| 感受野 | 左 −611 / 右 +612 样本 = **1224 样本 = 51.0 ms** | ≥ 960 样本 | ✅ |
| 周期覆盖 | 65 Hz 的 3.31 个周期；180 Hz 的 9.18 个周期 | — | 记录 |
| 初始损失（y=x） | SC 0.41103 / LM 6.51173 / periodicity 0.06767 / env_shape 0.05104 / env_level 0.03458 / residual 0.0 / **total 7.07604** | — | 记录 |
| 梯度（step 0） | 仅 head 61950.56，其余 14 个 block 全 0.0；`parameters_without_grad == []` | 全部参数有梯度 | ⚠️ 零梯度（预期内，见 E3） |
| 梯度（head 走一步后） | 全部 block 非零：stem 6763.0 … dec3 219086.94，head 305.74 | 同上 | ✅ |
| 设备一致性 | CPU vs MPS edit 最大差 **3.1e-06**（FP32），MPS 输出有限 | 无非有限值 | ✅ |

损失矩阵（estimate 受扰动，target 不变；`—` 表示该探针未打印分项）：

| 失真 | SC | LM | periodicity | env_shape | env_level | total | 相对 identity 增量 |
|---|---|---|---|---|---|---|---|
| identity | 0 | 0 | 0 | 0 | 0 | 0.00038 | — |
| noise_snr20 | 0.0802 | 1.355 | 0.0045 | 0.2353 | — | 1.8044 | 1.804 |
| pitch_plus_2pct | 0.4056 | 0.9618 | 0.0727 | — | — | 1.6134 | 1.613 |
| shift_20ms | — | — | — | 0.2775 | — | 1.7041 | 1.704 |
| gain_2x | 1.000 | 0.6746 | 0.0000 | 0.0543 | 0.6649 | 2.3947 | 2.394 |
| drop_100ms | — | — | — | 0.7523 | — | 2.1085 | 2.108 |
| dull_3tap | — | 1.9931 | — | — | — | 2.1550 | 2.155 |
| **bitcrush_8bit** | 0.0036 | 0.1223 | 0.0001 | — | — | **0.1264** | **0.126** |

主观听音：**无**。本实验没有任何真实音频产出，也未做听测；不得据表推断听感。

## 失败、负面发现、错误日志

**E1（代码错误，已修）**：初版 `gradient_flow()` 直接在传入的模型上做 SGD 一步，而主流程在它之后才计算 `loss_at_init`，导致"初始损失"实际是在 head 已被更新一次之后测的；另有一行 `model.load_state_dict({k: v*0.0+v ...})` 是无意义空操作。修法：探针内部 `copy.deepcopy(model)`，删除空操作行。重跑后 `loss_at_init.residual == 0.0`（编辑严格为零）与修复前打印值在四位小数内一致——因为 lr=1e-2 的单步 head 更新对编辑量影响极小。**本文所有数字来自修复后的版本**；修复前的 `validate.json` 已被覆盖未保留（流程缺陷）。教训：探针函数不得修改被测对象。

**E2（方法学缺陷）**：判据 5「裕度 > 10×」被设计成比值，但 identity 基线 = 0.00038（残差项 1e-3 × 0.377），任何失真都能轻松超过 10×，包括几乎没被惩罚的 bitcrush（0.126/0.00038 ≈ 332×）。**该判据无效**，应改为绝对增量阈值。本文因此改用"相对 identity 的绝对增量"列，并明确 bitcrush 的 0.126 是**未通过**实质可分性的证据。

**E3（负面发现，设计代价）**：零初始化残差头使 step 0 的上游梯度严格为 0。这不是 bug（`parameters_without_grad` 为空，head 一步后全部打通），但它会让人误判"模型不学习"。训练脚本必须记录前 N 步的 per-block 梯度范数。

**E3b（E3 的实测后果，已修）**：`tests/test_model.py::test_unet_learns_a_non_identity_edit_and_backprops` 原先不设种子、只跑 3 步；单独跑通过，全量跑**失败**（`1 failed, 99 passed`）。原因是前几步只有 head 在动，损失曲线强烈依赖随机初始状态，测试因此依赖执行顺序。修法：`torch.manual_seed(0)` + 步数 3→10，并在注释里说明原因。修后连续 3 次全量 pytest 均 `100 passed, 1 skipped`。这条记录保留在此，因为"只有 head 先动"这个特性还会影响任何短步数的训练探针。

**E4（负面发现，最重要）**：8-bit 量化的总损失增量仅 **0.126**，而其他失真是 1.6–2.4。当前损失几乎不惩罚宽带量化噪声，而量化/嗡声正是机械音的典型缺陷 → 单靠这套损失训练，模型很可能学不会"去嗡"。候选补丁（谱平坦度、高频带能量比、对抗项）必须各自开实验，不得直接改默认权重。

**E5（量级失衡）**：初始总损失 7.076 中 log_magnitude 占 6.512（**92%**），其余五项合计 < 2%。默认权重 1.0 只是起点；训练前需重标定（LM 权重约 0.1 或改为每频带归一化），否则早期梯度几乎只优化 LM。

**E6（设计使然的异常值）**：gain_2x 的 SC = 1.000 精确等于 1，因为 SC 以 `||m_t||` 归一化，整体幅度翻倍时相对误差恰为 1。若需要增益无关性需改 SC 定义，属 P2 消融，本实验未改。

**E7（勘误：标题）**：标题写 "four-term loss suite"，实际是六项（SC、LM、periodicity、envelope_shape、envelope_level、residual）。标题保留原文不改，此处记录不一致。

## 结论 / 下次实验

- v0.1 的**数值**是可靠的：恒等起点、任意长度、51 ms 感受野、CPU/MPS 一致、梯度可通。可以进入训练阶段（前提是 P1 数据就绪）。
- **不能**据此宣称模型有用：本实验没有训练、没有真实音频、没有听测；所有指标都是合成夹具上的探针。
- 51 ms 感受野覆盖音节内微结构，但**不足以修改长时韵律**（语调、重音、语速），架构文档已把它列为 v0.1 的已知局限。
- 训练前必须解决：E5（损失量级重标定）、E4（量化伪影盲区）、E3（记录前 N 步梯度）。
- 下次实验（按优先级）：
  1. EXP-0009（可选、纯数值）：谱平坦度/高频带能量项作为补丁 A/B 的离线对比，用同一夹具矩阵判断 bitcrush 增量能否提升到与其他失真同量级；
  2. P2 S0：单条 1 s 片段过拟合 200 步（需 P1 数据或先用手工合成对），验证损失能下降且残差非零；
  3. S1–S2 需真实数据与许可，当前阻塞。
