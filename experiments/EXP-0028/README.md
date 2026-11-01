# EXP-0028 — GPU memory of the P2 training recipe

## 假设
（运行前由工具写入，原文保留）The P2 runs never recorded GPU memory, so the headroom for larger channels or batch is unknown. With the EXP-0027 recipe (crop 1.6 s, 24 kHz, ResidualUNet 16 channels, batch 8, FP32) on the 4090 Laptop 16 GB card, torch.cuda.max_memory_allocated stays under 1.0 GB and max_memory_reserved under 2.0 GB, and doubling channels or batch size at most quadruples allocated memory. If the baseline already exceeds 1 GB, the 8x-headroom assumption behind the next training plan is wrong and must be reported before scaling.

## 变量 / 对照组 / 数据划分
唯一变量是「通道数 × batch」的三档矩阵，其余配置固定为 EXP-0027 的配方（crop 1.6 s、24 kHz、overfit 8 对、seed 42、lr 3e-3、20 步）。

| tag | channels | batch_size |
|---|---|---|
| ch16_b8 | 16 | 8 |
| ch32_b8 | 32 | 8 |
| ch16_b16 | 16 | 16 |

- 数据：`derived/aishell3-pairs/p2-train40x10-v2`（217 对缓存，overfit 固定取前 8 对）。
- 对照组：无独立对照组；ch16_b8 即 EXP-0027 的配方，作为放大基线。
- 步数只有 20，**本实验不评估收敛**，只看显存；末步 total 仅作「确实跑了 20 步」的旁证。

## 完整复现命令 / 环境与哈希
```
cd ~/VHE && PYTHONPATH=~/VHE/src .venv/bin/python experiments/EXP-0028/probe_memory.py \
  --cache ~/datasets/VHE-Datasets/derived/aishell3-pairs/p2-train40x10-v2 \
  --out-root ~/datasets/VHE-Datasets/derived/aishell3-runs/EXP-0028 \
  --steps 20 --out ~/datasets/VHE-Datasets/derived/aishell3-runs/EXP-0028/memory_report.json
```
- 主机：`ssh -p 2222 hands@10.0.0.190`（Windows 11 + WSL2，Ubuntu 26.04.1）
- GPU：NVIDIA GeForce RTX 4090 Laptop GPU，16375.5 MiB
- Python 3.12.14、torch 2.11.0+cu128、CUDA runtime 12.8、FP32
- manifest sha256 `1a8138c4d13e547d1d266a7aaf50d3a002a8e32dad78b0555f1b417c77074f31`
- 报告 `logs/memory_report.json` sha256 `58d00d233063b2034920e61072dffd0f9b6f45c1ca362dfc4d8793d9872a0f03`
- 实测时 EXP-0028 尚未提交到远端，脚本以 `/tmp/exp0028_probe_memory.py` 运行，内容与 `experiments/EXP-0028/probe_memory.py` 相同。

## 预注册通过/失败条件
1. ch16_b8 的 `max_memory_allocated` < 1.0 GB → 否则「8 倍余量」判断错误。
2. 翻倍通道或翻倍 batch 后 allocated ≤ 4 倍 baseline。
3. 三档都跑完、无 OOM、输出长度不变。

## 实际结果与主观听音
| tag | channels | batch | params | max_allocated (MB) | max_reserved (MB) | 末步 total |
|---|---|---|---|---|---|---|
| ch16_b8 | 16 | 8 | 251,201 | **1347.4** | 1620.0 | 3.7439 |
| ch32_b8 | 32 | 8 | 1,001,089 | **2566.1**（1.90×） | 2700.0 | 3.5234 |
| ch16_b16 | 16 | 16 | 251,201 | **2716.4**（2.02×） | 3106.0 | 3.6456 |

| 判据 | 结果 |
|---|---|
| 1. ch16_b8 allocated < 1.0 GB | **失败**：1347.4 MB = 1.32 GB |
| 2. 翻倍后 ≤ 4× baseline | 通过（1.90×、2.02×） |
| 3. 三档跑完无 OOM、长度不变 | 通过 |

主观听音：无音频产出，不适用。

## 失败、负面发现、错误日志
- **判据 1 失败**，且失败的正是「余量」这件事：16 GB / 1.35 GB ≈ 12×，留 20% 安全边界后约 9–10×，不是假设里的「随便 8 倍」。
- 1.35 GB 对 25 万参数的小模型偏大，说明显存被激活而不是参数吃掉：1.6 s × 24 kHz = 38400 样本，U-Net 在多个分辨率上保留激活，batch 8 就占 1.3 GB。省显存要缩 crop 或减通道，不是换更小的模型。
- 每样本约 168 MB（ch16）。按此**线性外推**（未实测）：ch16/batch32 ≈ 5.4 GB、ch32/batch32 ≈ 10 GB、ch64/batch16 ≈ 10 GB。
- 翻倍只涨 2 倍而非 4 倍，与「激活 ∝ batch × 分辨率」一致，逐层缓存没有超线性项。
- 未做：crop 长度扫描（0.8/1.6/3.2 s）、ch64、AMP/bf16、梯度检查点、EXP-0025/0026/0027 的逐步显存插桩（仍是记录缺口）。

## 结论 / 下次实验
- 显存放得下 P2 的放大需求，但必须先测再跑。任何「加通道/加 batch」的计划都要先用本脚本量一次。
- 目前可安全使用：ch16/batch16（2.7 GB）、ch32/batch8（2.6 GB）。下一步若要提高容量，优先测 crop 0.8 s 或梯度检查点，而不是直接上 ch64。
