# EXP-0000 — CPU机械元音基线与复现验证

## 假设
同一参数、同一 seed 的上游机械声音代理合成器生成的 WAV 在同一执行环境中逐字节一致；可为后续 VHE 输入提供固定 baseline。

## 变量 / 对照
两次生成完全相同命令，仅输出文件路径不同。`a,i,u`，24kHz，1.2 秒，f0=220Hz，模板 `yang`，seed=42。未使用任何真人录音、GPU、神经网络。

## 结果
- `PYTHONPATH=src:. python -m pytest -q`: **7 passed**（当前执行环境）。
- 生成文件两个，均 28,800 个单声道 PCM 样本。
- 两次 WAV 的 SHA256 都是 `cefbcaec99a276683989f05abc537c6dfb8edfeef164be7728ee22775ada7985`。
- 生成状态：**机械基线，尚无 VHE，自然度提升未被测试**。

## 失败试错（必须保留）
- 首次尝试 `python -m pip install -e '.[dev]' --no-build-isolation -q` 失败：所在沙箱环境无法联网访问依赖索引，`ruff>=0.7` 无法安装。此问题属于环境依赖获取，并非代码验证失败。改为使用已安装的 `numpy`/`pytest` 及 `PYTHONPATH=src:.` 测试成功。
- 这次验证 **不意味着 Windows / RTX 4090 上已经运行成功**；需要本地 Agent 复测。

## 听音与质量
未做正式主观盲听，不能宣称「类似真人」。音源实现是谐波与共振峰代理模型，并非真正全声道物理建模。

## 下次实验
P1 前半：AISHELL-3 adapter、按 speaker split、错误显式报告、20 对配对数据人工审核；无授权真实数据时只做夹具测试。
