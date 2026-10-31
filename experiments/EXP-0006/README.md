# EXP-0006 — P1-C 成对数据构造：按实测起点渲染的机械对应物与显式阻塞

## 假设
把每个音节按「从真实音频测得的起点」单独渲染并放回等长缓冲区，可以让机械信号与真实信号严格等长、并把音节间停顿保留在原位置；后端口不能发声的声母/韵母必须记为 `blocked`，不得近似或静默删除。

## 变量 / 对照组 / 数据划分
- 唯一实现变量：新增 `src/vhe/data/pairing.py`（分析 → 渲染 → 配对记录）。合成器、F0、分段、重采样代码均未改动。
- 对照（负对照）：同一 fixture 用「单个拼接请求」渲染（本实验第一版实现），用于证明改动不是装饰——该版本机械信号只有 18240 样本，比真实信号短 4560 样本，并迫使真实音频被截断。
- 数据：**全部为合成信号**。fixture 由合成器渲染 3 个孤立元音 a/i/u（各 0.2 s、100 ms 间隔、50 ms 前导静音）拼接而成，用来冒充「3 个真实音节」。**本机没有 AISHELL-3，本实验没有任何真实语音结论。**
- 划分：不涉及训练/验证/测试划分（本实验只做数据构造）。

## 完整复现命令 / 环境与哈希
```bash
.venv/bin/python experiments/EXP-0006/measure_pairing.py   # -> logs/pairing_report.json
.venv/bin/python -m pytest -q                              # -> logs/pytest.txt
```
- 环境：macOS 27.2 arm64（Apple M5, 32 GB），Python 3.14.6，NumPy 2.x，PyTorch 2.15.0（仅 CPU/MPS，未使用）。
- 随机性：合成器 seed 固定 42；本实验重复运行逐字节一致。
- 输入哈希：`measure_pairing.py` SHA256 = `9533588b94949f5a16c0f7ed11f2dc206ac75b186fc87f45307e9f1086598d21`。
- 输出哈希：`logs/pairing_report.json` = `1390212f7be6c45e67001701bea18a2ad0f276935eaada8ff8b2ec6ee0f47988`；`logs/pytest.txt` = `46aef67fc4b490372de1ae0398531e605690f9a752230315aa101efc4cf16f6a`。
- 代码提交：见 `metadata.json` 的 `git.commit`（记录时工作区 dirty，配对模块本身即在该 dirty 状态中验证）。

## 预注册通过/失败条件
1. 3 个元音 fixture：`status == "ok"`，`real.size == mechanical.size`，分段数 = 3。
2. 音节间停顿保留：机械信号在停顿窗（样本 6000–7200）RMS 为 0，且与真实信号同位置对齐。
3. `ma1/ni3/guang3`：`status == "blocked"`，不产生 mechanical，每个音节给出具体 inventory 理由。
4. 44.1 kHz 输入：记录 `resampled`，输出仍与机械信号等长。
5. `write_pair` 写出 real.wav/mechanical.wav/pair.json，含两侧 SHA256，重复写同一目录抛 `FileExistsError`。
6. `pytest -q` 全绿（真实 CUDA 缺失的 skip 不计）。

## 实际结果

| 项目 | 结果 | 判据 | 结论 |
|---|---|---|---|
| 3 元音配对 | `ok`，22800 / 22800 样本 | 等长 | 通过 |
| 停顿窗 RMS（真实 / 机械） | 0.0 / 0.0 | 保留停顿 | 通过 |
| 分段 | 3 段，speech 0.62 s，3 voiced，0 issue | =3 | 通过 |
| F0 统计 | voiced ratio 0.6848，中位数 199.96 Hz | 与 200 Hz 输入一致 | 通过 |
| 44.1 kHz 输入 | `ok`，22800 / 22800，含 `resampled` + `syllablewise_render` | 等长 | 通过 |
| `ma1/ni3/guang3` | `blocked`，3/3 音节不可渲染 | 显式阻塞 | 通过 |
| pair.json | SHA256 `dace35fd…`，含两侧音频 SHA256 | 可溯源 | 通过 |
| 测试 | 85 passed, 1 skipped | 全绿 | 通过 |

关键哈希：
- 元音配对 request fingerprint `3ea14fbc26ffd3ab7b9dc755eb41ff44c6066f7e73a1af970273117bc3cb1023`；real `945db6ed…`，mechanical `f7606ca8…`。
- 44.1 kHz 配对 fingerprint `3a6ba09d…`；real `434e3702…`，mechanical `3f2ff456…`。

阻塞理由（原文）：`onset 'm' not in backend inventory ['a', 'i', 'u']`、`onset 'n' …`、`onset 'g' …`。

## 失败、负面发现、错误日志
1. **blocked 状态被误标成 rejected**：`build_pair` 的初始 `status` 是 `"rejected"`，inventory 不足的分支直接 `return` 而没有改状态，导致「后端口不能发音」与「音频分析失败」在记录里无法区分。修正为显式置 `blocked`。
2. **单请求拼接会压缩掉停顿，并迫使真实音频被截断**：第一版把所有音节塞进一个 `SynthesisRequest`，但 `PhonemeSequence` 没有静音槽位，机械信号只有 18240 样本（0.76 s：0.05 前导 + 3×0.2 音节 + 尾静音），真实信号 22800 样本（0.95 s，多出 3×0.1 s 停顿）。代码当时用 `min()` 截断两者，**这会静默改变真实音频的内容与时长**，属于产品边界问题，已删除该逻辑，改为按实测起点逐音节放置。
3. **元音相连时能量分段会把 3 个音节并成 1 段**：连续发声没有能量谷，`segment_by_energy` 只给出 1 段，配对因此被拒（`segment_syllable_mismatch`）。这不是 bug 而是能量法的固有限制；测试 `test_continuous_vowels_collapse_into_one_segment` 把它固定为已知行为。真实中文连读语音大概率触发。
4. **真实语料不可用且存在结构性阻塞**：本机 `data/raw/aishell3`、`data/raw/AISHELL-3`、`~/data/aishell3` 均不存在（`real_corpus_available: false`），未获授权下载 19 GB。更关键的是：**即使有语料，AISHELL-3 的普通话音节几乎都带声母/复韵母，当前后端只能发 a/i/u**，`build_pair` 会把它们整体判为 `blocked`。因此 P1 的「20 对真人对听」在现后端下无法达成，不是工程未完成而是能力缺口。
5. **记录创建晚于实现**（与此前 EXP-0003/0004 相同的流程偏差），已在 metadata 注明。

## 已知限制
- 逐音节渲染让每个音节边缘带 30 ms 淡入淡出，边界比真实语音更「塌」，记录为 `syllablewise_render`(warning)。
- 声调不参与渲染：`a1` 与 `a4` 在现后端只差 F0 曲线，不差任何音段特征。
- 能量分段对连续语流不可靠；没有强制对齐器。
- 本实验没有任何「自然度」「音色保持」结论，VHE 仍未训练。

## 结论 / 下次实验
配对管线在合成元音上满足全部预注册判据：等长、停顿保留、非法音节显式阻塞、44.1 kHz 输入可追溯、写盘可溯源且拒绝覆盖。**真实数据侧仍被两件事同时阻塞：语料缺失，以及后端音素表只有 a/i/u。**

下一步优先级：
1. 扩后端音素表（声母/复韵母），否则 P1 验收永远无法在真实语料上完成；
2. 拿到语料后先跑 `python -m vhe manifest`，再对至少 20 条做人工听音，单独记为 EXP-0007，不得沿用本实验结论；
3. 用真实语料重测能量分段与 F0 的失败率，把失败样本暴露成统计，不得静默过滤。
