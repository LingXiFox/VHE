# 里程碑与停止条件

| 阶段 | 交付 | 验收门槛 | 状态 |
|---|---|---|---|
| P0 | 本地仓库、最小机械声、测试、实验模板、**合成器契约** | CPU 单测绿、两次相同 seed 输出哈希一致、实验目录不可覆盖；CUDA/MPS 真机 smoke 单独验收，不冒充通过 | 本机已验收（EXP-0002 MPS 真机、EXP-0003 契约重构 35 passed）；**Windows CUDA 真机未验证** |
| P1 | 中文数据 manifest、对齐、成对机械音频 | 至少人工检查 20 对；错误分段有显式报告；按说话人拆分 | 进行中：AISHELL-3 适配器 + manifest CLI 通过合成夹具（EXP-0004）；信号前端（44.1k→24k 重采样、自相关 F0、能量分段）通过合成信号（EXP-0005，71 passed）；**真实语料不在本机，真实解析/对齐/听音验收阻塞**；成对构造已实现并通过合成元音（EXP-0006，85 passed）；**真实语料阻塞**；P1.2 已补 26 音素中文后端（EXP-0009 能力验收 + EXP-0010 审计），擦音谱带为待标定的手设值；P1.3 完成真实语料索引与说话人划分（EXP-0012/0013：88035 句 / 218 人 / 0 解析错误 / 30.95% 音节覆盖，**整句可渲染仅 0.21%**），F0 估计器仍含八度错误（EXP-0014，已由 EXP-0015 修复）；P1.4 完成数据审计缺陷修复、拒绝式 F0 跟踪、音素扩表 26→36（EXP-0016，音节覆盖 30.95%→69.24%，整句 0.21%→6.04%）；对齐与听音验收仍未做；P1.5 完成管线集成与 20 条真实配对（EXP-0017/0018/0019：拒绝式 F0 进管线、v4 manifest 可加载 5314 句、16/20 出预览但 **0 条 training-ready**，能量 DP 在连续语流上不可信，模板与真人相距约 7 dB） |
| P2 | VHE 浅层模型和一批次 overfit | 能过拟合小批次且没有长度/形状错误；训练可恢复 | 进行中：v0.1 模型与六项损失已实现并完成数值验证（EXP-0008，100 passed）；强制对齐器已实现并在合成信号上验证（EXP-0020，边界误差 12 ms），真实语料上可收敛（EXP-0021），逐状态转移重估的时长消融为负面结论（EXP-0022：边界仅移动 1.07 ms，vowel/stop 1.63 未达判据），**配对管线接入与听音验收未完成**，故训练仍未开始 |
| P3 | 跨 speaker/template 测试 | 对照组、音高、音色、自然度全面报告 | 待做 |
| P4 | GAN/VAE/残差控制消融 | 清晰归因，含失败实验与盲测 | 待做 |

**Stop conditions**：没有成对对齐质量验证、不清楚数据许可、测试集泄漏、GPU OOM 无记录、出现静默跳过音频/音素、通过变声达到虚假自然度时，暂停进入下一阶段。

## P1.3 数据准备（2026-11-01，本机 Apple M5 / CPU）

| 项 | 状态 | 证据 |
|---|---|---|
| 说话人零泄漏 180/19/19（seed 20261101） | 完成 | EXP-0012，`tests/test_data_corpus.py` |
| 全量文本/拼音扫描（只读 WAV 头） | 完成，0 错误 | EXP-0012，manifest `651bbbac…` |
| 26 音素覆盖率与扩展优先级 | 完成：音节 30.95%、整句 0.21%、前 10 项排序已出 | EXP-0012 |
| G2P 正字法（缩写/ü/舌尖/儿化） | 完成：1920 distinct token 0 失败 | EXP-0013 |
| 标注一致性审计 | 完成：63262 行、2 冲突、4 未标注（冲突待打标记） | EXP-0013 |
| F0 估计器小样本验证 | **失败**：210/300 Hz 八度错误，滑音仍有 12% | EXP-0014 |
| 按需加载接口（dataset + 配对） | 完成：`CorpusPairs.from_manifest` 不预生成音频 | `tests/test_data_dataset.py` |
| Windows CUDA 迁移准备 | 未做真机；接口不依赖 torch，数据路径 CPU 可跑 | — |

## P1.4 音素扩表与 F0 修复（2026-11-01，本机 Apple M5 / CPU）

| 项 | 状态 | 证据 |
|---|---|---|
| 数据审计缺陷修复 | 完成：2 处 `prosody_conflict` 落盘进 manifest；"4 条未标注"经复核为 label 文件头 4 行 `#` 注释（EXP-0013 已追加勘误）；格式错误/缺文本/冲突四分类 | `b9047f5`，`derived/aishell3-audit/v2/`、`derived/aishell3-scan/v3/` |
| F0 多候选 + 连续性跟踪 | 完成：默认 65–900 Hz 不缩窄，合成 8/8 hit50=1.00、八度错误 0.00；真实 12 条八度跳变率 0.0027（legacy 0.0248） | EXP-0015，`tests/test_data_pitch.py` |
| 低置信度显式拒绝 | 完成：`f0_hz=0` + reason（silence/no_candidate/low_confidence/octave_ambiguous/continuity_gated），不插值；真实拒绝率 0.4835（含 28% 静音/清音帧） | EXP-0015 |
| 音素扩表（10 单元 / 3 批） | 完成：26 → 36，`e ian ing iou` / `apical_retroflex uo ong en` / `zh h` | EXP-0016 |
| 覆盖率复算（不重解码） | 完成：音节 30.95% → **69.24%**，整句 0.21% → **6.04%** | `logs/coverage_stage{0..3}.json` |
| 旧渲染回归 | 完成：8 条 stage-0 音节 sha256 每批后不变 | `logs/render_baseline.json`、`regression.all_same=true` |
| 听音验收 | 未做：全程只有合成信号与数值指标 | — |
| Windows CUDA 迁移准备 | 仍未真机；数据路径不依赖 torch | — |

**跨端验收（待两台真机验证）**：Windows `--device cuda` 与 Mac `--device mps` 各保存 doctor、smoke、pytest 输出到新的实验记录；代码已添加，不能据 CPU 成功宣称 GPU 已验证。

## P1.5 管线集成与真实配对（2026-11-01，本机 Apple M5 / CPU）

| 项 | 状态 | 证据 |
|---|---|---|
| 拒绝式 F0 进入配对管线 | 完成：`build_pair` 默认 `pitch_estimator="viterbi"`，legacy YIN 保留为消融开关 | `tests/test_data_pairing.py` |
| 机械音高曲线质量记录 | 完成：coverage / gaps / 八度歧义帧 / 接缝 confidence；插值必标记并阻断 `training_ready` | EXP-0018 报告字段 |
| 能量 DP 对齐（替代纯能量分段） | 完成但有明确边界：`confidence` 只反映能量谷深度，连续元音上不可自证 | `src/vhe/data/align.py`、`tests/test_data_align.py` |
| manifest 可渲染性随音素表刷新 | 完成：v3 → v4（`1a8138c4…`），5314 句 newly_renderable、0 反向失效 | `vhe refresh`、`tests/test_data_corpus.py` |
| 实际可加载句数与覆盖率一致 | 完成：5314 句（train 4333 / dev 528 / test 453，218 人 3.37 h） | `CorpusPairs.from_manifest` |
| 真实配对 20 条 | 完成：16 条出预览、3 条 blocked（<80 ms 窗）、1 条 rejected；**0 条 training-ready** | EXP-0018 v3 |
| 拒绝帧是否被静默插值 | 未发生：拒绝帧 `f0=0` + reason；跨间隙插值只在明确标记后发生并阻断训练资格 | EXP-0015 / EXP-0018 |
| F0 拒绝率归因 | 完成：281 个 `no_candidate` 里只有 4 帧（1.4%）疑似有声漏检 | EXP-0017 |
| 模板/音色身份 | 完成分析：F0 不能选音色（10/16 一致），预设与真人相距约 7 dB，margin 仅 1.48 dB | EXP-0019 |
| 20 对人工验收材料 | 已生成（音频、谱图、F0 曲线、质量标记、试听表），**主人未试听 → 全部"未验收"** | `derived/aishell3-pairs/v2/`、`logs/listening_sheet_v3.md` |
| 听音结论 | 无。不得声称自然度提升 | — |
| Windows CUDA | 仍未真机验证 | — |

**P1.5 结论**：管线端到端可跑，但`training_ready` 为 0，P1 的"至少人工检查 20 对"门槛未达成。进入 P2 前必须先有 phoneme-level 强制对齐（或人工边界）并完成听音验收。

## Windows CUDA 真机验收（2026-11-01，WSL2 / RTX 4090 Laptop）

| 项 | 结果 | 证据 |
|---|---|---|
| 环境 | Ubuntu 26.04.1 WSL2、28 核、21 GiB RAM、842 GB 空闲、`/dev/dxg` 直通 | `ssh -p 2222 hands@10.0.0.190` |
| GPU | RTX 4090 Laptop GPU 16376 MiB，driver 621.01，`nvidia-smi` 在 `/usr/lib/wsl/lib` | `doctor` 输出 |
| Python/torch | venv Python 3.12.14（uv），torch **2.11.0+cu128**，CUDA runtime 12.8 | `doctor` 输出 |
| `vhe doctor --device cuda` | **通过**：`selected_device=cuda`、`cuda_available=true`、`precision_policy=float32` | 终端输出 |
| `vhe smoke --device cuda` | **通过**：forward/backward/optimizer step，峰值显存 1.62 MB | 终端输出 |
| `pytest -q` | **404 passed / 2 skipped**（skip 均为 MPS 专用用例；CUDA 专用用例在本机转为通过） | 终端输出 |
| 数据镜像 | 88035 wav / 27,208,419,174 B；5314 条可渲染文件逐文件 sha256 一致 | 聚合摘要 `d26b7f68…` |
| 训练 | **未开始**：P2 前置（forced alignment、听音验收）仍未达成 | — |

注意：这是 WSL2 路径，不等于原生 Windows 图形会话验收；两者若要都算通过必须分别跑。

## P2 前置一：phoneme-level 强制对齐器（2026-11-01，本机 Apple M5 / CPU）

| 项 | 状态 | 证据 |
|---|---|---|
| 前端 | 完成：NumPy MFCC 13 维 + delta + delta-delta，25 ms 帧 / 10 ms 跳，16 kHz，逐句 CMVN | `src/vhe/data/features.py`、`tests/test_data_features.py`（9 例） |
| 模型 | 完成：左到右链式 HMM，36 音素 x 3 状态，对角 GMM，转移固定 0.5/0.5，Baum-Welch 用充分统计量累加（内存不随语料增长） | `src/vhe/data/hmm.py`、`tests/test_data_hmm.py`（9 例，含全路径枚举对照） |
| 对齐器 | 完成：能量门裁首尾静音（HMM 不建模静音），Viterbi 路径按音素块切分，输出 span/confidence/logprob | `src/vhe/data/aligner.py`、`tests/test_data_aligner.py`（12 例） |
| 合成验证 | 通过：单分量最大边界误差 12 ms（平均 5.9 ms）、门控裁掉 15 帧填充、训练单调 | EXP-0020 |
| 合成消融的负面结论 | **Baum-Welch 未改善边界**（5.9 → 9.7 ms，似然仍单调上升），双分量更差（26 ms）→ 默认 `n_components=1` | EXP-0020 |
| 真实语料验证 | 完成（负面）：226 句 / 16 说话人训练，60 句未见说话人评估，能收敛、无非法跨度，但各音素类时长挤在 0.05–0.135 s（元音/塞音 1.43），与能量谷 DP 的 50 ms 一致率仅 30.7% | EXP-0021 |
| 时长模型 | 已实现逐状态自环概率（Beta(2,2) 先验、0.05–0.98 clamp、未触及状态保留），默认关闭 | `src/vhe/data/hmm.py` |
| 时长模型消融 | **负面**：重估后 vowel/stop 1.59 → 1.63（判据 ≥2.0 失败），地板比例 5.8% → 6.0%（判据 ≤5% 失败），置信度 0.7959 → 0.8004（勉强过），似然单调、DP 一致率 30.6% 不变；边界平均只移动 **1.07 ms**（中位 0.0、最大 40）而逐句 logprob 抬高 20 nats → 发射项主导解码，转移先验不是时长瓶颈 | EXP-0022 |
| 接入配对管线 | **未做**：`build_pair` 仍用 `g2p.syllable_plan` 的音节内名义时长 | — |
| 听音验收 | 仍未做（与 P1.5 相同的 20 对材料） | — |

**结论**：对齐器在可分离合成信号上可复现已知边界，在真实语料上能收敛且不产生非法跨度。EXP-0021 曾把"各类时长挤在 0.05–0.135 s"归因于固定 0.5 转移概率，**EXP-0022 已证伪该因果**：固定 0.5 时模型隐含 60 ms，解码仍给出元音 115.7 ms / 塞音 72.7 ms；逐状态重估把似然抬高约 20 nats/句，边界只移动 1.07 ms。时长区分度不足（vowel/stop 1.63）的瓶颈在发射模型，需要显式时长模型或更大发射容量，而不是转移先验。vowel/stop 1.63 是否已接近真实仍无法判定——参照物能量谷 DP 未经人工验证，能判定它的只有听音验收。配对管线接入、CLI 与听音验收仍未完成；训练本身仍未开始。
