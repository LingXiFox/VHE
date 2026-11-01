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

## 2026-10-31 | EXP-0003 P0-B 合成器契约重构（字节级等价）

- 变更：`src/vhe/synth.py`（126 行单体）→ `src/vhe/synth/{contract,formant_proxy,registry,__init__}.py` + `src/vhe/audio.py`。
  新增 `SynthesisRequest`（template/phonemes/pitch/timing/sample_rate/seed）、`SynthesisResult`（含 backend 版本与 request fingerprint）、
  `PitchCurve`（线性插值、区间外 clamp）、`Timing`（前后静音 ≤5 s）、`SynthBackend` 协议与注册表。
- 结果：`pytest -q` **35 passed, 1 skipped**（跳过项为无 NVIDIA CUDA）；重构前 18 passed。
- 字节等价：同一 CLI 命令连续两次输出 SHA256 均为 `cefbcaec99a276683989f05abc537c6dfb8edfeef164be7728ee22775ada7985`，与 EXP-0000/EXP-0002 完全一致。
- 失败记录（保留）：重构中把 `write_wav` 的量化改成 float64，导致 13/28800 个样本相差 1 LSB；已改回 float32 量化并以测试固定。
  契约测试自身也写错两次（过零率冒充音高、未归一化自相关偏向长滞后；WAV 往返 1 LSB 容差不成立），改法与原因见实验记录。
- 流程偏差（不隐去）：实验记录在实现完成后、验证运行前创建，晚于 AGENTS.md 要求的"运行前"。
- **VHE 仍未训练**，无任何自然度或音色保持证据。证据：`experiments/EXP-0003/`。

## 2026-10-31 | EXP-0004 AISHELL-3 适配器（合成夹具验证，真实语料阻塞）

- 新增 `src/vhe/data/aishell3.py`：目录发现、`content.txt`/`spk-info.txt`/`pinyin.txt` 解析、只读 WAV 头的校验、
  SHA256 溯源、按说话人分组切 dev（官方 test 不动）、JSONL manifest 读写；新增 `python -m vhe manifest`。
- 失败可见：拒绝项一律落成 `ParseIssue`（missing_wav / not_mono / not_16bit / empty_text / too_short / too_long /
  unexpected_sample_rate(warn) / pinyin_mismatch(warn) / bad_line / unreadable_wav），不静默过滤。
- 结果：适配器 **14 passed**，全库 **49 passed, 1 skipped**（CUDA）；manifest SHA256 `d49953ec…`（10 行），
  同一输入可复现，重复写入被拒绝（退出码 2）。
- 真实 AISHELL-3 不在本机，且未获授权下载 19 GB 语料 → **真实解析率、听音抽检、P1 真实验收均未完成**。
- 失败记录（保留）：`empty_text` 曾被 `strip` 吃掉 TAB 退化成 bad_line；测试路径漏了说话人目录导致假失败；
  夹具重写 WAV 触发拒绝覆盖。记录创建晚于实现（流程偏差，同 EXP-0003）。
- **VHE 仍未训练**，无自然度或音色保持证据。证据：`experiments/EXP-0004/`。

## 2026-10-31 | EXP-0005 信号前端：有理数重采样、自相关 F0、能量分段（仅合成信号）

- 新增 `src/vhe/data/resample.py`（多相 windowed-sinc，Kaiser β=8.6，按 `Fraction(src,dst)` 精确映射相位）、
  `src/vhe/data/f0.py`（FFT 自相关 + 无偏修正 + 抛物线插值 + 三倍频保护）、`src/vhe/data/segment.py`（能量谷分段 + 质量 issue）。
- 结果：通带 0.5/1/2/4 kHz 偏差 ≤0.0001 dB，15 kHz 残留 **-87.28 dB**（线性插值对照 **-0.34 dB**，别名原样通过）；
  稳态 F0 100–400 Hz 误差 0.02%–0.84%，100→400 Hz 扫频平均误差 0.14%；4 突发精确分为 4 段且全 voiced、0 issue；
  10 ms 点击被丢弃并记 `segment_too_short`；纯静音记 `no_speech`(error)。`pytest -q` **71 passed, 1 skipped**。
- 失败记录（保留）：多相索引把 `Fraction(dst,src)` 的分子/分母当 (step, phases)，上采样直接 `IndexError`；
  F0 无偏修正偏袒长滞后，220 Hz 与 400 Hz 分别读成 73.33 Hz、133.25 Hz（=f0/3），加"最短强局部峰"规则后修正；
  通带用峰值测量在 4 kHz 得到 -1.25 dB（6 样本/周期、相位误差 `20log10(cos30°)`），改整周期 RMS 后为 0 dB；
  分段时长曾加满窗导致 20 ms 点击算成 65 ms，改为 `frames*hop`；15 ms 点击仍卡在 60 ms 边界，记为已知限制。
- **仅合成信号**：本机无 AISHELL-3，真实语音上的 F0/分段精度与人工听音均未验证。证据：`experiments/EXP-0005/`。

## 2026-10-31 | EXP-0006 成对数据构造：按实测起点渲染 + 显式阻塞（仅合成元音）

- 新增 `src/vhe/data/pairing.py`：`parse_syllable`（声母/韵母/声调拆分，无模糊匹配）、`build_pair`（重采样 → F0 →
  能量分段 → 音节解析 → 逐音节渲染回填）、`write_pair`（real.wav / mechanical.wav / pair.json，拒绝覆盖）。
- 结果：3 元音 fixture `ok`，22800 / 22800 样本严格等长，停顿窗 RMS 0.0 / 0.0，3 段全 voiced，0 issue；
  44.1 kHz 输入重采样后仍等长；`ma1/ni3/guang3` → `blocked`，3/3 音节给出 `onset 'm' not in backend inventory` 理由；
  `pytest -q` **85 passed, 1 skipped**。report SHA256 `1390212f…`。
- 失败记录（保留）：① `blocked` 分支忘记改 `status`，被误记成 `rejected`；② 第一版用单个拼接请求，
  `PhonemeSequence` 没有静音槽位 → 机械信号只有 18240 样本（0.76 s）而真实 22800（0.95 s），旧代码用 `min()`
  **截断真实音频**，已删除该逻辑改为按实测起点逐音节回填；③ 连续元音无能量谷 → 3 音节并成 1 段，固有限制，
  已写成测试 `test_continuous_vowels_collapse_into_one_segment`；④ 本机无 AISHELL-3（三个候选路径均不存在），
  未授权下载，且即使有语料，当前后端只能发 `a/i/u`，普通话音节会整体 `blocked`。
- **结论边界**：没有任何真实语音、自然度或音色保持结论；P1 的 20 对真人对听在当前后端下不可达。证据：`experiments/EXP-0006/`。

## 2026-10-31 | EXP-0007 缺陷审计与修复：F0、分段、音高插值、配对对齐（仅合成探针）

- 施工顺序偏差（保留）：探针脚本先写、证据跑完才补 `tools/new_experiment.py` 建档，已在 metadata `errata` 记录。
- 发现并修复 6 项、降级 1 项：
  ① `f0.py` 明确积分窗与全长相关系数不一致 → 差分全为 0；重写为 CMND，16/24/48 kHz × 65–880 Hz 误差 ≤0.01%
     （修复前 65/80 Hz 分别读成 969.70/941.18/914.29 Hz），静音残差 voiced 0.515→0.0。
  ② `segment.py` 先合并后精修 → 60 ms 真停顿被吃成 ~10 ms；改为**先样本级精修每个 run 再合并**，4 个 burst 的
     onset/duration 误差从 −30/+40 ms 降到 **0.0 ms**，20 ms 点击现在被正确判 `segment_too_short`。
  ③ 合成器 `fade_out` 覆盖 `fade_in`，短段（10/20/29 ms）直接 broadcast 崩溃：改 `fade = min(0.03*sr, n//2)`。
  ④ `PitchCurve.sample` 在 Hz 域线性插值，两八度中点偏 **386 cents**；改为 log2 域插值（中点 250→200 Hz），
     常数曲线保持逐位不变。
  ⑤ `pairing.py` 的 `request_fingerprint` 被单音节渲染指纹覆盖 → 改为 composite sha256；删除死代码 `_timing`。
  ⑥ 配对缺少增益匹配：按音节真实 RMS 归一（`gains_db` 记录），并加 **峰值封顶 0.999**（超限时记
     `gain_limited_by_peak`）。
- 结果：三音节 RMS 比 0.995/1.979/1.278 → **1.0/1.0/1.0**；机械声在真实有声处的零样本 2.18%→1.27%。
  `audit_after.json` 的 sha256 在干净树上重跑一致（确定性已证）。`pytest -q` **86 passed, 1 skipped**。
- 未修/回归（保留）：缺失基频 200+4×400 Hz 仍读 400.91 Hz（CMND 首个谷在 2× 周期，记为可辨识性限制）；
  200→300 Hz 阶跃的边界帧伪影从 1 帧扩到 2 帧（`[99.9, 289.3]`→`[100.0, 100.0]`）；
  短段 fade 变陡使包络跳变相对变大（按比例，属预期）。证据：`experiments/EXP-0007/`。

## 2026-10-31 | EXP-0008 v0.1 模型与损失套件的数值验证（无训练、无真人音频）

- 新增 `src/vhe/model.py::ResidualUNet`（波形残差 U-Net，1,001,089 参数，零初始化输出层）与
  `src/vhe/losses.py`（多分辨率 STFT、自相关周期结构、能量轮廓、残差惩罚），并补 19 个测试。
- 结果：初始 `max|y−x| = 0.0`；长度 777/1000/24000/24013 全部原样；实测感受野 **1224 样本 = 51.0 ms**
  （65 Hz 的 3.31 周期 / 180 Hz 的 9.18 周期）；CPU vs MPS 编辑差 3.1e-06；`pytest -q` **100 passed, 1 skipped**。
- 重要负面发现：
  ① **8-bit 量化的总损失增量只有 0.126**，噪声/变调/移位/增益/丢段/变闷为 1.6–2.4 → 现有损失几乎不惩罚
     宽带量化噪声，而这正是机械音的典型缺陷，训练前必须补谱平坦度/高频带项或对抗项（各自开实验）。
  ② 初始总损失 7.076 中 `log_magnitude` 占 6.512（**92%**），默认权重未标定。
  ③ 零初始化残差头使 step 0 只有 `head` 有梯度（61950.56，其余 14 个 block 为 0），一步后全部打通；
     该特性还导致一个不设种子的 3 步测试在全量跑时失败，已加种子并延到 10 步。
- 预注册判据 5（10× 比值裕度）被证明**退化无效**（identity 基线 0.00038），改用绝对增量；勘误写在实验目录。
- **VHE 仍未训练**：本阶段没有真实音频、没有听测，不能宣称自然度或音色保持。证据：`experiments/EXP-0008/`。

## 2026-10-31 | EXP-0011 F0 估计器八度上偏：机制、两次尝试与最终规则（仅合成探针）

- 起因：P1.2 的 Mandarin 后端在 /a/ 220 Hz 上被读成 414–455 Hz，频谱核对（nperseg=1024）真值 211–234 Hz
  → 是估计器缺陷，不是音频缺陷；为此单开实验，没混进后端验收记录。
- 机制（已确认）：YIN CMND 取"首个 <0.15 的谷"。诊断帧 `cmnd(54)=0.062`（半周期）、`cmnd(109)=0.003`（整周期），
  argmin 本来就指向正确答案，被选择规则丢弃。
- 两次尝试：① 从全局最小重启 → **实测否决**，把八度上偏换成八度下偏（comb 87 / vowel 13 / robust 40 帧）；
  ② 有界二次谷搜索（只在 `(lag, 2*lag+2]` 内找局部最小，要求 `<=0.5×current` 且降幅 `>=0.02`）→ 采用。
- 结果：comb 精度 0.4333→0.7667、vowel 0.9351→**1.0**（555/555）、robust 0.5853→0.8710，纯音/静音/噪声门控全部不变。
- 保留的缺陷：comb 仍有 329/1410 帧（23%）落在"次周期"lag（CMND 0.11–0.15），既非 f0 也非倍频；10 dB 噪声下
  65/504 帧仍上偏；`estimate_f0` 不输出置信度。**附录 E5**：默认 65–900 Hz 搜索下 /a/ 210 Hz 读 842 Hz、
  300 Hz 读 902 Hz（F1≈800 落在 3f0/4f0），`fmax_hz=500` 才恢复——EXP-0009 的 F0 网格因此不能代表默认可用的精度。
- 阈值 0.02 / 0.5 是在本批 2469 帧上扫描后选的，未预注册，也未在真实语音上验证。证据：`experiments/EXP-0011/`。

## 2026-10-31 | EXP-0009 / EXP-0010 Mandarin 后端能力验收与算法审计（发现真实缺陷并修复）

- 新增 `src/vhe/synth/mandarin.py`（26 音素源-滤波器后端）与 `phonology.py`（音素表、共振峰目标、Chao 声调轮廓），
  契约加 `inventory`（`vowels` / `mandarin`），后端与请求音素表不一致时报错而不是当成另一种表解释。
- EXP-0009（能力）：26 音素全部可渲染、长度精确、峰值 ≤1；F0 误差 0.209–0.227%（65–880 Hz 网格，用了 `fmax_hz=1.5*f0`）；
  有声/无声自相关 0.937–0.957 vs 0.067–0.086；模板 yang 峰 [820,1078,1312] vs yin [586,680,867] Hz；
  CPU/MPS 往返 bit-equal。**未做听音**，不宣称发音正确或自然度。
- EXP-0010（审计）抓到真实缺陷：`_noise` 的带通原来是"以 low 为中心的凸包"，`/s/` 只到 6.4 kHz、`/sh/` 2.7 kHz 以上全丢、
  `/f/` 96% 能量挤在 1–2 kHz。改成真正的 rise×fall 带通后`/s/` 谱质心 4077→8518 Hz，但**修复引入第二个缺陷**：
  fall 边宽 3.5 kHz 让 `/s/` 能量伸到 13.5 kHz，0.45·sr 以上能量占比 2.1e-4 → **1.05e-1**。审计第二轮抓到，
  把 fall 终点 clamp 到 `0.45*sample_rate` 后回到 2.12e-4。
- 判据结果：(a) 混叠 通过（最终态）、(d) 复韵母谱通量 0.0015–0.0048 vs 硬拼接对照 0.107 通过、(e) 上下文尾段 dBFS 极差 0.95 dB 通过；
  **(b) 失败**：`/f/` 22% 能量在 8–12 kHz、质心 6156 Hz 高于 `/sh/` 5051 Hz——擦音谱带仍是手设占位值，不用调参掩盖；
  **(c) 失败**：ba/ka/sa 的边界帧 RMS 跳变比段内最大跳变大约 0.05，但采样级 click_ratio ≤1.08，说明是快速物理起音而非不连续，
  判据本身（5 ms 帧）需要改。两个失败都保留在原处。
- 流程偏差：EXP-0009 的 README 判据是跑完才落盘（判据当时只在探针/测试代码里），EXP-0010 的判据文本先写进 metadata。
  三态日志（pre_band_fix / bandpass_only / final）全部保留，没有覆盖。证据：`experiments/EXP-0009/`、`experiments/EXP-0010/`。

## 2026-11-01 | EXP-0012 AISHELL-3 语料索引、说话人划分与 26 音素覆盖率（本机全量文本扫描，不含波形）

- 前提核实：官方 `train/`+`test/` 目录是**按句切分**——`test/` 的 214 个说话人目录中 170 个与 `train/` 重名（并集恰 218 人）。据此不能拿官方 test 做 speaker-independent 评估，本轮改为说话人划分 180/19/19（seed 20261101），句子路径完全不动。
- 实现：`src/vhe/data/corpus.py`（`scan_corpus` 只读 `content.txt` 与 WAV 头、`write/read_corpus_manifest`、`audit_prosody_labels`、`rank_next_units` 精确边际排序）+ `vhe scan` / `vhe audit` 子命令。
- 结果：**88035 句 / 218 人 / 0 解析错误 / 997,997 音节 / 85.6178 h**，音节总数与 `phone_set.txt` 词频总和逐项吻合；train 72966 句/180 人、dev 8189/19、test 6880/19。
- 覆盖率：音节级 **0.309544**；整句可渲染仅 **187/88035 = 0.0021**。假设（<50%）成立，但整句维度远低于预期，是本轮最重要的负面结果：现有 26 音素表几乎无法整句合成。
- 优先扩展（按"加入后可整音节解锁"精确边际，非单点频次）：final:e 74797 → final:ian +47442 → final:apical_retroflex +37864 → final:ing +37436 → final:iou +34896 → final:uo +34783 → onset:zh +34606 → final:ong +29562 → final:en +26197 → onset:h +25585。
- **失败记录**：v1 扫描 9193 句 `bad_pinyin`，根因 `resolve_final` 的 j/q/x 分支提前 return 未还原 `iu` 缩写（`'jiu3': j + 'iu' is not a Mandarin final`）；另有 y/w 被当作声母导致 `onset:y` 86439 / `onset:w` 37524 虚假缺口。两处修复后 v2 才 0 错误。v1 产物（manifest sha256 `9ed0cc56…`，覆盖率 0.2728）原样保留在 `derived/aishell3-scan/v1/`，不覆盖。
- 证据：`experiments/EXP-0012/`；manifest `651bbbac1468305c47b8fee2be417561a152097cca0edc5dba5313ef2c1e0093`（不入 Git）。

## 2026-11-01 | EXP-0013 拼音正字法 vs 实际发音 + 韵律标注审计

- 正字法解析层（`mandarin.split_token`）：缩写 iou/uei/uen 还原、j/q/x + u → ü、y/w 作滑音、舌尖元音 apical_dental/apical_retroflex、唇音 + o → uo、儿化标记单列。**1920 个 distinct token 全部解析成功（0 失败）**，全量 997,997 音节 0 个 `bad_pinyin`。
- 触发证据：iou 34896、uei 27950、apical_retroflex 53935、apical_dental 20702、ü(v) 23661——这些单元在旧表里全部属于"阻塞"，说明旧统计确实是被解析能力限制住的。
- 标注审计（`label_train-set.txt` vs `content.txt`）：checked 63262、matched 63260、**mismatched 2**（`SSB19560288` index 4 `ni2`/ `ni3`；`SSB15550353` index 2 `zheng1`/`zeng1`）、unlabelled 4。
- 政策：文本以 `content.txt` 为准，label 只做韵律交叉核验；两处冲突在配对时打 `prosody_conflict` 标记，不得静默择一。未标注的 4 条 ID 本轮未落盘（下次补）。
- 证据：`experiments/EXP-0013/`，审计输出 `derived/aishell3-audit/v1/prosody_audit.json`。

## 2026-11-01 | EXP-0014 F0 估计器：合成真值与 12 条真实语音（小样本，不跑全量）

- 合成真值：8 条轮廓（5 条常数 + 上行/下行滑音 + 颤音）由 `MandarinFormantSynth` 渲染 /a/，F0 真值即传入 `PitchCurve`，按 ±50 cents 命中率与八度帧比例评分。
- **假设失败**：默认 65–900 Hz 下 210 Hz 命中 0.02、八度帧 0.98、中位误差 2404 cents（≈4f0）；300 Hz 同理 1903 cents（≈3f0）。`fmax_hz=500` 可修常数音（命中 1.00），但**下行滑音仍有 12% 八度帧、颤音 6%**——缺帧间连续性，EXP-0011 的"有界二次谷搜索"不足以外推到滑音。
- 真实语音：test split 每位说话人 1 条、12 人、24 kHz。中位 F0 120.9–289.9 Hz（与性别相容，中位值的中位数 225.9 Hz），发声帧比例 0.415–0.702，相邻帧八度跳变率 0–5.6%（均值 2.48%），2 条各 1 帧越界。
- **实现缺陷（保留证据）**：第一版把无声帧（`f0_hz=0`、`voicing=0`）当测量值，得出发声比例 1.0、中位 F0 0.0；第一版取样还全取自同一说话人。修正后重跑，旧报告保留为 `logs/f0_report_v1_unfiltered.json`。原假设中"44.1k vs 24k 中位差 <3%"本轮未测，不能声称通过。
- 结论：**F0 估计器暂不可直接用于成对构造**；若必须用只能 `fmax_hz ≤ 500` 并给配对打"F0 可能含八度错误"标记。下次 EXP-0015：多候选（lag/2lag/3lag）子谐波校验 + 帧间连续性平滑，用同一套轮廓与 12 条真实语音回归。
- 证据：`experiments/EXP-0014/`（`probe_f0.py`、`logs/f0_report.json`）。

## 2026-11-01 | EXP-0015 多候选 CMND + 谐波支撑 + Viterbi 连续性：F0 拒绝式跟踪

- 假设：显式多候选周期（含 2×/3×/4× 与半频）+ 谐波一致性 + 帧间连续性，能在**不缩窄 `fmax`** 的默认 65–900 Hz 下修复 EXP-0014 的倍频错误，并对无把握的帧显式拒绝而不是插值。
- 实现：`src/vhe/data/pitch.py`（`frame_candidates` / `prefer_shortest_period` / `_viterbi` / `track_pitch` / `octave_error_kind`；`cmnd_curve` 由 `vhe.data.f0` 提供）。评分 `(1-aperiodicity) * (0.7 + 0.3*harmonic_support)`，Viterbi 转移含连续性成本与未发声态，`f0_hz=0` + reason（ok/silence/no_candidate/low_confidence/octave_ambiguous/continuity_gated）。
- 合成（8 条与 EXP-0014 完全相同的轮廓，默认 900 Hz）：**8/8 hit50 = 1.00、八度错误 0.00**；对照 legacy 默认 210/300 Hz 仅 0.02（八度帧 0.98）、下行滑音 0.67、颤音 0.64；legacy `fmax=500` 下行滑音仍 0.88/颤音 0.94。
- 真实（同样 12 条 / 12 人 / 24 kHz）：总帧 2052，拒绝率 0.4835（其中静音 577 帧），非静音帧拒绝 0.2698；相邻帧八度跳变率均值 **0.0027**（EXP-0014 legacy 0.0248，约 9× 降低）；12 人中位 F0 的中位数 226.6 Hz（legacy 225.9）。
- **失败记录**：v1 用 40 ms 分析窗时下行滑音 300→130 只有 0.84 命中、16% 八度帧（40 ms 内音高变化约 6%，真周期不再是 CMND 谷，静止的 880 Hz 共振峰获胜），改 30 ms 后 1.00/0.00，其余 7 条不变 → 保留 `logs/pitch_report_v1.json` 作窗长消融。另修掉 7 个实现缺陷（整数倍候选被误判歧义、基础 dip 按深度/长度选取各失败一次、`cmnd_curve` off-by-one 造成 +5…40 cents 偏高、家族折叠排序写反、谐波支撑等权导致共振峰周期胜出、歧义判据用评分窗过宽）。
- 流程勘误：EXP-0015 目录最初手工创建，事后用 `tools/new_experiment.py` 重建骨架并原样回填探针与日志（md5 校验一致），未删除任何失败结果。
- 结论：跟踪器显著优于 EXP-0014，但**仍不宣称"可直接用于成对构造"**：无真实 F0 真值、拒绝率偏高、SSB04350040 中位 F0 与 legacy 差 0.18 倍频程原因未查明。未做听音。
- 证据：`experiments/EXP-0015/`（`probe_pitch.py`、`logs/pitch_report_v1.json`、`logs/pitch_report_v2.json`）。

## 2026-11-01 | EXP-0016 按实测边际分批扩表：26 → 36 音素，音节 30.95% → 69.24%

- 假设：按声学类别成批加入**整单元**韵母（每个韵母一个 `PhonemeSpec`，带完整共振峰轨迹，绝不拆成 i+a+n），会让 v3 manifest 上的音节覆盖率单调上升、不会让非法音节变得可渲染、已支持音素的渲染逐字节不变；整句覆盖率仍会远低于音节覆盖率（一句需要每个音节都可渲染）。
- 批次与声学实现（`src/vhe/synth/phonology.py`）
  - stage 1：`e`（[ɤ] 稳态，F1 500 / F2 1280 / F3 2500）、`ian`（F2 2250→1900 后接鼻尾，`nasal_from=0.60`）、`ing`（F2 2250→2050 后接鼻尾，`nasal_from=0.50`）、`iou`（F2 2200→1000 复韵母）。
  - stage 2：`apical_retroflex`（[ʅ]，F1 300 / F2 1600 / F3 1800 全程平，新增 `approximant` 带摩擦浊音声源分支）、`uo`（F1 400→500）、`ong`（鼻尾，`nasal_from` 生效）、`en`（F2 1500→1700 + 鼻尾）。
  - stage 3：声母 `zh`（affricate，`noise_band=(2000,6000)`、`noise_gain=0.30`、`burst=(3000,1500,0.004,1.0)`、`closure_s=0.035`、`vot_s=0.030`）、`h`（[x] fricative，`noise_band=(900,4000)`、`noise_gain=0.20`）。普通话 h 不拼 i/y，故 [x] 覆盖全部 h 音节，不用 [ç] 近似替代。
- 覆盖率（同一 v3 manifest `10539f61…`，只重算条目、不重解码 WAV）：音节 **0.309544 → 0.503755 → 0.621630 → 0.692366**；整句 **0.002124（187/88035）→ 0.012279 → 0.037894 → 0.060362（5314/88035）**。
- 渲染级证据（`probe_phonemes.py`，每单元 yang/yin 双模板、16-bit PCM、sha256）：滑音方向（ian/ing/iou 段尾 F2 ≤ 段首 −200 Hz、e 首尾差 ≤100 Hz）、模板比 1.10–1.40（`PRESETS` 的 1.08/0.87 = 1.24）、鼻尾低带能量 2.1–2.2×、zh 闭塞/爆发 0.00306（闭塞未提前泄露）、zh vs j 的 (2–6k)/(6–9k) 比 1.57 vs 0.41、h 低中带占比 0.594 vs x 的 0.0；音节层 F0 中位 199.7–200.3 Hz（误差 <0.3%）、长度精确、`track_pitch` voiced 0.857–0.929、reasons 全 `ok`。
- 回归：8 条 stage-0 音节在每批之后 sha256 全部不变（`regression.all_same=true`，`logs/render_baseline.json`）。
- **失败记录（全部保留）**：滑音度量两次作废（FFT 长度与窗口不匹配；带能量比随谐波落点剧烈跳动，参照元音也到 3.57）→ 定稿 f0=100 Hz 谐波峰值定位；`phonemes_stage1_v1.json` 把 l+iou 写成 `liou`（拼音正字法错误）；探针把单元 id 当拼音（`shapical_retroflex`）；`phonemes_stage2_v3.json` 的模板比被 200 Hz 谐波网格量化成 1.089–1.619（作废，v4 重测 1.209–1.319）；两处峰值断言误报（1600 vs 1800、/i/ 的 F3 实测 2400）→ 改为区间/带比断言并把 F3 移回声明层；扩表使 4 条既有断言失效（未知音素样例、w+uo / zh+a 缺口、`shi2` 可渲染）→ 全部重定向到仍阻塞的单元（`ch`、`ua`、`zi2`），断言文本与判据未改。
- 结论：三批均通过预注册条件（覆盖率单调上升、旧渲染不变、每单元四类渲染级证据）。整句 6.04% 仍是结构性限制——按音节乘积，一句里有一个未实现音节即整句阻塞。剩余阻塞按 stage-3 `next_units`：`final:uei`（+27534，需 ch/r）、`final:v`（+23661）、`final:in`（+22536）、`onset:ch`（+21214）、`final:eng`（+25944，需 r）、`final:apical_dental`（+20702）、`final:iao`（+20371）、`final:iang`（+18755）、`onset:r`（+17877）、`final:ie`（+17485）。`v`（ü）与 `apical_dental`（[ɿ]）需要新的声学设定，不能拿 u / i 顶替。
- 未做：听音评分（不能声称更像真人）、Windows CUDA 真机验证。
- 证据：`experiments/EXP-0016/`（`probe_coverage.py`、`probe_phonemes.py`、`logs/coverage_stage{0..3}.json`、`logs/phonemes_stage{1,2,3}*.json`、`artifacts/stage{1,2,3}/*.wav`）。

## 2026-11-01 | P1.5 管线集成：拒绝式 F0 进配对、能量 DP 对齐、manifest 可渲染性刷新

- F0：`build_pair(pitch_estimator="viterbi")` 成为默认，legacy YIN 保留为消融开关（`tests/test_data_pairing.py` 有对照测试）；曲线记 `frames/voiced_frames/coverage/interior_gaps/longest_gap_s/octave_ambiguous_frames/dip_db/confidence`，coverage < 0.35 打 `low_pitch_coverage`，长间隙只做**标注过的**线性插值并阻断 `training_ready`。
- 对齐：新增 `src/vhe/data/align.py` 的能量谷 DP。代价 = `smooth/local_peak − 0.5×clip(rise/local_peak)`；加 rise 项的实测理由：纯能量谷会把边界停在 /b/ 长闭塞中段，第 2 个音节窗覆盖 0.0 浊音帧 → `insufficient_pitch_points`，加 rise 后三窗 coverage 0.72/0.72/0.91。`confidence` 只反映能量谷深度，**不代表音素边界正确**（连续 a/i/u 上 DP 的 min_confidence 仍 0.74–0.82）。
- 可渲染性：`MANIFEST_VERSION=3` 加 `inventory` 指纹（`af1e7361…`，36 音素 / 19 onsets / 17 finals）与 `refresh_renderable`；`vhe refresh` 把 v3 刷成 v4（sha256 `1a8138c4d13e547d1d266a7aaf50d3a002a8e32dad78b0555f1b417c77074f31`），stale 5127 句全部 `newly_renderable`、0 句反向失效。
- `CorpusPairs.from_manifest(revalidate=True)` 默认按当前库存重判并记 `staleness`；实测可加载 **5314 句 / 218 说话人 / 3.37 h**（train 4333 / dev 528 / test 453），与覆盖率报告一致。
- 测试：`pytest -q` 360 passed / 1 skipped（skip = `tests/test_model.py:80` 需真 CUDA）。

## 2026-11-01 | EXP-0017 `no_candidate` 帧专项：4/281 疑似有声漏检

- 目的：把 `track_pitch` 的 281 个 `no_candidate` 帧分清是清音/气声/边缘，还是真正的有声漏检；阈值用 `ok` 帧的分位数校准，不用手调常数。
- 每帧独立证据：HPS（65–400 Hz、0.5 Hz 网格、5 次谐波）、实倒谱峰 prominence、80–4000 Hz 谱平坦度、相对电平、最小 CMND。
- 结果（12 句 / 2052 帧）：labels **aperiodic 174 / boundary_or_weak 103 / suspected_miss 4**；`suspected_miss` 占 `no_candidate` 的 1.4%、占非静音帧的 2.0%。位于两 `ok` 帧之间的只有 4 帧，全部是 `aperiodic`。
- 交叉表说明机理：电平越高越"非周期"（>−20 dB 组 median flatness 0.129、min_cmnd 0.778），−20…−45 dB 组才是边缘/弱有声（median flatness 0.008–0.018）。即高电平被拒的帧不是漏检，是汽声/摩擦成分把周期性抹掉了。
- SSB04350040：新跟踪器中位 250.0 Hz、legacy 280.3 Hz，逐帧中位偏差 5.4 cents、18.2% 帧 >50 cents；t=0.675 s 新 214.2 vs legacy 318.2（HPS 213.5、倒谱 210.5，峰 prominence 9.63），t=0.685 s 新 209.1 vs 417.7（HPS 206.5、倒谱 200.0）→ 独立证据支持新跟踪器；**无真值不宣称准确**。
- **失败记录**：v1 用固定阈值（hps_z≥2、cep≥3、flatness≤0.25）把 220/281 判成 `suspected_miss`，作废保留。
- 结论：27% 总拒绝率里只有 1.4% 是"真正有声却被拒"，其余是非周期帧（气声/清音/边界），因此**不降低 CMND 阈值**；要进一步提高有声召回只能靠真值或 forced alignment。
- 证据：`experiments/EXP-0017/`。

## 2026-11-01 | EXP-0018 真实配对 20 条：16 条出预览、0 条 training-ready

- 目的：用真实语音走完"分析 → 边界 → 机械合成 → 质量报告"全链路，检验能不能开始 P2。
- v1（未指定 backend）：落到 `FormantProxySynth`（`vowels` 库存）→ 20/20 `backend_inventory_insufficient`。产物移到 `derived/aishell3-pairs/v1_blocked_proxy_backend/`，报告保留为 `logs/pair_report_v1_blocked.json`。
- v2：暴露 **`parse_syllable` 缺陷**——它把底层韵母（`iou`/`apical_retroflex`/`uo`）当成正字法写法传给 `syllable_phonemes`，拼出 `yiou`/`shapical_retroflex`/`wuo` 并被拒。实测 v4 manifest 的 5314 句可渲染句中 **3160 句（59%）实际不能渲染**（4408 音节：`shapical_retroflex` 1597、`yiou` 1237、`wuo` 900、`zhapical_retroflex` 503、`yian` 171）。修法：用 `token.written or token.final` 查表、保留 `underlying_final` 作记录；修复后 5314 句 / 35153 音节 **0 阻塞**，且在 1920 个 distinct token 上 `missing_units` 与 `parse_syllable` **0 不一致**（新增交叉测试）。
- v3（有效）：**16 ok / 3 blocked / 1 rejected，0 training-ready**。reasons：`segmentation_count_disagrees` 19、`status_blocked` 3、`error_issues` 4、`no_pitch_frames` 1、`low_boundary_confidence` 1、`no_alignment` 1、`pitch_gap_interpolated` 1。38 处接缝 confidence 中位 1.000、最小 0.214；50 个音节窗 coverage 中位 0.892、<0.5 仅 1 个；机械增益 −26.02…−6.09 dB。
- 3 条 blocked 全是 `syllable_plan_blocked`：DP 给出的窗长 0.0795 s 略低于 `ONSET_MIN_S + NUCLEUS_MIN_S = 0.08`（`de5`/`tao2`/`ce4`）；1 条 rejected 是 `ku4 pai4` 的 voiced_ratio 0.2997 < 0.30。
- **0 training-ready 的根因**：普通 AISHELL 句是连续语流，能量分段器对 4–6 音节句只找到 **1 段** → `segmentation_count_disagrees`。DP 的 min_confidence 在连续元音上仍达 1.0，证明"置信度"不能自证边界正确；只有人工边界或真正的 forced aligner 能过闸。
- 结论：预览只能标 `preview_only`，**不得**当训练数据。要做 P2 必须先引入带声学模型的强制对齐（或人工校正），并对齐结果再做一轮人工抽检。
- 证据：`experiments/EXP-0018/`（`probe_pairs.py`、`logs/pair_report_v{1,v2,v3}.json`、`logs/listening_sheet_v3.md`、`derived/aishell3-pairs/v2/`）。

## 2026-11-01 | EXP-0019 模板身份 vs 真人包络：F0 不能选音色，预设与真人相距约 7 dB

- 目的：检查"按 F0 选模板"能不能算音色匹配，以及两个原创预设离真人有多远。**只做分析，不训练、不拟合模板、不造新配对**。
- 度量：level-normalised log-mel LTAS（24 带 / 100–7000 Hz / 1024 帧 / 256 hop / 只取响帧），dB RMS 距离；同一份真实音频 + 同一条 F0 曲线渲染 yang 与 yin，两者的唯一差别是模板（有 hash 断言）。
- 结果（EXP-0018 v3 的 16 条 ok 对）：F0 启发式与"更近预设"一致 **10/16 = 0.625**；median `d(real,yang)` 6.92 dB、`d(real,yin)` 8.42 dB、`d(yang,yin)` 6.03 dB；margin 1.48 dB、`gap/separation` **1.01**。
- **预注册的定量预测不成立**："预设之间的距离远小于预设到真人的距离"实测比值 1.01，不是 <0.7。选对预设只挽回 1.48 dB，而失配本身约 7 dB。
- **操作层结论成立**：5 个低 F0 说话人（125–200 Hz）反而更接近 `yang`（更大的共振峰尺度），与"低音=大声道=暗"的直觉相反 → F0 不携带音色信息，"按 F0 配对"不能称为音色匹配。
- 1-D formant_scale 拟合不可辨识：v1 网格 0.75–1.20 有 9/16 顶到边界，v2 扩到 0.70–1.30 仍有 7/16 顶到 1.30（中位 1.175）；scale 与 tilt 混淆，只能做方向性解读（多数说话人需要比两个预设更大的尺度）。
- 由此定下 P2 的五条可测试方案：模板按构造固定（seeded cycle，与录音无关）、用**合成原创音色**扩到 4 个身份（绝不按目标说话人拟合）、identity-preservation 指标 `d(VHE(x),x) < d(VHE(x),render(x,τ'))`、identity-leakage 守卫（`d(VHE(x),real)` 不得低于 `d(x,real)` 超界）、换模板时 F0/时长/边界不变。
- 未做：听音、训练、二维 (scale, tilt) 拟合。下一步 EXP-0020：加两个原创模板、四身份重跑本探针。
- 证据：`experiments/EXP-0019/`（`probe_identity.py`、`logs/identity_report_v1.json`、`logs/identity_report_v2.json`）。
