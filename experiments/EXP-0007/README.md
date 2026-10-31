# EXP-0007 — Signal front-end audit: F0, segmentation, envelope, pitch interpolation, pairing alignment

状态：已完成（before/after 两轮，均在同一台机器上实跑）。结论：7 项假设中 6 项确认并已修复，1 项（峰值归一化）确认为设计缺陷但修复方式是把补偿下沉到配对层，另发现 3 项未修的可辨识性/回归问题，见「失败、负面发现」。

## 假设

Seven suspected defect classes in the signal front-end produce measurable, silent errors: (1) FormantProxySynth envelope fails for voiced regions shorter than 2x30 ms; (2) autocorrelation F0 misreports 65-100 Hz as ~900 Hz; (3) F0 voicing gate is absolute, so any low-level periodic residue counts as voiced; (4) energy segmentation shifts onsets by -30 ms and inflates durations by +40 ms; (5) PitchCurve interpolates f0 linearly in Hz, up to +386 cents away from the perceptual midpoint; (6) per-syllable peak normalisation erases the loudness contour, which VHE cannot recover; (7) pairing overwrites the composite request fingerprint with the last syllable's fingerprint.

## 变量 / 对照组 / 数据划分

- 唯一变量：`src/vhe` 的代码版本。before = commit `692c0ca`（worktree 副本，`PYTHONPATH=/tmp/vhe_head/src`）；after = 本次修复后的工作树。
- 对照组：同一探针脚本、同一 numpy/进程、同一合成夹具；两轮之间只改 `src/vhe/data/{f0,segment,pairing}.py`、`src/vhe/synth/{contract,formant_proxy}.py` 与对应测试。
- 数据划分：本实验不使用任何真实录音，不含 AISHELL-3，全部是脚本内生成的合成夹具（纯音、谐波堆、静音、白噪声、矩形 burst、click、脉冲）。因此不存在 train/val/test 划分；这些数字只能证明前端的数值性质，不能推断真人语音上的表现。
- 夹具参数（写死在 `audit_signal.py`）：采样率 16/24/48 kHz；纯音 65/80/100/150/220/300/440/600/700/880 Hz；包络度量用 f0=200 Hz（24 kHz 下正好 120 样本/周期），矩形窗取整周期、hop=window//8，避免相位相关脉动污染；burst 真值 onset 0.37/1.0/0.5/0.5 s、时长 0.30/0.15/0.08/0.30 s、段间停顿 0/0.02/0.06 s。

## 完整复现命令 / 环境与哈希

```bash
# before（历史版本，只读）
git worktree add /tmp/vhe_head 692c0ca
PYTHONPATH=/tmp/vhe_head/src .venv/bin/python experiments/EXP-0007/audit_signal.py \
    --out experiments/EXP-0007/logs/audit_before.json

# after（当前树）
.venv/bin/python experiments/EXP-0007/audit_signal.py \
    --out experiments/EXP-0007/logs/audit_after.json
```

- 环境：macOS 27.2 arm64（Apple Silicon），Python 3.14.6，PyTorch 2.15.0（本实验不使用 torch），NumPy 2.5.4，CPU 执行，无 CUDA/MPS。
- 代码版本：before = `692c0ca2cb3447a00512f89b7a5c42b20f185608`（`git status` 干净）；after = `692c0ca` + 本实验的修复补丁（见下方「修复清单」）。
- 哈希（sha256）：
  - `experiments/EXP-0007/audit_signal.py` = `82162a31…2e4c5`
  - `experiments/EXP-0007/logs/audit_before.json` = `ea10fd24…e2852`
  - `experiments/EXP-0007/logs/audit_after.json` = `b5622d14…f404f`
- 随机性：探针无随机源（无 seed 需求）；相位、频率、幅度全部解析给定。

## 预注册通过/失败条件

对每一类缺陷，判据在跑之前写定：

1. 合成器短段：10–200 ms 的有声段渲染不抛异常，且首样本不在静音（`first == 0.0` 为通过）。失败判据：任何 `ValueError` 或首样本被 fade-out 吃掉。
2. F0：16/24/48 kHz × 10 个频率，相对误差 ≤0.5%；静音输入 voiced_ratio = 0。
3. 分段：4 个 burst 的 onset 与 duration 误差 ≤5 ms。
4. 音高插值：0.5 s 内 110→440 Hz，中点（t=0.25 s）相对几何均值误差 ≤10 cents。
5. 配对：三音节 RMS 比在 1±0.05 内；fingerprint 为三音节 composite；mechanical 峰值 ≤0.999。

## 实际结果与主观听音

原始数据见 `logs/audit_before.json` / `logs/audit_after.json`，下表只摘关键项。

| 探针 | before | after |
| --- | --- | --- |
| 短段 0.010/0.020/0.029 s | `ValueError: could not broadcast input array from shape (720,) into shape (240,)`（0.020 → 480，0.029 → 696） | 全部 ok，`first=0.0`，env_jump 0.2335/0.1828/0.1260 |
| 短段 0.030 s | ok 但 `first=0.3437`（fade-out 覆盖 fade-in） | `first=0.0`，env_jump 0.1203 |
| 短段 0.045 s | env_jump 0.2789 | 0.0854 |
| 短段 0.061/0.200 s | 0.0672/0.0558 | 0.0672/0.0558（未变，属正常长度） |
| 峰值归一化 | yin/yang 峰值均 0.85，RMS 0.4879/0.3183 | 不变（0.4879/0.3183）——修复在配对层，不在合成器 |
| F0 65/80 Hz @16k | 969.70 / 969.70 | 65.0 / 80.0（误差 −0.0%） |
| F0 65/80 Hz @24k | 941.18 / 941.18 | 65.0 / 80.0 |
| F0 65/80 Hz @48k | 914.29 / 914.29 | 65.0 / 80.0 |
| F0 880 Hz @16/24/48k | 880.08 / 880.01 / 879.95 | 880.04 / 880.01 / 880.00 |
| 静音 1e-6 / 1e-3 | voiced 0.515 / 0.515 | 0.0 / 0.0 |
| 白噪声 | voiced 0.0 | 0.0 |
| 分段 onset/duration | 全部 −30 ms / +40 ms | 全部 0.0 ms / 0.0 ms |
| 段间断停 0 / 20 / 60 ms | 1 / 1 / 1 段 | 1 / 1 / 2 段（60 ms > merge_gap 50 ms，修复后终于分得开） |
| 10/20 ms click | 10 ms 丢弃；20 ms 被当成 ~60 ms 保留 | 两者均判 `segment_too_short` 丢弃 |
| 音高中点 | 250.0 Hz（+386.3 cents） | 200.0 Hz（0.0 cents） |
| 配对 RMS 比 | 0.995 / 1.979 / 1.278 | 1.0 / 1.0 / 1.0（gains +0.064 / −5.879 / −1.866 dB） |
| 配对 fingerprint | 等于最后一个音节 | 等于三音节 composite |
| mechanical 在 real 有声处为零 | 2.18% | 1.27% |
| mechanical 在 real 静音处有能量 | 9.77% | 2.65% |
| 配对峰值 | 未记录 | 0.8563（<0.999） |

主观听音：尚未进行。本实验只做数值探针，没有导出音频片段，也没有做 AB 听测；任何"更自然"的说法在此不成立。

## 失败、负面发现、错误日志

1. **缺失基频未被修复**：输入 200 Hz 基频缺失（200+4×400 Hz 谐波堆）时，before 报 400.27 Hz，after 报 400.91 Hz。真周期是 200 Hz，但 CMND 的首个谷落在 400 Hz 周期上——这是 YIN 类算法在多谐波缺失基频下的可辨识性限制，探针无法区分 400 Hz 纯音与 200 Hz 缺基频信号。判据要求 ≤0.5% 相对误差，实测偏差 100%（相对真值），记为**未通过**，不修，写入文档并在 P1 数据检查中作为已知误判类型统计。
2. **阶跃边界帧变多（回归）**：200→300 Hz 阶跃，before 边界两帧为 [99.9, 289.3]，after 为 [100.0, 100.0]。原先一帧被拉偏、一帧正确；现在两帧都读成 100 Hz（其实是 200 与 300 的最小公倍数 600 的邻域被判为周期）。即伪影从 1 帧扩散到 2 帧。记录在案，不掩盖：这是 CMND 谷选择更"保守"带来的副作用，需要后续用能量/频谱变化检测来定位过渡帧。
3. **短段 fade 变陡**：fade 改为 `min(30 ms, n//2)` 后，10–30 ms 段的 envelope jump 变大（0.1260–0.2335），因为 5 ms 内就要爬完整条斜坡。这是"不静音短音节"的必然代价；听感待验证，数值上记为负面发现。
4. **20 ms click 现在被丢弃**：before 因为帧级膨胀把 20 ms 判成 ~60 ms 而通过，after 如实测得 20 ms 并触发 `segment_too_short`。这不是退步，是过滤器终于起作用；但它意味着 60 ms 以下的真实短促音（如塞音爆破）无法进入配对流程，属于数据侧限制。
5. **流程偏差（勘误）**：探针脚本 `audit_signal.py` 是在 `tools/new_experiment.py` 建目录**之前**写的，`metadata.json` 的 `command`/`notes` 当时留空。违反"实验运行前先建档"的规约，此处如实记录，不改写 metadata 的原始字段，只在 `metadata.json` 追加 `runs` 与 `errata` 说明。
6. **试错记录（未进最终实现）**：(a) 第一版 CMND 用固定积分窗 `W = size - lag_max`，但相关系数用全长 `(n-τ)` 项，能量与相关不匹配 → diff 全被截断为 0，argmin 落在随机 lag；(b) 第二版 cumsum 从 `lag_min` 起算，导致 `cmnd(lag_min) = lag_min` 整体偏大，880 Hz 被读成 440 Hz；(c) 包络度量最初用 Hann 窗 + 24 样本窗，波形尖峰造成 0.2 以上的假跳变，改用整周期矩形窗后才可复现。以上都已作废，保留在此说明为什么最终实现长成这样。

## 修复清单

- `src/vhe/data/f0.py`：YIN 式 CMND（`_difference` / `_estimate_frame`），加法能量项 + FFT 自相关 + 首个谷搜索 + 抛物线亚样本细化；新增 `silence_floor_db=-60.0`、`absolute_floor=1e-3` 的相对静音门。
- `src/vhe/data/segment.py`：新增 `_refine_edges()` 样本级边缘精修（门限 `max(0.05*峰值, 1e-4)`，搜索半径 `frame_ms+hop_ms`），并把"精修 → 合并"的顺序改成"逐个 run 精修 → 按精修后间隔合并"；`merged` 元素改为 `[start_s, end_s, start_frame, stop_frame]`。
- `src/vhe/synth/formant_proxy.py`：`fade = min(max(1, int(0.03*sr)), n_voiced//2)`，`if fade > 0` 保护，version → `0.2.0`。
- `src/vhe/synth/contract.py`：`PitchCurve.sample` 改为在 `log2(f0)` 上插值。
- `src/vhe/data/pairing.py`：逐音节 RMS 增益匹配（`gains_db` 记录，clip 0.05–20，再按 `0.999/peak` 封顶并记 `gain_limited_by_peak`）、`request_fingerprints` 记录全部渲染指纹、删除覆盖 composite 指纹的行、越界渲染记 `segment_truncated`、删除死代码 `_timing`、`seed=seed+index`。
- 测试：`tests/test_synth_contract.py` 两处断言随语义更新（中点 275→220 Hz、rising pitch 期望改几何级数）并新增常数曲线逐位断言；`tests/test_data_pairing.py` 新增增益记录与封顶用例。全量 `pytest -q`：86 passed, 1 skipped（仅 CUDA 真机用例）。

## 结论 / 下次实验

- 7 项假设：6 项确认并修复；第 6 项（峰值归一化）确认为合成器设计而非 bug，修复下沉到配对层（逐音节增益），合成器仍保持 0.85 峰值。
- 未修但已知：缺失基频误判（100% 误差）、阶跃边界伪影扩到 2 帧、短段 fade 更陡、<60 ms 短促音进不了配对。
- 这些数字全部来自合成夹具，**不能**推断真人语音上的 F0/分段精度；下一步（EXP-0008）需要用带真值的合成"准真人"信号做端到端 Loss 数值验证，P1 数据检查阶段需在 AISHELL-3 子集上统计同一组指标（需主人许可后才下载）。
