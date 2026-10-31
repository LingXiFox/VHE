# EXP-0003 — P0-B 合成器契约重构与字节级等价回归

## 预注册假设

把单体 `src/vhe/synth.py`（126 行、固定等长音素、单一音高）重构为显式契约
（`SynthesisRequest` / `SynthesisResult` / `SynthBackend`）后：

1. 旧 CLI 路径（`a,i,u` / `yang` / 220 Hz / 1.2 s / 24 kHz / seed 42）仍然生成与历史基线
   **逐字节相同**的 WAV：sha256 `cefbcaec99a276683989f05abc537c6dfb8edfeef164be7728ee22775ada7985`。
2. 新增能力（每段独立时长、分段线性音高曲线、前后静音、后端注册表、请求指纹）在**不被显式使用**时
   不改变默认输出。
3. 契约层能够拒绝夹带隐藏条件输入（text、拼音、speaker id、参考音频），并可被单元测试固定。

## 唯一变量与对照

- 唯一变量：合成器内部结构（单体函数 → 契约 + 后端实现）。数值算法（谐波叠加、共振峰插值、
  vibrato、30 ms 淡入淡出、breathiness 噪声、0.85 峰值归一）刻意保持不变。
- 对照：EXP-0000（Linux/CPU）与 EXP-0002（macOS/MPS）记录的基线哈希。
- 数据：无数据集，仅合成元音。不涉及真人录音、不涉及 VHE 训练。

## 环境（实测）

| 项 | 值 |
|---|---|
| OS | macOS 27.2 (Darwin 27.2.0, arm64), Apple M5, 32 GB |
| Python | 3.14.6（仓库内 .venv，uv 管理） |
| PyTorch | 2.15.0（本轮只依赖 numpy 的 synth 路径） |
| 起点 commit | 890f009（工作区 dirty：重构未提交） |
| 精度 | float32 |

## 完整复现命令

```bash
.venv/bin/python -m pytest -q -v
.venv/bin/python -m vhe synth --phonemes a,i,u --preset yang --f0 220 \
  --duration 1.2 --sr 24000 --seed 42 --output /tmp/vhe_check/exp0003_run_a.wav
.venv/bin/python -m vhe synth --phonemes a,i,u --preset yang --f0 220 \
  --duration 1.2 --sr 24000 --seed 42 --output /tmp/vhe_check/exp0003_run_b.wav
shasum -a 256 /tmp/vhe_check/exp0003_run_a.wav /tmp/vhe_check/exp0003_run_b.wav
```

原始输出见 `logs/pytest.txt`、`logs/baseline_equivalence.txt`。

## 预注册通过条件

- `pytest -q` 全绿（仅允许「无 NVIDIA CUDA 硬件」条件跳过）。
- 两次 CLI 输出 SHA256 均等于历史基线常数（不是"看起来相同"，是字节相同）。
- 契约测试能检出：夹带字段、未支持音素、未覆盖的音高曲线、非法采样率、静音不改变浊音段样本。
  **失败判据**：任何一项不符，则本次重构被判定为改变行为，必须回滚或另开实验记录差异，不得静默更新基线常数。

## 实际结果

- 测试：**35 passed, 1 skipped**（跳过项 `tests/test_model.py:35` 需要真实 NVIDIA CUDA）。
  重构前为 18 passed / 1 skipped；新增 `tests/test_synth_contract.py` 17 项，另有 `vhe.audio` 相关断言并入其中。
- 字节等价：`exp0003_run_a.wav` 与 `exp0003_run_b.wav` 的 SHA256 均为
  `cefbcaec99a276683989f05abc537c6dfb8edfeef164be7728ee22775ada7985`，
  与 EXP-0000 / EXP-0002 完全一致（28800 samples, 24 kHz, 1.2 s）。
- 默认请求指纹（a 0.4 s / i 0.4 s / u 0.4 s, yang, 220 Hz 常数, 24 kHz, seed 42）：
  `6ce3178192aa11dd54507ab303068f1c0cf24463e94294273e72644cb2075188`（canonical JSON 的 sha256）。
- 新增公开 API：`SynthesisRequest`（字段仅 template/phonemes/pitch/timing/sample_rate/seed）、
  `SynthesisResult`（audio/sample_rate/backend_name/backend_version/request_fingerprint，`as_batch()` → `[1,1,N]`）、
  `VoiceTemplate`、`PhonemeSegment`/`PhonemeSequence`、`PitchCurve`（线性插值、区间外 clamp 不外推）、
  `Timing`（前/后静音 ≤5 s）、`get_backend()`/`list_backends()`、`write_wav`/`read_wav`/`peak_normalize`。
- 契约性质由测试固定：静音只加零样本、浊音段逐样本不变（8000 → 8000+1600+800，中间段 `array_equal`）；
  同请求两次渲染逐样本相同；改 seed 则波形改变但指纹不变；改 template/pitch/phonemes 则指纹改变。

## 失败、负面发现、错误日志

1. **重构期间真实引入过回归，已定位并修复。** 第一次重写 `write_wav` 时把音频先转成 `float64` 再量化，
   导致 **13/28800 个样本相差 1 LSB**（0.045%），基线哈希随之改变。根因：v0.0.1 在 float32 下乘 32767
   后 `astype("<i2")` 向零截断；转 float64 后舍入点不同。修复：量化前强制 `float32`，并用
   `tests/test_synth_contract.py::test_legacy_wrapper_reproduces_v001_baseline_bytes` 固定该常量。
   该中间态的完整 WAV 哈希**未归档**（那次运行未落盘保留），但样本级差异已实测确认（13 处 ±1 LSB），
   此处按事实描述，不作修饰。
2. **测试本身两次写错，均已记录而非删除。**
   - `test_rising_pitch_curve_raises_pitch_track` 最初用过零率当音高代理：`crossings[0]=569 > crossings[-1]=401`。
     失败原因：谐波丰富的信号里过零率反映的是高频谐波而非基频。改用自相关后仍失败，
     估计值 `[421.05, 410.26, 444.44, 400.0]` 非单调——原因是**未归一化**的自相关随滞后增大而能量衰减，
     偏向长滞后（低八度/次谐波）。最终改为按重叠段能量归一化的自相关 + 512 样本窗，
     得到 `[150.9, 231.9, 320.0, 400.0]`，与解析值 `[151.25, 233.75, 316.25, 398.75]` 在 8% 内一致。
   - `test_wav_roundtrip_within_one_lsb` 断言 1 LSB 容差，实测最大误差 `5.4955482e-05` > `3.05e-05`。
     失败原因：写用 32767 且向零截断、读用 32768，16-bit 存储本身有约 2 LSB 固有误差。
     已把容差改为 `2/32768` 并注明理由；**没有**为了迁就测试改回 float64。
3. **流程偏差（如实记录）**：AGENTS.md 要求"实验运行前"建立实验记录，本记录在重构实现完成后、
   验证运行之前创建（`created_utc` 与 `git.status` 的 dirty 状态可见）。重构属设计实现而非运行，
   但顺序仍与本规范有出入，记录于此不隐去。
4. 仍未验证 Windows RTX 4090 真机 CUDA；本实验与 GPU 无关，不构成 CUDA 验收。

## 结论 / 下次实验

假设 1–3 全部成立：契约化重构**没有**改变既有机械声输出（字节级证据），并新增了可审计的显式控制面
（分段时长、音高曲线、静音、后端版本、请求指纹），为 P1「从真实录音构造对齐控制 → 渲染机械声」提供接口。
**VHE 仍未训练**，本实验不提供任何自然度或音色保持证据。
下一实验（EXP-0004）：P1-A AISHELL-3 适配器——在无真实数据的前提下用合成夹具验证解析、质量校验、
按 speaker 分组划分与 manifest 哈希；真实数据侧保持阻塞并如实报告。
