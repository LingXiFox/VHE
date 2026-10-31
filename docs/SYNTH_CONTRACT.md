# 合成器契约（上游 x_mechanical 生产方）

状态：P0-B 已实现并验收（EXP-0003）。适用代码：`src/vhe/synth/`、`src/vhe/audio.py`。

## 1. 这份契约解决什么问题

VHE 的唯一输入是机械人声音频张量。为了让「机械声」可复现、可审计、可替换，
同时**不把条件信息偷渡进 VHE**，本项目把「谁可以看什么」写成显式类型：

```
VoiceTemplate + PhonemeSequence + PitchCurve + Timing   (上游控制面，允许来自真实录音的分析)
                    │
                    ▼  SynthBackend.render(request)
              SynthesisResult (audio float32 [N], sample_rate, backend_name/version, request_fingerprint)
                    │
                    ▼  result.as_batch()
              x_mechanical: float32 [1,1,N]  ──► VHE（只此一项）
```

- 上游合成器**可以**使用：音色模板、音素、F0 曲线、时长、静音。
- VHE **只能**收到渲染后的波形。没有任何 API 允许把 text / pinyin / phone / MIDI /
  speaker id / 参考音频塞进 `SynthesisRequest`；`tests/test_synth_contract.py::test_request_has_no_hidden_conditioning_inputs`
  用字段白名单把这条边界钉死。新增字段必须先改该测试，即在代码评审里显式暴露。
- 训练阶段允许用真实录音分析出对齐控制信息（F0、段长）来**构造**机械声，但这些信息止步于 request。

## 2. 单位与形状（统一，不做隐式换算）

| 物理量 | 类型 / 单位 | 说明 |
|---|---|---|
| `duration_s`, `lead_silence_s`, `tail_silence_s`, `times_s` | float，秒 | 从 0 开始，不出现样本数/毫秒混用 |
| `f0_hz` | float，Hz | **不用** MIDI 音高号，不用半音；范围 65–900 Hz |
| `sample_rate` | int，Hz | 仅允许 `(16000, 24000, 48000)` |
| `audio` | float32，形状 `[N]`，值域 [-1, 1] | 单声道；16-bit PCM 落盘时按 float32 量化 |
| VHE 输入 | float32，形状 `[B,1,N]` | `as_batch()` 只做 reshape，不做归一化、不附加元数据 |

采样率不在白名单、f0 越界、段长为 0、曲线没覆盖浊音区，一律**抛错**，不做静默裁剪或重采样。

## 3. `SynthesisRequest`（`src/vhe/synth/contract.py`）

| 字段 | 类型 | 默认 | 校验 |
|---|---|---|---|
| `template` | `VoiceTemplate` | 必填 | 见下 |
| `phonemes` | `PhonemeSequence` | 必填 | 非空；每段 `duration_s` ∈ [0.01, 60]；总时长 ≤ 60 s；`inventory` ∈ `INVENTORIES`（默认 `"vowels"`），每段音素必须在该表内 |
| `pitch` | `PitchCurve` | 必填 | `times_s` 严格递增、≥0；`f0_hz` ∈ [65,900]；`times_s[-1] ≥ 浊音总时长` |
| `timing` | `Timing` | 0/0 | 前后静音各 ∈ [0, 5] s |
| `sample_rate` | int | 24000 | ∈ (16000, 24000, 48000) |
| `seed` | int | 42 | 控制 breathiness 白噪，从而控制可复现性 |

派生属性：`voiced_duration_s`（各段时长之和）、`total_duration_s`（含静音）、
`fingerprint()`（上述全部参数做 canonical JSON 后取 sha256；**不含**波形，也不含时间戳）。

音素表 `INVENTORIES`（`src/vhe/synth/contract.py`）：`"vowels"` = `a/i/u`（`formant_proxy`），
`"mandarin"` = 26 个普通话音素（`mandarin_formant`，见 `src/vhe/synth/phonology.py`）。
`inventory` 进 `fingerprint()`，所以同一串音素在两个音素表下是不同请求。后端的
`inventory` 与请求不一致时报错，不会把请求当另一种表解释。

`VoiceTemplate`（音色控制，非 VHE 输入）：`formant_scale` [0.65,1.40]、`spectral_tilt` [0.6,3.0]、
`breathiness` [0,0.1]、`vibrato_cents` [0,100]、`vibrato_hz` [0,10]。
预设：`yang`=(1.08, 1.20, 0.012)、`yin`=(0.87, 1.50, 0.006)。

## 4. `PitchCurve` 语义

- 分段线性插值；`t` 超出控制点范围时 **clamp，不外推**。
- 这是**基频轮廓**；vibrato 由后端的音色参数叠加，所以指纹里两者是分开记录的两件事。
- 真实录音驱动时，F0 曲线的控制点密度必须与时长一起写进 manifest，否则机械声与被拟合的录音不可复算。
- 曲线只需要覆盖浊音区（`times_s[-1] ≥ voiced_duration_s`）；尾部的静音不需要音高。

## 5. `Timing` 语义

- 静音就是**零样本**，不是低电平噪声，也不是音素。
- 只在前/后添加，不改变浊音段任何样本（测试用 `np.array_equal` 逐样本校验）。
- 静音上限 5 s：防止把「模型没声音」这类失败伪装成长静音样本。

## 6. `SynthesisResult`

| 字段 | 含义 |
|---|---|
| `audio` | float32 `[N]`；构造时校验 dtype、单声道、有限值 |
| `sample_rate` | 与 request 一致 |
| `backend_name` / `backend_version` | 例：`formant_proxy` / `0.1.0`，进 manifest，用于追溯"这批机械声是谁生成的" |
| `request_fingerprint` | 请求参数的 sha256，用于去重和配对索引 |

## 7. `SynthBackend` 协议（`typing.Protocol`，运行时 `isinstance` 可检查）

实现方必须满足：

1. `name` / `version` / `supported_phonemes` / `inventory` 四个属性；
2. `render(request) -> SynthesisResult`，且入口先 `request.validate()`；
3. **确定性**：同一 request 在同一平台/同一版本上逐样本一致（本仓库以 SHA256 断言）；
4. **不静默替换**：不支持的音素必须抛错，不能换成近似音；
5. **不读外部状态**：不读参考录音、网络、随机全局状态；随机性只来自 `request.seed`；
6. 版本号语义：*任何改变输出样本的行为变化*都要提升 `version`，并开新实验记录。

注册表：`get_backend(name=None)` 取默认（`formant_proxy`），未知名字抛 `KeyError`；`list_backends()` 列出可用后端。

## 8. 现有后端

两个后端都只做源-滤波器合成，都不是声道物理引擎。``FormantProxySynth`` 是元音基线，
``MandarinFormantSynth`` 是 26 音素的中文发声后端；两者共享契约但音素表互斥。

### 8.1 `FormantProxySynth` v0.1.0 的算法与局限

算法：按段中点线性插值三条共振峰轨迹（乘 `formant_scale`）→ 对 f0 轮廓叠加正弦 vibrato（cents 转
`2**(cents/1200)`）→ 相位用 `cumsum(f0)` 累积 → 谐波叠加，每个谐波幅度为三个高斯共振峰响应
（带宽 110/150/210 Hz，权重 1.0/0.72/0.38）乘 `harmonic**(-spectral_tilt)` → 叠加 breathiness 白噪 →
30 ms 线性淡入淡出 → 0.85 峰值归一。

局限（必须写进任何使用它的报告）：

- 这是**源-滤波器代理**，不是声道物理模型、不是声码器、不代表任何真实发音器官。
- 只有元音 `a/i/u`，**没有辅音、声调、韵律、停顿建模**；共振峰是演示级占位值，不是说话人无关的音系学数据。
- 无辅音意味着它无法代表真实中文语音的机械声分布；P2 之前不得据此宣称任何自然度结论。
- 谐波上限用 `max(base_f0)` 计算，保证上升音高时不产生混叠。

### 8.2 `MandarinFormantSynth` v0.1.0 的算法与局限

算法：音素表给出每段的共振峰目标（F1/F2/F3，复韵母按段内分数在两个目标间插值）、浊音/噪声
激励强度、噪声带宽、爆破时刻、闭塞段与 VOT；轨迹在 25 ms 内平滑、幅度在 8 ms 内平滑 →
谐波求和时每帧按谐波 RMS 归一化（电平只由浊音轨道控制）→ 噪声走 STFT 带通（边沿软过渡，
fall 终点 clamp 到 `0.45*sample_rate`）并按浊音 RMS 校准 → 爆破/送气在指定时刻叠加 →
整体归一到 `TARGET_VOICED_RMS=0.20`、峰值上限 0.98、5 ms 淡入淡出。声调不靠后端自造，
由 `tone_curve_points` 把 Chao 级数换算成半音偏移后并入传入的 F0 曲线。

支持：a/i/u/ai/ei/au/ou/an/ang、m/n/l、b/p/d/t/g/k、f/s/sh/x、z/c/j/q。

局限（必须写进任何使用它的报告）：

- 擦音谱带（`/f/` 1500–8000 Hz 等）与爆破增益是**手设占位值**，未用真实语音标定；EXP-0010 实测
  `/f/` 22% 能量落在 8–12 kHz、质心高于 `/sh/`，属已知缺陷。
- 塞音只有闭塞→爆破→VOT 三段的固定时长，没有真实的协同发音；复韵母是线性插值，不是双目标
  轨迹；鼻音用固定反极点近似，没有鼻腔分岔。
- 不做韵律：不接受音节边界、重音、语速，也不生成停顿。
- `zh/ch/r/h` 与 `ian/ie/uo/üan` 等不在表内，请求即报错；`g2p.py` 会把它们标成 `blocked`
  而不是用近似音素顶替。
- 26 音素"能渲染"不等于"发音正确"：EXP-0009 只验收了长度、F0、有声/无声对比、模板差异与
  CPU/MPS 一致性，没有听音评分。

## 9. 不变量（回归门槛）

- 基线字节等价：`synthesize(["a","i","u"], f0_hz=220, duration_s=1.2, sample_rate=24000, template=PRESETS["yang"], seed=42)`
  落盘（16-bit PCM）后 SHA256 = `cefbcaec99a276683989f05abc537c6dfb8edfeef164be7728ee22775ada7985`，
  由 `tests/test_synth_contract.py::test_legacy_wrapper_reproduces_v001_baseline_bytes` 固定。
  **改这个常量等于改行为**，必须开新实验并解释为什么旧基线作废。
- `write_wav` 必须在 **float32** 下量化：早期版本先转 float64，导致 13/28800 个样本相差 1 LSB（见 EXP-0003 失败记录）。
- 16-bit 往返固有误差约 2 LSB（写 ×32767 截断、读 ÷32768），测试容差按此设定。
- 音素表隔离：`formant_proxy` 收到 `inventory="mandarin"` 必须抛 `ValueError` 且信息里带
  `renders the 'vowels' inventory, got 'mandarin'`；`mandarin_formant` 反向同理。任何"把 `m`
  当元音渲染"或 `KeyError` 泄漏都算回归（`tests/test_synth_mandarin.py`）。

## 10. 与 P1 的衔接

P1 的成对数据构造流程固定为：

1. 从授权录音读出 `wav`；
2. 分析得到 F0 曲线、能量分段（音节级段长）、质量标记；
3. 用**同一份控制信息**构造 `SynthesisRequest`，渲染 `x_mechanical`。注意 `PhonemeSequence` 没有静音槽位，多个音节塞进一个请求会把音节间停顿压掉；P1 的做法是**逐音节渲染并按实测起点回填**到等长缓冲区（代价是每段 30 ms 淡入淡出，`pairing` 记为 `syllablewise_render` warning，见 EXP-0006）；
4. 机械/真人配对共享 `request_fingerprint` 与 speaker 分组，写入 manifest；
5. 分析失败（F0 无浊音、段长为零、信噪比过低）必须进 `failures` 列表并在统计里可见，**不得静默丢弃**。

## 11. 已知缺口

- 辅音/声调已由 `mandarin_formant` 覆盖 26 音素，但擦音/爆破参数未用真实语音标定（见 8.2），
  且 `zh/ch/r/h` 与 `ian/ie/uo/üan` 等缺失；补表必须新增音素清单 + 新后端版本 + 新实验。
- `estimate_f0` 在默认 65–900 Hz 搜索范围下会被 F1≈800 Hz 的强谐波锁定（210 Hz 读 842 Hz，
  300 Hz 读 902 Hz，见 EXP-0011 附录 E5），真实数据管线不能预知 f0 规避。
- `SynthesisResult` 不携带响度（LUFS）或峰值信息；配对时会由数据处理层统一测算。
- 未实现批量化渲染（`[B]` 个请求）；当前靠多次调用 + concat，P2 再看是否需要。
