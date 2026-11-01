# EXP-0016 — 按实测边际收益分批次扩展普通话音素表

## 假设

现在的 `mandarin` 音素表只有 26 个单元，对 AISHELL-3 的音节覆盖率 30.95%、整句覆盖率 0.21%
（EXP-0012 / `derived/aishell3-scan/v3`）。若按「精确边际覆盖」排出的顺序，每批加入少量**整单元**
（一个韵母一个 `PhonemeSpec`，不拆成 i+a+n），则：

1. 音节覆盖率单调上升，且新增可渲染的音节必然是「缺失单元 ⊆ 本批新增单元」的音节，不会让本来
   非法的音节变得可渲染；
2. 已支持音素的渲染逐字节不变（扩表不能悄悄改动旧的声学参数）；
3. 新单元确实携带它宣称的对立（滑音方向、模板依赖、鼻化只出现在韵尾、声母边界保留真实闭塞）。

不假设整句覆盖率会同步接近音节覆盖率：一句里每个音节都要可渲染，覆盖率是乘积关系。

## 变量 / 对照组 / 数据划分

- 唯一变量：`MANDARIN_PHONEMES` + `g2p.SYNTH_FINAL_TO_PHONEME` 的成员（按批增长）。
- 对照组：stage 0（26 音素表），用同一份 manifest、同一套 g2p 重算，不重新扫描音频。
- 数据：`/Volumes/LingXiWorkspace/VHE-Datasets/derived/aishell3-scan/v3/manifest.jsonl`
  （sha256 `10539f61d015bc81dff88cb0ee5ba72b640ab703ee57373324f7c446a624c80f`，88035 句 /
  997997 音节 / 218 说话人 / 180-19-19 划分 seed 20261101）。
- 不训练 VHE，不生成配对机械音频，不重解码任何 WAV：覆盖率只用 manifest 条目重算
  （`corpus.stats_from_entries`），音频只来自合成器的探针。

## 完整复现命令 / 环境与哈希

```bash
# 每批：覆盖率复算（拒绝覆盖已有报告）
python experiments/EXP-0016/probe_coverage.py \
    --manifest /Volumes/LingXiWorkspace/VHE-Datasets/derived/aishell3-scan/v3/manifest.jsonl \
    --stage 1 --out experiments/EXP-0016/logs/coverage_stage1.json

# 每批：合成探针（WAV + 数值分析 + 旧渲染回归）
python experiments/EXP-0016/probe_phonemes.py --stage 1 --units e ian ing iou \
    --json experiments/EXP-0016/logs/phonemes_stage1_v2.json \
    --wav-dir experiments/EXP-0016/artifacts/stage1
```

环境：macOS 27.2 arm64（M5 MacBook Air 32 GB），CPython 3.14.6，numpy only（合成器无 scipy），
无 GPU。渲染用 `SynthesisRequest.seed` 默认值；F0 探针 200 Hz、时长 0.30 s、24 kHz。

## 预注册通过/失败条件

- 覆盖率：音节覆盖率上升，且 stage N 的可渲染集合是 stage N-1 的超集（由 `missing_units` 的定义
  保证，逐条对比报告里的 `renderable` 计数与 `blocked_units` 递减）。
- 回归：`probe_phonemes.py` 的 8 条 stage-0 渲染 sha256 全部不变（`all_same: true`）。
- 声学：`tests/test_synth_mandarin.py` 的 stage 段全部通过，且断言测的是渲染音频而不是查表
  （滑音方向用 f0=100 Hz 谐波峰值定位，鼻化用 200–320 Hz 低带占比的鼻/口对照）。
- 任一条件不满足即判该批失败，记录原因，不删除该批数据。

## 批次日志

### stage 1 — `e` `ian` `ing` `iou`（2026-11-01）

选择依据：stage-0 `next_units` 里这四个单元**单独加入**即可解锁音节（`also_needs` 为空或只差本批
其他单元），边际收益 74797 / 47442 / 37436 / 34896 音节。

| 单元 | 类别 | 轨迹 (F1,F2,F3) Hz | 单独解锁音节 |
| --- | --- | --- | --- |
| `e` | vowel | 500 / 1280 / 2500（稳态） | +74797 |
| `ian` | nasal（`nasal_from=0.60`） | (300,2250,3050) → (560,1900,2600) → (470,1750,2550) | +47442 |
| `ing` | nasal（`nasal_from=0.50`） | (300,2250,3050) → (400,2050,2900) → (330,1750,2700) | +37436 |
| `iou` | diphthong | (300,2200,3000) → (520,1000,2500) → (380,860,2350) | +34896 |

覆盖率（同一 manifest，不重解码）：

| | stage 0（26） | stage 1（30） |
| --- | --- | --- |
| 音节覆盖率 | 0.309544（308924/997997） | **0.503755（502746/997997）** |
| 整句覆盖率 | 0.002124（187/88035） | **0.012279（1081/88035）** |
| 剩余阻塞 top | e, apical_retroflex, zh, ian, h | apical_retroflex, zh, h, uo, en, ong, ch, uei |

整句覆盖率涨了 5.8 倍（187 → 1081 句），但绝对水平仍然很低——这是「每个音节都要可渲染」的乘积
效应，不是本批的缺陷；stage 2 继续按边际排序推进。

合成探针（`logs/phonemes_stage1_v2.json`，yang 模板渲染、f0=100 Hz 测共振峰）：

| 单元（载体） | F2 段首→段尾 | F1 段首→段尾 | 模板比 yang/yin | F0 中位 | voiced |
| --- | --- | --- | --- | --- | --- |
| e（零声母） | 1400 → 1400 | 500 → 500 | 1.27 | 200.37 | 1.00 |
| ian（d+ian） | 2200 → 1900 | 400 → 500 | 1.22 | 200.36 | 0.893 |
| ing（b+ing） | 2300 → 2000 | 300 → 300 | 1.21 | 200.34 | 0.893 |
| iou（l+iou） | 1600 → 1000 | 400 → 400 | 1.23 | 200.37 | 1.00 |

- 模板比 1.21–1.27 与 `PRESETS` 的 formant_scale 比 1.08/0.87 = 1.24 一致：新单元跟着音色模板走，
  没有被写死成某个模板的声线。
- 鼻化只在韵尾：ian 段尾低带占比 0.0063（去掉鼻极为 0.0015，4.2×），ing 0.1061 vs 0.0489（2.2×），
  段首两者相同 → 鼻极不是全程加成。
- 声母边界保留：`syllable_plan("d","ian",0.30)` = `(("d",0.056),("ian",0.244))`，渲染 7200 样本，
  0–40 ms RMS 0.0133 vs 120–200 ms 0.211（闭塞段 < 0.1× 元音），voiced 0.893，F0 200.36 Hz。
- 旧渲染回归：8 条 stage-0 音节（ma1/sha1/ban3/dei3/gou3/bi1/dan1/ai4）的 float32 字节 sha256
  全部与扩表前一致（`regression.all_same = true`）。

### stage 2 — `apical_retroflex` `uo` `ong` `en`（2026-11-01）

选择依据：stage-1 `next_units` 中边际最大的四个韵母（37864 / 34783 / 29562 / 26197 音节）。
`apical_retroflex` 单独加入也能解锁 37864 个音节，但其中含 zh/ch/r 声母的部分要等声母到位；
本批只加韵母，不解锁的部分留给 stage 3。

| 单元 | 类别 | 轨迹 (F1,F2,F3) Hz | 语音描写 | 单独解锁 |
| --- | --- | --- | --- | --- |
| `apical_retroflex` | approximant（新增 kind） | 300 / 1600 / 1800（稳态） | [ʅ] 舌尖后元音 | +37864 |
| `uo` | diphthong | (320,750,2300) → (450,850,2400) → (480,880,2400) | [wo] 后圆唇复韵母 | +34783 |
| `ong` | nasal（`nasal_from=0.55`） | (380,900,2400) → (430,820,2450) → (350,900,2350) | [ʊŋ] 软腭鼻尾 | +29562 |
| `en` | nasal（`nasal_from=0.62`） | (520,1350,2500) → (470,1450,2500) → (330,1650,2500) | [ən] 央中元音+齿龈鼻尾 | +26197 |

`apical_retroflex` 不能复用 /i/：它是带摩擦的浊近音，需要同时发声（浊音承载音节电平）与噪声
（残余摩擦）。为此在 `PhonemeSpec.kind` 增加 `"approximant"`，`_source` 给出
`voiced=1.0, noise=noise_gain(0.10)`；噪声带 2500–7000 Hz、增益小到不会反客为主。

覆盖率（同一 manifest，不重解码）：

| | stage 1（30） | stage 2（34） |
| --- | --- | --- |
| 音节覆盖率 | 0.503755（502746/997997） | **0.621630（620385/997997）** |
| 整句覆盖率 | 0.012279（1081/88035） | **0.037894（3336/88035）** |
| 剩余阻塞 top | apical_retroflex, zh, h, uo | onset:zh 53341, onset:h 39711, onset:ch 29513, final:uei 27950, final:eng 26316 |

整句覆盖率再涨 3.1 倍（1081 → 3336 句），相对 P1.4 基线 187 句已 17.8 倍。

合成探针（`logs/phonemes_stage2_v4.json`，yang 模板、f0=100 Hz 谐波峰值定位，0.100–0.200 s）：

| 单元（载体） | F1 段首→段尾 | F2 段首→段尾 | F3 段首→段尾 | 模板质心比 | F0 中位 | voiced |
| --- | --- | --- | --- | --- | --- | --- |
| apical_retroflex（sh+i） | 300 → 300 | 1600 → 1600 | 1800 → 1800 | 1.209 | 199.71 | 0.929 |
| uo（d+uo） | 400 → 500 | 900 → 900 | 2500 → 2600 | 1.305 | 200.32 | 0.893 |
| ong（d+ong） | 400 → 400 | 900 → 900 | 2600 → 2500 | 1.299 | 200.34 | 0.893 |
| en（b+en） | 500 → 400 | 1500 → 1700 | 2700 → 1800 | 1.319 | 200.23 | 0.857 |

- [ʅ] 的判据不是「像 /i/」而是「F3 低、F1 抬、带摩擦」：F3 区带比
  E(1600–2000)/E(2100–2600) = 8.09，而 /i/ 为 0.037（218×）；高带摩擦占比
  (2500–7000)/(300–1200) = 0.0194 vs /i/ 0.0040（4.85×）；F1 300 Hz 落在 250–400 内。
- `uo` 的 F1 抬升（400→500）与 `en` 的 F2 抬升（1500→1700）都与 IPA 描写方向一致；`ong`
  保持后元音特征（F1/F2 平），与 `en` 形成对照。
- 鼻化只在韵尾：ong 段尾低带占比 0.03461（去鼻极 0.01563，2.21×），en 0.01866 vs 0.00882（2.12×），
  两者段首的鼻/口对照完全相同 → 鼻极仍只作用于 coda。
- 模板依赖：质心比 1.209–1.319，覆盖 `PRESETS` formant_scale 比 1.24（详见失败日志第 7 条：
  v3 版指标在 f0=200 的音节里被谐波网格量化，已作废并重测）。
- 旧渲染回归：8 条 stage-0 音节 sha256 全部不变（`regression.all_same = true`）。

### stage 3 — `zh` `h`（2026-11-01）

选择依据：stage-2 `next_units` 里两个声母（zh 45238、h 25585 音节）。声母只增加
`MANDARIN_PHONEMES` 条目，`g2p.SUPPORTED_ONSETS` 由清单自动派生，没有第二份列表。

| 单元 | 类别 | 声学设定 | 单独解锁 |
| --- | --- | --- | --- |
| `zh` | affricate（清、不送气） | 闭塞 35 ms + 4 ms 爆发（3000/1500 Hz）+ VOT 30 ms，噪声带 2000–6000 Hz（与 sh 同部位） | +45238（需 eng/ua/uai/uan/uang/uei/uen） |
| `h` | fricative（清） | [x] 噪声带 900–4000 Hz，增益 0.20（x 为 3000–8000/0.28，低一个八度） | +25585（需 eng/ua/…） |

普通话 `h` 不拼 /i/、/y/（没有 hi/hü 音节），因此 [x] 覆盖它实际会遇到的韵母；德语 ich 的 [ç]
变体不在本表范围内，**没有**用近似音顶替，只是在清单里不存在对应音节。

覆盖率（同一 manifest，不重解码）：

| | stage 2（34） | stage 3（36） |
| --- | --- | --- |
| 音节覆盖率 | 0.621630（620385/997997） | **0.692366（690979/997997）** |
| 整句覆盖率 | 0.037894（3336/88035） | **0.060362（5314/88035）** |
| 剩余阻塞 top | onset:zh, onset:h, onset:ch, final:uei | onset:ch 29513, final:uei 27950, final:eng 26316, final:v 23661, final:in 22536 |

合成探针（`logs/phonemes_stage3.json`；声母单独渲染 0.12 s，音节为 zh+a / h+a）：

| 单元 | 闭塞 RMS | 爆发 RMS | 闭塞/爆发 | 摩擦 RMS | 低中带 1–3 kHz | 中带 2–6 kHz | 高带 6–9 kHz |
| --- | --- | --- | --- | --- | --- | --- | --- |
| zh | 0.000223 | 0.07284 | 0.00306 | 0.06478 | 0.0239 | 0.6158 | 0.3842 |
| h | —（无闭塞） | — | — | 0.05132 | 0.5937 | 0.7463 | 0.0 |

- zh 的闭塞段比爆发低 327 倍：stop cue 没有被摩擦提前泄露；`zh` 与 `j` 的部位对比为
  （2–6 kHz）/（6–9 kHz）能量比 1.57 vs 0.41（测试断言 zh > 2× j）。
- h 的低中带占比 0.594，而 `x` 为 0.0（<5 倍即判失败）；高带 0.0。
- 音节层：zh+a F0 中位 200.33 Hz、voiced 0.893；h+a 199.86 Hz、voiced 0.857；长度精确 7200 样本
  （0.30 s）；onset 段噪声带占比 vs 元音段分别为 0.612/0.007（zh）与 0.358/0.035（h）。
- 旧渲染回归：8 条 stage-0 音节 sha256 全部不变（`regression.all_same = true`）。

## 失败、负面发现、错误日志

1. **度量本身失败两次**（与合成器无关，但会伪造结论，故保留）：
   - 第 1 版用整段频谱质心比较段首尾——FFT 长度写成 `np.fft.rfftfreq(n=1024)` 却配 960 样本窗口，
     且段尾取 `0.78*SR` 超出 0.30 s 时长，结论无效。
   - 第 2 版用带能量比 (2000–3500)/(300–1200)：f0=200 Hz 时该带只落 10–17 次谐波，比值随「谐波是否
     正好落在共振峰上」剧烈跳动（参照元音 ai/yin 也到 3.57），无法作为滑音判据。
   - 定稿度量：f0=100 Hz 探针 + 谐波峰值定位（`harmonic_peak`），窗口 0.055–0.095 s / 0.225–0.265 s。
2. `probe_phonemes.py` 首版把回归用例名当拼音解析（`ma1` → `a1`），直接触发
   `Unsupported phoneme 'a1'`；改为在脚本内固定 `REGRESSION_CASES`。同一版本还把
   `relative_to(ROOT)` 用在未 `resolve()` 的相对路径上，报 `is not in the subpath of`。
3. `logs/phonemes_stage1_v1.json` 保留：它的 `syllable` 字段把 `l+iou` 拼成 `liou`（拼音正字法是
   `liu`），属于标注缺陷；v2 改名为 `onset_plus_final` 并重跑。两份都在，未删。
4. 未做听音评分：本批只有合成信号与数值指标，不能声称「听上去更像人」。

5. **stage-2 探针的 v1 日志缺 F3**：`logs/phonemes_stage2.json` 只有 F1/F2，看不出 [ʅ] 的 F3 是否真的
   压低；v2 增加 `f3_early_hz/f3_late_hz` 重跑（`phonemes_stage2_v2.json`），v1 保留。
6. **探针把韵母 id 当拼音拼**：`apical_retroflex` 的拼音韵母是 `i`，直接拼成
   `shapical_retroflex`，触发 `pinyin 'shapical_retroflex' is not a Mandarin syllable: pinyin body
   must be ASCII letters`。改法：`PINYIN_FINAL = {"apical_retroflex": "i", "apical_dental": "i"}`，
   plan 与标签都用拼音韵母，记录里同时保留单元名与拼音两个字段。
7. **模板依赖指标作废一次（v3）**：`phonemes_stage2_v3.json` 把质心比算在 0.30 s 音节内、f0=200 Hz，
   结果被 200 Hz 谐波网格量化成 1.089–1.619，看起来像「en 偏离模板」。改为裸单元、0.30 s、f0=100 Hz
   的 0.100–0.200 s 窗口后为 1.209–1.319，与 `PRESETS` 的 1.08/0.87 = 1.24 一致。v3 保留，结论以 v4 为准。
   这不是合成器缺陷，是指标缺陷，但如果当时采信 v3 就会得出「新韵母跟错模板」的错误结论。
8. **测试断言写死峰值频率，两次误报**：
   - `assert harmonic_peak(..., 1600, 2100) == 1800.0` 实测 1600.0——F2 1500 Hz 的裙边在 1600 Hz
     强过 1800 Hz 的 F3。改为区间断言 1600–2000，并注明理由。
   - `front_f3 >= 2900` 对 /i/ 实测 2400.0——/i/ 的 F2 2250 Hz 肩部在 1800–3800 的峰值搜索里胜出，
     该断言测的其实是 F2。改为带能量比 E(1600–2000)/E(2100–2600)（[ʅ] 8.09 vs /i/ 0.037，218×），
     并把「/i/ 的 F3 > 2900」移回声明层（`MANDARIN_PHONEMES["i"].targets[0][3]`）。
9. **扩表使四条既有断言失效，全部重定向而不是删除**：`test_unknown_phoneme_is_rejected...` 的
   「未知音素」原来是 `zh`（现在合法）→ 改 `ch`；`test_y_and_w_are_not_missing_onsets` 的 `w+uo`
   与 `zh+a` 两个缺口被补齐 → 改 `w+ua`、`ch+a`；`test_build_pair_blocks_a_mandarin_syllable...`
   的 `shi2`（sh + 舌尖后元音）可渲染 → 改 `zi2`（舌尖前元音仍未实现）。断言文本与判据未变。
10. 仍未做听音评分：三个批次都只有合成信号与数值指标，不能声称「听上去更像人」。

## 结论 / 下次实验

stage 1-3 全部通过预注册条件：覆盖率单调上升（音节 30.95% → 50.38% → 62.16% → **69.24%**，
整句 0.21% → 1.23% → 3.79% → **6.04%**），每批的 8 条 stage-0 渲染 sha256 均未变化，
每个新单元都有「声学类别 + 轨迹方向 + 模板依赖 + 边界/F0」四类渲染级证据。

三批共新增 10 个单元（26 → 36）：韵母 `e ian ing iou`（stage 1）、`apical_retroflex uo ong en`
（stage 2，含新的 `approximant` 声源分支）、声母 `zh h`（stage 3）。剩余阻塞按 stage-3
`next_units` 排序：`final:uei`（+27534，需 ch/r）、`final:v`（+23661）、`final:in`（+22536）、
`onset:ch`（+21214）、`final:eng`（+25944，需 r）、`final:apical_dental`（+20702）、
`final:iao`（+20371）、`final:iang`（+18755）、`onset:r`（+17877）、`final:ie`（+17485）。
下一批仍按「整单元 + 声学类别正确 + 不近似替换」推进；`v`（ü）与 `apical_dental`（[ɿ]）都需要
新的声学设定（前圆唇、舌尖前近音），不能拿 `u` / `i` 顶替。

限制：整句覆盖率 6.04% 仍远低于音节覆盖率——一句里只要有一个音节未实现就整句阻塞；这是
「按音节乘积」的结构性结果，不是本批缺陷。未做听音评分，Windows CUDA 仍未验证。
