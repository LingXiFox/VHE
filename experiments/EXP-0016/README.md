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

## 结论 / 下次实验

stage 1 通过预注册条件：覆盖率单调上升（30.95% → 50.38%），旧渲染逐字节不变，四个单元的滑音方向、
模板依赖、鼻化位置、声母边界都有渲染级证据。下一步按 stage-0 的边际排序做 stage 2：
`apical_retroflex`（+37864，需要 ch/zh/r 一起才完全解锁）、`uo`（+34783）、`ong`（+29562）、
`en`（+26197）。`apical_retroflex` 是舌尖后元音 [ʅ]，不能拿 /i/ 顶替，需要新增「带摩擦的浊音」声源
分支。
