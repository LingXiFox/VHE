# EXP-0019 — Template identity matching versus the real speaker envelope

Status: analysis only. **No training, no template fitting, no new pairs.**

## 假设
Choosing between the two original presets by median F0 is not a timbre decision: the LTAS
distance between the presets is small compared with the distance from either preset to a real
AISHELL speaker, so identity preservation must be defined against the mechanical input envelope,
not against the real recording.

Two halves of that claim are separated below because they do not fare the same way: the
operational half ("F0 does not pick timbre") is supported, the quantitative half ("presets are
much closer to each other than to a speaker") is not — the two distances are about equal.

## 变量 / 对照组 / 数据划分
- Data: the 16 pairs EXP-0018 actually rendered (`logs/pair_report_v3.json`), same manifest
  (`derived/aishell3-scan/v4/manifest.jsonl`, sha256 `1a8138c4…`), same selection seed 42.
- Metric: level-normalised log-mel LTAS, 24 bands over 100-7000 Hz, 1024-sample Hann frames,
  256 hop, loud frames only (RMS >= 5% of the peak frame); distance = RMS difference in dB.
  Independence from level is deliberate: the synthesizer is 6-26 dB quieter than the recording.
- Control 1: `d(yang, yin)` on the `(real, yang)` and `(real, yin)` renders. The two renders
  share the same real audio (asserted by hash) and the same F0 curve, so the only difference is
  the template.
- Control 2: the EXP-0018 heuristic (median F0 >= 200 Hz → yang) versus the preset with the
  smaller `d(real, ·)`.
- Diagnostic: a 1-D grid over formant scale (0.70-1.30 step 0.05, tilt fixed at 1.35,
  breathiness 0.009) fitted to minimise `d(real, render(s))`. This measures where a speaker sits
  relative to the presets. It is **never** used to choose or build a pair.

## 完整复现命令 / 环境与哈希
```bash
python experiments/EXP-0019/probe_identity.py \
    --manifest /Volumes/LingXiWorkspace/VHE-Datasets/derived/aishell3-scan/v4/manifest.jsonl \
    --pairs-report experiments/EXP-0018/logs/pair_report_v3.json \
    --out experiments/EXP-0019/logs/identity_report_v2.json --grid
```
- logs: `logs/identity_report_v1.json` (grid 0.75-1.20, 9/16 fits pinned to the upper edge),
  `logs/identity_report_v2.json` (grid 0.70-1.30, 7/16 still at the edge — see failure log).
- environment: macOS arm64, Python 3.14.6, CPU only, pure numpy.
- runtime: ~110 render calls, under a minute; no audio corpus pass, no training.

## 预注册通过/失败条件
The hypothesis text above is the pre-registered claim (written by `tools/new_experiment.py`
before any run). The numeric thresholds below were written while recording the results, i.e.
**after** the run — they are reported as descriptive, not as a pre-registered gate:

- operational claim supported if the F0 heuristic matches the closer preset on <= 80% of pairs
  (chance would be 50%), i.e. it carries little timbre information.
- quantitative claim supported if `median d(preset, preset) < 0.7 × median d(real, best preset)`.
- diagnostic is only reported if fewer than half of the fits sit on a grid edge.

## 实际结果与主观听音
### Distances (v2, 16 pairs)
| id | speaker | gender | F0 Hz | heuristic | closer | agree | d(real,yang) | d(real,yin) | d(yang,yin) | margin | implied scale |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 02 | SSB1008 | female | 246.3 | yang | yang | yes | 4.51 | 6.61 | 4.56 | 2.11 | 1.05 |
| 04 | SSB0435 | female | 271.2 | yang | yang | yes | 6.21 | 9.03 | 6.98 | 2.82 | 1.30 |
| 05 | SSB1138 | female | 248.7 | yang | yang | yes | 6.10 | 8.81 | 6.26 | 2.71 | 1.10 |
| 08 | SSB0544 | male | 133.7 | yin | yang | **no** | 10.38 | 10.88 | 5.41 | 0.51 | 1.30 |
| 09 | SSB0288 | female | 185.3 | yin | yang | **no** | 5.32 | 7.55 | 5.99 | 2.23 | 1.15 |
| 10 | SSB1863 | male | 125.3 | yin | yang | **no** | 9.02 | 9.51 | 4.45 | 0.48 | 1.15 |
| 11 | SSB1891 | female | 183.9 | yin | yin | yes | 7.54 | 6.65 | 6.61 | 0.90 | 0.70 |
| 12 | SSB0601 | female | 283.6 | yang | yang | yes | 5.61 | 7.18 | 5.44 | 1.56 | 1.20 |
| 13 | SSB0395 | female | 197.6 | yin | yang | **no** | 7.58 | 10.56 | 5.73 | 2.98 | 1.30 |
| 14 | SSB1377 | female | 270.0 | yang | yang | yes | 7.82 | 9.01 | 7.49 | 1.19 | 1.30 |
| 15 | SSB1837 | female | 199.5 | yin | yang | **no** | 5.19 | 7.84 | 6.08 | 2.64 | 1.20 |
| 16 | SSB1593 | female | 206.8 | yang | yang | yes | 7.11 | 8.04 | 5.59 | 0.93 | 1.30 |
| 17 | SSB0339 | female | 223.3 | yang | yang | yes | 4.94 | 6.22 | 7.01 | 1.27 | 1.00 |
| 18 | SSB0758 | female | 225.4 | yang | yang | yes | 7.47 | 8.87 | 7.34 | 1.40 | 1.30 |
| 19 | SSB0145 | female | 248.1 | yang | yin | **no** | 6.72 | 6.35 | 6.30 | 0.37 | 0.90 |
| 20 | SSB1274 | female | 223.3 | yang | yang | yes | 11.56 | 13.14 | 5.12 | 1.59 | 1.15 |

Aggregates: agreement **10/16 = 0.625**; median `d(real,yang)` 6.92 dB, `d(real,yin)` 8.42 dB,
`d(yang,yin)` 6.03 dB, margin 1.48 dB, `gap/separation` **1.01**.

### Reading
1. **The operational claim holds.** The F0 heuristic reproduces the acoustically closer preset
   on only 10 of 16 pairs, so median F0 carries little timbre information; five low-F0 speakers
   (125-200 Hz) are actually *closer to `yang`* (the larger formant scale), the opposite of the
   "low voice = larger tract = darker preset" intuition. Choosing by F0 is not choosing by
   timbre, and EXP-0018's template assignment must not be described as a timbre match.
2. **The quantitative claim does not hold.** The presets differ by 6.03 dB, while the best preset
   still sits 6.92 dB from the real speaker: the ratio is 1.01, not < 0.7. Selecting the better
   preset wins only 1.48 dB (median) out of a ~7 dB mismatch. Both presets are far from every
   real speaker; the choice between them is a small perturbation of a large gap.
3. **Therefore** any statement like "the mechanical voice matches the speaker" is false at this
   stage. Identity has to mean "the model keeps the identity of its mechanical input", because
   that is the only identity the model is allowed to see.
4. **The metric is discriminative**: `d(yang,yin)` is 4.5-7.5 dB on identical content and F0,
   i.e. it separates identities well above noise, so it can serve as the identity-preservation
   metric in P2 — once measured output-vs-input instead of output-vs-real.

### Diagnostic: implied formant scale is not identifiable
The 1-D fit prefers scales above both presets (median 1.175). It is not trustworthy: with tilt
fixed at 1.35 the optimum trades scale against tilt, and 7 of 16 fits pin to the upper edge
(1.30) even after widening the grid from 1.20. The only defensible reading is directional: most
speakers want a larger formant scale than either preset, not that 1.175 is their scale. A
two-dimensional (scale, tilt) fit or an LPC-based formant estimate would be needed; not run here.

### 主观听音
None. This experiment renders nothing new for listening and makes no naturalness claim.

## 失败、负面发现、错误日志
1. v1 used a 0.75-1.20 grid; 9/16 fits pinned to the 1.20 edge. Kept as
   `logs/identity_report_v1.json`; v2 widened the range and still pins 7/16 at 1.30, so the
   non-identifiability is a property of the 1-D fit, not of the range.
2. The pre-registered quantitative claim (presets much closer to each other than to a speaker)
   is **not** supported; it is recorded as a failed prediction rather than reworded.
3. The numeric pass/fail thresholds were written after the run. They are labelled descriptive;
   the hypothesis text in `metadata.json` predates the run and is the only pre-registration.
4. Only 16 of 20 pairs are usable (3 blocked by sub-80 ms DP windows, 1 rejected as mostly
   unvoiced), and 17/20 are female, so the sample cannot speak for the corpus.
5. The metric uses loud frames only, so it describes vowels more than onsets; consonant identity
   is unmeasured here.
6. No listening, no training, and the grid fit is not a validated formant estimate.

## 结论 / 下次实验
"Pick the preset that matches the speaker" is not a valid construction rule at this stage: the
two originals are about as far from each other as each is from real speech, and F0 does not
predict which one is closer. Pairs must therefore keep a template that is fixed **before** the
real audio is inspected, and identity preservation must be defined and measured against that
mechanica1 input.

Proposal for P2 (testable, none of it started):

1. **Fix the template by construction.** Assign a template from a seeded cycle over the
   original presets, independent of the recording, and record it in the pair. Invariant:
   re-running selection with shuffled templates must not change any other pair field. A test can
   assert that the chosen template is uncorrelated with the speaker's F0 and LTAS.
2. **Widen the identity family with original voices, not target voices.** The presets cover
   formant scales 0.87-1.08 while most speakers sit above 1.15; add 2-3 *synthetic* templates
   (e.g. 1.15, 1.25 with a tilt axis) built from the same source, never from a real speaker. The
   synthesizer must never be fitted to the target recording: that is the identity-substitution
   shortcut AGENTS.md forbids.
3. **Identity-preservation metric.** For a held-out mechanical input x with identity τ, require
   `d(VHE(x), x) < d(VHE(x), render(x, τ'))` by a margin across τ, τ' and content; report the
   margin distribution, not a single average.
4. **Identity-leakage guard.** Also measure `d(VHE(x), real)` and require it not to fall below
   `d(x, real)` beyond a bound: an output that drifts toward the target speaker is the forbidden
   shortcut, and it would also fail on unseen original templates.
5. **Content/timing invariance.** With the template swapped, F0 curve, duration, phoneme
   boundaries and phoneme identity of the output must stay within tolerance of the input; test
   on the 8 EXP-0015 synthetic contours plus a small real set.

Next experiment: build the two extra original templates and re-run this probe with four
identities (EXP-0020), still without training.
