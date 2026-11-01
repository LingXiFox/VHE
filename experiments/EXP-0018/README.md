# EXP-0018 — Twenty real-mechanical pairs with diagnostic previews

Status: **built, not accepted**. No human has listened to any pair yet, so this record makes
no claim about naturalness and no pair is marked `training_ready`.

## 假设
With the viterbi tracker (EXP-0015) and the energy-DP aligner, a diverse 20-utterance sample
drawn from dev/test can be paired end to end without manual boundaries; most pairs will be
usable previews, but boundary confidence only measures energy-valley depth, so only a minority
should qualify as training-ready until phoneme boundaries are checked by hand.

The run also tests an unstated assumption of P1.5 §1⑤: that a manifest `renderable` flag means
the synthesizer can actually render the sentence. It did not — see "Defect found".

## 变量 / 对照组 / 数据划分
| item | value |
| --- | --- |
| sample | 20 utterances, dev+test only, 1-3 s, 2-6 syllables, no `quality_flags` |
| selection | greedy final coverage then onset coverage, one utterance per speaker, both genders, seed 42 |
| F0 estimator | `viterbi` (multi-candidate + continuity), fmax 900 Hz, never narrowed |
| alignment | `dp` (one energy-valley boundary per transcript syllable) |
| backend | `mandarin_formant` (MandarinFormantSynth), inventory 36 |
| template | heuristic: median F0 >= 200 Hz → `yang`, else `yin` (see EXP-0019) |
| target rate | 24 kHz (real audio resampled 44.1 → 24 kHz) |

Only one variable differs from the P1.5 defaults: nothing. This is an integration run.

Pre-registered success criteria: a pair is *rendered* when a mechanical half exists, and
*training-ready* only when every readiness reason is empty. A training-ready rate of zero was
declared in advance to mean "the DP is not a forced aligner", which is what happened.

## 完整复现命令 / 环境与哈希
```bash
# v1: no backend argument -> FormantProxySynth (inventory "vowels")
python experiments/EXP-0018/probe_pairs.py --manifest <v4 manifest> \
    --out-root .../derived/aishell3-pairs/v1 \
    --report experiments/EXP-0018/logs/pair_report_v1.json \
    --sheet experiments/EXP-0018/logs/listening_sheet_v1.md \
    --spk-info <AISHELL-3/spk-info.txt>

# v2: mandarin_formant backend, before the parse_syllable fix
python experiments/EXP-0018/probe_pairs.py ... --out-root .../aishell3-pairs/v1 \
    --report .../logs/pair_report_v2.json --sheet .../logs/listening_sheet_v2.md

# v3: after the fix, same selection
python experiments/EXP-0018/probe_pairs.py --manifest <v4 manifest> \
    --out-root /Volumes/LingXiWorkspace/VHE-Datasets/derived/aishell3-pairs/v2 \
    --report experiments/EXP-0018/logs/pair_report_v3.json \
    --sheet experiments/EXP-0018/logs/listening_sheet_v3.md \
    --spk-info /Volumes/LingXiWorkspace/VHE-Datasets/AISHELL-3/extracted/spk-info.txt --label v3
```

- manifest: `/Volumes/LingXiWorkspace/VHE-Datasets/derived/aishell3-scan/v4/manifest.jsonl`,
  sha256 `1a8138c4d13e547d1d266a7aaf50d3a002a8e32dad78b0555f1b417c77074f31`.
- previews (outside Git, the real half is licensed audio):
  `/Volumes/LingXiWorkspace/VHE-Datasets/derived/aishell3-pairs/v2/sample_XX/`
  (`real.wav`, `mechanical.wav`, `diagnostics.png`, `notes.json`), 20 folders, 4.4 MB.
- environment: macOS arm64, Python 3.14.6, CPU only (no CUDA, no MPS needed for this probe).

## 预注册通过/失败条件
- pass: a mechanical half exists for ≥15/20 and the run records per-pair readiness; no
  naturalness claim is made either way.
- fail: any pair is marked training-ready while a readiness reason is present, or any rejected
  F0 frame is silently interpolated into the mechanical request.
- defect: any utterance whose manifest `renderable` flag disagrees with the renderer.

## 实际结果与主观听音
### Defect found: the manifest lied about 59% of its renderable sentences
v2 produced 14 `ok` / 5 `blocked` / 1 `rejected`, and every blocked one named a syllable that
should have been renderable (`shi2`, `wo3`, `you3`). Root cause was
`src/vhe/data/pairing.py::parse_syllable`: it passed the *underlying* final (`apical_retroflex`,
`uo`, `iou`) into `syllable_phonemes(initial, final)`, which expects the *orthographic* spelling
and resolves it itself. The result was fabricated pinyin (`sh` + `apical_retroflex` =
`shapical_retroflex`, `w` + `uo`, `y` + `iou`) and a refusal.

Measured on the v4 manifest: **3160 of the 5314 utterances marked `renderable` (59%) could not be
rendered** (4408 syllables: 1597 `shapical_retroflex`, 1237 `yiou`, 900 `wuo`, 503
`zhapical_retroflex`, 171 `yian`). Coverage and the usable-sentence count disagreed, so the
P1.5 §1⑤ check had only compared counts, never an actual render.

Fix: `parse_syllable` now passes `token.written or token.final` and keeps the underlying final
in `Syllable.underlying_final`. After the fix all 5314 renderable utterances render (35153
syllables, 0 blocked), and across all 1920 distinct manifest tokens the `missing_units`
predicate and the synthesis path agree with zero exceptions. Both facts are regression tests
(`tests/test_data_pairing.py`, and
`tests/test_data_g2p.py::test_the_coverage_predicate_and_the_synthesiser_agree_on_every_token`).

Also fixed here: `PairResult.alignment["boundaries"]` now carries each join's own `dip_db` and
`confidence` (previously only start/end/duration, which is why the report's
`boundary_confidence.count` was 0), and the `mostly_unvoiced` message prints 4 decimals so it
no longer reads "0.30 < 0.30".

### v1 (failed, kept)
`logs/pair_report_v1_blocked.json`, previews moved to
`derived/aishell3-pairs/v1_blocked_proxy_backend/`. 20/20 `blocked` with
`backend_inventory_insufficient`: `build_pair` defaults to `FormantProxySynth`, whose inventory
is the five vowels, so every consonant was missing. The probe now passes
`get_backend("mandarin_formant")` explicitly. Nothing about the data was learned here; it only
proved the failure path works.

### v3 (valid)
| id | utt | spk | gender | F0 Hz | template | syl | status | min boundary conf | min pitch coverage | gains dB | ready |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 01 | SSB03160415 | SSB0316 | male | 137.6 | yin | 6 | blocked | 0.214 | 0.87 | -12.1..-9.9 | no |
| 02 | SSB10080285 | SSB1008 | female | 246.3 | yang | 5 | ok | 0.933 | 0.83 | -15.5..-6.1 | no |
| 03 | SSB00180288 | SSB0018 | female | 249.8 | yang | 6 | blocked | 0.462 | 0.75 | -10.9..-6.3 | no |
| 04 | SSB04350385 | SSB0435 | female | 271.2 | yang | 6 | ok | 0.643 | 0.25 | -26.0..-18.4 | no |
| 05 | SSB11380340 | SSB1138 | female | 248.7 | yang | 2 | ok | 1.000 | 0.81 | -20.7..-18.7 | no |
| 06 | SSB09130026 | SSB0913 | female | 269.0 | yang | 2 | rejected | - | - | - | no |
| 07 | SSB14520481 | SSB1452 | female | 239.2 | yang | 4 | blocked | 1.000 | 0.89 | -13.4..-13.3 | no |
| 08 | SSB05440115 | SSB0544 | male | 133.7 | yin | 2 | ok | 1.000 | 0.88 | -20.5..-8.9 | no |
| 09 | SSB02880339 | SSB0288 | female | 185.3 | yin | 2 | ok | 1.000 | 0.78 | -17.9..-15.9 | no |
| 10 | SSB18630221 | SSB1863 | male | 125.3 | yin | 2 | ok | 1.000 | 0.69 | -13.4..-10.6 | no |
| 11 | SSB18910490 | SSB1891 | female | 183.9 | yin | 2 | ok | 1.000 | 0.94 | -22.4..-12.6 | no |
| 12 | SSB06010331 | SSB0601 | female | 283.6 | yang | 2 | ok | 1.000 | 0.90 | -23.1..-22.3 | no |
| 13 | SSB03950362 | SSB0395 | female | 197.6 | yin | 2 | ok | 1.000 | 0.54 | -19.9..-16.3 | no |
| 14 | SSB13770005 | SSB1377 | female | 270.0 | yang | 3 | ok | 0.583 | 0.67 | -13.2..-10.4 | no |
| 15 | SSB18370358 | SSB1837 | female | 199.5 | yin | 2 | ok | 1.000 | 0.59 | -18.1..-14.6 | no |
| 16 | SSB15930434 | SSB1593 | female | 206.8 | yang | 2 | ok | 0.684 | 1.00 | -17.5..-13.4 | no |
| 17 | SSB03390129 | SSB0339 | female | 223.3 | yang | 2 | ok | 1.000 | 0.87 | -21.0..-18.9 | no |
| 18 | SSB07580481 | SSB0758 | female | 225.4 | yang | 2 | ok | 0.857 | 0.86 | -11.4..-10.0 | no |
| 19 | SSB01450094 | SSB0145 | female | 248.1 | yang | 2 | ok | 1.000 | 0.76 | -14.9..-14.6 | no |
| 20 | SSB12740255 | SSB1274 | female | 223.3 | yang | 3 | ok | 1.000 | 0.78 | -26.0..-19.0 | no |

- built 20: **16 ok / 3 blocked / 1 rejected**, **0 training-ready**.
- readiness reasons: `segmentation_count_disagrees` 19, `status_blocked` 3, `error_issues` 4,
  `no_pitch_frames` 1, `low_boundary_confidence` 1, `no_alignment` 1,
  `pitch_gap_interpolated` 1.
- boundary confidence: 38 joins, median 1.000, min 0.214, one below 0.35.
- pitch coverage: 50 syllable windows, median 0.892, one below 0.5, none at zero.
- gains: -26.02 .. -6.09 dB. The synthesizer is much quieter than the recording; the gain is
  recorded (never normalized into the model) so it can be undone.

#### Why three pairs are blocked
All three are `syllable_plan_blocked` with a window of 0.0795 s, just under the 80 ms floor
(`ONSET_MIN_S` 0.020 + `NUCLEUS_MIN_S` 0.060): sample_01 `de5` (neutral tone), sample_03
`tao2`, sample_07 `ce4`. The DP placed those joins inside a continuous 0.79 s stretch for four
syllables (SSB14520481: 0.32-0.74 / 0.74-0.93 / 0.93-1.01 / 1.01-1.11 s), compressing the last
two instead of distributing the span. A real forced aligner or operator boundaries would put
them elsewhere; the refusal itself is correct — a 24 ms nucleus is not a syllable, and rendering
it as one would be exactly the "close enough" substitution this project forbids.

#### Why nothing is training-ready
`segmentation_count_disagrees` fires on 19/20 because ordinary AISHELL-3 utterances are
continuous speech: the energy segmenter finds **one** segment for a 4-6 syllable sentence, so
the independent cross-check the readiness gate requires cannot pass. Only operator-specified
boundaries ("manual") or a real forced aligner can clear it. This is the gate working as
designed: the DP's `min_confidence` reaches 1.0 on continuous vowels while the DP is
demonstrably wrong, so confidence alone must never certify a pair.

The one rejected sample (SSB09130026, `ku4 pai4`) has 0.2997 voiced frames, below the 0.30
floor: two voiceless stop onsets in 0.6 s. Refusing it is correct, and it shows the floor sits
right at the boundary for very short utterances.

### 主观听音
None. `logs/listening_sheet_v3.md` has all 20 rows with empty timbre/naturalness/content cells;
the previews exist but the owner has not listened. No naturalness claim may be made before that.

## 失败、负面发现、错误日志
1. v1 — 20/20 blocked by the default proxy backend (inventory "vowels"); artifacts kept in
   `derived/aishell3-pairs/v1_blocked_proxy_backend/`.
2. v2 — 5 blocked by fabricated pinyin from the `parse_syllable` defect (3160/5314 sentences
   affected corpus-wide); report kept as `logs/pair_report_v2.json`.
3. `boundary_confidence.count` was 0 in v2 because `_window_stats` dropped the per-join
   confidence; fixed in code and covered by a test, not by patching the report.
4. The `mostly_unvoiced` message printed "0.30 < 0.30" (rounding); fixed to 4 decimals.
5. v3 still blocks 3 pairs on sub-80 ms DP windows — recorded, not worked around.
6. No listening has been done; nothing here supports a naturalness claim.
7. Selection is 17 female / 3 male against a 175/43 corpus: gender balance was a greedy
   tie-break, not a quota. A balanced set needs its own selection pass.
8. The template heuristic (median F0 >= 200 Hz) is not a timbre match; EXP-0019 measures how
   weakly it predicts the closer preset (10/16) and how large the residual gap is.

## 结论 / 下次实验
The pipeline runs end to end: 16 of 20 chosen utterances produce a real/mechanical pair with
viterbi F0, DP boundaries, per-join evidence and a diagnostic image. It produces **zero**
training-ready pairs, which is the correct answer for continuous speech without forced
alignment. The run also invalidated a previous claim that coverage and the usable-sentence count
agree — they now do, and a test enforces it.

Next: EXP-0019 (template identity, no training), and a forced-aligner comparison before any
pair is promoted to training data.

## 勘误
`metadata.json` 的 `completed_utc` 原先写成 2026-11-01T07:30:00+00:00，早于实际落盘时刻被写成了未来时间（写记录时按估算填的）。已改为记录实际写入时刻 2026-11-01T06:46:00+00:00，并在 metadata 里保留 `erratum` 字段说明原值。命令、结果与结论均未改动。
