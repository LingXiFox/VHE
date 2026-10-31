# EXP-0011 — F0 estimator reads harmonic-rich voiced frames one octave high

Status: completed. Code fix applied, records and tests in the same commit series.

## 假设

Filed before the first probe run, kept as filed (two of its claims turned out wrong; see E2/E5):

> The YIN CMND estimator in `src/vhe/data/f0.py` picks the first dip below 0.15, so a
> harmonic-rich voiced frame whose dominant partial is an even multiple of f0 can be read one
> octave high (f0=220 Hz with a formant near 880 Hz reads 441 Hz). Hypothesis: choosing the
> smallest lag whose CMND is within 0.02 of the global minimum removes octave-up errors on
> harmonic-comb stimuli and on synthesized /a/ at 200-240 Hz, keeps pure-tone accuracy, does
> not introduce octave-down errors, and leaves the silence/noise gating results unchanged.

How it was found: while verifying the P1.2 Mandarin backend, the `tone2` contour (200 -> 238 Hz)
came back from `estimate_f0` as 414-455 Hz in the middle of the syllable. A spectral check
(nperseg=1024) put the true F0 at 211-234 Hz, so the audio was right and the estimator was
wrong. Not a threshold issue: in the offending frame `cmnd(54) = 0.062` at half the period and
`cmnd(109) = 0.003` at the period, and `argmin` already pointed at 109. The selection rule
discarded the answer it had.

## 变量 / 对照组 / 数据划分

* Independent variable: the dip-selection rule in `src/vhe/data/f0.py`. Everything else —
  estimator maths, 40 ms frames, 10 ms hop, `fmin_hz=65`, `fmax_hz=900`,
  `voicing_threshold=0.45`, rates, seeds — is identical between arms.
* Control arm: the pre-fix `_estimate_frame` from `git show HEAD:src/vhe/data/f0.py`
  (commit `b379237`). A worktree at HEAD cannot be used because the Mandarin backend is
  uncommitted there, so the control tree is a copy of the current `src/` with only that one
  file restored to `b379237`.
* Stimulus families, all deterministic, no real audio:
  * `comb`: 6 f0 (100..300 Hz) x 5 resonances (500..1250 Hz) = 30 cases, 0.5 s at 24 kHz,
    1/k^1.1 tilt shaped by one Gaussian resonance, fundamental held at -12 dB relative to the
    strongest partial so the nominal f0 actually exists.
  * `vowel`: `MandarinFormantSynth` /a/ /i/ /u/ at 110/200/220/240/350 Hz = 15 cases. These are
    the signals VHE pairs, so they carry the weight.
  * `robust`: comb(220,500) and vowel /a/ 220 Hz plus white noise at 30/20/10 dB SNR, seeds
    11/12 = 12 cases.
  * `guards`: 65/220/440/880 Hz tones, digital silence, white noise = 6 cases.
* Scoring per frame: `ok` within 5% of f0, `octave_up` within 5% of 2*f0, `octave_down`
  within 5% of f0/2, otherwise `other`. Accuracy = ok / frames.

## 完整复现命令 / 环境与哈希

Environment: Python 3.14.6, NumPy (no SciPy), macOS 27.2 arm64, CPU only. Git base commit
`b379237`, tree dirty with the P1.2 synth work (see metadata.json for the exact status).

```sh
# control arm (v0 rule)
rm -rf /tmp/vhe_exp11_before && mkdir -p /tmp/vhe_exp11_before
cp -R src /tmp/vhe_exp11_before/src
git show HEAD:src/vhe/data/f0.py > /tmp/vhe_exp11_before/src/vhe/data/f0.py
PYTHONPATH=/tmp/vhe_exp11_before/src .venv/bin/python experiments/EXP-0011/probe_octave_v2.py \
    --out experiments/EXP-0011/logs/v2_before.json

# treatment arm
PYTHONPATH=src .venv/bin/python experiments/EXP-0011/probe_octave_v2.py \
    --out experiments/EXP-0011/logs/v2_after.json
```

Hashes: `probe_octave_v2.py` 5ac1fec7c8738a18d42778d86c25087cb262a82b745bb5d644c0546afc72a28d,
`probe_octave.py` 7f47b15e5ffee623a57b82fc6266959ba92267f5d491aa0d96016f4f4409d4d5,
`v2_before.json` 6bcbde736bec3e2d4fff63639c90b775570b6b64d7912fc0439a8501584f8178,
`v2_after.json` 6b3c362439ea94329cffa84d5939cce244cbfc1c2a13275617797de048045d19,
`probe_before.json` 65f93dcb36d446b816d5a62279e46c272d02219a9f71057bd399447e6acd1720,
`probe_after.json` 9e512e481842cfae8cf5d9b1c272bab0e56e982407e65cc1d3d2df917eb21792.

## 预注册通过/失败条件

1. Octave-up frames on the comb family: 0 (filed as "removed"; a 90% reduction was not
   accepted as a pass).
2. Octave-up frames on the vowel family: 0.
3. Pure tones 65/220/440/880 Hz: 100% of voiced frames within 5%, median unchanged.
4. Octave-down frames: 0 in both clean families.
5. Silence and white noise: 0 voiced frames, as before.

## 实际结果与主观听音

No listening test: the probe is numeric and the subject is a scalar estimator. Listening
evidence for the Mandarin backend is a separate experiment (EXP-0009).

v0 rule (`logs/v2_before.json`):

| group | frames | ok | accuracy | octave_up | octave_down | other |
|-------|--------|----|----------|-----------|-------------|-------|
| comb | 1410 | 611 | 0.4333 | 141 | 0 | 658 |
| vowel | 555 | 519 | 0.9351 | 36 | 0 | 0 |
| robust | 504 | 295 | 0.5853 | 209 | 0 | 0 |
| guards | 188 | 188 | 1.0 | 0 | 0 | 0 |

Bounded rule (`logs/v2_after.json`):

| group | frames | ok | accuracy | octave_up | octave_down | other |
|-------|--------|----|----------|-----------|-------------|-------|
| comb | 1410 | 1081 | 0.7667 | 0 | 0 | 329 |
| vowel | 555 | 555 | 1.0 | 0 | 0 | 0 |
| robust | 504 | 439 | 0.8710 | 65 | 0 | 0 |
| guards | 188 | 188 | 1.0 | 0 | 0 | 0 |

All 15 vowel cases are now exact: a_110 110.00, a_200 199.99, a_220 220.00, a_240 239.99,
a_350 349.97, i_* and u_* within 0.02%. The four completely wrong comb cases
(220Hz_F500 440.37, 240Hz_F500 479.99, 300Hz_F700 599.26, 300Hz_F1250 599.74) are gone.

Rule variants and the parameter sweep:

| variant | rule | outcome |
|---------|------|---------|
| v0 | first dip below 0.15, walk downhill | the defect |
| v1 | restart from the global CMND minimum when deeper by 0.01 (absolute) | fixed clean cases, broke noisy ones (E2) |
| v2 | search only `(best, 2*best]`, require <= 0.5x *and* >= 0.02 lower | adopted |

| tol | ratio | comb up/down/other | vowel up/down | robust up/down |
|-----|-------|--------------------|---------------|----------------|
| disabled (v0) | - | 141/0/658 | 36/0 | 209/0 |
| 0.008 | 0.5 | 0/0/329 | 0/3 | 65/0 |
| 0.008 | 0.35 | 0/0/329 | 1/3 | 69/0 |
| 0.008 | 0.25 | 0/0/329 | 1/2 | 84/0 |
| 0.02 | 0.5 | 0/0/329 | 0/0 | 65/0 |
| 0.02 | 0.35 | 0/0/329 | 1/0 | 69/0 |
| 0.02 | 0.25 | 0/0/329 | 1/0 | 84/0 |
| 0.04 | 0.5 | 0/0/329 | 0/0 | 65/0 |
| 0.04 | 0.35 | 0/0/329 | 1/0 | 69/0 |
| 0.04 | 0.25 | 0/0/329 | 1/0 | 84/0 |
| 0.08 | 0.5 | 0/0/423 | 36/0 | 209/0 |
| 0.08 | 0.35 | 0/0/423 | 36/0 | 209/0 |
| 0.08 | 0.25 | 0/0/423 | 36/0 | 209/0 |

0.02 / 0.5 chosen: smallest absolute margin that clears the vowel family, and identical to the
0.04 row on these stimuli, so the choice sits on a plateau rather than a knife edge. 0.08 is
past the point where the repair stops firing at all, which is why the vowel errors return.

Verdict: criteria 1, 2, 3, 5 met; criterion 4 met for the adopted rule but violated by the
first implementation (E2).

## 失败、负面发现、错误日志

E1 — the first probe scored a flaw of the probe, not of the estimator. `probe_octave.py`
(v1, logs `probe_before.json`/`probe_after.json`) let the fundamental sit up to 29 dB below the
dominant partial. A frame that is essentially a pure tone at 2*f0 *is* identifiable at 2*f0, so
799 of 1410 comb frames were scored "other" while the estimator was reporting a real harmonic.
`probe_octave_v2.py` fixes the stimulus (fundamental floor -12 dB) and reports the CMND at the
nominal period and at half of it. v1 logs are kept unchanged.

E2 — the fix as first implemented made a different error worse. "Restart from the global CMND
minimum" is unsafe: for an exactly periodic frame several lags reach ~0 and floating point
decides, and under 10 dB noise the global minimum drifts to twice the period. Measured on the
v2 families: comb 87 octave-down frames, vowel 13, robust 40 — zero before the fix.
`probe_after.json` therefore documents a rule that is no longer in the code; all numbers above
come from `probe_octave_v2.py`.

E3 — implementation bug caught by the sweep, before any number was treated as a result. The
bounded search compared array indices instead of lags, so with `lag_min=26` it only reached
lag ~74 while the period under test was 100: the repair never fired and the first sweep looked
identical to the unfixed estimator.

Still wrong (not fixed here):

* 329 of 1410 comb frames (23%) remain "other": the first dip fires at a *sub-multiple* lag
  (26, 36, 48 samples for 200-240 Hz f0) whose CMND sits at 0.11-0.15. These are neither f0,
  2*f0 nor f0/2, so no octave rule can repair them; it is a separate defect class needing its
  own experiment.
* Under 10 dB noise 65 of 504 frames stay octave-up: the half-period dip becomes
  indistinguishable from the period dip. `estimate_f0` exposes no confidence, so the pairing
  pipeline cannot currently down-weight those frames.
* The two thresholds were chosen on these 2469 frames. AISHELL-3 has not been run (no corpus
  on disk, see docs/DATA_POLICY.md), so the choice is not validated on real speech.

## 结论 / 下次实验

The defect was real, its mechanism is understood (first-dip rule + even-harmonic dominance),
and the bounded search fixes it on every clean vowel case and on the noise set without
introducing the octave-down failure that the naive global-minimum fix caused. Tests added:
`tests/test_data_frontend.py::test_f0_does_not_read_a_harmonic_comb_one_octave_high`
(220/500, 240/500, 300/700, 2% band) and
`tests/test_data_frontend.py::test_f0_does_not_read_a_synthesized_a_one_octave_high`
(Mandarin /a/ at 220 Hz, 2% band).

Next experiment: give `estimate_f0` a confidence output (chosen-dip depth plus the margin to
the next-best dip) and measure how much of the residual 23% comb / 13% noisy error can be
*reported* instead of silently wrong, before considering a second estimator.

## 附录 E5 — 谐波锁定（P1.2 验收期间追加，2026-10-31）

P1.2 的 Mandarin 后端验收（EXP-0009）在 /a/ 220 Hz 上看到 414–455 Hz 的读数，频谱核对
（nperseg=1024）为 211–234 Hz，因此追加这组测量。这不是 v2 八度规则能处理的情形：首个
CMND<0.15 的谷落在 F1≈800 Hz 对应的 3f0/4f0 上，f0 本身从未被选中。

| f0 (Hz) | 默认 65–900 Hz 搜索结果 | fmax_hz=500 时 | 机制 |
| --- | --- | --- | --- |
| 210 | 842.0 | 210.5 | F1≈800 落在 4·f0 |
| 300 | 902.0 | 300.7 | F1≈800 落在 3·f0（且 3f0 进入 900 Hz 上限） |

- 记录形式：`tests/test_synth_mandarin.py::test_f0_estimator_locks_onto_a_high_harmonic_at_210_and_300_hz`，
  docstring 注明若该断言失败说明估计器变好，应更新本实验而不是删断言。
- 后果：真实数据管线不能预知 f0，不能用 `fmax_hz` 规避；EXP-0009 的 F0 网格用的是
  `fmax_hz=1.5*f0`，因此那组 ≤0.23% 的误差不代表默认可用的精度。
- 与 v2 修复的关系：v2 只在「首个谷已经是 f0 或 f0/2」时才能救回；谐波锁定是首个谷直接
  落在 3f0/4f0 上，属于另一类失败，未在本实验修复。

