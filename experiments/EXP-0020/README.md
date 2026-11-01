# EXP-0020 — Synthetic validation of the chain-HMM forced aligner

Status: synthetic validation only. **No corpus audio, no training of VHE, no listening.**

## 假设
A flat-start Baum-Welch chain HMM with 3 states per phoneme recovers phone boundaries on
spectrally separable synthetic phones to within 2 frames (20 ms) at a 10 ms hop, and Baum-Welch
iterations improve on the flat-start segmentation.

The second half of that claim is **not** supported (see 实际结果 3): on this fixture flat start is
already as good as, or slightly better than, the EM-trained model. The record keeps the two
halves separate instead of restating the hypothesis after the fact.

## 变量 / 对照组 / 数据划分
- Synthetic speaker: six band-limited noises, a=400-900, i=1800-2600, s=4000-6000,
  sh=2500-3200, t=900-1400, n=200-350 Hz; 100-200 ms each, 10 ms raised-cosine edges, 150 ms
  silence padding on both sides. There is no F0, no formant transition, no coarticulation and no
  noise: the only boundary cue is the change of band.
- Train: 30 utterances of 3-5 phones, seeds 1000..1029 (`rng(11)` chooses the phone sequences).
- Evaluate: 8 held-out utterances of 4 **distinct** phones (seeds 2000..2007). Adjacent identical
  phones are excluded because their internal boundary has no acoustic cue and any split is
  equally valid.
- Independent variables: number of Gaussian components (1, 2) x Baum-Welch iterations (0, 1, 4).
  `iterations=0` is the flat-start control: parameters come from the frame-pooling initialisation
  only.
- Fixed: 3 states per phone, hop 10 ms, 16 kHz, frame 25 ms, 13 MFCC + delta + delta-delta with
  per-utterance CMVN, seed 42 for k-means and flat start.
- Metric: absolute error of every interior phone boundary (seconds), maximised per utterance then
  across utterances; plus the minimum Viterbi span confidence.

## 完整复现命令 / 环境与哈希
```bash
python experiments/EXP-0020/probe_aligner_synthetic.py \
    --out experiments/EXP-0020/logs/aligner_synthetic.json
```
- environment: macOS arm64 (Apple Silicon), Python 3.14.6, CPU only, pure NumPy. No torch.
- runtime: 6 configurations x (30 utterances x 0-4 EM passes), a few seconds.
- log: `logs/aligner_synthetic.json` (all per-utterance errors).
- code: `src/vhe/data/{features,hmm,aligner}.py`; regression guard
  `tests/test_data_{features,hmm,aligner}.py` (30 cases).

## 预注册通过/失败条件
The hypothesis text was written by `tools/new_experiment.py` before this run. The numeric gate
below (20 ms) came from an earlier interactive probe of the same fixture, so it is a
**regression threshold**, not an independent pre-registration; it is labelled as such.

- supported if max boundary error < 0.020 s on all held-out utterances and every training run is
  monotone in likelihood.
- the EM half is supported only if 4 iterations beat the flat-start control on mean boundary
  error.

## 实际结果与主观听音
| components | iterations | monotone | final loglik | max err (s) | mean err (s) | min confidence |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0 (flat start) | yes | — | 0.0120 | 0.0059 | 0.937 |
| 1 | 1 | yes | 16378.3 | 0.0117 | 0.0067 | 0.978 |
| 1 | 4 | yes | 25428.9 | 0.0118 | 0.0097 | 0.957 |
| 2 | 0 (flat start) | yes | — | 0.0260 | 0.0122 | 0.899 |
| 2 | 1 | yes | 20481.5 | 0.0260 | 0.0124 | 0.885 |
| 2 | 4 | yes | 29602.6 | 0.0260 | 0.0155 | 0.947 |

1. **Boundary recovery holds.** One component reaches 12 ms maximum error, i.e. just over one
   10 ms frame, and the energy gate crops 15 frames of padding to (14, 15) frames. The
   pre-registered 20 ms gate passes on every utterance.
2. **Two components are worse**, not better: 26 ms maximum and a systematically higher mean. With
   30 utterances the extra component mostly fits speaker-specific sub-bands, and the state
   posterior becomes less decisive (confidence 0.899 vs 0.937 on flat start). This is the reason
   `DEFAULT_COMPONENTS = 1`.
3. **Baum-Welch does not improve boundaries here.** Likelihood rises monotonically
   (16378 -> 25429 nats) while the mean boundary error drifts *up* from 5.9 ms to 9.7 ms. EM is
   maximising the likelihood of the frame-to-state assignment it starts from; on spectrally
   disjoint phones the flat split is already at the optimum, so the extra iterations only sharpen
   the transitions. The hypothesis half that predicted an EM gain is therefore **falsified on
   this fixture** — which also means this fixture cannot be used to choose iteration counts.
4. **The fixture is too easy to be discriminative.** Success at 12 ms comes mostly from the
   pooling initialisation, not from the model. Real Mandarin has overlapping spectra
   (t/k vs a/i), coarticulation across every boundary and channel noise, so the corpus experiment
   is the only evidence that matters for P2.

### 主观听音
None. No audio was rendered and no naturalness claim is made.

## 失败、负面发现、错误日志
Development of the aligner produced several real bugs before this run; each is recorded because
the fix is now load-bearing:

1. **State-id scope.** Flat start built a per-utterance layout (local ids) and then indexed
   vocabulary-wide sample arrays with it, so initial parameters were assigned to the wrong phones.
   Fixed by always using the vocabulary layout; regression test
   `test_state_layout_offsets_are_global`.
2. **Repeated phone overwrote statistics.** `weights_sum[indices] += ...` with a repeated index
   (是是) keeps only the last buffered write; Baum-Welch was non-monotone (K=1: -2372 -> -3415).
   Fixed with `np.add.at`; regression test `test_accumulator_scatter_handles_repeated_states`.
3. **Variance floor too small.** An absolute 1e-3 floor let one dimension collapse into a spike,
   costing ~500 nats per frame whenever the chain was forced through it (one utterance: -28053).
   Fixed by flooring at `max(1e-3, 0.05 * global_variance[dim])`.
4. **Unvisited states written as zeros.** The M-step wrote mean 0 / variance floor for states the
   corpus never visited, creating a spike that swallowed frames once used. Fixed by masking with
   the occupancy sum; regression test `test_update_keeps_unvisited_states`.
5. **`iterations=0` silently ran one iteration** (`range(max(1, iterations))`), so the flat-start
   control was impossible. Fixed to `range(max(0, iterations))`; the control row above only
   exists because of that fix.
6. Two adjacent identical phones are unidentifiable by construction; an earlier version of the
   test scored them and reported 182 ms error. The fixture now excludes them and the limitation
   is documented in `tests/test_data_aligner.py`.

## 结论 / 下次实验
The aligner's mechanics are correct and reproduce known boundaries on separable synthetic
phones, but this experiment cannot rank training configurations and is not evidence about real
speech. Two consequences for P2:

1. The real-corpus run (a few hundred AISHELL-3 utterances, 4 EM passes, held-out speakers) is
   the experiment that decides whether this aligner is usable; it must report agreement with the
   energy-valley segmentation, per-phoneme duration distributions and confidence, not just
   likelihood.
2. Because EM did not improve boundaries here, the corpus run must include the flat-start control
   too. If EM does not help there either, the cheaper model wins and the iteration count stays at
   0-1.

Next: EXP-0021, corpus subset alignment on the WSL2/CUDA machine.
