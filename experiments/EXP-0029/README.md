# EXP-0029 — Analysis-resynthesis mechanical source

## Why

EXP-0023's listening check failed: the owner heard "unintelligible, with electronic
noise" on the formant synthesiser's output. Objective probes agreed — the /f/
centroid sits above /sh/ (EXP-0010), the mechanical spectra are flatter than human
speech (0.023 against 0.051), and syllable joins click (16 samples jumping more
than 0.3 in one utterance). Training a humaniser on a source nobody can understand
would teach it to map an unidentifiable target.

So the mechanical source is replaced, not patched: `vhe.data.robotize` builds the
mechanical signal from the reference recording by

1. estimating F0 and rendering a pulse-train excitation (noise in unvoiced frames),
2. flattening the reference magnitude onto a small number of mel bands, and
3. imposing that band envelope on the excitation's phase, then overlap-adding.

AGENTS.md permits analysis of the human recording for pair construction; the model
still sees only `x_mechanical`. The consequence has to be stated plainly: because
the coarse envelope comes from the reference, the pair is closer than a pair from a
formant synthesiser. The learning problem becomes "restore natural excitation and
spectral detail", not "synthesise speech from a phoneme plan". Numbers below make
that distance explicit.

## Pre-registered criteria (written before the runs)

1. Mean log-mel distance between real and mechanical < 2.5 — the formant source
   scored 5.59, so the bar is "clearly closer while not identical".
2. Fundamental preserved within 5 %, measured by autocorrelation, not by
   `estimate_f0` (see the caveat below).
3. Energy above 6 kHz at least half the reference's — fricatives must not be lost.
4. Listening: the owner can make out the syllables. **Pending.**

## Data and environment

- AISHELL-3 manifest v4 (`sha256 1a8138c4d13e547d1d266a7aaf50d3a002a8e32dad78b0555f1b417c77074f31`),
  split `dev`, renderable utterances, seed 42.
- Renders written outside Git, under
  `/Volumes/LingXiWorkspace/VHE-Datasets/derived/aishell3-listen/`.
- Analysis on the Mac (CPU, NumPy); no model training involved.

## Runs

| run | bands | envelope | pairs | log-mel | real / robot HF >6 kHz | real / robot flatness | robot periodicity |
|---|---|---|---|---|---|---|---|
| 1 | 24 | filterbank weights | 6 | 1.662 | 0.015 / **0.001** | 0.401 / 0.313 | 0.356 |
| 2 | 24 | per-bin band mean | 6 | **0.933** | 0.015 / 0.024 | 0.401 / 0.474 | 0.356 |
| 3 | 12 | per-bin band mean | 3 | 1.546 | 0.008 / 0.019 | 0.390 / 0.494 | 0.229 |

Run 3 was tried as the default because its lower periodicity looked more mechanical.
It was rejected: on a synthetic 220 Hz vowel the 12-band envelope collapses the first
five harmonics into one band and the output's measured fundamental drops to
**110.6 Hz** (period doubling), with loudness 2.6 dB below the reference. The default
stays at 24 bands, which passes the fundamental and loudness tests;
`tests/test_data_robotize.py` pins that behaviour.

Runs 1 and 2 match the reference loudness exactly (RMS −32.0 dB on both sides) and
keep the length sample for sample; run 3 is 2.6 dB quiet and is not used.

Criterion 1 passes in runs 2 and 3 (0.933 / 1.546 against 2.5). Criterion 2 holds by
autocorrelation: on three utterances the mechanical fundamental was 240.0 / 242.4 /
242.4 Hz against 244.9 / 252.6 / 247.4 Hz in the reference (2 % low), and a synthetic
220 Hz vowel comes back at 230.8 Hz, inside the integer-lag quantisation. Criterion 3
fails in run 1 (0.1 % of the energy above 6 kHz against 1.5 %) and passes in runs 2
and 3 (2.4 % / 1.9 %). Criterion 4 is open.

## Failures and bugs found on the way

1. **Edge amplification (fixed).** The first overlap-add divided by the window
   square sum without padding, so the first and last frames were covered by one
   window each and the normaliser collapsed: a signal peaking at 4.6 came out
   peaking at **86.7**. Fixed by padding `n_fft` samples on both sides; regression
   test `test_robotize_does_not_amplify_the_edges`.
2. **High-band collapse (fixed, run 1).** Mapping band energies back through the
   filterbank weights shrank wide bands: energy above 6 kHz fell to 0.1 % against
   1.5 % in the reference. Fixed by assigning each bin to its strongest band and
   using the per-frame band mean; run 1 is kept as the failing record.
3. **Ambiguous pitch metric (fixed in the probe).** Taking the argmax of the
   autocorrelation reports half the fundamental for a periodic signal, because
   twice the period scores just as high and integer lag quantisation decides. The
   test and probe now take the first lag above 90 % of the maximum.
4. **`estimate_f0` misreads the mechanical signal.** It reports 89–126 Hz where the
   autocorrelation says 240 Hz: the buzzy pulse train makes a subharmonic win. This
   does not affect the pipeline (F0 is always estimated on the human recording) but
   it means `median_f0_hz` in these reports must not be read as a pitch error.
5. **Process violation.** As in EXP-0024, the probe was written and run before the
   metadata existed; this file was written afterwards. Recorded here rather than
   hidden.

## What this does not show

- Nothing about intelligibility or "does it sound mechanical": that is criterion 4
  and needs the owner's ears.
- The distance is small (0.93–1.55 log-mel), so a trained model may reach a low
  loss by copying the envelope. Any later claim about humanisation must show the
  model changes excitation, not only magnitudes.

## Listening material

- `EXP-0029-b12/ab/` — 3 files (default setting, bands = 12).
- `EXP-0029-robotize-v2/ab/` — 6 files (bands = 24, closer to the reference).
- Each A/B file is real speech, 0.7 s silence, then the mechanical copy.
