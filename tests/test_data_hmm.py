"""Tests for the chain HMM used by forced alignment (P2).

These tests prove the dynamic programs agree with brute-force enumeration on small chains, that
Baum-Welch is monotone and never destroys parameters it has no data for, and that sufficient
statistics scatter correctly when a phone repeats. They say nothing about alignment accuracy on
real speech — that is measured with the synthetic-phone and corpus experiments.
"""

from __future__ import annotations

import numpy as np
import pytest

from vhe.data.hmm import ChainHmm, HmmAccumulator, kmeans_init, logsumexp


def make_hmm(n_states=3, dim=2, seed=0):
    rng = np.random.default_rng(seed)
    means = rng.normal(0.0, 2.0, size=(n_states, 1, dim))
    variances = rng.uniform(0.5, 1.5, size=(n_states, 1, dim))
    weights = np.ones((n_states, 1))
    return ChainHmm(means=means, variances=variances, weights=weights)


def enumerate_paths(total_frames, n_states):
    """Every non-decreasing path from state 0 to state n_states-1 (step 0 or +1)."""
    paths = []

    def walk(path):
        frame = len(path) - 1
        state = path[-1]
        if frame == total_frames - 1:
            if state == n_states - 1:
                paths.append(tuple(path))
            return
        for following in (state, state + 1):
            if following < n_states:
                walk(path + [following])

    walk([0])
    return paths


def path_logprob(state_ll, path):
    transition = np.log(0.5) * (len(path) - 1)
    return float(sum(state_ll[frame, state] for frame, state in enumerate(path)) + transition)


def test_forward_backward_matches_path_enumeration():
    hmm = make_hmm()
    frames = np.random.default_rng(1).normal(size=(5, 2))
    state_ll = hmm.state_loglik(frames)
    total, alpha, beta = hmm.forward_backward(state_ll)
    scores = np.array([path_logprob(state_ll, path) for path in enumerate_paths(5, 3)])
    assert total == pytest.approx(float(logsumexp(scores)), rel=1e-9)
    # States that cannot be reached yet (or can no longer reach the end) carry -inf on purpose.
    assert np.isfinite(alpha[0, 0]) and np.isfinite(beta[-1, -1])
    assert np.isneginf(alpha[0, 1:]).all()
    assert alpha[-1, -1] == pytest.approx(total)


def test_viterbi_matches_best_enumerated_path():
    hmm = make_hmm(seed=2)
    frames = np.random.default_rng(3).normal(size=(6, 2))
    state_ll = hmm.state_loglik(frames)
    score, path = hmm.viterbi(state_ll)
    scored = sorted(((path_logprob(state_ll, p), p) for p in enumerate_paths(6, 3)),
                    reverse=True)
    assert score == pytest.approx(scored[0][0], rel=1e-9)
    assert tuple(int(state) for state in path) == scored[0][1]


def test_posteriors_sum_to_one_per_frame():
    hmm = make_hmm(seed=4)
    frames = np.random.default_rng(5).normal(size=(12, 2))
    _total, gamma = hmm.occupancy(hmm.state_loglik(frames))
    posterior = np.exp(gamma)
    assert np.allclose(posterior.sum(axis=1), 1.0)
    assert (posterior >= 0).all()


def test_baum_welch_is_monotone():
    rng = np.random.default_rng(6)
    frames = np.concatenate([rng.normal(shift, 0.5, size=(40, 2))
                             for shift in (-3.0, 0.0, 3.0)])
    hmm = ChainHmm(means=np.array([[[-3.0, 0.0]], [[0.0, 0.0]], [[3.0, 0.0]]]),
                   variances=np.ones((3, 1, 2)), weights=np.ones((3, 1)))
    history = []
    for _ in range(4):
        accumulator = HmmAccumulator(3, 1, 2)
        history.append(accumulator.add(hmm, frames))
        hmm = hmm.update(accumulator.weights_sum, accumulator.mean_sum,
                         accumulator.var_sum, accumulator.occupancy_sum)
    assert all(later >= earlier - 1e-6 for earlier, later in zip(history, history[1:]))


def test_update_keeps_unvisited_states():
    hmm = make_hmm(seed=7)
    before = hmm.means[2].copy()
    weights_sum = np.array([[2.0], [2.0], [0.0]])
    mean_sum = np.array([[[1.0, 1.0]], [[2.0, 2.0]], [[0.0, 0.0]]])
    var_sum = np.array([[[3.0, 3.0]], [[6.0, 6.0]], [[0.0, 0.0]]])
    occupancy_sum = np.array([4.0, 4.0, 0.0])
    updated = hmm.update(weights_sum, mean_sum, var_sum, occupancy_sum)
    assert np.allclose(updated.means[2], before)
    assert np.allclose(updated.variances[2], hmm.variances[2])
    assert np.allclose(updated.means[0], [[0.5, 0.5]])


def test_accumulator_scatter_handles_repeated_states():
    hmm = make_hmm(seed=8)
    frames = np.random.default_rng(9).normal(size=(20, 2))
    accumulator = HmmAccumulator(3, 1, 2)
    accumulator.add(hmm, frames, state_indices=[0, 0])
    # Reference: the two-state sub-chain's own statistics. Both sub-states scatter into global
    # row 0, so the accumulator must hold their sum; an in-place add on the repeated index
    # would silently keep only the second write.
    sub = hmm.subset([0, 0])
    _total, gamma = sub.occupancy(sub.state_loglik(frames))
    posterior = np.exp(gamma)
    expected_weight = float(posterior.sum(axis=0).sum())
    expected_mean = (posterior[:, :, None] * frames[:, None, :]).sum(axis=(0, 1))
    assert accumulator.weights_sum[0, 0] == pytest.approx(expected_weight)
    assert accumulator.occupancy_sum[0] == pytest.approx(expected_weight)
    assert np.allclose(accumulator.mean_sum[0, 0], expected_mean)
    assert accumulator.frames == 20


def test_subset_preserves_order_and_parameters():
    hmm = make_hmm(seed=10)
    subset = hmm.subset([2, 0])
    assert np.allclose(subset.means[0], hmm.means[2])
    assert np.allclose(subset.means[1], hmm.means[0])
    assert subset.n_states == 2


def test_npz_roundtrip():
    hmm = ChainHmm(means=np.random.default_rng(11).normal(size=(3, 1, 2)),
                   variances=np.ones((3, 1, 2)), weights=np.ones((3, 1)),
                   global_variance=np.array([1.0, 2.0]))
    restored = ChainHmm.from_npz(hmm.to_npz())
    assert np.allclose(restored.means, hmm.means)
    assert np.allclose(restored.variance_floor_vector(), hmm.variance_floor_vector())


def test_kmeans_init_recovers_separated_clusters():
    rng = np.random.default_rng(12)
    frames = np.concatenate([rng.normal(-4.0, 0.2, size=(50, 1)),
                             rng.normal(4.0, 0.2, size=(50, 1))])
    weights, means, variances = kmeans_init(frames, 2, rng)
    assert weights.sum() == pytest.approx(1.0)
    assert np.allclose(np.sort(means[:, 0]), [-4.0, 4.0], atol=0.2)
    assert (variances > 0).all()
