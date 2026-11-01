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


def brute_force_transition_counts(state_ll, paths):
    """Expected stay/advance counts under uniform 0.5 transitions, from enumerated paths."""
    scores = np.array([path_logprob(state_ll, path) for path in paths])
    weights = np.exp(scores - float(logsumexp(scores)))
    n_states = state_ll.shape[1]
    stays = np.zeros(n_states)
    advances = np.zeros(n_states)
    for weight, path in zip(weights, paths):
        for here, following in zip(path, path[1:]):
            if here == following:
                stays[here] += weight
            else:
                advances[here] += weight
    return stays, advances


def test_transition_counts_match_path_enumeration():
    hmm = make_hmm(seed=13)
    frames = np.random.default_rng(14).normal(size=(5, 2))
    state_ll = hmm.state_loglik(frames)
    stays, advances = hmm.transition_counts(state_ll)
    expected_stays, expected_advances = brute_force_transition_counts(
        state_ll, enumerate_paths(5, 3))
    assert np.allclose(stays, expected_stays, atol=1e-9)
    assert np.allclose(advances, expected_advances, atol=1e-9)
    # The chain ends in its last state, so it can never leave it.
    assert advances[-1] == 0.0
    assert stays.sum() + advances.sum() == pytest.approx(4.0)


def test_transition_counts_reuse_given_alpha_beta():
    hmm = make_hmm(seed=15)
    state_ll = hmm.state_loglik(np.random.default_rng(16).normal(size=(7, 2)))
    total, alpha, beta = hmm.forward_backward(state_ll)
    cached = hmm.transition_counts(state_ll, alpha=alpha, beta=beta, total=total)
    recomputed = hmm.transition_counts(state_ll)
    assert np.allclose(cached[0], recomputed[0]) and np.allclose(cached[1], recomputed[1])


def test_stay_probability_is_learned_per_state():
    # Three well separated states, but the middle one only owns two frames: a scalar stay
    # probability would have to pay for that with the same value everywhere.
    rng = np.random.default_rng(17)
    frames = np.concatenate([rng.normal(-4.0, 0.3, size=(30, 1)),
                             rng.normal(0.0, 0.3, size=(2, 1)),
                             rng.normal(4.0, 0.3, size=(30, 1))])
    hmm = ChainHmm(means=np.array([[[-4.0]], [[0.0]], [[4.0]]]), variances=np.ones((3, 1, 1)),
                   weights=np.ones((3, 1)))
    for _ in range(3):
        accumulator = HmmAccumulator(3, 1, 1)
        accumulator.add(hmm, frames)
        hmm = hmm.update(accumulator.weights_sum, accumulator.mean_sum, accumulator.var_sum,
                         accumulator.occupancy_sum, accumulator.self_counts,
                         accumulator.exit_counts)
    stays = hmm.self_probabilities
    assert stays.shape == (3,)
    assert np.ptp(stays) > 0.2, stays
    # Longer occupation -> higher stay probability; expected duration is 1/(1-stay).
    assert stays[0] > stays[1] and stays[2] > stays[1]


def test_transition_estimation_stays_monotone():
    rng = np.random.default_rng(18)
    frames = np.concatenate([rng.normal(shift, 0.6, size=(25, 2)) for shift in (-3.0, 0.0, 3.0)])
    hmm = ChainHmm(means=np.array([[[-3.0, 0.0]], [[0.0, 0.0]], [[3.0, 0.0]]]),
                   variances=np.ones((3, 1, 2)), weights=np.ones((3, 1)))
    history = []
    for _ in range(4):
        accumulator = HmmAccumulator(3, 1, 2)
        history.append(accumulator.add(hmm, frames))
        hmm = hmm.update(accumulator.weights_sum, accumulator.mean_sum, accumulator.var_sum,
                         accumulator.occupancy_sum, accumulator.self_counts,
                         accumulator.exit_counts)
    assert all(later >= earlier - 1e-6 for earlier, later in zip(history, history[1:]))


def test_transition_update_leaves_untouched_states_alone():
    hmm = ChainHmm(means=np.zeros((3, 1, 2)), variances=np.ones((3, 1, 2)),
                   weights=np.ones((3, 1)), self_probability=np.array([0.5, 0.5, 0.9]))
    updated = hmm.update(np.full((3, 1), 2.0), np.zeros((3, 1, 2)), np.full((3, 1, 2), 6.0),
                         np.array([4.0, 4.0, 4.0]),
                         self_counts=np.array([30.0, 10.0, 0.0]),
                         exit_counts=np.array([10.0, 30.0, 0.0]))
    stays = updated.self_probabilities
    assert stays[0] == pytest.approx((30.0 + 2.0) / (40.0 + 4.0))
    assert stays[1] == pytest.approx((10.0 + 2.0) / (40.0 + 4.0))
    assert stays[2] == pytest.approx(0.9)  # never visited: previous value, not the prior


def test_transition_update_clamps_extreme_counts():
    hmm = ChainHmm(means=np.zeros((2, 1, 2)), variances=np.ones((2, 1, 2)),
                   weights=np.ones((2, 1)))
    updated = hmm.update(np.full((2, 1), 2.0), np.zeros((2, 1, 2)), np.full((2, 1, 2), 6.0),
                         np.array([4.0, 4.0]),
                         self_counts=np.array([1e6, 0.0]), exit_counts=np.array([0.0, 1e6]))
    stays = updated.self_probabilities
    assert stays[0] == pytest.approx(0.98) and stays[1] == pytest.approx(0.05)


def test_subset_and_npz_keep_per_state_probability():
    hmm = ChainHmm(means=np.zeros((3, 1, 2)), variances=np.ones((3, 1, 2)),
                   weights=np.ones((3, 1)), self_probability=np.array([0.2, 0.5, 0.9]))
    subset = hmm.subset([2, 0])
    assert np.allclose(subset.self_probabilities, [0.9, 0.2])
    restored = ChainHmm.from_npz(hmm.to_npz())
    assert np.allclose(restored.self_probabilities, [0.2, 0.5, 0.9])


def test_scalar_and_wrong_shape_stay_probability():
    hmm = ChainHmm(means=np.zeros((3, 1, 2)), variances=np.ones((3, 1, 2)),
                   weights=np.ones((3, 1)), self_probability=0.7)
    assert np.allclose(hmm.self_probabilities, 0.7)
    with pytest.raises(ValueError):
        ChainHmm(means=np.zeros((3, 1, 2)), variances=np.ones((3, 1, 2)),
                 weights=np.ones((3, 1)), self_probability=np.array([0.5, 0.5]))
    with pytest.raises(ValueError):
        ChainHmm(means=np.zeros((3, 1, 2)), variances=np.ones((3, 1, 2)),
                 weights=np.ones((3, 1)), self_probability=np.array([0.5, 0.5, 1.0]))



def test_accumulator_counts_chain_termination_as_exit():
    hmm = make_hmm(seed=19)
    frames = np.random.default_rng(20).normal(size=(12, 2))
    state_ll = hmm.state_loglik(frames)
    _stays, arcs = hmm.transition_counts(state_ll)
    assert arcs[-1] == 0.0  # no arc leaves the last state: it ends the chain
    accumulator = HmmAccumulator(3, 1, 2)
    accumulator.add(hmm, frames)
    assert accumulator.exit_counts[-1] >= 1.0  # the one termination event
    total = accumulator.self_counts.sum() + accumulator.exit_counts.sum()
    assert total == pytest.approx(12.0)

