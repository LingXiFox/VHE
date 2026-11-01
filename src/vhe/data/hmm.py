"""Left-to-right chain HMM with diagonal-GMM emissions (forced-alignment core, P2).

Topology: S states in one chain, each state may stay or advance by exactly one state. There
are no skip arcs and no phone-boundary arcs to special-case — a phone of p states is simply p
consecutive chain states, so the dynamic program is O(T*S) and a boundary is wherever the
Viterbi path crosses from one phone's block into the next.

Why this topology: an aligner only needs state posteriors and a best path. Skip arcs and
optional-silence arcs add likelihood on read speech but also invent paths the transcript does
not contain, and a wrong path here becomes a wrong *duration* in the mechanical render.

Transitions were originally fixed (stay 0.5 / advance 0.5) on the argument that 3-state phones
then expect ~60 ms, "the right order for Mandarin", so re-estimating them buys nothing. EXP-0021
falsified that: with fixed transitions every phoneme class collapsed into 0.05-0.135 s (vowel
over stop 1.43) because the prior, not the data, set the duration. Stay probabilities can now be
per state, and Baum-Welch re-estimates them from expected transition counts (EXP-0022).

A scalar stay probability remains the default: it is the flat-start control the ablation needs
and the value a chain falls back to when transitions are not being estimated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

LOG_TWO_PI = float(np.log(2.0 * np.pi))
DEFAULT_VARIANCE_FLOOR = 1e-3
DEFAULT_FLOOR_RATIO = 0.05
DEFAULT_SELF_PROBABILITY = 0.5
#: Beta(a, a) prior on each state's stay probability: a=2.0 pulls a barely visited state back to
#: 0.5 instead of letting two frames of evidence decide the duration model for a whole phone.
DEFAULT_TRANSITION_PRIOR = 2.0
#: Clamp for the re-estimated stay probability (1 frame/state is the physical floor of this
#: topology; 0.98 would make the chain unable to leave a state on the data it was trained on).
MIN_SELF_PROBABILITY = 0.05
MAX_SELF_PROBABILITY = 0.98


def logsumexp(values: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically safe log-sum-exp; a fully -inf slice stays -inf."""
    peak = np.max(values, axis=axis, keepdims=True)
    peak = np.where(np.isfinite(peak), peak, 0.0)
    summed = np.sum(np.exp(values - peak), axis=axis, keepdims=True)
    with np.errstate(divide="ignore"):
        return np.squeeze(peak + np.log(summed), axis=axis)


def _logsumexp_pair(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    peak = np.maximum(first, second)
    peak = np.where(np.isfinite(peak), peak, 0.0)
    with np.errstate(divide="ignore"):
        return peak + np.log(np.exp(first - peak) + np.exp(second - peak))


def kmeans_init(frames: np.ndarray, n_components: int, rng: np.random.Generator,
                iterations: int = 12, variance_floor: float = DEFAULT_VARIANCE_FLOOR
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Diagonal-GMM parameters for one state from Lloyd's algorithm.

    Deterministic given the rng: the first component is seeded from the frame farthest from
    the mean, the rest from distinct random frames, which keeps the classic "two centroids
    land on the same frame" collapse rare without k-means++ machinery.
    """
    data = np.asarray(frames, dtype=np.float64)
    if data.ndim != 2 or data.shape[0] == 0:
        raise ValueError("kmeans_init expects a non-empty [N, D] matrix")
    count, dim = data.shape
    n_components = max(1, min(int(n_components), count))
    if n_components == 1:
        mean = data.mean(axis=0)
        return (np.ones(1), mean[None, :], np.maximum(data.var(axis=0), variance_floor)[None, :])
    centre = data.mean(axis=0)
    seeds = [int(np.argmax(np.sum((data - centre) ** 2, axis=1)))]
    while len(seeds) < n_components:
        candidate = int(rng.integers(0, count))
        if candidate not in seeds:
            seeds.append(candidate)
    centroids = data[np.asarray(seeds)].copy()
    assignment = np.zeros(count, dtype=np.int64)
    for _ in range(iterations):
        distances = ((data[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        updated = np.argmin(distances, axis=1)
        if np.array_equal(updated, assignment) and _ > 0:
            assignment = updated
            break
        assignment = updated
        for index in range(n_components):
            members = data[assignment == index]
            if members.size:
                centroids[index] = members.mean(axis=0)
            else:  # an empty cluster keeps its previous centre but gets a jitter guard
                centroids[index] = centre + rng.normal(0.0, 0.01, size=dim)
    weights = np.zeros(n_components)
    means = np.zeros((n_components, dim))
    variances = np.zeros((n_components, dim))
    global_var = np.maximum(data.var(axis=0), variance_floor)
    for index in range(n_components):
        members = data[assignment == index]
        weights[index] = max(members.shape[0], 1) / count
        if members.shape[0] < 2:
            means[index] = centroids[index]
            variances[index] = global_var
        else:
            means[index] = members.mean(axis=0)
            variances[index] = np.maximum(members.var(axis=0), variance_floor)
    weights /= weights.sum()
    return weights, means, variances


@dataclass
class ChainHmm:
    """Emission parameters plus the fixed stay/advance transition table."""

    means: np.ndarray        # [S, K, D]
    variances: np.ndarray    # [S, K, D]
    weights: np.ndarray      # [S, K]
    #: Stay probability: one scalar for the whole chain or one value per state. Per-state values
    #: are what make the chain a duration model at all - a scalar forces every phone to expect
    #: 2 frames per state (60 ms for a 3-state phone) no matter what it sounds like.
    self_probability: Union[float, np.ndarray] = DEFAULT_SELF_PROBABILITY
    variance_floor: float = DEFAULT_VARIANCE_FLOOR
    #: Per-dimension variance of the whole feature set. The M-step floors each dimension at
    #: `floor_ratio` of it, not at an absolute constant: on whitened features one dimension can
    #: legitimately sit near zero inside a state, and an absolute floor lets it collapse into a
    #: spike that then penalises every frame the chain forces through it (observed: 28k nats lost
    #: on one utterance before this floor existed).
    global_variance: Optional[np.ndarray] = None
    floor_ratio: float = DEFAULT_FLOOR_RATIO

    def __post_init__(self) -> None:
        self.means = np.asarray(self.means, dtype=np.float64)
        self.variances = np.asarray(self.variances, dtype=np.float64)
        self.weights = np.asarray(self.weights, dtype=np.float64)
        if self.means.ndim != 3:
            raise ValueError("means must have shape [S, K, D]")
        if self.variances.shape != self.means.shape or self.weights.shape != self.means.shape[:2]:
            raise ValueError("variances [S,K,D] and weights [S,K] must match means")
        if np.isscalar(self.self_probability):
            if not 0.0 < float(self.self_probability) < 1.0:
                raise ValueError("self_probability must be in (0, 1)")
            self.self_probability = float(self.self_probability)
        else:
            stays = np.asarray(self.self_probability, dtype=np.float64)
            if stays.shape != (self.means.shape[0],):
                raise ValueError(f"self_probability must be a scalar or [{self.means.shape[0]}]")
            if not ((stays > 0.0) & (stays < 1.0)).all():
                raise ValueError("self_probability must be in (0, 1)")
            self.self_probability = stays
        if self.global_variance is not None:
            self.global_variance = np.asarray(self.global_variance, dtype=np.float64)
            if self.global_variance.shape != (self.dim,):
                raise ValueError("global_variance must have shape [D]")

    # ------------------------------------------------------------------ properties

    @property
    def n_states(self) -> int:
        return int(self.means.shape[0])

    @property
    def n_components(self) -> int:
        return int(self.means.shape[1])

    @property
    def dim(self) -> int:
        return int(self.means.shape[2])

    @property
    def self_probabilities(self) -> np.ndarray:
        """Stay probability per state, broadcast from a scalar if that is what was given."""
        return np.broadcast_to(np.asarray(self.self_probability, dtype=np.float64),
                               (self.n_states,))

    def transition_logs(self) -> tuple[np.ndarray, np.ndarray]:
        stays = self.self_probabilities
        log_self = np.log(stays)
        log_next = np.log1p(-stays)
        log_next = np.where(np.arange(self.n_states) == self.n_states - 1, -np.inf, log_next)
        return log_self, log_next  # the chain ends in its last state

    # ------------------------------------------------------------------ emissions

    def component_loglik(self, frames: np.ndarray) -> np.ndarray:
        """[T, S, K] log p(x_t | state s, component k) + log weight."""
        data = np.asarray(frames, dtype=np.float64)
        if data.ndim != 2 or data.shape[1] != self.dim:
            raise ValueError(f"frames must be [T, {self.dim}]")
        flat_var = self.variances.reshape(-1, self.dim)
        flat_mean = self.means.reshape(-1, self.dim)
        precision = 1.0 / flat_var
        quadratic = ((data ** 2) @ precision.T
                     - 2.0 * (data @ (flat_mean * precision).T)
                     + (flat_mean ** 2 * precision).sum(axis=1))
        log_det = np.log(flat_var).sum(axis=1)
        emission = -0.5 * (self.dim * LOG_TWO_PI + log_det + quadratic)
        with np.errstate(divide="ignore"):
            emission = emission + np.log(self.weights.reshape(-1))[None, :]
        return emission.reshape(data.shape[0], self.n_states, self.n_components)

    def state_loglik(self, frames: np.ndarray) -> np.ndarray:
        """[T, S] log p(x_t | state s), marginalised over components."""
        return logsumexp(self.component_loglik(frames), axis=2)

    # ------------------------------------------------------------------ inference

    def forward_backward(self, state_ll: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        """Return (total log P(x), alpha [T,S], beta [T,S]). State 0 starts the chain."""
        total_frames, n_states = state_ll.shape
        log_self, log_next = self.transition_logs()
        alpha = np.full((total_frames, n_states), -np.inf)
        alpha[0, 0] = state_ll[0, 0]
        for step in range(1, total_frames):
            previous = alpha[step - 1]
            stay = previous + log_self
            advance = np.concatenate(([-np.inf], previous[:-1] + log_next[:-1]))
            alpha[step] = _logsumexp_pair(stay, advance) + state_ll[step]
        total = float(alpha[-1, -1])
        beta = np.full((total_frames, n_states), -np.inf)
        beta[-1, -1] = 0.0
        for step in range(total_frames - 2, -1, -1):
            following = beta[step + 1] + state_ll[step + 1]
            stay = log_self + following
            advance = np.append(log_next[:-1] + following[1:], -np.inf)
            beta[step] = _logsumexp_pair(stay, advance)
        return total, alpha, beta

    def occupancy(self, state_ll: np.ndarray) -> tuple[float, np.ndarray]:
        total, alpha, beta = self.forward_backward(state_ll)
        return total, alpha + beta - total

    def viterbi(self, state_ll: np.ndarray) -> tuple[float, np.ndarray]:
        """Return (score, path[T]) for the single best state sequence."""
        total_frames, n_states = state_ll.shape
        log_self, log_next = self.transition_logs()
        delta = np.full((total_frames, n_states), -np.inf)
        back = np.zeros((total_frames, n_states), dtype=np.int8)  # 0 stay, 1 advance
        delta[0, 0] = state_ll[0, 0]
        for step in range(1, total_frames):
            previous = delta[step - 1]
            stay = previous + log_self
            advance = np.concatenate(([-np.inf], previous[:-1] + log_next[:-1]))
            take_advance = advance > stay
            delta[step] = np.where(take_advance, advance, stay) + state_ll[step]
            back[step] = take_advance.astype(np.int8)
        path = np.zeros(total_frames, dtype=np.int64)
        path[-1] = n_states - 1
        for step in range(total_frames - 1, 0, -1):
            path[step - 1] = path[step] - int(back[step, path[step]])
        return float(delta[-1, -1]), path

    def transition_counts(self, state_ll: np.ndarray, *, alpha: Optional[np.ndarray] = None,
                          beta: Optional[np.ndarray] = None, total: Optional[float] = None
                          ) -> tuple[np.ndarray, np.ndarray]:
        """Expected stay and advance counts per state, [S] each.

        xi_t(i, i) = alpha_t(i) * a_ii * p(x_{t+1} | i) * beta_{t+1}(i) / P(x) and likewise for
        the advance arc i -> i+1, so the loop is over frames with all states vectorised. Reusing
        the alpha/beta of an occupancy call keeps one forward-backward per utterance.

        An arc is not one row of a single array: the advance arc i -> i+1 takes alpha from state
        i at frame t, but its emission and beta from state i+1 at frame t+1. Using one shifted
        row for all three terms credits the wrong source state and splits the transition mass
        across the wrong arcs - the totals still sum to T-1, so only an exact count exposes it.

        The last state never advances, so its exit rate is not observable from these arcs alone;
        HmmAccumulator adds the one termination event that ends each chain (see add()).
        """
        if alpha is None or beta is None or total is None:
            total, alpha, beta = self.forward_backward(state_ll)
        log_self, log_next = self.transition_logs()
        stays = np.zeros(self.n_states)
        advances = np.zeros(self.n_states)
        with np.errstate(under="ignore"):
            for step in range(state_ll.shape[0] - 1):
                base = alpha[step] + state_ll[step + 1] + beta[step + 1] - total
                stays += np.exp(base + log_self)
                # Arc i -> i+1: alpha from state i, emission and beta from state i+1.
                advance_base = (alpha[step, :-1] + state_ll[step + 1, 1:]
                                + beta[step + 1, 1:] - total)
                advances[:-1] += np.exp(advance_base + log_next[:-1])
        return stays, advances

    # ------------------------------------------------------------------ training

    def variance_floor_vector(self) -> np.ndarray:
        """Per-dimension floor applied by the M-step."""
        if self.global_variance is None:
            return np.full(self.dim, max(self.variance_floor, self.floor_ratio))
        return np.maximum(self.variance_floor, self.floor_ratio * self.global_variance)

    def subset(self, indices) -> "ChainHmm":
        """Copy of this chain restricted to a subset of states, order preserved.

        Alignment uses it to run a chain over only the states one utterance needs: the full
        vocabulary chain has more states than a short utterance has frames.
        """
        picked = np.asarray(list(indices), dtype=np.int64)
        return ChainHmm(means=self.means[picked], variances=self.variances[picked],
                        weights=self.weights[picked],
                        self_probability=self.self_probabilities[picked].copy(),
                        variance_floor=self.variance_floor,
                        global_variance=self.global_variance, floor_ratio=self.floor_ratio)

    def update(self, weights_sum: np.ndarray, mean_sum: np.ndarray, var_sum: np.ndarray,
               occupancy_sum: np.ndarray, self_counts: Optional[np.ndarray] = None,
               exit_counts: Optional[np.ndarray] = None,
               transition_prior: float = DEFAULT_TRANSITION_PRIOR) -> "ChainHmm":
        """One Baum-Welch M-step from sufficient statistics accumulated by HmmAccumulator.

        When expected transition counts are given the stay probabilities are re-estimated with a
        Beta(prior, prior) smoothing; otherwise the current ones are carried over unchanged.
        """
        safe_occ = np.maximum(occupancy_sum, 1e-8)[:, None]
        weights = np.maximum(weights_sum, 1e-8) / safe_occ
        weights /= weights.sum(axis=1, keepdims=True)
        safe_weight = np.maximum(weights_sum, 1e-8)[:, :, None]
        means = mean_sum / safe_weight
        variances = np.maximum(var_sum / safe_weight - means ** 2,
                               self.variance_floor_vector()[None, None, :])
        # A state or component the corpus never visited has no statistics: keep what flat start
        # gave it instead of writing zeros (mean 0, variance = floor), which would make it a
        # spike that swallows every frame once it is finally used.
        seen = occupancy_sum > 1e-6
        seen_components = weights_sum > 1e-6
        means = np.where(seen[:, None, None], means, self.means)
        variances = np.where(seen[:, None, None], variances, self.variances)
        weights = np.where(seen_components, weights, self.weights)
        weights /= weights.sum(axis=1, keepdims=True)
        stays: Union[float, np.ndarray] = self.self_probability
        if self_counts is not None and exit_counts is not None:
            numerator = np.asarray(self_counts) + transition_prior
            denominator = np.asarray(self_counts) + np.asarray(exit_counts) + 2.0 * transition_prior
            estimated = np.clip(numerator / denominator, MIN_SELF_PROBABILITY,
                                MAX_SELF_PROBABILITY)
            # A state the corpus never left or entered keeps its previous value: (0 + a) / (0 + 2a)
            # would silently reset an untouched state to the prior instead of leaving it alone.
            touched = (np.asarray(self_counts) + np.asarray(exit_counts)) > 1e-6
            stays = np.where(touched, estimated, self.self_probabilities)
        return ChainHmm(means=means, variances=variances, weights=weights,
                        self_probability=stays,
                        variance_floor=self.variance_floor,
                        global_variance=self.global_variance, floor_ratio=self.floor_ratio)

    # ------------------------------------------------------------------ persistence

    def to_npz(self) -> dict[str, np.ndarray]:
        payload = {"means": self.means, "variances": self.variances, "weights": self.weights,
                   "self_probability": np.asarray(self.self_probability, dtype=np.float64),
                   "variance_floor": np.asarray(self.variance_floor, dtype=np.float64),
                   "floor_ratio": np.asarray(self.floor_ratio, dtype=np.float64)}
        if self.global_variance is not None:
            payload["global_variance"] = self.global_variance
        return payload

    @classmethod
    def from_npz(cls, data) -> "ChainHmm":
        stays = np.asarray(data["self_probability"], dtype=np.float64)
        return cls(means=data["means"], variances=data["variances"], weights=data["weights"],
                   self_probability=float(stays) if stays.ndim == 0 else stays,
                   variance_floor=float(data["variance_floor"]),
                   global_variance=(np.asarray(data["global_variance"])
                                    if "global_variance" in data else None),
                   floor_ratio=float(data["floor_ratio"]) if "floor_ratio" in data else 0.05)


class HmmAccumulator:
    """Sufficient statistics for one Baum-Welch pass; never holds the whole corpus."""

    def __init__(self, n_states: int, n_components: int, dim: int) -> None:
        self.weights_sum = np.zeros((n_states, n_components))
        self.mean_sum = np.zeros((n_states, n_components, dim))
        self.var_sum = np.zeros((n_states, n_components, dim))
        self.occupancy_sum = np.zeros(n_states)
        self.self_counts = np.zeros(n_states)
        self.exit_counts = np.zeros(n_states)
        self.frames = 0
        self.loglik = 0.0

    def add(self, hmm: ChainHmm, frames: np.ndarray, state_indices=None) -> float:
        """Run E-step on one utterance and accumulate; returns its log likelihood.

        `state_indices` restricts the chain to the states this utterance visits (the phone
        sequence it was transcribed with) and scatters the statistics back to their global
        rows, so a 60-frame utterance is not asked to walk all 100+ vocabulary states.
        """
        indices = None
        if state_indices is not None:
            indices = np.asarray(list(state_indices), dtype=np.int64)
            chain = hmm.subset(indices)
        else:
            chain = hmm
        component_ll = chain.component_loglik(frames)
        state_ll = logsumexp(component_ll, axis=2)
        # One forward-backward feeds both the occupancy and the transition counts.
        total, alpha, beta = chain.forward_backward(state_ll)
        gamma = np.exp(alpha + beta - total)
        stays, advances = chain.transition_counts(state_ll, alpha=alpha, beta=beta, total=total)
        # The chain ends in its last state, and ending is that state's one exit event. Without it
        # a phone that finishes the utterance looks like it never leaves, its stay probability
        # saturates, and its implied duration (1/(1-p) per state) becomes nonsense.
        advances[-1] += 1.0
        # p(k | t, s) = w_k N(x|.) / sum_k w_k N(x|.)
        component_post = np.exp(component_ll - state_ll[:, :, None])
        weighted = gamma[:, :, None] * component_post
        if indices is None:
            self.weights_sum += weighted.sum(axis=0)
            self.mean_sum += np.einsum("tsk,td->skd", weighted, frames)
            self.var_sum += np.einsum("tsk,td->skd", weighted, frames ** 2)
            self.occupancy_sum += gamma.sum(axis=0)
            self.self_counts += stays
            self.exit_counts += advances
        else:
            # np.add.at, not `+= indexed`: a phone may repeat inside one transcript
            # (e.g. 是是), which repeats its state ids and makes buffered fancy indexing drop
            # all but the last write.
            np.add.at(self.weights_sum, indices, weighted.sum(axis=0))
            np.add.at(self.mean_sum, indices, np.einsum("tsk,td->skd", weighted, frames))
            np.add.at(self.var_sum, indices, np.einsum("tsk,td->skd", weighted, frames ** 2))
            np.add.at(self.occupancy_sum, indices, gamma.sum(axis=0))
            np.add.at(self.self_counts, indices, stays)
            np.add.at(self.exit_counts, indices, advances)
        self.frames += frames.shape[0]
        self.loglik += total
        return total
