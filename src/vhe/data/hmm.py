"""Left-to-right chain HMM with diagonal-GMM emissions (forced-alignment core, P2).

Topology: S states in one chain, each state may stay or advance by exactly one state. There
are no skip arcs and no phone-boundary arcs to special-case — a phone of p states is simply p
consecutive chain states, so the dynamic program is O(T*S) and a boundary is wherever the
Viterbi path crosses from one phone's block into the next.

Why this topology: an aligner only needs state posteriors and a best path. Skip arcs and
optional-silence arcs add likelihood on read speech but also invent paths the transcript does
not contain, and a wrong path here becomes a wrong *duration* in the mechanical render.

Transition probabilities are fixed (stay 0.5 / advance 0.5): with 10 ms frames and 3-state
phones this puts the expected phone duration near 60 ms, which is the right order for
Mandarin, and re-estimating three million transition counts per speaker buys nothing at
boundary level. Everything else (weights, means, variances) is learned by Baum-Welch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

LOG_TWO_PI = float(np.log(2.0 * np.pi))
DEFAULT_VARIANCE_FLOOR = 1e-3
DEFAULT_FLOOR_RATIO = 0.05
DEFAULT_SELF_PROBABILITY = 0.5


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
    self_probability: float = DEFAULT_SELF_PROBABILITY
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
        if not 0.0 < self.self_probability < 1.0:
            raise ValueError("self_probability must be in (0, 1)")
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

    def transition_logs(self) -> tuple[np.ndarray, np.ndarray]:
        log_self = np.full(self.n_states, np.log(self.self_probability))
        log_next = np.full(self.n_states, np.log(1.0 - self.self_probability))
        log_next[-1] = -np.inf  # the chain ends in its last state
        return log_self, log_next

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
                        weights=self.weights[picked], self_probability=self.self_probability,
                        variance_floor=self.variance_floor,
                        global_variance=self.global_variance, floor_ratio=self.floor_ratio)

    def update(self, weights_sum: np.ndarray, mean_sum: np.ndarray, var_sum: np.ndarray,
               occupancy_sum: np.ndarray) -> "ChainHmm":
        """One Baum-Welch M-step from sufficient statistics accumulated by HmmAccumulator."""
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
        return ChainHmm(means=means, variances=variances, weights=weights,
                        self_probability=self.self_probability,
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
        return cls(means=data["means"], variances=data["variances"], weights=data["weights"],
                   self_probability=float(data["self_probability"]),
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
        total, gamma = chain.occupancy(state_ll)
        gamma = np.exp(gamma)
        # p(k | t, s) = w_k N(x|.) / sum_k w_k N(x|.)
        component_post = np.exp(component_ll - state_ll[:, :, None])
        weighted = gamma[:, :, None] * component_post
        if indices is None:
            self.weights_sum += weighted.sum(axis=0)
            self.mean_sum += np.einsum("tsk,td->skd", weighted, frames)
            self.var_sum += np.einsum("tsk,td->skd", weighted, frames ** 2)
            self.occupancy_sum += gamma.sum(axis=0)
        else:
            # np.add.at, not `+= indexed`: a phone may repeat inside one transcript
            # (e.g. 是是), which repeats its state ids and makes buffered fancy indexing drop
            # all but the last write.
            np.add.at(self.weights_sum, indices, weighted.sum(axis=0))
            np.add.at(self.mean_sum, indices, np.einsum("tsk,td->skd", weighted, frames))
            np.add.at(self.var_sum, indices, np.einsum("tsk,td->skd", weighted, frames ** 2))
            np.add.at(self.occupancy_sum, indices, gamma.sum(axis=0))
        self.frames += frames.shape[0]
        self.loglik += total
        return total
