"""LIBERO-NADE primitives: candidate-set construction + proposal math.

Three pure helpers, used by both ``test_model.py`` (suite-level NADE) and
``collect_buffer.py``:

* :func:`build_candidate_set` — pool + optional Gaussian-perturbed clones.
* :func:`build_proposal` — epsilon-greedy importance-sampling distribution::

      q(k) = (1 - eps) * p_fail[k]  +  eps / M

  (if every ``p_fail[k]`` is below ``criticality_threshold``, fall back to
  uniform — the IS weight is then identically 1.)
* :func:`importance_weight` — per-draw weight against the uniform reference
  ``p(k) = 1 / M`` so downstream weighted estimators stay unbiased.

The orchestration (env reset, init-state apply, settle) lives in
``test_model._apply_init`` and ``run_task_with_draws``. Suite-level sampling
lives in ``test_model._sample_suite_draws``.

Reference: other_source/criticality/test/maniskill_ordinary_nade.py
Key difference: this is **episode-level** (one IS weight per reset, not
per step), and the candidate set is task-specific rather than a fixed
force grid.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

_THIS = Path(__file__).resolve()
_CODE_ROOT = _THIS.parents[2]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from adversarial_training.utils.init_state import compute_perturb_indices


# ---------------------------------------------------------------------------
# Candidate set
# ---------------------------------------------------------------------------


@dataclass
class NADEConfig:
    epsilon: float = 0.01
    weight_clip: float = 100.0
    criticality_threshold: float = 0.5
    alpha: float = 3.0  # softmax temperature for proposal distribution
    # Candidate-set construction:
    pool_repeats: int = 1                     # use each pool entry this many times
    perturbations_per_pool: int = 4           # number of Gaussian-perturbed clones per pool entry
    perturb_fraction: Optional[float] = None  # None -> auto-detect object dims via pool variance
    perturb_std: float = 0.02


def build_candidate_set(
    pool: np.ndarray,
    config: NADEConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(candidates, source_modes, pool_indices)``.

    candidates    : (M, D)  float32
    source_modes  : (M,)    object array of 'pool' / 'perturb_pool'
    pool_indices  : (M,)    int32, the originating pool index for each candidate
    """
    pool = np.asarray(pool, dtype=np.float32)
    P, D = pool.shape

    cands: List[np.ndarray] = []
    modes: List[str] = []
    pidx: List[int] = []

    # 1) Plain pool entries.
    for _ in range(max(1, int(config.pool_repeats))):
        for i in range(P):
            cands.append(pool[i].copy())
            modes.append("pool")
            pidx.append(i)

    # 2) Perturbed clones.
    if config.perturbations_per_pool > 0:
        indices = compute_perturb_indices(pool, config.perturb_fraction)
        n_idx = int(indices.size)
        for i in range(P):
            for _ in range(int(config.perturbations_per_pool)):
                vec = pool[i].copy()
                noise = rng.normal(0.0, float(config.perturb_std), size=n_idx).astype(np.float32)
                vec[indices] = vec[indices] + noise
                cands.append(vec)
                modes.append("perturb_pool")
                pidx.append(i)

    cand_arr = np.stack(cands, axis=0).astype(np.float32)
    return cand_arr, np.asarray(modes, dtype=object), np.asarray(pidx, dtype=np.int32)


# ---------------------------------------------------------------------------
# Proposal distribution
# ---------------------------------------------------------------------------


def build_proposal(
    p_fail: np.ndarray,
    epsilon: float,
    criticality_threshold: float,
    alpha: float = 3.0,
) -> Tuple[np.ndarray, bool]:
    """Return ``(q, used_criticality)``.

    Falls back to uniform whenever every candidate is below the threshold,
    which makes the IS weight identically 1 and matches plain Monte Carlo
    on safe regions of the candidate set.
    """
    M = len(p_fail)
    if M == 0:
        return np.array([], dtype=np.float64), False

    if np.max(p_fail) < float(criticality_threshold):
        return np.full(M, 1.0 / M, dtype=np.float64), False
    else:
        # shifted = p_fail - np.max(p_fail)
        # p_eff = np.exp(shifted * alpha)
        p_eff = p_fail
        s = p_eff.sum()
        weighted = p_eff / s
        uniform = np.full(M, 1.0 / M, dtype=np.float64)
        q = (1.0 - float(epsilon)) * weighted + float(epsilon) * uniform
        q = q / q.sum()
        return q, True


def importance_weight(
    chosen_idx: int, q: np.ndarray, M: int, clip: float,
) -> float:
    """IS weight for one sample, assuming reference distribution is uniform on the candidate set."""
    p = 1.0 / float(M)
    w = p / float(q[chosen_idx])
    return float(np.clip(w, 0.0, float(clip)))
