"""LIBERO initial-state sampling, representation, and (de)serialization.

The criticality model and the accelerated-testing sampler treat an init
state as a fixed-length vector. The natural representation is LIBERO's
own flat MuJoCo state — exactly what ``env.set_init_state`` consumes — so
that round-tripping is trivial.

Three sampling modes are supported:
    * ``pool``         — uniform draw over the LIBERO-shipped 50-init pool.
    * ``generated``    — uniform draw over a pre-generated *extended* pool
                         created by ``utils/generate_inits.py`` (re-seeded
                         ``env.reset()`` calls). Same source distribution as
                         the official pool, but disjoint vectors — used as
                         the training-data pool to keep train/test
                         non-overlapping.
    * ``perturb_pool`` — pool sample plus Gaussian noise on a configurable
                         subset of state-vector dims (legacy; not the
                         default any more). Dim selection is variance-based
                         by default: any dim whose variance across the pool
                         is nonzero is "an object dim".

Heavier strategies (e.g. learned proposal distributions) can be added later
by subclassing :class:`BaseInitSampler`.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class InitState:
    """One sampled LIBERO initial state."""

    suite_name: str
    task_index: int
    vector: np.ndarray              # the float32 vector handed to env.set_init_state
    source_mode: str = "pool"       # 'pool' | 'perturb_pool' | ...
    pool_index: Optional[int] = None
    perturbation: Optional[np.ndarray] = None  # delta applied on top of the pool sample

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suite_name": self.suite_name,
            "task_index": int(self.task_index),
            "vector": np.asarray(self.vector, dtype=np.float32),
            "source_mode": self.source_mode,
            "pool_index": -1 if self.pool_index is None else int(self.pool_index),
            "perturbation":
                np.asarray(self.perturbation, dtype=np.float32) if self.perturbation is not None
                else np.zeros_like(self.vector, dtype=np.float32),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "InitState":
        pool_index = int(d["pool_index"])
        pert = np.asarray(d["perturbation"]) if "perturbation" in d else None
        if pert is not None and not np.any(pert):
            pert = None
        return cls(
            suite_name=str(d["suite_name"]),
            task_index=int(d["task_index"]),
            vector=np.asarray(d["vector"], dtype=np.float32),
            source_mode=str(d.get("source_mode", "pool")),
            pool_index=pool_index if pool_index >= 0 else None,
            perturbation=pert,
        )


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------


class BaseInitSampler:
    """Subclass to add new sampling strategies."""

    def __init__(self, suite_name: str, task_index: int, pool: np.ndarray):
        self.suite_name = suite_name
        self.task_index = task_index
        # LIBERO returns either np.ndarray (N, D) or torch.Tensor — coerce.
        self.pool = np.asarray(pool, dtype=np.float32)

    def sample(self, rng: np.random.Generator) -> InitState:
        raise NotImplementedError


class PoolSampler(BaseInitSampler):
    """Cycle through (or uniformly draw from) the LIBERO-provided pool."""

    def sample(self, rng: np.random.Generator) -> InitState:
        idx = int(rng.integers(0, len(self.pool)))
        return InitState(
            suite_name=self.suite_name,
            task_index=self.task_index,
            vector=self.pool[idx].copy(),
            source_mode="pool",
            pool_index=idx,
        )


class GeneratedPoolSampler(BaseInitSampler):
    """Uniform draw over a pre-generated extended init pool.

    Same distribution as LIBERO's shipped 50-init pool — both come from
    ``env.reset()`` with different seeds — but the concrete vectors are
    disjoint. This is the recommended training-data sampler when the
    official pool is reserved for SOTA-comparable evaluation.
    """

    def sample(self, rng: np.random.Generator) -> InitState:
        idx = int(rng.integers(0, len(self.pool)))
        return InitState(
            suite_name=self.suite_name,
            task_index=self.task_index,
            vector=self.pool[idx].copy(),
            source_mode="generated",
            pool_index=idx,
        )


def compute_perturb_indices(
    pool: np.ndarray,
    perturb_fraction: Optional[float] = None,
    variance_threshold: float = 1e-8,
) -> np.ndarray:
    """Pick which state-vector dims to perturb.

    * ``perturb_fraction is None`` (default): variance-based detection. Any
      dim with nonzero variance across the pool is treated as an "object
      dim" — robot dims share the home pose across every pool entry and so
      have zero variance, which means this isolates per-object qpos/qvel
      exactly, without hard-coding the Franka layout.
    * ``perturb_fraction`` set explicitly: legacy "tail fraction" mode —
      perturb the last ``D * fraction`` dims contiguously.
    """
    pool = np.asarray(pool, dtype=np.float32)
    D = int(pool.shape[-1])
    if perturb_fraction is None:
        var = pool.var(axis=0)
        idx = np.flatnonzero(var > float(variance_threshold))
        if idx.size == 0:
            # Single-entry pool / fully constant: nothing to detect — fall
            # back to perturbing everything so the caller still gets noise.
            idx = np.arange(D)
        return idx.astype(np.int64)
    start = int(D * (1.0 - float(perturb_fraction)))
    return np.arange(start, D, dtype=np.int64)


class PerturbPoolSampler(BaseInitSampler):
    """Pool sample + Gaussian noise on auto-detected object dims (default)
    or a hand-specified contiguous tail slice."""

    def __init__(
        self,
        suite_name: str,
        task_index: int,
        pool: np.ndarray,
        perturb_fraction: Optional[float] = None,
        perturb_std: float = 0.02,
        slice_override: Optional[Tuple[int, int]] = None,
    ):
        super().__init__(suite_name, task_index, pool)
        if slice_override is not None:
            self.indices = np.arange(
                int(slice_override[0]), int(slice_override[1]), dtype=np.int64,
            )
        else:
            self.indices = compute_perturb_indices(self.pool, perturb_fraction)
        self.perturb_std = float(perturb_std)

    def sample(self, rng: np.random.Generator) -> InitState:
        idx = int(rng.integers(0, len(self.pool)))
        vec = self.pool[idx].copy()
        delta = np.zeros_like(vec)
        noise = rng.normal(0.0, self.perturb_std, size=self.indices.size).astype(np.float32)
        delta[self.indices] = noise
        vec = vec + delta
        return InitState(
            suite_name=self.suite_name,
            task_index=self.task_index,
            vector=vec,
            source_mode="perturb_pool",
            pool_index=idx,
            perturbation=delta,
        )


def make_sampler(
    mode: str,
    suite_name: str,
    task_index: int,
    pool: np.ndarray,
    **kwargs,
) -> BaseInitSampler:
    if mode == "pool":
        return PoolSampler(suite_name, task_index, pool)
    if mode == "generated":
        return GeneratedPoolSampler(suite_name, task_index, pool)
    if mode == "perturb_pool":
        return PerturbPoolSampler(suite_name, task_index, pool, **kwargs)
    raise ValueError(f"Unknown init sampler mode '{mode}'")


def generated_pool_path(cache_dir: Path, suite_name: str, task_index: int) -> Path:
    return Path(cache_dir) / suite_name / f"task_{task_index:02d}.npy"


def load_generated_pool(
    cache_dir: Path,
    suite_name: str,
    task_index: int,
) -> np.ndarray:
    """Load a (K, D) generated init pool from ``utils/generate_inits.py``.

    Raises FileNotFoundError with a helpful message if the cache is missing
    — that's the signal to run the generator first.
    """
    path = generated_pool_path(cache_dir, suite_name, task_index)
    if not path.exists():
        raise FileNotFoundError(
            f"No generated init pool at {path}.\n"
            f"  Run:  python -m adversarial_training.utils.generate_inits "
            f"--cache_dir {cache_dir} --suites {suite_name}"
        )
    arr = np.load(path)
    return np.asarray(arr, dtype=np.float32)


# ---------------------------------------------------------------------------
# Env interaction
# ---------------------------------------------------------------------------


# Per LIBERO eval client. Wait for objects to settle after set_init_state.
NUM_SETTLE_STEPS = 10
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]


def apply_init_state(env: Any, init: InitState) -> Dict[str, Any]:
    """Reset ``env`` to ``init`` and run the canonical settling period."""
    env.reset()
    obs = env.set_init_state(init.vector)
    for _ in range(NUM_SETTLE_STEPS):
        obs, _reward, _done, _info = env.step(LIBERO_DUMMY_ACTION)
    return obs


# ---------------------------------------------------------------------------
# Disk I/O
# ---------------------------------------------------------------------------


def save_init_states(path: Path, inits: List[InitState]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump([i.to_dict() for i in inits], f, protocol=pickle.HIGHEST_PROTOCOL)


def load_init_states(path: Path) -> List[InitState]:
    with open(path, "rb") as f:
        raw = pickle.load(f)
    return [InitState.from_dict(d) for d in raw]
