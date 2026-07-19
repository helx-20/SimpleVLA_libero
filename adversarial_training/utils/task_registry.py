"""Per-task-suite metadata for LIBERO.

Mirrors `other_source/examples/baselines/ppo/task_registry.py` but for the
4 LIBERO suites instead of ManiSkill tasks.

Each LIBERO suite ships with 10 sub-tasks; we treat every (suite, task_index)
pair as a distinct "task" for the multi-task criticality model.

The flat init-state dimensionality is task-specific (varies with the number
of free objects in the scene) and **not known until LIBERO is queried**, so
``init_state_dim`` starts at None and is filled in lazily on first use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class LiberoTaskSpec:
    suite_id: int
    suite_name: str
    num_tasks: int = 10
    max_episode_steps: int = 600
    # Filled in lazily by ``populate_init_state_dims`` once LIBERO is importable.
    init_state_dims: Optional[List[int]] = None


TASK_REGISTRY: Dict[str, LiberoTaskSpec] = {
    "libero_spatial": LiberoTaskSpec(suite_id=0, suite_name="libero_spatial", max_episode_steps=800),
    "libero_object":  LiberoTaskSpec(suite_id=1, suite_name="libero_object",  max_episode_steps=800),
    "libero_goal":    LiberoTaskSpec(suite_id=2, suite_name="libero_goal",    max_episode_steps=800),
    "libero_10":      LiberoTaskSpec(suite_id=3, suite_name="libero_10",      max_episode_steps=900),
}


def get_task_spec(suite_name: str) -> LiberoTaskSpec:
    if suite_name not in TASK_REGISTRY:
        raise KeyError(f"Unknown LIBERO suite '{suite_name}'. Known: {list(TASK_REGISTRY)}")
    return TASK_REGISTRY[suite_name]


def all_suites() -> List[str]:
    return list(TASK_REGISTRY.keys())


def populate_init_state_dims(suite_name: str) -> List[int]:
    """Query LIBERO for the flat init-state dim of every task in a suite.

    Caches the result on the registry entry so repeat calls are free.
    """
    spec = get_task_spec(suite_name)
    if spec.init_state_dims is not None:
        return spec.init_state_dims

    from libero.libero import benchmark
    task_suite = benchmark.get_benchmark_dict()[suite_name]()
    dims: List[int] = []
    for task_id in range(task_suite.n_tasks):
        inits = task_suite.get_task_init_states(task_id)
        dims.append(int(inits.shape[-1]) if hasattr(inits, "shape") else int(len(inits[0])))

    spec.init_state_dims = dims
    spec.num_tasks = task_suite.n_tasks
    return dims


def global_task_id(suite_name: str, task_index: int) -> int:
    """Stable id across all suites — used as the MoE task token."""
    spec = get_task_spec(suite_name)
    return spec.suite_id * 100 + task_index


def all_task_keys() -> List[tuple]:
    """All (suite_name, task_index) pairs in a deterministic order."""
    out: List[tuple] = []
    for suite_name in all_suites():
        spec = get_task_spec(suite_name)
        for i in range(spec.num_tasks):
            out.append((suite_name, i))
    return out


def flat_task_id(suite_name: str, task_index: int) -> int:
    """Dense 0..N-1 id, used to index ModuleLists in the MoE classifier."""
    keys = all_task_keys()
    try:
        return keys.index((suite_name, task_index))
    except ValueError as e:
        raise KeyError(f"({suite_name}, {task_index}) not in registry") from e


def from_flat_task_id(fid: int) -> tuple:
    return all_task_keys()[fid]
