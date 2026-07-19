"""Thin LIBERO env factory + obs packer.

Mirrors ``get_libero_env`` / ``eval_libero`` obs-packing logic from
``evaluation/libero/libero_client.py`` so the data-collection /
accelerated-testing scripts share env handling with the official
evaluation pipeline.

This module is import-safe without SimpleVLA / torch present — collect.py
can run ``--dry_run`` even on machines that lack the model deps.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict

import numpy as np


LIBERO_ENV_RESOLUTION = 256


def make_libero_env(suite_name: str, task_index: int, *, resolution: int = LIBERO_ENV_RESOLUTION, seed: int = 0):
    """Return ``(env, task_description, init_states)``."""
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_suite = benchmark.get_benchmark_dict()[suite_name]()
    task = task_suite.get_task(task_index)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file

    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)

    init_states = _coerce_pool(task_suite.get_task_init_states(task_index))
    return env, task.language, init_states


def _coerce_pool(pool: Any) -> np.ndarray:
    """LIBERO returns either np.ndarray (N, D) or a torch Tensor — normalize."""
    try:
        import torch
        if isinstance(pool, torch.Tensor):
            pool = pool.cpu().numpy()
    except ImportError:
        pass
    return np.asarray(pool, dtype=np.float32)


def load_official_pool(suite_name: str, task_index: int) -> np.ndarray:
    """Fetch the LIBERO-shipped init pool without opening an env.

    Used by test_model.py (benchmark) and collect_buffer.py (fallback). The
    50-init array is the canonical eval distribution — anything that should
    be SOTA-comparable must score against this exact set.
    """
    from libero.libero import benchmark
    task_suite = benchmark.get_benchmark_dict()[suite_name]()
    return _coerce_pool(task_suite.get_task_init_states(task_index))


# ---------------------------------------------------------------------------
# Observation packing — matches what SimpleVLA expects.
# ---------------------------------------------------------------------------


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = math.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def pack_libero_obs(env_obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Convert a raw LIBERO obs dict into the SimpleVLA input format.

    Mirrors evaluation/libero/libero_client.py:eval_libero — images
    flipped 180 degrees, state is 8-D ``[eef_pos(3), axis_angle(3), gripper_qpos(2)]``.
    """
    img = np.ascontiguousarray(env_obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(env_obs["robot0_eye_in_hand_image"][::-1, ::-1])
    state = np.concatenate([
        env_obs["robot0_eef_pos"],
        _quat2axisangle(env_obs["robot0_eef_quat"]),
        env_obs["robot0_gripper_qpos"],
    ]).astype(np.float32)
    return {"image": img, "wrist_image": wrist, "state": state}
