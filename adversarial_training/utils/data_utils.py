"""I/O helpers for criticality data.

We store one shard per (suite, task_index) holding *all* episodes for that
task. Each shard is an ``.npz`` file with parallel arrays:

    init_state    (N, D_task)         float32, the LIBERO MuJoCo init state
    success       (N,)                bool, env.step returned done before timeout
    episode_len   (N,)                int32, number of policy steps
    source_mode   (N,)                S16, e.g. b"pool" or b"perturb_pool"
    pool_index    (N,)                int32, -1 if not from the pool
    is_weight     (N,)                float32, importance weight (1.0 for stage1)

When trajectories are kept (``collect.save_trajectories=True`` or the
accelerated-testing buffer mode), an additional companion file
``<shard>.traj.pkl`` stores per-episode dicts with image + state + action
arrays. Trajectories are large (~MBs / episode), so they're optional.

``save_shard`` appends by default: if a shard already exists for the same
(suite, task_index), new records are concatenated onto the old ones. Pass
``append=False`` to wipe the previous shard. Writes are staged through a
``.tmp`` file and ``os.replace``'d into place, so a crash mid-write can
never corrupt the accumulated buffer.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

import numpy as np


SHARD_KEYS = ("init_state", "success", "episode_len", "source_mode", "pool_index", "is_weight")


@dataclass
class EpisodeRecord:
    suite_name: str
    task_index: int
    init_state: np.ndarray
    success: bool
    episode_len: int
    source_mode: str = "pool"
    pool_index: int = -1
    is_weight: float = 1.0
    trajectory: Optional[Dict[str, np.ndarray]] = None     # optional, see module doc


# ---------------------------------------------------------------------------
# Shard layout helpers
# ---------------------------------------------------------------------------


def shard_path(root: Path, suite_name: str, task_index: int) -> Path:
    return Path(root) / suite_name / f"task_{task_index:02d}.npz"


def traj_path(shard: Path) -> Path:
    return shard.with_suffix(".traj.pkl")


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def save_shard(
    root: Path,
    suite_name: str,
    task_index: int,
    records: List[EpisodeRecord],
    save_trajectories: bool = False,
    append: bool = True,
) -> Path:
    """Persist a list of EpisodeRecord into a per-task shard.

    When ``append=True`` (default) and the shard already exists, the new
    records are concatenated onto the previous ones. The trajectory pkl is
    merged the same way; records without a trajectory (i.e. this or an
    earlier call had ``save_trajectories=False``) get a ``None`` slot, so
    the pkl always lines up 1:1 with the npz.

    Writes go to ``<shard>.tmp`` and are ``os.replace``'d into the final
    name so a crash mid-write can never corrupt the accumulated buffer.
    """
    if not records:
        raise ValueError("save_shard() received an empty record list")

    out = shard_path(root, suite_name, task_index)
    out.parent.mkdir(parents=True, exist_ok=True)

    new_init = np.stack([r.init_state.astype(np.float32) for r in records], axis=0)
    new_succ = np.array([r.success for r in records], dtype=np.bool_)
    new_len  = np.array([r.episode_len for r in records], dtype=np.int32)
    new_src  = np.array([r.source_mode for r in records], dtype="S16")
    new_pidx = np.array([r.pool_index for r in records], dtype=np.int32)
    new_w    = np.array([r.is_weight for r in records], dtype=np.float32)
    new_trajs: List[Optional[Dict[str, np.ndarray]]] = (
        [r.trajectory for r in records] if save_trajectories
        else [None] * len(records)
    )

    if append and out.exists():
        z_old = np.load(out, allow_pickle=False)
        if z_old["init_state"].shape[1] != new_init.shape[1]:
            raise ValueError(
                f"append: existing init_state dim {z_old['init_state'].shape[1]} "
                f"!= new dim {new_init.shape[1]} for {out}"
            )
        init_state = np.concatenate([z_old["init_state"], new_init], axis=0)
        success    = np.concatenate([z_old["success"],    new_succ], axis=0)
        ep_len     = np.concatenate([z_old["episode_len"], new_len], axis=0)
        source     = np.concatenate([z_old["source_mode"], new_src], axis=0)
        pool_idx   = np.concatenate([z_old["pool_index"],  new_pidx], axis=0)
        is_w       = np.concatenate([z_old["is_weight"],   new_w], axis=0)

        n_old = int(z_old["success"].shape[0])
        tp_old = traj_path(out)
        if tp_old.exists():
            with open(tp_old, "rb") as f:
                old_trajs = pickle.load(f)
            if len(old_trajs) != n_old:
                # Pkl/npz desync — fall back to None padding rather than crash.
                old_trajs = [None] * n_old
        else:
            old_trajs = [None] * n_old
        all_trajs = list(old_trajs) + list(new_trajs)
    else:
        init_state, success, ep_len = new_init, new_succ, new_len
        source, pool_idx, is_w = new_src, new_pidx, new_w
        all_trajs = list(new_trajs)

    # Stage through .tmp so a crash mid-write leaves the previous shard intact.
    # NB: np.savez_compressed auto-appends ".npz" when passed a path that doesn't
    # already end in .npz — which would silently rename our .tmp file. Pass an
    # open file handle to disable that behavior.
    npz_tmp = out.with_suffix(out.suffix + ".tmp")
    with open(npz_tmp, "wb") as f:
        np.savez_compressed(
            f,
            init_state=init_state,
            success=success,
            episode_len=ep_len,
            source_mode=source,
            pool_index=pool_idx,
            is_weight=is_w,
        )
    os.replace(npz_tmp, out)

    tp = traj_path(out)
    if any(t is not None for t in all_trajs):
        tp_tmp = tp.with_suffix(tp.suffix + ".tmp")
        with open(tp_tmp, "wb") as f:
            pickle.dump(all_trajs, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tp_tmp, tp)
    elif tp.exists():
        # No record (old or new) has a trajectory — drop a stale pkl if present.
        tp.unlink()

    return out


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def load_shard(shard: Path, with_trajectories: bool = False) -> List[EpisodeRecord]:
    shard = Path(shard)
    suite_name = shard.parent.name
    task_index = int(shard.stem.split("_")[1])

    z = np.load(shard, allow_pickle=False)
    n = len(z["success"])

    trajs: List[Optional[Dict[str, np.ndarray]]] = [None] * n
    if with_trajectories:
        tp = traj_path(shard)
        if tp.exists():
            with open(tp, "rb") as f:
                trajs = pickle.load(f)

    out: List[EpisodeRecord] = []
    for i in range(n):
        out.append(EpisodeRecord(
            suite_name=suite_name,
            task_index=task_index,
            init_state=z["init_state"][i],
            success=bool(z["success"][i]),
            episode_len=int(z["episode_len"][i]),
            source_mode=z["source_mode"][i].decode() if isinstance(z["source_mode"][i], bytes) else str(z["source_mode"][i]),
            pool_index=int(z["pool_index"][i]),
            is_weight=float(z["is_weight"][i]),
            trajectory=trajs[i],
        ))
    return out


def iter_shards(root: Path) -> Iterator[Path]:
    for shard in sorted(Path(root).glob("*/task_*.npz")):
        yield shard


def iter_episodes(root: Path, with_trajectories: bool = False) -> Iterator[EpisodeRecord]:
    for shard in iter_shards(root):
        yield from load_shard(shard, with_trajectories=with_trajectories)


# ---------------------------------------------------------------------------
# Flattening (consumed by stage1_train.py)
# ---------------------------------------------------------------------------


def flatten_for_training(root: Path) -> Dict[str, Any]:
    """Stack every shard into model-ready arrays.

    Each task may have a different init_state dim, so we return a *dict of
    per-task arrays* keyed by (suite_name, task_index) plus a flat index of
    which (suite, task) each sample comes from.
    """
    by_task: Dict[tuple, Dict[str, list]] = {}
    for rec in iter_episodes(root, with_trajectories=False):
        key = (rec.suite_name, rec.task_index)
        bucket = by_task.setdefault(key, {"init_state": [], "label": [], "is_weight": []})
        bucket["init_state"].append(rec.init_state.astype(np.float32))
        bucket["label"].append(int(not rec.success))   # criticality target: 1 = failure
        bucket["is_weight"].append(float(rec.is_weight))

    out: Dict[str, Any] = {"by_task": {}}
    for key, bucket in by_task.items():
        out["by_task"][key] = {
            "init_state": np.stack(bucket["init_state"], axis=0),
            "label": np.asarray(bucket["label"], dtype=np.int64),
            "is_weight": np.asarray(bucket["is_weight"], dtype=np.float32),
        }
    return out
