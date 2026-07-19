"""Stage 1 — Data collection.

Roll the pre-trained SimpleVLA policy on LIBERO with sampled initial
states; for each episode, record:

    (suite_name, task_index, init_state, success, episode_len,
     source_mode, pool_index, is_weight=1.0)

Output: one ``.npz`` shard per (suite, task_index) under
``output_dir/<suite_name>/task_<i>.npz``.

The criticality model (stage1_train.py) consumes these shards directly.

Reference: other_source/criticality/stage1/stage1_collect.py — but episode
granularity instead of step granularity, and SimpleVLA in place of the
ManiSkill MoE policy.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

# Make ``adversarial_training.utils.*`` importable when launched as a script.
_THIS = Path(__file__).resolve()
_CODE_ROOT = _THIS.parents[2]   # .../
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from adversarial_training.utils import data_utils
from adversarial_training.utils.data_utils import EpisodeRecord
from adversarial_training.utils.init_state import apply_init_state, load_generated_pool, make_sampler
from adversarial_training.utils.libero_env import make_libero_env, pack_libero_obs
from adversarial_training.utils.task_registry import get_task_spec

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect LIBERO rollouts for criticality training.")
    p.add_argument("--config", type=Path, default=Path("./adversarial_training/configs/default.yaml"))
    p.add_argument("--suites", nargs="+", default=None, help="Override suites listed in the config.")
    p.add_argument("--episodes_per_suite", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--output_dir", type=Path, default=None)
    p.add_argument("--save_trajectories", action="store_true",
                   help="Also dump per-episode obs/action arrays. Large.")
    p.add_argument("--init_sampler", type=str, default=None,
                   choices=["pool", "perturb_pool", "generated"])
    p.add_argument("--generated_pool_cache", type=Path, default=None,
                   help="Override collect.generated_pool_cache. Required when init_sampler=generated.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry_run", action="store_true",
                   help="Skip policy load + env step; just walk the task list.")
    return p.parse_args()


def load_cfg(args: argparse.Namespace) -> Dict[str, Any]:
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    c = cfg.get("stage1_collect", {})

    if args.suites is not None:                 c["suites"] = args.suites
    if args.episodes_per_suite is not None:     c["episodes_per_suite"] = args.episodes_per_suite
    if args.num_workers is not None:            c["num_workers"] = args.num_workers
    if args.output_dir is not None:             c["output_dir"] = str(args.output_dir)
    if args.init_sampler is not None:           c["init_sampler"] = args.init_sampler
    if args.generated_pool_cache is not None:   c["generated_pool_cache"] = str(args.generated_pool_cache)

    # Allow fallback to a top-level continual.generated_pool_cache so both training-data
    # collectors (stage1 + collect_buffer) can share one cache without duplicating the path.
    if "generated_pool_cache" not in c:
        cont = cfg.get("continual", {})
        if cont.get("generated_pool_cache"):
            c["generated_pool_cache"] = cont["generated_pool_cache"]

    c.setdefault("policy", {})
    c.setdefault("save_trajectories", bool(args.save_trajectories))
    c.setdefault("seed", args.seed)
    c.setdefault("dry_run", bool(args.dry_run))
    return c


# ---------------------------------------------------------------------------
# Per-worker collection
# ---------------------------------------------------------------------------


def _build_task_assignments(cfg: Dict[str, Any]) -> List[Tuple[str, int, int]]:
    """Return (suite, task_index, episodes_for_this_task) tuples."""
    from libero.libero import benchmark as _bench
    benches = _bench.get_benchmark_dict()

    assignments: List[Tuple[str, int, int]] = []
    for suite in cfg["suites"]:
        suite_obj = benches[suite]()
        n_tasks = suite_obj.n_tasks
        episodes_per_task = max(1, int(cfg["episodes_per_suite"]) // n_tasks)
        for task_idx in range(n_tasks):
            assignments.append((suite, task_idx, episodes_per_task))
    return assignments


def _round_robin_split(items: List[Any], num_workers: int) -> List[List[Any]]:
    buckets: List[List[Any]] = [[] for _ in range(num_workers)]
    for i, item in enumerate(items):
        buckets[i % num_workers].append(item)
    return buckets


def collect_one_task(
    suite_name: str,
    task_index: int,
    n_episodes: int,
    policy,
    cfg: Dict[str, Any],
    rng: np.random.Generator,
) -> List[EpisodeRecord]:
    env, task_desc, init_pool = make_libero_env(suite_name, task_index, seed=int(rng.integers(0, 2**31 - 1)))
    spec = get_task_spec(suite_name)
    max_steps = int(spec.max_episode_steps)

    sampler_mode = cfg.get("init_sampler", "generated")
    if sampler_mode == "generated":
        cache_dir = cfg.get("generated_pool_cache")
        if not cache_dir:
            raise RuntimeError(
                "init_sampler=generated requires collect.generated_pool_cache (or a "
                "top-level continual.generated_pool_cache). Run utils/generate_inits.py "
                "first and point this key at its --cache_dir."
            )
        pool_for_sampler = load_generated_pool(Path(cache_dir), suite_name, task_index)
    else:
        pool_for_sampler = np.asarray(init_pool)

    sampler_kwargs: Dict[str, Any] = {}
    if sampler_mode == "perturb_pool":
        if "perturb_fraction" in cfg:
            pf = cfg["perturb_fraction"]
            sampler_kwargs["perturb_fraction"] = None if pf is None else float(pf)
        if "perturb_std" in cfg:
            sampler_kwargs["perturb_std"] = float(cfg["perturb_std"])
    sampler = make_sampler(
        sampler_mode,
        suite_name,
        task_index,
        pool_for_sampler,
        **sampler_kwargs,
    )

    out: List[EpisodeRecord] = []
    try:
        for ep in range(n_episodes):
            init = sampler.sample(rng)
            obs = apply_init_state(env, init)

            policy.reset()
            success = False
            steps = 0
            traj: Optional[Dict[str, List[np.ndarray]]] = None
            if cfg["save_trajectories"]:
                traj = {"image": [], "wrist_image": [], "state": [], "action": []}

            while steps < max_steps:
                obs_packed = pack_libero_obs(obs)
                action = policy.step(obs_packed, task_desc)

                if traj is not None:
                    traj["image"].append(obs_packed["image"])
                    traj["wrist_image"].append(obs_packed["wrist_image"])
                    traj["state"].append(obs_packed["state"])
                    traj["action"].append(np.asarray(action, dtype=np.float32))

                obs, _r, done, _info = env.step(action.tolist())
                steps += 1
                if done:
                    success = True
                    break

            record = EpisodeRecord(
                suite_name=suite_name,
                task_index=task_index,
                init_state=init.vector,
                success=success,
                episode_len=steps,
                source_mode=init.source_mode,
                pool_index=-1 if init.pool_index is None else init.pool_index,
                is_weight=1.0,
                trajectory={k: np.asarray(v) for k, v in traj.items()} if traj is not None else None,
            )
            out.append(record)
    finally:
        try:
            env.close()
        except Exception:
            pass

    return out


def collect_dry_run(suite_name: str, task_index: int, n_episodes: int) -> List[EpisodeRecord]:
    """Emit dummy records so the rest of the pipeline can be tested without LIBERO/SimpleVLA."""
    rng = np.random.default_rng(seed=hash((suite_name, task_index)) & 0xFFFFFFFF)
    dim = 79          # placeholder; real dim comes from LIBERO
    out = []
    for ep in range(n_episodes):
        out.append(EpisodeRecord(
            suite_name=suite_name,
            task_index=task_index,
            init_state=rng.normal(size=dim).astype(np.float32),
            success=bool(rng.random() > 0.3),
            episode_len=int(rng.integers(50, 400)),
            source_mode="pool",
            pool_index=int(rng.integers(0, 50)),
            is_weight=1.0,
            trajectory=None,
        ))
    return out


# ---------------------------------------------------------------------------
# Worker process entry
# ---------------------------------------------------------------------------


def _worker_main(
    worker_id: int,
    gpu_id: Optional[int],
    tasks: List[Tuple[str, int, int]],
    cfg: Dict[str, Any],
    seed: int,
) -> None:
    # Must happen BEFORE any torch import, so torch's cuda:0 becomes this physical GPU.
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"[w{worker_id}] pinned to GPU {gpu_id}")

    rng = np.random.default_rng(seed + worker_id)
    out_root = Path(cfg["output_dir"])

    policy = None
    if not cfg["dry_run"]:
        from adversarial_training.utils.policy import SimpleVLAPolicy
        pcfg = cfg["policy"]
        policy = SimpleVLAPolicy(
            checkpoint=pcfg["checkpoint"],
            norm_stats=pcfg.get("norm_stats"),
            smolvlm_model=pcfg.get("smolvlm_model", "HuggingFaceTB/SmolVLM-500M-Instruct"),
        )

    for suite_name, task_index, n_episodes in tasks:
        t0 = time.time()
        if cfg["dry_run"]:
            records = collect_dry_run(suite_name, task_index, n_episodes)
        else:
            records = collect_one_task(suite_name, task_index, n_episodes, policy, cfg, rng)

        data_utils.save_shard(
            out_root, suite_name, task_index, records,
            save_trajectories=cfg["save_trajectories"],
        )
        succ = sum(int(r.success) for r in records)
        dur = time.time() - t0
        print(f"[w{worker_id}] {suite_name}/task_{task_index:02d}  {succ}/{len(records)} success  ({dur:.1f}s)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args)

    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)

    assignments = _build_task_assignments(cfg) if not cfg["dry_run"] else [
        (suite, t, max(1, int(cfg["episodes_per_suite"]) // 10))
        for suite in cfg["suites"] for t in range(10)
    ]
    print(f"Collecting {sum(n for *_, n in assignments)} episodes "
          f"across {len(assignments)} (suite, task) pairs.")

    num_workers = max(1, int(cfg.get("num_workers", 1)))

    # Resolve GPU assignment: cfg['gpus'] is an optional list of physical GPU ids.
    # If omitted, default to GPUs 0..num_workers-1. Workers > len(gpus) wrap around.
    gpus_cfg = cfg.get("gpus")
    if gpus_cfg is None:
        gpu_assignment = list(range(num_workers))
    else:
        gpu_assignment = [int(gpus_cfg[i % len(gpus_cfg)]) for i in range(num_workers)]

    if num_workers == 1:
        _worker_main(0, gpu_assignment[0], assignments, cfg, int(cfg["seed"]))
        return

    # spawn: each child re-imports the module, so CUDA_VISIBLE_DEVICES set inside
    # the child takes effect before torch is touched. fork on Linux would inherit
    # any CUDA context the parent already initialized and pin all workers to it.
    ctx = mp.get_context("spawn")
    buckets = _round_robin_split(assignments, num_workers)
    procs = []
    for wid, bucket in enumerate(buckets):
        if not bucket:
            continue
        p = ctx.Process(
            target=_worker_main,
            args=(wid, gpu_assignment[wid], bucket, cfg, int(cfg["seed"])),
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join()


if __name__ == "__main__":
    main()
