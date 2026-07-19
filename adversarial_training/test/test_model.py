"""Accelerated testing loop.

Binds together :class:`LiberoNADE` + :class:`SimpleVLAPolicy` + the trained
multi-task criticality MoE. For each assigned ``(suite, task_index)``:

    1. Load LIBERO env + init-state pool.
    2. Build a per-task scorer that maps candidate init states to
       ``P(failure)`` via the criticality model.
    3. Wrap the env in :class:`LiberoNADE` so ``nade.reset()`` returns
       importance-sampled inits with a real ``is_weight``.
    4. Roll SimpleVLA, capture success + (optionally) the full trajectory.
    5. Save one ``.npz`` shard per task into ``output_dir`` — same schema as
       Stage 1 collection, so the existing flatten/load helpers Just Work.

Output modes:

    * ``metric_only`` — keep init/success/IS_weight only. Cheap to store;
      enough for unbiased weighted crash-rate estimates of the policy.
    * ``buffer``     — also dump trajectories. Required for the continual-
      learning pipeline, which reuses these rollouts as training data.

Reference: ``other_source/criticality/test/test_model.py``
Key port differences:
    * Per-episode (not per-step) importance sampling.
    * Local in-process SimpleVLA, no WebSocket server.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import yaml

_THIS = Path(__file__).resolve()
_CODE_ROOT = _THIS.parents[2]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from adversarial_training.test.libero_nade import (
    NADEConfig, build_candidate_set, build_proposal, importance_weight,
)
from adversarial_training.utils import data_utils
from adversarial_training.utils.data_utils import EpisodeRecord
from adversarial_training.utils.init_state import LIBERO_DUMMY_ACTION, NUM_SETTLE_STEPS
from adversarial_training.utils.libero_env import (
    LIBERO_ENV_RESOLUTION, load_official_pool, make_libero_env, pack_libero_obs,
)
from adversarial_training.utils.task_registry import (
    all_task_keys, get_task_spec,
)


# ---------------------------------------------------------------------------
# CLI / config
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Accelerated testing of SimpleVLA on LIBERO via init-state importance sampling.",
    )
    p.add_argument("--config", type=Path, default=Path("./adversarial_training/configs/default.yaml"))
    p.add_argument("--criticality_ckpt", type=Path, default=None)
    p.add_argument("--output_dir", type=Path, default=None)
    p.add_argument("--suites", nargs="+", default=None)
    p.add_argument("--episodes_per_suite", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--mode", choices=["metric_only", "buffer"], default=None)
    p.add_argument("--seed", type=int, default=None)
    # Policy overrides — let the user point at a continual-learning checkpoint
    # without editing the YAML when evaluating a fresh model.
    p.add_argument("--policy_checkpoint", type=str, default=None,
                   help="Override test.policy.checkpoint (the SimpleVLA model under test).")
    p.add_argument("--policy_norm_stats", type=str, default=None,
                   help="Override test.policy.norm_stats.")
    p.add_argument("--policy_smolvlm_model", type=str, default=None,
                   help="Override test.policy.smolvlm_model (VLM backbone path/HF id).")
    p.add_argument("--dry_run", action="store_true",
                   help="Use a random scorer + skip the policy; only checks plumbing.")
    return p.parse_args()


def load_cfg(args: argparse.Namespace) -> Dict[str, Any]:
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    t = cfg.get("test", {})
    c = cfg.get("collect", {})

    # Policy: deep-merge test.policy on top of collect.policy so users can
    # override only the keys they want (typically `checkpoint`) for eval.
    policy = dict(c.get("policy", {}))
    policy.update(dict(t.get("policy", {})))
    if args.policy_checkpoint is not None:    policy["checkpoint"] = args.policy_checkpoint
    if args.policy_norm_stats is not None:    policy["norm_stats"] = args.policy_norm_stats
    if args.policy_smolvlm_model is not None: policy["smolvlm_model"] = args.policy_smolvlm_model
    t["policy"] = policy

    if args.criticality_ckpt is not None: t["criticality_ckpt"] = str(args.criticality_ckpt)
    if args.output_dir is not None:       t["output_dir"] = str(args.output_dir)
    if args.suites is not None:           t["suites"] = args.suites
    if args.episodes_per_suite is not None: t["episodes_per_suite"] = args.episodes_per_suite
    if args.num_workers is not None:      t["num_workers"] = args.num_workers
    if args.mode is not None:             t["mode"] = args.mode
    if args.seed is not None:             t["seed"] = args.seed

    t.setdefault("mode", "buffer")
    t.setdefault("num_workers", 1)
    t.setdefault("seed", 123)
    t.setdefault("hidden_dim", 1024)
    t.setdefault("expansion", 4)
    t.setdefault("depth", 12)
    t.setdefault("dropout", 0.1)
    t.setdefault("epsilon", 0.01)
    t.setdefault("weight_clip", 100.0)
    t.setdefault("criticality_threshold", 0.5)
    t.setdefault("pool_repeats", 1)
    # Default 0: benchmark candidate set = official 50, period. SOTA-comparable.
    # Bump this and tune perturb_* for ablations that intentionally widen the pool.
    t.setdefault("perturbations_per_pool", 0)
    t.setdefault("perturb_fraction", None)
    t.setdefault("perturb_std", 0.02)
    # Master switch for importance sampling. False -> per-task uniform draws,
    # IS weight = 1, criticality scorer not loaded. True -> suite-level NADE.
    t.setdefault("use_nade", True)
    t.setdefault("dry_run", bool(args.dry_run))
    return t


# ---------------------------------------------------------------------------
# Criticality scorer
# ---------------------------------------------------------------------------


ScorerFn = Callable[[np.ndarray], np.ndarray]


def _load_criticality_model(
    ckpt: Path,
    hidden_dim: int,
    expansion: int,
    depth: int,
    dropout: float,
):
    """Reconstruct the deep-MLP classifier and load weights.

    Padded input dim ``max_D`` is read back from the shape of
    ``input_proj.weight`` in the checkpoint.
    """
    import torch
    from adversarial_training.utils.criticality_model import (
        CriticalityModel, CriticalityModelConfig,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(ckpt, map_location=device, weights_only=False)

    if "input_proj.weight" not in state:
        raise RuntimeError(f"No input_proj.weight tensor in {ckpt}")
    max_D = int(state["input_proj.weight"].shape[1])

    model = CriticalityModel(CriticalityModelConfig(
        input_dim=max_D,
        hidden_dim=int(hidden_dim),
        expansion=int(expansion),
        depth=int(depth),
        dropout=float(dropout),
    )).to(device).eval()
    model.load_state_dict(state)
    return model, device, max_D


def build_scorer(model, device, max_D: int) -> ScorerFn:
    """Closure: ``(M, D) candidates -> (M,) P(failure)``.

    Right-pads each candidate to ``max_D`` before scoring, so the same
    model handles every task.
    """
    import torch

    def _score(candidates: np.ndarray) -> np.ndarray:
        x = np.asarray(candidates, dtype=np.float32)
        if x.shape[1] < max_D:
            pad = np.zeros((x.shape[0], max_D - x.shape[1]), dtype=np.float32)
            x = np.concatenate([x, pad], axis=1)
        elif x.shape[1] > max_D:
            raise ValueError(f"Candidate dim {x.shape[1]} exceeds model input_dim {max_D}")
        t = torch.from_numpy(x).to(device)
        return model.criticality_score(t).detach().cpu().numpy()
    return _score


def build_dry_run_scorer(seed: int) -> ScorerFn:
    """Deterministic random scorer for plumbing tests (no torch / no ckpt)."""
    rng = np.random.default_rng(seed)

    def _score(candidates: np.ndarray) -> np.ndarray:
        return rng.uniform(0.0, 1.0, size=candidates.shape[0]).astype(np.float64)
    return _score


# ---------------------------------------------------------------------------
# Per-task rollout
# ---------------------------------------------------------------------------


def _build_task_assignments(cfg: Dict[str, Any]) -> List[Tuple[str, int, int]]:
    """Mirror stage1_collect: split episodes_per_suite evenly across tasks."""
    from libero.libero import benchmark as _bench
    benches = _bench.get_benchmark_dict()

    assignments: List[Tuple[str, int, int]] = []
    for suite in cfg["suites"]:
        suite_obj = benches[suite]()
        n_tasks = suite_obj.n_tasks
        per_task = max(1, int(cfg["episodes_per_suite"]) // n_tasks)
        for task_idx in range(n_tasks):
            assignments.append((suite, task_idx, per_task))
    return assignments


def _round_robin_split(items: List[Any], num_workers: int) -> List[List[Any]]:
    buckets: List[List[Any]] = [[] for _ in range(num_workers)]
    for i, item in enumerate(items):
        buckets[i % num_workers].append(item)
    return buckets


def _nade_config(cfg: Dict[str, Any]) -> NADEConfig:
    pf = cfg.get("perturb_fraction")
    return NADEConfig(
        epsilon=float(cfg["epsilon"]),
        weight_clip=float(cfg["weight_clip"]),
        criticality_threshold=float(cfg["criticality_threshold"]),
        alpha=float(cfg.get("alpha", 3.0)),
        pool_repeats=int(cfg["pool_repeats"]),
        perturbations_per_pool=int(cfg["perturbations_per_pool"]),
        perturb_fraction=None if pf is None else float(pf),
        perturb_std=float(cfg["perturb_std"]),
    )


# ---------------------------------------------------------------------------
# Suite-level NADE
# ---------------------------------------------------------------------------


@dataclass
class _Draw:
    """One sampled init for an episode, with its bookkeeping."""
    vector: np.ndarray
    source_mode: str
    pool_index: int
    is_weight: float


def _sample_uniform_draws(
    pool: np.ndarray,
    nade_cfg: NADEConfig,
    n_draws: int,
    rng: np.random.Generator,
) -> List["_Draw"]:
    """Per-task uniform draws over the NADE candidate set (no scoring).

    Used by the ``use_nade=False`` path. IS weight is 1 because the proposal
    equals the reference distribution.

    Sampling is *sequential without replacement* (``ep % M``), matching
    ``evaluation/libero_client.py`` which iterates ``initial_states[ep % 50]``
    — every init state is visited exactly once when ``n_draws == M`` so the
    success-rate estimator has no with-replacement variance.
    """
    cands, modes, pidx = build_candidate_set(pool, nade_cfg, rng)
    M = int(cands.shape[0])
    if M == 0 or n_draws <= 0:
        return []
    n = int(n_draws)
    idx_arr = np.arange(n, dtype=np.int64) % M
    out: List[_Draw] = []
    for k in range(idx_arr.shape[0]):
        idx = int(idx_arr[k])
        mode = str(modes[idx])
        out.append(_Draw(
            vector=cands[idx].copy(),
            source_mode=mode,
            pool_index=int(pidx[idx]) if mode in {"pool", "perturb_pool"} else -1,
            is_weight=1.0,
        ))
    return out


def _sample_suite_draws(
    pools_per_task: List[np.ndarray],
    scorer: ScorerFn,
    nade_cfg: NADEConfig,
    n_draws: int,
    rng: np.random.Generator,
) -> Dict[int, List[_Draw]]:
    """Draw ``n_draws`` episodes via one suite-level NADE proposal.

    Concatenates each task's candidate set into one joint pool, scores them
    with the criticality model in a single pass, builds one proposal ``q``,
    then samples ``n_draws`` candidates with replacement. Returned dict maps
    local task index (position within ``pools_per_task``) to the draws routed
    to it.
    """
    cands_pt:  List[np.ndarray] = []
    modes_pt:  List[np.ndarray] = []
    pidx_pt:   List[np.ndarray] = []
    for pool in pools_per_task:
        c, m, p = build_candidate_set(pool, nade_cfg, rng)
        cands_pt.append(c); modes_pt.append(m); pidx_pt.append(p)

    p_fails = [np.asarray(scorer(c), dtype=np.float64) for c in cands_pt]
    p_fail  = np.concatenate(p_fails) if p_fails else np.array([], dtype=np.float64)
    M = int(p_fail.size)
    grouped: Dict[int, List[_Draw]] = {}
    if M == 0 or n_draws <= 0:
        return grouped

    q, used = build_proposal(
        p_fail, nade_cfg.epsilon, nade_cfg.criticality_threshold, nade_cfg.alpha,
    )
    flat_idx = rng.choice(M, size=int(n_draws), p=q)

    # Map each flat index back to (task_local, within-task idx).
    sizes = np.asarray([c.shape[0] for c in cands_pt], dtype=np.int64)
    cum   = np.concatenate([[0], np.cumsum(sizes)])
    task_local = np.searchsorted(cum[1:], flat_idx, side="right")
    within     = flat_idx - cum[task_local]

    for k in range(flat_idx.shape[0]):
        ti = int(task_local[k]); wi = int(within[k]); fi = int(flat_idx[k])
        w = importance_weight(fi, q, M, nade_cfg.weight_clip) if used else 1.0
        mode = str(modes_pt[ti][wi])
        grouped.setdefault(ti, []).append(_Draw(
            vector=cands_pt[ti][wi].copy(),
            source_mode=f"nade:{mode}",
            pool_index=int(pidx_pt[ti][wi]) if mode in {"pool", "perturb_pool"} else -1,
            is_weight=float(w),
        ))
    return grouped


def _roll_episode_from_obs(
    env, obs, task_desc: str, policy, max_steps: int, save_traj: bool,
) -> Tuple[bool, int, Optional[Dict[str, Any]]]:
    """Roll one episode; ``obs`` is the post-init observation.

    Records both per-step ``(image, wrist_image, state, action)`` for BC
    (consumed by ``prepare_data.py``) and per-chunk PPO bookkeeping
    (consumed by ``prepare_ppo_data.py``). A chunk fires every
    ``policy.replan_steps`` env steps; we accumulate that chunk's reward
    and capture ``done`` when it closes.
    """
    policy.reset()
    success = False
    steps = 0
    traj: Optional[Dict[str, Any]] = None
    if save_traj:
        traj = {"image": [], "wrist_image": [], "state": [], "action": []}

    ppo_chunks: List[Dict[str, Any]] = []
    current_chunk: Optional[Dict[str, Any]] = None
    chunk_reward = 0.0
    done = False

    while steps < max_steps:
        obs_packed = pack_libero_obs(obs)
        action, decision = policy.step_with_record(obs_packed, task_desc)

        if decision is not None:
            if current_chunk is not None:
                current_chunk["reward"] = chunk_reward
                current_chunk["done"] = False
                ppo_chunks.append(current_chunk)
            current_chunk = decision
            chunk_reward = 0.0

        if traj is not None:
            traj["image"].append(obs_packed["image"])
            traj["wrist_image"].append(obs_packed["wrist_image"])
            traj["state"].append(obs_packed["state"])
            traj["action"].append(np.asarray(action, dtype=np.float32))

        obs, r, done, _info = env.step(action.tolist())
        chunk_reward += float(r)
        steps += 1
        if done:
            success = True
            break

    if current_chunk is not None:
        current_chunk["reward"] = chunk_reward
        current_chunk["done"] = bool(done)
        ppo_chunks.append(current_chunk)

    return success, steps, _pack_trajectory(traj, ppo_chunks)


def _apply_init(env, init_vector: np.ndarray):
    """Reset env + apply init + settle.  Mirrors LiberoNADE.reset()."""
    env.reset()
    obs = env.set_init_state(np.asarray(init_vector, dtype=np.float32))
    for _ in range(NUM_SETTLE_STEPS):
        obs, _r, _d, _i = env.step(LIBERO_DUMMY_ACTION)
    return obs


def run_one_task(
    suite_name: str,
    task_index: int,
    n_episodes: int,
    pool: np.ndarray,
    policy,
    cfg: Dict[str, Any],
    rng: np.random.Generator,
) -> List[EpisodeRecord]:
    """Per-task uniform path (``use_nade=False``).

    Builds the NADE candidate set for this task, draws ``n_episodes``
    uniform samples, then delegates to :func:`run_task_with_draws` to roll
    them. IS weight is 1 throughout.
    """
    draws = _sample_uniform_draws(
        np.asarray(pool, dtype=np.float32), _nade_config(cfg), n_episodes, rng,
    )
    return run_task_with_draws(suite_name, task_index, draws, policy, cfg, rng)


def run_task_with_draws(
    suite_name: str,
    task_index: int,
    draws: List["_Draw"],
    policy,
    cfg: Dict[str, Any],
    rng: np.random.Generator,
) -> List[EpisodeRecord]:
    """Suite-level path: caller has already sampled ``draws`` from the joint
    proposal. We just open the env once and roll each draw.
    """
    if not draws:
        return []
    # Constant env seed across all tasks — matches evaluation/libero_client.py,
    # which passes the same CLI ``--seed`` (default 7) to every task's env.seed().
    # Using rng.integers here injects per-task variation in robosuite's controller
    # noise sources that evaluation/ doesn't have, drifting SR by ~0.1-0.2pp.
    resolution = int(cfg.get("env_resolution", LIBERO_ENV_RESOLUTION))
    env, task_desc, _ = make_libero_env(
        suite_name, task_index, seed=int(cfg["seed"]), resolution=resolution,
    )
    spec = get_task_spec(suite_name)
    max_steps = int(spec.max_episode_steps)
    save_traj = (cfg["mode"] == "buffer")

    out: List[EpisodeRecord] = []
    log_every = bool(cfg.get("log_every_episode", False))
    n_total = len(draws)
    try:
        for ep_i, draw in enumerate(draws):
            t_ep = time.time()
            obs = _apply_init(env, draw.vector)
            success, steps, traj = _roll_episode_from_obs(
                env, obs, task_desc, policy, max_steps, save_traj,
            )
            out.append(EpisodeRecord(
                suite_name=suite_name,
                task_index=task_index,
                init_state=np.asarray(draw.vector, dtype=np.float32),
                success=success,
                episode_len=steps,
                source_mode=draw.source_mode,
                pool_index=int(draw.pool_index),
                is_weight=float(draw.is_weight),
                trajectory=traj,
            ))
            if log_every:
                cum_succ = sum(int(r.success) for r in out)
                tag = "OK  " if success else "FAIL"
                print(
                    f"    {suite_name}/task_{task_index:02d}  ep {ep_i + 1:4d}/{n_total:<4d}  "
                    f"{tag}  steps={steps:4d}  w={draw.is_weight:.3f}  "
                    f"cum={cum_succ}/{ep_i + 1}  ({time.time() - t_ep:.1f}s)",
                    flush=True,
                )
    finally:
        try:
            env.close()
        except Exception:
            pass
    return out


def _pack_trajectory(
    traj: Optional[Dict[str, Any]],
    ppo_chunks: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Pack per-step lists to ndarrays; attach the ppo_chunks list as-is.

    Returns ``None`` when neither stream has any data (uniform/metric_only
    rollout that didn't request trajectories and produced no chunks).
    """
    if traj is None and not ppo_chunks:
        return None
    out: Dict[str, Any] = {}
    if traj is not None:
        out.update({k: np.asarray(v) for k, v in traj.items()})
    if ppo_chunks:
        out["ppo_chunks"] = ppo_chunks
    return out


def run_dry_run_task(
    suite_name: str, task_index: int, n_episodes: int,
    scorer: ScorerFn, cfg: Dict[str, Any], rng: np.random.Generator,
) -> List[EpisodeRecord]:
    """No LIBERO / no policy — builds a synthetic candidate set so the
    proposal + IS-weight bookkeeping can be exercised end-to-end.
    """
    dim = 79
    fake_pool = rng.normal(size=(20, dim)).astype(np.float32)

    # We can't truly run LiberoNADE.reset() without an env; reproduce the
    # candidate set + proposal math here directly.
    from adversarial_training.test.libero_nade import (
        build_candidate_set, build_proposal, importance_weight,
    )
    cnf = _nade_config(cfg)
    cands, modes, pidx = build_candidate_set(fake_pool, cnf, rng)
    M = cands.shape[0]

    out: List[EpisodeRecord] = []
    for _ in range(n_episodes):
        p_fail = np.asarray(scorer(cands), dtype=np.float64)
        q, used = build_proposal(p_fail, cnf.epsilon, cnf.criticality_threshold, cnf.alpha)
        idx = int(rng.choice(M, p=q))
        w = importance_weight(idx, q, M, cnf.weight_clip) if used else 1.0
        # Synthetic outcome: more likely to fail when scorer says so.
        success = bool(rng.random() > float(p_fail[idx]))
        out.append(EpisodeRecord(
            suite_name=suite_name,
            task_index=task_index,
            init_state=cands[idx].copy(),
            success=success,
            episode_len=int(rng.integers(50, 400)),
            source_mode=f"nade:{modes[idx]}",
            pool_index=int(pidx[idx]),
            is_weight=float(w),
            trajectory=None,
        ))
    return out


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def _load_worker_policy(cfg: Dict[str, Any]):
    from adversarial_training.utils.policy import SimpleVLAPolicy
    pcfg = cfg["policy"]
    return SimpleVLAPolicy(
        checkpoint=pcfg["checkpoint"],
        norm_stats=pcfg.get("norm_stats"),
        smolvlm_model=pcfg.get("smolvlm_model", "HuggingFaceTB/SmolVLM-500M-Instruct"),
        logstd_init=pcfg.get("logstd_init"),
    )


def _load_worker_scorer(cfg: Dict[str, Any]) -> ScorerFn:
    model, device, max_D = _load_criticality_model(
        Path(cfg["criticality_ckpt"]),
        hidden_dim=int(cfg["hidden_dim"]),
        expansion=int(cfg["expansion"]),
        depth=int(cfg["depth"]),
        dropout=float(cfg["dropout"]),
    )
    return build_scorer(model, device, max_D)


def _pin_gpu(worker_id: int, gpu_id: Optional[int]) -> None:
    """Set ``CUDA_VISIBLE_DEVICES`` before any torch import.

    Mirrors ``stage1_collect._worker_main``'s GPU pinning. Must run at the
    very top of the worker entrypoint — torch caches the device list at
    first import, so a later assignment is silently ignored.
    """
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"[w{worker_id}] pinned to GPU {gpu_id}")


def _worker_main(
    worker_id: int,
    gpu_id: Optional[int],
    tasks: List[Tuple[str, int, int]],
    cfg: Dict[str, Any],
    seed: int,
) -> None:
    """Per-task uniform path (``use_nade=False`` or ``dry_run``).

    Never needs the criticality model: uniform draws don't score candidates,
    and ``dry_run`` uses its own synthetic scorer for the proposal math.
    """
    _pin_gpu(worker_id, gpu_id)
    rng = np.random.default_rng(seed + worker_id)
    out_root = Path(cfg["output_dir"])

    policy = None
    if not cfg["dry_run"]:
        policy = _load_worker_policy(cfg)

    for suite_name, task_index, n_episodes in tasks:
        t0 = time.time()
        if cfg["dry_run"]:
            dry_scorer = build_dry_run_scorer(
                seed + worker_id * 1000 + hash((suite_name, task_index)) & 0xFFFF
            )
            records = run_dry_run_task(
                suite_name, task_index, n_episodes, dry_scorer, cfg, rng,
            )
        else:
            pool = load_official_pool(suite_name, task_index)
            records = run_one_task(
                suite_name, task_index, n_episodes, pool, policy, cfg, rng,
            )

        data_utils.save_shard(
            out_root, suite_name, task_index, records,
            save_trajectories=(cfg["mode"] == "buffer"),
        )
        succ = sum(int(r.success) for r in records)
        mean_w = float(np.mean([r.is_weight for r in records])) if records else 0.0
        dur = time.time() - t0
        print(
            f"[w{worker_id}] {suite_name}/task_{task_index:02d}  "
            f"{succ}/{len(records)} success  mean_w={mean_w:.3f}  ({dur:.1f}s)"
        )


def _worker_main_suite(
    worker_id: int,
    gpu_id: Optional[int],
    suite_assignments: List[Tuple[str, int]],
    cfg: Dict[str, Any],
    seed: int,
    pool_loader: Callable[[str, int], np.ndarray] = load_official_pool,
) -> None:
    """Suite-level NADE path. ``suite_assignments`` = list of (suite, n_draws_for_this_worker).

    For each suite the worker independently samples its share of episodes
    from the suite-wide proposal, groups draws by task, and rolls them.
    Multiple workers may write to the same task shard — ``save_shard`` uses
    atomic ``.tmp -> rename`` + ``append=True`` so the concat is safe.
    """
    _pin_gpu(worker_id, gpu_id)
    from libero.libero import benchmark as _bench

    rng = np.random.default_rng(seed + worker_id)
    out_root = Path(cfg["output_dir"])
    save_traj = (cfg["mode"] == "buffer")
    nade_cfg = _nade_config(cfg)

    policy = _load_worker_policy(cfg)
    scorer = _load_worker_scorer(cfg)
    bench_dict = _bench.get_benchmark_dict()

    for suite_name, n_draws in suite_assignments:
        if n_draws <= 0:
            continue
        t0 = time.time()
        n_tasks = bench_dict[suite_name]().n_tasks
        pools_per_task = [pool_loader(suite_name, ti) for ti in range(n_tasks)]
        grouped = _sample_suite_draws(
            pools_per_task, scorer, nade_cfg, n_draws, rng,
        )

        total_succ = 0
        all_w: List[float] = []
        for ti, draws in grouped.items():
            records = run_task_with_draws(
                suite_name, ti, draws, policy, cfg, rng,
            )
            data_utils.save_shard(
                out_root, suite_name, ti, records,
                save_trajectories=save_traj,
                append=True,
            )
            total_succ += sum(int(r.success) for r in records)
            all_w.extend(float(r.is_weight) for r in records)

        mean_w = float(np.mean(all_w)) if all_w else 0.0
        dur = time.time() - t0
        # Per-task distribution: useful when checking NADE concentration.
        per_task_counts = ", ".join(
            f"t{ti}:{len(d)}" for ti, d in sorted(grouped.items())
        )
        print(
            f"[w{worker_id}] {suite_name}  draws={n_draws}  "
            f"{total_succ}/{n_draws} success  mean_w={mean_w:.3f}  "
            f"({per_task_counts})  ({dur:.1f}s)"
        )


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------


def compute_weighted_metrics(out_root: Path) -> Dict[str, Any]:
    """Crash rate under the uniform-init reference distribution.

    ``E_unif[failure]  =  mean( IS_weight * (1 - success) )``

    Reported both globally and per (suite, task).
    """
    per_task: Dict[str, Dict[str, float]] = {}
    all_w, all_fail = [], []
    for shard in data_utils.iter_shards(out_root):
        recs = data_utils.load_shard(shard, with_trajectories=False)
        if not recs:
            continue
        w = np.array([r.is_weight for r in recs], dtype=np.float64)
        fail = np.array([0 if r.success else 1 for r in recs], dtype=np.float64)
        key = f"{recs[0].suite_name}/task_{recs[0].task_index:02d}"
        per_task[key] = {
            "n": int(len(recs)),
            "raw_fail_rate": float(fail.mean()),
            "weighted_fail_rate": float((w * fail).mean()),
            "mean_is_weight": float(w.mean()),
        }
        all_w.extend(w.tolist()); all_fail.extend(fail.tolist())

    aw = np.asarray(all_w); af = np.asarray(all_fail)
    aggregate = {
        "n": int(len(all_w)),
        "raw_fail_rate": float(af.mean()) if af.size else 0.0,
        "weighted_fail_rate": float((aw * af).mean()) if af.size else 0.0,
        "mean_is_weight": float(aw.mean()) if aw.size else 0.0,
    }
    return {"aggregate": aggregate, "per_task": per_task}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _suite_worker_split(
    suites: List[str], episodes_per_suite: int, num_workers: int,
) -> List[List[Tuple[str, int]]]:
    """For the suite-level path, divide each suite's episode quota across workers.

    Returns ``buckets[worker_id]`` = ``[(suite, n_draws_for_this_worker), ...]``.
    Uses remainder-first allocation so the total stays exactly equal to
    ``episodes_per_suite * len(suites)``.
    """
    buckets: List[List[Tuple[str, int]]] = [[] for _ in range(num_workers)]
    for suite in suites:
        base = episodes_per_suite // num_workers
        rem  = episodes_per_suite - base * num_workers
        for w in range(num_workers):
            n = base + (1 if w < rem else 0)
            if n > 0:
                buckets[w].append((suite, n))
    return buckets


def _resolve_gpu_assignment(cfg: Dict[str, Any], num_workers: int) -> List[Optional[int]]:
    """``cfg['gpus']`` (list of physical GPU ids) -> per-worker assignment.

    Mirrors stage1_collect:
    - Missing key: default to 0..num_workers-1.
    - Shorter list: wrap around.
    """
    gpus_cfg = cfg.get("gpus")
    if gpus_cfg is None:
        return [i for i in range(num_workers)]
    return [int(gpus_cfg[i % len(gpus_cfg)]) for i in range(num_workers)]


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args)

    out_root = Path(cfg["output_dir"])
    out_root.mkdir(parents=True, exist_ok=True)
    use_nade = bool(cfg.get("use_nade", True))
    num_workers = max(1, int(cfg.get("num_workers", 1)))
    gpu_assignment = _resolve_gpu_assignment(cfg, num_workers)
    # spawn: each child re-imports the module, so CUDA_VISIBLE_DEVICES set
    # at the top of the worker takes effect before torch is initialized.
    ctx = mp.get_context("spawn")

    # Suite-level NADE (real run only). Dry-run + use_nade=False fall through
    # to the per-task path.
    if use_nade and not cfg["dry_run"]:
        suite_buckets = _suite_worker_split(
            list(cfg["suites"]), int(cfg["episodes_per_suite"]), num_workers,
        )
        total = int(cfg["episodes_per_suite"]) * len(cfg["suites"])
        print(
            f"Accelerated testing (suite-level NADE): {total} episodes "
            f"across {len(cfg['suites'])} suites.  mode={cfg['mode']}  "
            f"workers={num_workers}  gpus={gpu_assignment}"
        )
        if num_workers == 1:
            _worker_main_suite(0, gpu_assignment[0], suite_buckets[0], cfg, int(cfg["seed"]))
        else:
            procs = []
            for wid, bucket in enumerate(suite_buckets):
                if not bucket:
                    continue
                p = ctx.Process(
                    target=_worker_main_suite,
                    args=(wid, gpu_assignment[wid], bucket, cfg, int(cfg["seed"])),
                )
                p.start(); procs.append(p)
            for p in procs:
                p.join()
    else:
        if cfg["dry_run"]:
            # Mirror the dry-run task list from collect.py so we don't need LIBERO.
            assignments = [
                (suite, t, max(1, int(cfg["episodes_per_suite"]) // 10))
                for suite in cfg["suites"] for t in range(10)
            ]
        else:
            assignments = _build_task_assignments(cfg)

        print(
            f"Accelerated testing ({'uniform' if not use_nade else 'dry-run'}): "
            f"{sum(n for *_, n in assignments)} episodes "
            f"across {len(assignments)} (suite, task) pairs.  mode={cfg['mode']}  "
            f"gpus={gpu_assignment}"
        )

        if num_workers == 1:
            _worker_main(0, gpu_assignment[0], assignments, cfg, int(cfg["seed"]))
        else:
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

    metrics = compute_weighted_metrics(out_root)
    metrics_path = out_root / "weighted_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(
        f"[test_model] aggregate weighted fail-rate "
        f"= {metrics['aggregate']['weighted_fail_rate']:.4f}  "
        f"(raw = {metrics['aggregate']['raw_fail_rate']:.4f}, "
        f"n = {metrics['aggregate']['n']})"
    )
    print(f"[test_model] wrote {metrics_path}")


if __name__ == "__main__":
    main()
