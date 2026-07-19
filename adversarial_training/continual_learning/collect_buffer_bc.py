"""BC buffer collector — cross-suite NADE importance sampling over LIBERO demos.

Pipelines:
  1. Enumerate every demo across all suites, extract its initial MuJoCo state
     (full qpos+qvel, matching ``env.set_init_state``).
  2. Score all init-state vectors with the criticality model in one batch.
  3. Build one joint NADE proposal over the entire candidate set.
  4. Sample ``episodes_total`` draws with replacement — high-crit inits are
     drawn more often (importance sampling), but every demo has a chance.
  5. Copy each draw's demo into a single-demo HDF5 under ``output_dir``,
     preserving the IS weight as an HDF5 attribute.
  6. Emit a metadata JSON consumable by ``bc_offline.py``.

No threshold filtering — the NADE proposal handles the soft prioritisation.
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import h5py
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import yaml


def _silence_libero(fn, *args, **kwargs):
    """Call *fn* with stderr redirected to /dev/null.

    LIBERO prints ``[Warning]: datasets path ... does not exist!`` directly to
    stderr on every ``benchmark.get_benchmark_dict()`` call.  This wrapper
    suppresses that noise for the duration of *fn*.
    """
    with open(os.devnull, "w") as fnull, \
         contextlib.redirect_stderr(fnull):
        return fn(*args, **kwargs)

_THIS = Path(__file__).resolve()
_CODE_ROOT = _THIS.parents[2]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from adversarial_training.test.libero_nade import (
    build_proposal,
    importance_weight,
)
from adversarial_training.utils.task_registry import get_task_spec

# ===================================================================
# CLI / config
# ===================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collect BC buffer via cross-suite NADE over LIBERO demos.")
    p.add_argument("--config", type=Path,
                   default=Path("./adversarial_training/configs/default.yaml"))
    p.add_argument("--criticality_ckpt", type=Path, default=None)
    p.add_argument("--libero_dataset_dir", type=Path, default=None)
    p.add_argument("--output_dir", type=Path, default=None)
    p.add_argument("--suites", nargs="+", default=None)
    p.add_argument("--episodes_total", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def load_cfg(args: argparse.Namespace) -> Dict[str, Any]:
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    bc   = dict(cfg.get("bc_collect", {}))
    cont = cfg.get("continual", {})

    buf = dict(cont.get("buffer_collect", {}))
    for key in ("criticality_ckpt", "crit_hidden_dim", "crit_expansion",
                "crit_depth", "crit_dropout",
                "suites", "libero_dataset_dir", "output_dir",
                "episodes_total", "epsilon", "weight_clip", "alpha", "seed"):
        if key not in bc and key in buf:
            bc[key] = buf[key]

    if args.criticality_ckpt is not None:
        bc["criticality_ckpt"] = str(args.criticality_ckpt)
    if args.libero_dataset_dir is not None:
        bc["libero_dataset_dir"] = str(args.libero_dataset_dir)
    if args.output_dir is not None:
        bc["output_dir"] = str(args.output_dir)
    if args.suites is not None:
        bc["suites"] = args.suites
    if args.episodes_total is not None:
        bc["episodes_total"] = args.episodes_total
    if args.seed is not None:
        bc["seed"] = args.seed

    bc.setdefault("suites", ["libero_10", "libero_goal",
                              "libero_object", "libero_spatial"])
    bc.setdefault("seed", 234)
    bc.setdefault("crit_hidden_dim", 128)
    bc.setdefault("crit_expansion", 1)
    bc.setdefault("crit_depth", 4)
    bc.setdefault("crit_dropout", 0.0)
    bc.setdefault("episodes_total", 2000)
    bc.setdefault("epsilon", 0.01)
    bc.setdefault("weight_clip", 10.0)
    bc.setdefault("alpha", 3.0)
    bc.setdefault("dry_run", bool(args.dry_run))

    if not bc.get("libero_dataset_dir"):
        bc["libero_dataset_dir"] = cont.get(
            "libero_dataset_dir",
            "/mnt/hlx/SimpleVLA_libero_data/datasets")
    if not bc.get("output_dir"):
        bc["output_dir"] = cont.get(
            "accelerated_buffer",
            "/mnt/hlx/SimpleVLA_libero_data/datasets/bc_buffer")
    return bc


# ===================================================================
# Criticality model
# ===================================================================

ScorerFn = Callable[[np.ndarray], np.ndarray]


def _load_criticality_model(cfg: Dict[str, Any]) -> Tuple[Any, Any, int]:
    import torch
    from adversarial_training.utils.criticality_model import (
        CriticalityModel, CriticalityModelConfig,
    )
    ckpt = Path(cfg["criticality_ckpt"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(ckpt, map_location=device, weights_only=False)
    if "input_proj.weight" not in state:
        raise RuntimeError(f"No input_proj.weight tensor in {ckpt}")
    max_D = int(state["input_proj.weight"].shape[1])
    model = CriticalityModel(CriticalityModelConfig(
        input_dim=max_D,
        hidden_dim=int(cfg["crit_hidden_dim"]),
        expansion=int(cfg["crit_expansion"]),
        depth=int(cfg["crit_depth"]),
        dropout=float(cfg["crit_dropout"]),
    )).to(device).eval()
    model.load_state_dict(state)
    print(f"[collect_buffer_bc] criticality model: "
          f"input_dim={max_D}  hidden={cfg['crit_hidden_dim']}  "
          f"expansion={cfg['crit_expansion']}  depth={cfg['crit_depth']}  "
          f"dropout={cfg['crit_dropout']}")
    return model, device, max_D


def build_scorer(model, device, max_D: int) -> ScorerFn:
    import torch

    def _score(candidates: np.ndarray) -> np.ndarray:
        x = np.asarray(candidates, dtype=np.float32)
        if x.shape[1] < max_D:
            pad = np.zeros((x.shape[0], max_D - x.shape[1]), dtype=np.float32)
            x = np.concatenate([x, pad], axis=1)
        t = torch.from_numpy(x).to(device)
        return model.criticality_score(t).detach().cpu().numpy()
    return _score


def build_dry_scorer(seed: int) -> ScorerFn:
    rng = np.random.default_rng(seed)

    def _score(candidates: np.ndarray) -> np.ndarray:
        return rng.uniform(0.0, 1.0, size=candidates.shape[0]).astype(np.float64)
    return _score


# ===================================================================
# Demo discovery — iterate every demo across all suites
# ===================================================================

@dataclass
class _DemoEntry:
    """Lightweight handle for one demo in the LIBERO dataset."""
    h5_path: str
    demo_key: str
    suite_name: str
    task_index: int
    task_description: str        # e.g. "pick up the black bowl on the ..."
    init_state: np.ndarray       # full MuJoCo qpos+qvel
    pool_index: int              # index into the flat candidate list


def _collect_all_demos(
    libero_data_dir: str,
    suites: List[str],
    dry_run: bool = False,
) -> List[_DemoEntry]:
    """Walk every task in every suite and collect one ``_DemoEntry`` per demo."""
    from libero.libero import benchmark

    entries: List[_DemoEntry] = []
    pool_idx = 0

    for suite_name in suites:
        h5_paths = sorted(glob.glob(
            os.path.join(libero_data_dir, suite_name, "*.hdf5")))

        task_suite = _silence_libero(
            lambda: benchmark.get_benchmark_dict()[suite_name]())
        n_tasks = task_suite.n_tasks

        for task_index in range(n_tasks):
            # Find the right HDF5 for this task
            task = task_suite.get_task(task_index)
            bddl_stem = os.path.splitext(task.bddl_file)[0]
            h5_path = None
            for p in h5_paths:
                if bddl_stem in os.path.basename(p):
                    h5_path = p
                    break
            if h5_path is None and task_index < len(h5_paths):
                h5_path = h5_paths[task_index]
            if h5_path is None:
                continue

            task_language = getattr(task, "language", "") or ""
            with h5py.File(h5_path, "r") as f:
                for demo_key in sorted(f["data"].keys()):
                    states = f["data"][demo_key]["states"][:]
                    init_state = states[0].astype(np.float32).copy()
                    entries.append(_DemoEntry(
                        h5_path=h5_path,
                        demo_key=demo_key,
                        suite_name=suite_name,
                        task_index=task_index,
                        task_description=task_language,
                        init_state=init_state,
                        pool_index=pool_idx,
                    ))
                    pool_idx += 1

    return entries


# ===================================================================
# HDF5 writer
# ===================================================================

def _copy_demo(
    entry: _DemoEntry,
    output_dir: Path,
    demo_index: int,
    is_weight: float,
) -> Path:
    out_dir = output_dir / entry.suite_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"task_{entry.task_index:02d}_demo_{demo_index:04d}.hdf5"

    with h5py.File(entry.h5_path, "r") as src:
        demo = src["data"][entry.demo_key]
        # Use task description from the LIBERO task suite (entry.task_description),
        # NOT from model_file XML (which doesn't exist in these HDF5s).
        task_desc = entry.task_description

        with h5py.File(out_path, "w") as dst:
            dst_grp = dst.create_group("data").create_group("demo_0")
            for key in ("actions",
                        "obs/agentview_rgb", "obs/eye_in_hand_rgb",
                        "obs/ee_pos", "obs/ee_ori",
                        "obs/gripper_states", "obs/joint_states"):
                if key in demo:
                    dst_grp.create_dataset(
                        key, data=np.asarray(demo[key]), compression="gzip")
            dst_grp.attrs["task_description"] = task_desc
            dst_grp.attrs["suite_name"] = entry.suite_name
            dst_grp.attrs["task_index"] = entry.task_index
            dst_grp.attrs["is_weight"] = float(is_weight)
    return out_path


# ===================================================================
# Metadata
# ===================================================================

def build_meta_json(output_dir: Path, suites: List[str]) -> Dict[str, Any]:
    datalist: List[Dict[str, Any]] = []
    for suite in suites:
        suite_dir = output_dir / suite
        if not suite_dir.is_dir():
            continue
        for h5_path in sorted(glob.glob(str(suite_dir / "*.hdf5"))):
            fname = os.path.basename(h5_path)
            m = re.match(r"task_(\d+)_demo_(\d+)\.hdf5", fname)
            task_idx = int(m.group(1)) if m else -1
            task_desc = f"{suite}_task_{task_idx:02d}"
            is_weight = 1.0
            try:
                with h5py.File(h5_path, "r") as f:
                    dg = f["data"]["demo_0"]
                    task_desc = str(dg.attrs.get("task_description", task_desc))
                    is_weight = float(dg.attrs.get("is_weight", 1.0))
            except Exception:
                pass
            datalist.append({
                "path": h5_path, "task": task_desc,
                "subset": suite, "task_index": task_idx,
                "weight": is_weight,
            })
    return {
        "dataset_name": "libero_hdf5",
        "data_dir": str(output_dir),
        "datalist": datalist,
        "num_episodes": len(datalist),
        "observation_key": ["obs/agentview_rgb", "obs/eye_in_hand_rgb"],
        "action_key": "actions",
        "state_dim": 8,
        "action_dim": 7,
        "fps": 10,
    }


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    args = parse_args()
    cfg = load_cfg(args)

    out_root = Path(cfg["output_dir"])
    out_root.mkdir(parents=True, exist_ok=True)
    libero_dir = cfg["libero_dataset_dir"]
    suites = list(cfg["suites"])
    n_total = int(cfg["episodes_total"])
    dry_run = bool(cfg.get("dry_run", False))
    rng = np.random.default_rng(int(cfg["seed"]))

    print(
        f"[collect_buffer_bc] cross-suite NADE over LIBERO demos\n"
        f"  source={libero_dir}  suites={suites}\n"
        f"  episodes_total={n_total}  epsilon={cfg['epsilon']}  "
        f"weight_clip={cfg['weight_clip']}  alpha={cfg['alpha']}\n"
        f"  output → {out_root}")

    # ── Load scorer ──
    scorer: ScorerFn
    if dry_run:
        scorer = build_dry_scorer(int(cfg["seed"]))
    else:
        model, device, max_D = _load_criticality_model(cfg)
        scorer = build_scorer(model, device, max_D)

    # ── Collect all demos ──
    entries = _collect_all_demos(libero_dir, suites, dry_run)
    M = len(entries)
    print(f"[collect_buffer_bc] {M} total demos across {len(suites)} suites")
    if M == 0:
        print("[collect_buffer_bc] no demos found — aborting")
        return

    # ── Batch-score all init states ──
    #   Init-state dims vary across tasks — right-pad to the max before scoring.
    max_dim = max(len(e.init_state) for e in entries)
    all_vectors = np.stack([
        np.pad(e.init_state, (0, max_dim - len(e.init_state)))
        for e in entries
    ], axis=0).astype(np.float32)
    p_fail = np.asarray(scorer(all_vectors), dtype=np.float64)
    print(f"[collect_buffer_bc] scored {M} init states  "
          f"P(fail) ∈ [{p_fail.min():.4f}, {p_fail.max():.4f}]  "
          f"mean={p_fail.mean():.4f}  "
          f"frac≥0.5={float((p_fail >= 0.5).mean()):.1%}")

    # ── Build joint NADE proposal ──
    q, _used = build_proposal(
        p_fail,
        epsilon=float(cfg["epsilon"]),
        criticality_threshold=float(cfg.get("criticality_threshold", -1.0)),
    )

    # ── Sample ──
    draws = rng.choice(M, size=n_total, p=q, replace=True)

    # ── Per-suite minimum enforcement ──
    per_suite_min = int(cfg.get("per_suite_min", 0))
    if per_suite_min > 0:
        suite_indices: Dict[str, list] = {}
        for i, e in enumerate(entries):
            suite_indices.setdefault(e.suite_name, []).append(i)
        for s in suites:
            current = int(sum(1 for idx in draws if entries[idx].suite_name == s))
            deficit = max(0, per_suite_min - current)
            if deficit > 0:
                pool = suite_indices.get(s, [])
                if pool:
                    extra = rng.choice(pool, size=deficit, replace=True)
                    draws = np.concatenate([draws, extra])
                    print(f"[collect_buffer_bc] +{deficit} extra draws for {s} "
                          f"(was {current}, min={per_suite_min})")

    # Count occurrences per demo for stats
    unique, counts = np.unique(draws, return_counts=True)
    max_dup = int(counts.max())

    per_suite: Dict[str, int] = {}
    per_task: Dict[Tuple[str, int], int] = {}
    for idx in draws:
        e = entries[idx]
        per_suite[e.suite_name] = per_suite.get(e.suite_name, 0) + 1
        per_task[(e.suite_name, e.task_index)] = per_task.get(
            (e.suite_name, e.task_index), 0) + 1

    print(f"[collect_buffer_bc] sampled {len(draws)} draws  "
          f"unique={len(unique)}/{M}  max repeats={max_dup}")
    for s in suites:
        n_s = per_suite.get(s, 0)
        t_with = sum(1 for (sn, _), c in per_task.items()
                     if sn == s and c > 0)
        print(f"  {s}: {n_s} draws  ({t_with} tasks)")

    # ── Copy demos ──
    demo_counter: Dict[Tuple[str, int], int] = {}
    for draw_idx in draws:
        e = entries[draw_idx]
        key = (e.suite_name, e.task_index)
        di = demo_counter.get(key, 0)
        demo_counter[key] = di + 1

        w = importance_weight(draw_idx, q, M, float(cfg["weight_clip"]))
        _copy_demo(e, out_root, demo_index=di, is_weight=float(w))

    total_files = sum(demo_counter.values())
    print(f"[collect_buffer_bc] wrote {total_files} HDF5 files")

    # ── Metadata ──
    meta = build_meta_json(out_root, suites)
    meta_path = out_root / "bc_train_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[collect_buffer_bc] metadata → {meta_path}  "
          f"({meta['num_episodes']} demos)")


if __name__ == "__main__":
    main()
