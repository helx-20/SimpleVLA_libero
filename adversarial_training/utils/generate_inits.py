"""Generate an extended initial-state pool per (suite, task).

LIBERO ships 50 hand-picked initial states per task (``.init`` files); the
benchmark protocol uses those exact 50 for evaluation. For training the
criticality model and producing the continual-learning rollout buffer we
want *more* inits, drawn from the same physically-valid distribution but
disjoint from the eval set.

The mechanism is the same one LIBERO used to create the official pool:
each task's BDDL declares per-object placement regions; ``env.reset()``
runs the placement initializer (with collision rejection) and produces a
fresh valid scene. Re-seeding ``env`` each call gives independent draws.

For each (suite, task_index) we:

    1. Open the env once.
    2. Loop K times: ``env.seed(seed_k); env.reset(); state = env.sim.get_state().flatten()``.
    3. Stack into (K, D) float32 and write to ``cache_dir/<suite>/task_NN.npy``.

The script is idempotent — a task whose cache file already exists is
skipped unless ``--overwrite`` is passed.

CLI::

    python -m adversarial_training.utils.generate_inits \\
        --cache_dir cache/generated_inits \\
        --num 500 \\
        --suites libero_10 libero_goal libero_object libero_spatial
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, List

import numpy as np

_THIS = Path(__file__).resolve()
_CODE_ROOT = _THIS.parents[2]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from adversarial_training.utils.init_state import generated_pool_path
from adversarial_training.utils.libero_env import load_official_pool, make_libero_env


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate extended init-state pools per (suite, task).")
    p.add_argument("--cache_dir", type=Path, required=True,
                   help="Output dir; one .npy per (suite, task) under <cache_dir>/<suite>/task_NN.npy")
    p.add_argument("--num", type=int, default=500,
                   help="Number of init states to generate per task.")
    p.add_argument("--suites", nargs="+",
                   default=["libero_10", "libero_goal", "libero_object", "libero_spatial"])
    p.add_argument("--base_seed", type=int, default=10_000,
                   help="Seeds used are base_seed + k. Pick something disjoint from any benchmark seeds.")
    p.add_argument("--overwrite", action="store_true",
                   help="Regenerate even if the cache file already exists.")
    return p.parse_args()


def _get_flat_state(env: Any) -> np.ndarray:
    """Return the flat MuJoCo state vector — same format LIBERO's pool uses."""
    sim = getattr(env, "sim", None)
    if sim is None:
        inner = getattr(env, "env", None)
        sim = getattr(inner, "sim", None) if inner is not None else None
    if sim is None:
        raise RuntimeError(
            "Could not locate `sim` attribute on the LIBERO env. "
            "LIBERO API may have changed; expected env.sim or env.env.sim."
        )
    return np.asarray(sim.get_state().flatten(), dtype=np.float32)


def generate_one_task(
    suite_name: str,
    task_index: int,
    num: int,
    base_seed: int,
) -> np.ndarray:
    env, _task_desc, _official_pool = make_libero_env(suite_name, task_index, seed=base_seed)
    # Sanity check: generated state dim must match the official pool's dim,
    # otherwise downstream `env.set_init_state(vec)` will silently misalign.
    try:
        D_official = int(_official_pool.shape[1])
    except Exception:
        D_official = -1

    out: List[np.ndarray] = []
    try:
        for k in range(int(num)):
            env.seed(int(base_seed) + k)
            env.reset()
            vec = _get_flat_state(env)
            if D_official > 0 and vec.shape[0] != D_official:
                raise RuntimeError(
                    f"{suite_name}/task_{task_index:02d}: generated state dim "
                    f"{vec.shape[0]} != official pool dim {D_official}. "
                    f"Aborting to avoid corrupted cache."
                )
            out.append(vec)
    finally:
        try:
            env.close()
        except Exception:
            pass
    return np.stack(out, axis=0).astype(np.float32)


def main() -> None:
    args = parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    from libero.libero import benchmark as _bench
    benches = _bench.get_benchmark_dict()

    total = 0
    skipped = 0
    for suite_name in args.suites:
        suite_obj = benches[suite_name]()
        n_tasks = int(suite_obj.n_tasks)
        for task_index in range(n_tasks):
            out_path = generated_pool_path(args.cache_dir, suite_name, task_index)
            if out_path.exists() and not args.overwrite:
                skipped += 1
                print(f"  {suite_name}/task_{task_index:02d}: cached -> skip ({out_path})")
                continue

            t0 = time.time()
            pool = generate_one_task(
                suite_name, task_index, num=args.num, base_seed=args.base_seed,
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_path, pool)
            total += pool.shape[0]
            dur = time.time() - t0
            print(f"  {suite_name}/task_{task_index:02d}: wrote {pool.shape} -> {out_path}  ({dur:.1f}s)")

    print(f"[generate_inits] done. wrote {total} inits, skipped {skipped} cached tasks.")


if __name__ == "__main__":
    main()
