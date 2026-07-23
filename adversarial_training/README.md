# Criticality-Driven Accelerated Testing & Continual Learning for SimpleVLA on LIBERO

This module ports the ManiSkill-based pipeline in `other_source/criticality/`
onto LIBERO + SimpleVLA. The key difference from the reference:

> **LIBERO has no step-level environment perturbation.**
> Disturbances are **episode-level initial-state samples** —
> i.e. randomized initial poses of objects in the scene.
> The criticality model learns `P(failure | initial_state)`,
> and accelerated testing importance-samples the initial-state distribution.

## Train / eval init-pool split

LIBERO ships **50 hand-picked initial states** per task; those are the
canonical eval set used by every reported SOTA number.

- **Eval** (`test/test_model.py`) draws from the official 50-init pool.
  Anything that wants to be SOTA-comparable must score against this exact set.
- **Training data** (`stage1/stage1_collect.py` + `continual_learning/collect_buffer_bc.py`)
  draws from an **extended generated pool** produced by
  `utils/generate_inits.py` — same distribution (re-seeded `env.reset()` with
  collision rejection), but the concrete vectors are disjoint from the
  official 50. This is what prevents training data from leaking into the
  benchmark.

The generated pool is built once and cached at
`continual.generated_pool_cache` (default
`/mnt/.../datasets/generated_inits/<suite>/task_NN.npy`).

## Pipeline

```
utils/generate_inits.py        env.seed + env.reset to build the extended pool
        │                      writes <cache>/<suite>/task_NN.npy (K, D)
        ▼
stage1/stage1_collect.py       roll SimpleVLA on the *generated* pool
        │                      one row per episode: (init_state, success)
        ▼
stage1/stage1_train.py         train MLP criticality model
        │                      P(failure | init_state) per task
        ▼
test/test_model.py             accelerated testing on the *official* pool
        │                      NADE importance sampling → SOTA-comparable
        │                      weighted fail-rate
        │
continual_learning/
  collect_buffer_bc.py         NADE-weighted importance sampling over LIBERO
        │                      demos (no env rollouts needed). Produces
        │                      single-demo HDF5 shards + bc_train_meta.json.
        │
  collect_buffer_random.py     Uniform random sampling (no criticality model
        │                      needed) — baseline / control group. Every demo
        │                      has equal probability. All weights = 1.0.
        ▼
  bc_offline.py                BC fine-tuning of SimpleVLA:
                                 L = MSE(action, action_gt)
                               with per-demo IS weight from the buffer.
                               VLM backbone can be frozen to prevent
                               catastrophic forgetting.
  run_bc.sh                    wrapper for the two steps above.
```

## Layout

- `utils/task_registry.py` — per-suite metadata (max steps, init dim, ...)
- `utils/init_state.py` — `PoolSampler` / `GeneratedPoolSampler` / `PerturbPoolSampler`,
  plus the `apply_init_state(env, init)` helper and `load_generated_pool(...)`.
- `utils/libero_env.py` — env factory + obs packer; `load_official_pool(...)`
  fetches the 50-init benchmark pool without opening an env.
- `utils/generate_inits.py` — CLI tool for building the extended init cache.
- `utils/policy.py` — local SimpleVLA wrapper.
- `utils/criticality_model.py` — multi-task MLP classifier.
- `utils/data_utils.py` — shard I/O. `save_shard(..., append=True)` is atomic
  (`.tmp` + `os.replace`) and merges trajectory `.pkl` companions.
- `test/test_model.py` — accelerated NADE testing on the official 50-init pool.
- `test/libero_nade.py` — NADE proposal building and IS-weight computation.
- `stage1/stage1_collect.py` — collect rollouts on the generated pool.
- `stage1/stage1_train.py` — train the criticality MLP model.
- `continual_learning/collect_buffer_bc.py` — criticality-guided (NADE) demo sampler.
- `continual_learning/collect_buffer_random.py` — uniform random demo sampler (baseline).
- `continual_learning/bc_offline.py` — BC fine-tuning script.
- `continual_learning/run_bc.sh` — end-to-end BC pipeline wrapper.
- `continual_learning/run_bc_train_only.sh` — training-only wrapper.
- `configs/default.yaml` — hyperparameters shared across stages.
- `analysis.py`, `draw_RHF.py` — analysis and visualization utilities.

## Status

All phases are implemented:
- **Criticality training** (`stage1_collect.py` + `stage1_train.py`) — collect rollouts on
  the generated pool and train the MLP criticality model.
- **Accelerated testing** (`test/test_model.py`) — NADE importance sampling on the
  official 50-init pool.
- **Continual learning** — BC fine-tuning pipeline using criticality-guided demo
  sampling (`collect_buffer_bc.py` + `bc_offline.py`).

## Run order

```bash
# 0) Build the extended generated init pool (idempotent; skips cached tasks).
python -m adversarial_training.utils.generate_inits \
    --cache_dir /mnt/hlx/SimpleVLA_libero_data/datasets/generated_inits \
    --num 500 \
    --suites libero_10 libero_goal libero_object libero_spatial

# 1) Collect criticality training rollouts on the generated pool.
python adversarial_training/stage1/stage1_collect.py \
    --config adversarial_training/configs/default.yaml

# 2) Train criticality MLP.
python adversarial_training/stage1/stage1_train.py \
    --config adversarial_training/configs/default.yaml
    
# 3) Accelerated testing on the official 50-init pool (SOTA-comparable).
python adversarial_training/test/test_model.py \
    --config adversarial_training/configs/default.yaml

# 4) Continual learning: collect_buffer → bc_offline.
#    NADE importance-sampled buffer:
bash adversarial_training/continual_learning/run_bc.sh
#    Uniform random baseline (no criticality model needed):
python adversarial_training/continual_learning/collect_buffer_random.py \
    --libero_dataset_dir /mnt/hlx/SimpleVLA_libero/datasets/metas \
    --output_dir /mnt/hlx/SimpleVLA_libero_data/datasets/bc_buffer_random \
    --episodes_total 800
#    Then train on the random buffer:
accelerate launch \
    --num_processes 4 --mixed_precision bf16 \
    adversarial_training/continual_learning/bc_offline.py \
    --bc_meta /mnt/hlx/SimpleVLA_libero_data/datasets/bc_buffer_random/bc_train_meta.json \
    --output_dir /mnt/hlx/SimpleVLA_libero_data/runs/bc_random \
    --iters 100000
# Reuse a previously-collected buffer:
SKIP_COLLECT=1 bash adversarial_training/continual_learning/run_bc.sh
# Train only (buffer already exists):
bash adversarial_training/continual_learning/run_bc_train_only.sh
```

## Config keys, by stage

| Stage | Key block | Init-pool source |
|---|---|---|
| `generate_inits.py` | CLI flags | writes to `continual.generated_pool_cache` |
| `stage1_collect.py` | `collect:` (default `init_sampler: generated`) | generated pool |
| `stage1_train.py` | `train:` | n/a (reads npz shards) |
| `test_model.py` | `test:` | official 50-init pool (`load_official_pool`) |
| `collect_buffer_bc.py` | `bc_collect:` | LIBERO official demos + criticality model |
| `collect_buffer_random.py` | CLI flags only | LIBERO official demos (uniform random) |
| `bc_offline.py` | `bc_training:` | n/a (reads HDF5 buffer + meta JSON) |

`collect.generated_pool_cache` and `continual.buffer_collect.generated_pool_cache`
both fall back to top-level `continual.generated_pool_cache`, so by default
all training-data collectors share one cache path.

## Picking which policy each stage rolls

The rollout scripts (`stage1_collect`, `test_model`) each load a SimpleVLA
checkpoint. Across continual-learning rounds you typically want a
**different** checkpoint per round — the just-fine-tuned model — so the
rollouts reflect the current policy.

The policy block is **deep-merged** on top of `collect.policy`: any key you
set on the stage's own block (or via CLI) overrides; everything else
inherits. For example, to point only `checkpoint` at a new run without
re-specifying `norm_stats` / `smolvlm_model`:

```yaml
collect:
  policy:
    checkpoint: /mnt/.../models/SimVLA-LIBERO       # baseline
    norm_stats: ./norm_stats/libero_norm.json
    smolvlm_model: "/mnt/.../models/SmolVLM-500M-Instruct"

test:
  policy:
    checkpoint: /mnt/.../runs/bc_continual/ckpt-10000   # inherits norm_stats, smolvlm_model
```

Or override on the CLI without touching the YAML:

```bash
python adversarial_training/test/test_model.py \
    --policy_checkpoint /mnt/.../runs/bc_continual/ckpt-10000

python adversarial_training/continual_learning/collect_buffer_bc.py \
    --policy_checkpoint /mnt/.../runs/bc_continual/ckpt-10000
```

Both scripts also accept `--policy_norm_stats` and `--policy_smolvlm_model`.

## BC continual-learning pipeline

The continual-learning loop uses **behavior cloning (BC)** rather than PPO.
There are two buffer collection strategies available:

### 1. NADE importance-sampled buffer (default)

**`collect_buffer_bc.py`** — enumerates all LIBERO official demos across
suites, scores each demo's initial state with the criticality model, builds
a joint NADE proposal, and importance-samples `episodes_total` demos.
A `per_suite_min` lower bound ensures every suite gets enough samples.
Output: single-demo HDF5 shards under `bc_collect.output_dir` +
`bc_train_meta.json`.

### 2. Uniform random buffer (baseline / control group)

**`collect_buffer_random.py`** — uniformly random sampling from all LIBERO
demos, **without** a criticality model. Every demo has equal probability
of being selected, and all IS weights are set to 1.0. This serves as a
baseline to measure the benefit of criticality-guided sampling.

```bash
python adversarial_training/continual_learning/collect_buffer_random.py \
    --libero_dataset_dir /mnt/hlx/SimpleVLA_libero/datasets/metas \
    --output_dir /mnt/hlx/SimpleVLA_libero_data/datasets/bc_buffer_random \
    --episodes_total 800 \
    --seed 42
```

### Training

**`bc_offline.py`** — loads the filtered demo buffer and fine-tunes
SimpleVLA with standard BC loss `MSE(action, action_gt)`. The VLM backbone
is frozen by default (`freeze_vlm: true`, implemented via `learning_coef=0`)
to prevent catastrophic forgetting on small fine-tuning datasets.

```bash
# Full pipeline (NADE sampling)
bash adversarial_training/continual_learning/run_bc.sh

# Skip buffer collection (reuse existing)
SKIP_COLLECT=1 bash adversarial_training/continual_learning/run_bc.sh

# Training only
bash adversarial_training/continual_learning/run_bc_train_only.sh
```

> **Note:** The `bc_training` block in `configs/default.yaml` sets default
> values for `bc_meta`, `output_dir`, `iters`, etc.  These YAML values
> **override** CLI arguments — if you need to use a different buffer or
> output dir, either (a) edit the YAML, (b) temporarily comment out the
> relevant key, or (c) pass `--config /dev/null` to skip YAML entirely.

## Notes on the variance-based perturb dims

`PerturbPoolSampler` (legacy path; only relevant when `init_sampler=perturb_pool`)
defaults to **variance-based dim detection**: any dim with nonzero variance
across the pool is treated as an object dim. Robot dims share the home pose
across every pool entry and so have zero variance, which isolates per-object
qpos/qvel exactly without hard-coding the Franka layout.

Setting `perturb_fraction: 0.35` falls back to the old "last 35% of dims"
heuristic.

## Buffer append semantics

`stage1_collect.py` re-runs do **not** overwrite existing shards. `save_shard`
appends the new records to any existing `.npz` (and merges the `.traj.pkl`
companion when trajectories are saved), writing atomically via `.tmp` +
`os.replace`. The `init_state` dim is validated for consistency on append.
