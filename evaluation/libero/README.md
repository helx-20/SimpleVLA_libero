# Evaluation on LIBERO

## Overview

This directory contains evaluation scripts for SimVLA on LIBERO benchmarks. There are three evaluation modes:

| Mode | Script | Description |
|------|--------|-------------|
| **Single model** | `libero_client.py` | One model on all episodes |
| **Routed** | `run_eval_routed.sh` | Base model for easy episodes, FT model for hard episodes (criticality-gated) |
| **A/B comparison** | `libero_client.py --ab_compare` | Both Base and FT on same hard episodes for direct comparison |

---

## 1. Environment Setup

LIBERO requires its own conda environment (`libero`), while SimVLA uses `simplevla`:

```bash
# LIBERO env
conda create -n libero python=3.8.13
conda activate libero
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /mnt/hlx/LIBERO
cd /mnt/hlx/LIBERO
pip install -r requirements.txt
pip install -e .

# SimVLA env (already set up)
conda activate simplevla
```

Set environment variables:

```bash
export LIBERO_ROOT=/mnt/hlx/LIBERO
export PYTHONPATH=${LIBERO_ROOT}:${PYTHONPATH}
```

---

## 2. Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--checkpoint` | YuankaiLuo/SimVLA-LIBERO | Base model (HF ID or local path) |
| `--ft_checkpoint` | - | Fine-tuned model for routed/AB mode |
| `--criticality_ckpt` | - | Criticality model checkpoint |
| `--criticality_threshold` | 0.5 | Routing threshold: crit > threshold → FT model |
| `--task_suite` | libero_spatial | One of: libero_spatial, libero_object, libero_goal, libero_10 |
| `--num_trials` | 50 | Episodes per task (50 init states → 50 episodes) |
| `--task_start / --task_end` | 0 / n_tasks | Task range for parallel sharding |
| `--ab_compare` | false | Enable A/B mode: test both models on same hard episodes |
| `--no_video` | false | Disable video recording for faster eval |

---

## 3. Single Model Evaluation

Test one model on all episodes (no routing):

```bash
conda activate simplevla
CUDA_VISIBLE_DEVICES=0 python -u libero_client.py \
    --client_type local \
    --checkpoint YuankaiLuo/SimVLA-LIBERO \
    --norm_stats ../../norm_stats/libero_norm.json \
    --smolvlm_model HuggingFaceTB/SmolVLM-500M-Instruct \
    --task_suite libero_spatial \
    --num_trials 50 \
    --no_video
```

---

## 4. Routed Evaluation

Base model on easy episodes (crit ≤ threshold), FT model on hard episodes (crit > threshold):

```bash
bash run_eval_routed.sh <base_ckpt> <ft_ckpt> <criticality_ckpt> [num_trials] [prefix] "<gpus>"

# Example: threshold=0.5
cd /mnt/hlx/SimpleVLA_libero/evaluation/libero
bash run_eval_routed.sh \
    YuankaiLuo/SimVLA-LIBERO \
    /mnt/hlx/SimpleVLA_libero_data/runs/bc_continual/ckpt-20000 \
    /mnt/hlx/SimpleVLA_libero_data/runs/criticality/best.pt \
    50 "eval_routed" "0 5 6 7"
```

Override threshold via env: `CRIT_THRESHOLD=0.5 bash run_eval_routed.sh ...`

---

## 5. A/B Comparison Evaluation

The most informative mode: on crit > threshold episodes, **both Base and FT** run on the **same init state**, enabling direct per-episode comparison.

```bash
conda activate simplevla
CUDA_VISIBLE_DEVICES=0 python -u libero_client.py \
    --client_type local \
    --checkpoint YuankaiLuo/SimVLA-LIBERO \
    --ft_checkpoint /path/to/ft/ckpt \
    --norm_stats ../../norm_stats/libero_norm.json \
    --smolvlm_model HuggingFaceTB/SmolVLM-500M-Instruct \
    --criticality_ckpt /path/to/criticality/best.pt \
    --criticality_threshold 0.5 \
    --ab_compare \
    --task_suite libero_10 \
    --num_trials 50 \
    --no_video
```

### Output Format

Each episode prints:

```
Easy:    [OK][base] Task 9 Ep 0: SUCCESS (steps=128)
Hard:    [OK][ft] [OK][base] Task 8 Ep 3: FT=SUCCESS Base=SUCCESS
         [FAIL][ft] [OK][base] Task 8 Ep 5: FT=FAILURE Base=SUCCESS   ← FT wins
         [OK][ft] [FAIL][base] Task 8 Ep 7: FT=SUCCESS Base=FAILURE   ← Base wins
```

Final summary per suite:

```
A/B Comparison Results — libero_10
============================================================
  Easy episodes (crit≤0.5): 355
    Base only:  342/355 (96.3%)
  Hard episodes (crit>0.5): 145
    FT:         128/145 (88.3%)
    Base:       124/145 (85.5%)
  ---
  Base overall: 466/500 (93.2%)
  Routed (FT on hard, Base on easy): 470/500 (94.0%)
```

### Key Metrics

- **FT on hard vs Base on hard**: Direct comparison on same init states
- **FT独赢**: Episodes where FT succeeds but Base fails
- **Base独赢**: Episodes where Base succeeds but FT fails
- **Routed**: FT on hard + Base on easy = the hybrid system's performance
- **Δ**: Routed − Base overall = benefit of using FT model

---

## 6. Parallel Evaluation (8 GPUs)

Split tasks across GPUs for speed (~1.5h for all 4 suites):

```bash
source /mnt/miniconda3/bin/activate simplevla

# libero_10 on GPU 0 (all 10 tasks, or split for speed)
CUDA_VISIBLE_DEVICES=0 python -u libero_client.py \
    --ab_compare --task_suite libero_10 --num_trials 50 \
    --task_start 0 --task_end 10 ... &

# Other 3 suites on separate GPUs
CUDA_VISIBLE_DEVICES=5 python -u libero_client.py \
    --ab_compare --task_suite libero_goal --num_trials 50 ... &
CUDA_VISIBLE_DEVICES=6 python -u libero_client.py \
    --ab_compare --task_suite libero_object --num_trials 50 ... &
CUDA_VISIBLE_DEVICES=7 python -u libero_client.py \
    --ab_compare --task_suite libero_spatial --num_trials 50 ... &
```

### Task Splitting for libero_10

liberoo_10 is the slowest suite — split across 4 GPUs for balanced speed:

```bash
for gpu_tasks in "0:0:3" "1:3:6" "2:6:8" "3:8:10"; do
  gpu=${gpu_tasks%%:*}; rest=${gpu_tasks#*:}
  tstart=${rest%:*}; tend=${rest#*:}
  CUDA_VISIBLE_DEVICES=$gpu python -u libero_client.py \
      --ab_compare --task_suite libero_10 --num_trials 50 \
      --task_start $tstart --task_end $tend ... &
done
```

---

## 7. Expected Results (Reference)

Base model (YuankaiLuo/SimVLA-LIBERO) reference performance on 50 trials × 10 tasks = 500 episodes per suite:

| Suite | Base success | Hard ep (crit>0.5) |
|-------|-------------|-------------------|
| libero_10 | ~96.4% | ~41% |
| libero_goal | ~97.8% | ~6% |
| libero_object | ~99.2% | ~7% |
| libero_spatial | ~98.6% | ~9% |

FT model should exceed Base on hard episodes, with Routed overall > Base.

---

## 8. Threshold Selection

Optimal threshold varies by suite:

| Suite | Recommended | Reason |
|-------|------------|--------|
| libero_10 | 0.5 | FT massively outperforms Base on moderate-hard episodes |
| libero_goal | 0.5 | FT slightly better |
| libero_object | 0.5-0.7 | FT always better on hard |
| libero_spatial | 0.7 | FT only helps on very hard episodes; hurts on moderate |

---

## 9. Full Pipeline Example

```bash
# 1. Collect BC training data with importance sampling
python adversarial_training/continual_learning/collect_buffer_bc.py \
    --config adversarial_training/configs/default.yaml

# 2. Train BC model
bash adversarial_training/continual_learning/run_bc_train_only.sh

# 3. A/B evaluate
cd evaluation/libero
bash run_eval_routed.sh \
    YuankaiLuo/SimVLA-LIBERO \
    /path/to/ft_checkpoint \
    /path/to/criticality_ckpt \
    50 "eval_routed" "0 5 6 7"
```
