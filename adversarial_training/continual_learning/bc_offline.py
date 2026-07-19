"""Pure Behavior Cloning (BC) fine-tuning for SimpleVLA.

Thin wrapper around ``code/train_smolvlm.py``.  The only differences are:

* Data source   — ``--bc_meta`` (JSON produced by ``collect_buffer_bc.py``)
                   instead of ``--train_metas_path``.
* YAML defaults — the ``bc:`` block in the config YAML can set batch size,
                   LR, iters, etc.; CLI args always take precedence.

Everything else — model loading, SmolVLM dataloader, AdamW with per-group
LRS, freeze→warmup→cosine schedule, loss, accelerate multi-GPU, checkpointing —
is imported directly from ``train_smolvlm``.

Usage::

    accelerate launch \\
        --num_processes 4 --mixed_precision bf16 \\
        adversarial_training/continual_learning/bc_offline.py \\
        --config adversarial_training/configs/default.yaml \\
        --bc_meta ./datasets/bc_buffer/bc_train_meta.json \\
        --output_dir ./runs/bc_continual
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict

import torch
import yaml

_THIS = Path(__file__).resolve()
_CODE_ROOT = _THIS.parents[2]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from train_smolvlm import (          # noqa: E402
    get_logger,
    set_seed,
    build_optimizer,
    update_group_lrs,
    get_args_parser as _get_base_parser,
)
from datasets import create_smolvlm_dataloader              # noqa: E402
from models.modeling_smolvlm_vla import SmolVLMVLA          # noqa: E402
from models.processing_smolvlm_vla import SmolVLMVLAProcessor  # noqa: E402

from accelerate import Accelerator, DistributedDataParallelKwargs


# ============================================================
# Argument parser  (extends the base parser from train_smolvlm)
# ============================================================
def get_args_parser():
    parser = _get_base_parser()

    # BC-specific overrides / additions
    parser.add_argument("--config", type=str,
                        default="adversarial_training/configs/default.yaml",
                        help="YAML config with bc: block for defaults")
    parser.add_argument("--bc_meta", type=str, default=None,
                        help="Path to BC training metadata JSON "
                             "(overrides --train_metas_path)")

    # Fix defaults inherited from train_smolvlm that don't apply to LIBERO
    for action in parser._actions:
        if action.dest == "action_mode":
            action.default = "libero_joint"
            break
    return parser


# ============================================================
# YAML → CLI fallback  (CLI always wins)
# ============================================================
def _apply_yaml_overrides(args: argparse.Namespace) -> None:
    config_path = getattr(args, "config", None)
    if config_path is None or not Path(config_path).exists():
        return
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    bc = cfg.get("bc_training", {})

    _maybe(args, "bc_meta",           bc, "bc_meta")
    _maybe(args, "train_metas_path",  bc, "bc_meta")       # same source
    _maybe(args, "models",            bc, "resume_ckpt")
    _maybe(args, "output_dir",        bc, "output_dir")
    _maybe(args, "learning_rate",     bc, "learning_rate")
    _maybe(args, "batch_size",        bc, "batch_size")
    _maybe(args, "iters",             bc, "iters")
    _maybe(args, "freeze_steps",      bc, "freeze_steps")
    _maybe(args, "warmup_steps",      bc, "warmup_steps")
    _maybe(args, "save_interval",     bc, "save_interval")
    _maybe(args, "max_grad_norm",     bc, "max_grad_norm")
    _maybe(args, "num_workers",       bc, "num_workers")
    _maybe(args, "use_cosine_decay",  bc, "use_cosine_decay")
    _maybe(args, "min_lr_ratio",      bc, "min_lr_ratio")

    # freeze_vlm → learning_coef=0 (freeze SmolVLM backbone, prevents
    # catastrophic forgetting on small fine-tuning datasets).
    if bc.get("freeze_vlm", False):
        setattr(args, "learning_coef", 0.0)
    else:
        _maybe(args, "learning_coef",  bc, "learning_coef")


def _maybe(args, attr: str, section: dict, key: str) -> None:
    if key in section:
        setattr(args, attr, section[key])


# ============================================================
# Main
# ============================================================
def main(args):
    output_dir = Path(args.output_dir)

    # WandB
    try:
        import wandb
        WANDB_AVAILABLE = True
    except ImportError:
        wandb = None; WANDB_AVAILABLE = False

    wandb_api_key = os.environ.get("WANDB_API_KEY") or getattr(args, "wandb_api_key", None)
    wandb_project = os.environ.get("WANDB_PROJECT") or getattr(args, "wandb_project", None)
    use_wandb = WANDB_AVAILABLE and wandb_api_key

    log_with = ["tensorboard"]
    if use_wandb:
        log_with.append("wandb")
        os.environ["WANDB_API_KEY"] = wandb_api_key

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        log_with=log_with, project_dir=output_dir, kwargs_handlers=[ddp_kwargs])

    accelerator.init_trackers("BC-Offline", config={
        "learning_rate": args.learning_rate, "batch_size": args.batch_size,
        "iters": args.iters, "freeze_steps": args.freeze_steps,
        "bc_meta": args.bc_meta,
    })
    accelerator.wait_for_everyone()

    logger = get_logger(__name__, output_dir=output_dir, accelerator=accelerator)
    set_seed(args.seed + accelerator.process_index)
    logger.info(f"BC offline training  |  bc_meta={args.bc_meta}")
    logger.info(f"  smolvlm={args.smolvlm_model_path}  "
                f"action_mode={args.action_mode}  num_actions={args.num_actions}")
    logger.info(f"  lr={args.learning_rate}  lr_coef={args.learning_coef}  "
                f"bs={args.batch_size}  iters={args.iters}")
    logger.info(f"  freeze={args.freeze_steps}  warmup={args.warmup_steps}")

    # ── Model ──
    from models.configuration_smolvlm_vla import SmolVLMVLAConfig
    from models.action_hub import build_action_space

    action_space_kwargs = {}
    if getattr(args, "norm_stats_path", None):
        action_space_kwargs["norm_stats_path"] = args.norm_stats_path

    load_path = args.models
    # Support both local directories AND HuggingFace model IDs.
    is_local = (
        load_path and os.path.isdir(load_path) and
        os.path.exists(os.path.join(load_path, "model.safetensors"))
    )
    if load_path and (is_local or "/" in str(load_path) or not os.path.exists(str(load_path))):
        # load_path is a local directory, a HuggingFace repo id, or a path
        # that does not exist on the local filesystem (→ try HF).
        logger.info(f"Loading checkpoint: {load_path}")
        model = SmolVLMVLA.from_pretrained(load_path)
        if args.action_mode != model.action_mode:
            logger.warning(f"action_mode override: {model.action_mode} → {args.action_mode}")
            model.action_mode = args.action_mode
            model.action_space = build_action_space(args.action_mode, **action_space_kwargs)
        elif action_space_kwargs:
            model.action_space = build_action_space(args.action_mode, **action_space_kwargs)
    else:
        logger.info("Initializing from config (no checkpoint found)")
        config = SmolVLMVLAConfig(
            smolvlm_model_path=args.smolvlm_model_path,
            hidden_size=args.hidden_size,
            depth=args.depth,
            num_heads=args.num_heads,
            action_mode=args.action_mode,
            num_actions=args.num_actions,
            image_size=args.image_size,
        )
        model = SmolVLMVLA(config)
        if action_space_kwargs:
            model.action_space = build_action_space(args.action_mode, **action_space_kwargs)

    # ── Freeze VLM backbone ──
    # When learning_coef=0 (e.g. freeze_vlm=true), freeze the SmolVLM
    # backbone completely: no gradients, eval mode (no dropout).
    # This prevents catastrophic forgetting of visual features on small
    # fine-tuning datasets, and keeps VLM features identical between
    # training and inference (no train/eval dropout discrepancy).
    if getattr(args, "learning_coef", 0.1) == 0.0:
        for p in model.vlm.parameters():
            p.requires_grad_(False)
        model.vlm.eval()
        logger.info("VLM backbone FROZEN (learning_coef=0)")

    # ── Processor + DataLoader ──
    processor = SmolVLMVLAProcessor.from_pretrained(args.smolvlm_model_path)
    meta_path = args.bc_meta or args.train_metas_path          # bc_meta takes priority
    train_dataloader = create_smolvlm_dataloader(
        batch_size=args.batch_size, metas_path=meta_path,
        num_actions=model.num_actions, action_mode=model.action_mode,
        training=True, num_workers=args.num_workers,
        image_size=args.image_size)

    # ── Optimizer ──
    optim = build_optimizer(
        model=model, lr=args.learning_rate,
        weight_decay=args.weight_decay, betas=tuple(args.betas),
        lr_coef_vlm=args.learning_coef)
    model, optim = accelerator.prepare(model, optim)
    model.train()

    # ── Training loop ──
    start_step = 0
    if args.resume and load_path and os.path.isdir(load_path):
        state_json = os.path.join(load_path, "state.json")
        if os.path.exists(state_json):
            try:
                with open(state_json) as f:
                    start_step = int(json.load(f).get("global_step", 0))
                logger.info(f"Resume from step {start_step}")
            except Exception:
                pass

    global_step = start_step
    t0 = time.time()
    logger.info(f"Start BC training ({args.iters} iters, world_size={accelerator.num_processes})")

    for batch in train_dataloader:
        lang = processor.encode_language(batch["language_instruction"])
        batch.pop("language_instruction", None)
        inputs = {**batch, **lang}
        inputs = {k: v.cuda(non_blocking=True) for k, v in inputs.items()}

        update_group_lrs(optim, global_step, args)

        loss_dict: Dict[str, torch.Tensor] = model(**inputs)

        loss = sum(loss_dict.values())

        accelerator.backward(loss)
        if args.max_grad_norm:
            accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optim.step()
        optim.zero_grad()

        if global_step % args.log_interval == 0:
            logs = {k: v.detach().float().item() for k, v in loss_dict.items()}
            logs["loss_total"] = float(loss.detach().item())
            logs.update({f"lr_{g['name']}": g["lr"] for g in optim.param_groups})
            accelerator.log(logs, step=global_step)

            if accelerator.is_main_process:
                dt = (time.time() - t0) / args.log_interval
                t0 = time.time()
                logger.info(
                    f"[{global_step}/{args.iters}] "
                    f"loss={logs['loss_total']:.4f}  "
                    f"lr_core={logs.get('lr_transformer_core', 0):.2e}  "
                    f"lr_action={logs.get('lr_action_heads', 0):.2e}  "
                    f"lr_vlm={logs.get('lr_vlm', 0):.2e}  ({dt:.2f}s/it)")

        global_step += 1
        if accelerator.is_main_process and (
                global_step == args.iters or global_step % args.save_interval == 0):
            save_dir = os.path.join(output_dir, f"ckpt-{global_step}")
            accelerator.print(f"Saving → {save_dir}")
            accelerator.unwrap_model(model).save_pretrained(
                save_dir, safe_serialization=True)
            with open(os.path.join(save_dir, "state.json"), "w") as f:
                json.dump({"global_step": global_step}, f)

        if global_step >= args.iters:
            break

    accelerator.end_training()
    logger.info("BC training complete.")


# ============================================================
# Entry
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser("BC Offline", parents=[get_args_parser()])
    # --bc_meta replaces --train_metas_path; remove the required constraint
    for action in parser._actions:
        if action.dest == "train_metas_path":
            action.required = False
            action.default = None
            break
    args = parser.parse_args()

    _apply_yaml_overrides(args)

    # --bc_meta takes priority, fall back to --train_metas_path
    if not (getattr(args, "bc_meta", None) or getattr(args, "train_metas_path", None)):
        parser.error(
            "--bc_meta is required.  Point it at the JSON from collect_buffer_bc.py")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
