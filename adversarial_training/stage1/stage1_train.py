"""Stage 1 — Train the criticality model on LIBERO init-state data.

Consumes the per-(suite, task_index) shards produced by stage1_collect.py
and trains a task-agnostic deep residual MLP classifier
``padded_init_state -> P(failure)``.

All init-state vectors are right-padded with zeros to a global ``max_D``
before training. The model receives no task identifier.

Reference: other_source/criticality/stage1/stage1_train.py — but with
init_state-only inputs (no force vector) and per-(suite, task) granularity.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

_THIS = Path(__file__).resolve()
_CODE_ROOT = _THIS.parents[2]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from adversarial_training.utils.data_utils import flatten_for_training
from adversarial_training.utils.task_registry import all_task_keys


# ---------------------------------------------------------------------------
# CLI / config
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LIBERO criticality model.")
    p.add_argument("--config", type=Path, default=Path("./adversarial_training/configs/default.yaml"))
    p.add_argument("--data_dir", type=Path, default=None)
    p.add_argument("--output_dir", type=Path, default=None)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--rebuild_split", action="store_true",
                   help="Ignore cached splits in <data_dir>/_split.pkl")
    p.add_argument("--test_only", action="store_true",
                   help="Skip training; only evaluate the checkpoint at output_dir/best.pt")
    return p.parse_args()


def load_cfg(args: argparse.Namespace) -> Dict[str, Any]:
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    c = cfg.get("stage1_train", {})
    if args.data_dir is not None:   c["data_dir"] = str(args.data_dir)
    if args.output_dir is not None: c["output_dir"] = str(args.output_dir)
    c.setdefault("batch_size", 256)
    c.setdefault("num_epochs", 50)
    c.setdefault("learning_rate", 3e-4)
    c.setdefault("weight_decay", 1e-4)
    c.setdefault("val_ratio", 0.1)
    c.setdefault("test_ratio", 0.1)
    c.setdefault("hidden_dim", 1024)
    c.setdefault("expansion", 4)
    c.setdefault("depth", 12)
    c.setdefault("dropout", 0.1)
    c.setdefault("seed", 42)
    return c


# ---------------------------------------------------------------------------
# Data split (per-task stratified, then padded + concat into global splits)
# ---------------------------------------------------------------------------


TaskKey = Tuple[str, int]
PerTaskSplit = Dict[TaskKey, Dict[str, np.ndarray]]


def _stratified_split_one(
    X: np.ndarray, y: np.ndarray, w: np.ndarray,
    val_ratio: float, test_ratio: float, rng: np.random.Generator,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Per-task split: keeps positive / negative ratios stable across splits."""
    out_tr, out_va, out_te = {}, {}, {}
    parts = {"X": [[], [], []], "y": [[], [], []], "w": [[], [], []]}
    for label in (0, 1):
        idx = np.where(y == label)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_te = int(n * test_ratio)
        n_va = int(n * val_ratio)
        n_tr = n - n_te - n_va
        for bucket, sl in enumerate((idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:])):
            parts["X"][bucket].append(X[sl])
            parts["y"][bucket].append(y[sl])
            parts["w"][bucket].append(w[sl])
    for d, bucket in zip((out_tr, out_va, out_te), range(3)):
        d["init_state"] = np.concatenate(parts["X"][bucket], axis=0) if parts["X"][bucket] else np.zeros((0, X.shape[1]), dtype=np.float32)
        d["label"]      = np.concatenate(parts["y"][bucket], axis=0) if parts["y"][bucket] else np.zeros((0,), dtype=np.int64)
        d["is_weight"]  = np.concatenate(parts["w"][bucket], axis=0) if parts["w"][bucket] else np.zeros((0,), dtype=np.float32)
    return out_tr, out_va, out_te


def build_splits(data_dir: Path, val_ratio: float, test_ratio: float, seed: int
                 ) -> Tuple[PerTaskSplit, PerTaskSplit, PerTaskSplit]:
    rng = np.random.default_rng(seed)
    flat = flatten_for_training(data_dir)
    train, val, test = {}, {}, {}
    for key, bucket in flat["by_task"].items():
        X, y, w = bucket["init_state"], bucket["label"], bucket["is_weight"]
        if X.shape[0] == 0:
            continue
        tr, va, te = _stratified_split_one(X, y, w, val_ratio, test_ratio, rng)
        train[key], val[key], test[key] = tr, va, te
    return train, val, test


def cached_splits_path(data_dir: Path) -> Path:
    return Path(data_dir) / "_split.pkl"


def load_or_build_splits(cfg: Dict[str, Any], rebuild: bool
                         ) -> Tuple[PerTaskSplit, PerTaskSplit, PerTaskSplit]:
    cache = cached_splits_path(Path(cfg["data_dir"]))
    if cache.exists() and not rebuild:
        with open(cache, "rb") as f:
            tr, va, te = pickle.load(f)
        print(f"[stage1_train] loaded cached splits from {cache}")
        return tr, va, te
    tr, va, te = build_splits(
        Path(cfg["data_dir"]),
        val_ratio=float(cfg["val_ratio"]),
        test_ratio=float(cfg["test_ratio"]),
        seed=int(cfg["seed"]),
    )
    with open(cache, "wb") as f:
        pickle.dump((tr, va, te), f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[stage1_train] cached splits at {cache}")
    return tr, va, te


# ---------------------------------------------------------------------------
# Pad + concat to global arrays
# ---------------------------------------------------------------------------


def _resolve_max_dim(*splits: PerTaskSplit) -> int:
    """Largest init_state width across every task in every provided split."""
    widths = []
    for split in splits:
        for bucket in split.values():
            X = bucket["init_state"]
            if X.size:
                widths.append(int(X.shape[1]))
    if not widths:
        raise RuntimeError("No init_state rows found in any split — was data collected?")
    return max(widths)


def _pad_right(X: np.ndarray, max_D: int) -> np.ndarray:
    if X.shape[1] == max_D:
        return X.astype(np.float32, copy=False)
    if X.shape[1] > max_D:
        raise ValueError(f"Got init_state width {X.shape[1]} > max_D {max_D}")
    pad = np.zeros((X.shape[0], max_D - X.shape[1]), dtype=np.float32)
    return np.concatenate([X.astype(np.float32, copy=False), pad], axis=1)


def flatten_to_global(per_task: PerTaskSplit, max_D: int
                      ) -> Dict[str, np.ndarray]:
    """Pad each task's X to max_D and concat into one global (N, max_D) array.

    Also returns ``task_id`` so eval can produce per-task metrics from one loader.
    """
    keys = sorted(per_task.keys())
    key_to_id = {k: i for i, k in enumerate(keys)}
    Xs, ys, ws, tids = [], [], [], []
    for key in keys:
        b = per_task[key]
        if b["init_state"].size == 0:
            continue
        Xs.append(_pad_right(b["init_state"], max_D))
        ys.append(b["label"].astype(np.int64))
        ws.append(b["is_weight"].astype(np.float32))
        tids.append(np.full(b["label"].shape[0], key_to_id[key], dtype=np.int64))
    if not Xs:
        return {"X": np.zeros((0, max_D), dtype=np.float32),
                "y": np.zeros((0,), dtype=np.int64),
                "w": np.zeros((0,), dtype=np.float32),
                "task_id": np.zeros((0,), dtype=np.int64),
                "id_to_key": keys}
    return {
        "X":  np.concatenate(Xs, axis=0),
        "y":  np.concatenate(ys, axis=0),
        "w":  np.concatenate(ws, axis=0),
        "task_id": np.concatenate(tids, axis=0),
        "id_to_key": keys,
    }


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def make_loader(
    flat: Dict[str, np.ndarray],
    batch_size: int,
    shuffle: bool,
    balanced: bool = False,
):
    """Build a DataLoader.

    When ``balanced=True`` swap the regular shuffle for a
    ``WeightedRandomSampler`` that draws each sample with probability
    inversely proportional to its class count, giving roughly 50/50
    positive/negative batches. This is the standard fix for the
    collapse-to-majority pathology when positives are ~3% of the data.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

    ds = TensorDataset(
        torch.from_numpy(flat["X"]).float(),
        torch.from_numpy(flat["y"]).long(),
        torch.from_numpy(flat["w"]).float(),
        torch.from_numpy(flat["task_id"]).long(),
    )

    if balanced:
        y = flat["y"]
        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        # Fall back to plain shuffle when one class is missing — sampler probs
        # would be degenerate.
        if n_pos == 0 or n_neg == 0:
            print(f"[make_loader] balanced=True but only one class present "
                  f"(n_pos={n_pos}, n_neg={n_neg}); falling back to shuffle.")
        else:
            # P(draw this sample) ∝ 1 / class_count; normalisation is handled
            # by WeightedRandomSampler. num_samples=len(y) so an epoch still
            # walks the whole dataset's worth of draws.
            sample_w = np.where(y == 1, 1.0 / n_pos, 1.0 / n_neg).astype(np.float64)
            sampler = WeightedRandomSampler(
                weights=torch.from_numpy(sample_w),
                num_samples=len(y),
                replacement=True,
            )
            return DataLoader(ds, batch_size=batch_size, sampler=sampler,
                              num_workers=0, pin_memory=True)

    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=True)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def precision_recall_curve(y_true: np.ndarray, y_score: np.ndarray,
                           num_thresholds: int = 200):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_true.size == 0:
        return np.array([1.0]), np.array([0.0]), np.array([])

    thr = np.linspace(1.0, 0.0, num_thresholds, endpoint=False)
    P = int((y_true == 1).sum())
    prec_list, rec_list = [], []
    for t in thr:
        preds = (y_score >= t).astype(int)
        tp = int(((preds == 1) & (y_true == 1)).sum())
        fp = int(((preds == 1) & (y_true == 0)).sum())
        fn = P - tp
        p = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        prec_list.append(p); rec_list.append(r)
    return (np.concatenate(([1.0], prec_list)),
            np.concatenate(([0.0], rec_list)),
            thr)


def auc_trapezoid(x: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(x)
    return float(np.trapz(y[order], x[order]))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(model, loader, id_to_key: List[TaskKey], device
             ) -> Tuple[Dict[TaskKey, Dict[str, float]], Dict[str, float]]:
    import torch
    model.eval()

    all_true, all_score, all_tid = [], [], []
    total = correct = 0
    with torch.no_grad():
        for xb, yb, _wb, tb in loader:
            xb = xb.to(device); yb = yb.to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)
            total += yb.size(0); correct += int((preds == yb).sum().item())
            all_score.append(probs.cpu().numpy())
            all_true.append(yb.cpu().numpy())
            all_tid.append(tb.cpu().numpy())

    if not all_true:
        return {}, {"acc": 0.0, "auc_pr": 0.0, "n": 0, "pos_rate": 0.0}

    y_true  = np.concatenate(all_true)
    y_score = np.concatenate(all_score)
    tid     = np.concatenate(all_tid)

    per_task: Dict[TaskKey, Dict[str, float]] = {}
    for i, key in enumerate(id_to_key):
        m = tid == i
        if not np.any(m):
            continue
        prec, rec, _ = precision_recall_curve(y_true[m], y_score[m])
        per_task[key] = {
            "n": int(m.sum()),
            "auc_pr": auc_trapezoid(rec, prec),
            "pos_rate": float(y_true[m].mean()),
        }

    prec, rec, _ = precision_recall_curve(y_true, y_score)
    agg = {
        "acc": correct / max(total, 1),
        "auc_pr": auc_trapezoid(rec, prec),
        "n": int(y_true.size),
        "pos_rate": float(y_true.mean()),
    }
    return per_task, agg


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_loop(cfg: Dict[str, Any], resume: Optional[Path]) -> None:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    from adversarial_training.utils.criticality_model import (
        CriticalityModel, CriticalityModelConfig,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(cfg["seed"]))

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    train_dict, val_dict, test_dict = load_or_build_splits(cfg, rebuild=False)
    max_D = _resolve_max_dim(train_dict, val_dict, test_dict)
    print(f"[stage1_train] padded input dim (max_D): {max_D}")

    train_flat = flatten_to_global(train_dict, max_D)
    val_flat   = flatten_to_global(val_dict,   max_D)
    test_flat  = flatten_to_global(test_dict,  max_D)

    train_loader = make_loader(train_flat, int(cfg["batch_size"]), shuffle=True, balanced=True)
    val_loader   = make_loader(val_flat,   int(cfg["batch_size"]), shuffle=False)
    test_loader  = make_loader(test_flat,  int(cfg["batch_size"]), shuffle=False)

    n_pos_tr = int((train_flat["y"] == 1).sum())
    n_neg_tr = int((train_flat["y"] == 0).sum())
    print(f"[stage1_train] train pos/neg = {n_pos_tr}/{n_neg_tr} "
          f"(pos_rate={n_pos_tr / max(n_pos_tr + n_neg_tr, 1):.4f}); "
          f"using WeightedRandomSampler for ~50/50 batches.")

    model = CriticalityModel(CriticalityModelConfig(
        input_dim=max_D,
        hidden_dim=int(cfg["hidden_dim"]),
        expansion=int(cfg["expansion"]),
        depth=int(cfg["depth"]),
        dropout=float(cfg["dropout"]),
    )).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[stage1_train] model params: {n_params/1e6:.2f}M")

    if resume is not None:
        model.load_state_dict(torch.load(resume, map_location=device))
        print(f"[stage1_train] resumed from {resume}")

    optimizer = optim.AdamW(model.parameters(),
                            lr=float(cfg["learning_rate"]),
                            weight_decay=float(cfg["weight_decay"]))
    criterion = nn.CrossEntropyLoss(reduction="none")

    save_best = out_dir / "best.pt"
    save_last = out_dir / "last.pt"
    metrics_log = out_dir / "train_log.jsonl"
    best_metric = -1.0

    n_train = int(train_flat["X"].shape[0])
    n_tasks = len(train_flat["id_to_key"])
    print(f"[stage1_train] train rows: {n_train}  tasks: {n_tasks}")

    with open(metrics_log, "w") as log_f:
        for epoch in range(1, int(cfg["num_epochs"]) + 1):
            t0 = time.time()
            model.train()
            total = correct = 0
            for xb, yb, wb, _tb in train_loader:
                xb = xb.to(device); yb = yb.to(device); wb = wb.to(device)
                logits = model(xb)
                loss_vec = criterion(logits, yb)
                loss = (loss_vec * wb).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                preds = logits.argmax(dim=1)
                total += yb.size(0); correct += int((preds == yb).sum().item())

            train_acc = correct / max(total, 1)
            _, agg_val = evaluate(model, val_loader, val_flat["id_to_key"], device)
            dur = time.time() - t0

            log_line = {
                "epoch": epoch,
                "train_acc": train_acc,
                "val_acc": agg_val["acc"],
                "val_auc_pr": agg_val["auc_pr"],
                "val_pos_rate": agg_val["pos_rate"],
                "secs": dur,
            }
            log_f.write(json.dumps(log_line) + "\n"); log_f.flush()
            print(f"epoch {epoch:3d}/{cfg['num_epochs']}  "
                  f"train_acc={train_acc:.4f}  val_acc={agg_val['acc']:.4f}  "
                  f"val_auc_pr={agg_val['auc_pr']:.4f}  ({dur:.1f}s)")

            torch.save(model.state_dict(), save_last)
            if agg_val["auc_pr"] >= best_metric:
                best_metric = agg_val["auc_pr"]
                torch.save(model.state_dict(), save_best)
                print(f"  -> new best (val_auc_pr={best_metric:.4f}) saved to {save_best}")

    # Final test
    model.load_state_dict(torch.load(save_best, map_location=device))
    per_task_test, agg_test = evaluate(model, test_loader, test_flat["id_to_key"], device)
    print(f"\n[stage1_train][TEST] acc={agg_test['acc']:.4f}  auc_pr={agg_test['auc_pr']:.4f}  "
          f"pos_rate={agg_test['pos_rate']:.4f}")
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump({
            "aggregate": agg_test,
            "per_task": {f"{k[0]}/{k[1]}": v for k, v in per_task_test.items()},
            "input_dim": max_D,
        }, f, indent=2)


def test_only(cfg: Dict[str, Any]) -> None:
    import torch
    from adversarial_training.utils.criticality_model import (
        CriticalityModel, CriticalityModelConfig,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg["output_dir"])

    train_dict, val_dict, test_dict = load_or_build_splits(cfg, rebuild=False)
    max_D = _resolve_max_dim(train_dict, val_dict, test_dict)
    test_flat = flatten_to_global(test_dict, max_D)
    test_loader = make_loader(test_flat, int(cfg["batch_size"]), shuffle=False)

    model = CriticalityModel(CriticalityModelConfig(
        input_dim=max_D,
        hidden_dim=int(cfg["hidden_dim"]),
        expansion=int(cfg["expansion"]),
        depth=int(cfg["depth"]),
        dropout=float(cfg["dropout"]),
    )).to(device)
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device))
    per_task, agg = evaluate(model, test_loader, test_flat["id_to_key"], device)
    print(f"[stage1_train][TEST] acc={agg['acc']:.4f}  auc_pr={agg['auc_pr']:.4f}")
    for key, m in sorted(per_task.items()):
        print(f"  {key[0]}/task_{key[1]:02d}: n={m['n']}  auc_pr={m['auc_pr']:.4f}  pos_rate={m['pos_rate']:.3f}")


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args)
    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
    if args.test_only:
        test_only(cfg)
    else:
        train_loop(cfg, args.resume)


if __name__ == "__main__":
    main()
