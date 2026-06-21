# train_circle_cnn.py
"""
1D CNN with focal loss + signal augmentation — brake-only circle dataset.
Supports full-dataset lazy loading (FULL_DATASET=True) via ChunkWindowDataset:
  - Streams CHUNK_SIZE files at a time → memory stays low (~72 MB/chunk)
  - Per-chunk class balancing for approximate global balance
"""
from __future__ import annotations
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
from pathlib import Path
from datetime import datetime
import h5py
import wandb

from split import stratified_run_split, subsample_mat_paths
from dataset import _fault_class_from_name, _h5_read, class_balanced_indices, collect_mat_paths
from model_cnn import CnnMlpFaultIsolation

# ── constants ────────────────────────────────────────────────────────────────
SIGNALS = [
    "res_vx", "res_vy", "res_r",
    "res_slip_fl", "res_slip_fr", "res_slip_rl", "res_slip_rr",
    "res_ax", "res_ay",
]
WIN, STRIDE, WARMUP = 500, 100, 1.0
NUM_CLASSES  = 5
BATCH_SIZE   = 256
LR           = 3e-3
PATIENCE     = 25           # reduced: each epoch covers far more data
LR_PATIENCE  = 5            # reduced
MAX_EPOCHS   = 200
SEED         = 0
LABEL_MODE   = "any"
FOCAL_GAMMA  = 2.0
WEIGHT_DECAY = 1e-4
AUG_NOISE    = 0.02
AUG_SCALE    = 0.10

# ── full-dataset lazy loading ─────────────────────────────────────────────────
FULL_DATASET  = True
CHUNK_SIZE    = 50    # files per chunk  (50 × ~80 windows × 18 KB ≈ 72 MB)
NORM_FIT_N    = 300   # files to fit ChannelNorm
VAL_PRELOAD_N = 500   # val files to preload for epoch-level early stopping

CNN_SWEEP_SPEEDS     = set(range(10, 131, 12))
CNN_SWEEP_STEER_MAGS = {10, 50, 100, 150, 200, 250, 300}


# ── window extraction ────────────────────────────────────────────────────────
def _extract_raw(path: Path):
    fault_class = _fault_class_from_name(path)
    with h5py.File(path, "r") as f:
        time   = _h5_read(f, "data/time")
        fs_val = float(_h5_read(f, "data/FaultStart_s").flat[0])
        sigs   = np.stack([_h5_read(f, f"data/{s}") for s in SIGNALS])
    mask = time >= WARMUP
    time = time[mask]; sigs = sigs[:, mask]
    fault = time >= fs_val
    n = sigs.shape[1]
    wins, labs = [], []
    for s in range(0, n - WIN + 1, STRIDE):
        wins.append(sigs[:, s:s+WIN].astype(np.float32))
        labs.append(fault_class if fault[s:s+WIN].any() else 0)
    return np.stack(wins), np.array(labs, np.int64)


# ── preloaded dataset (small sets: norm fitting, val sample) ─────────────────
class RawWindowDataset(Dataset):
    def __init__(self, mat_paths: list):
        wins, labs = [], []
        skipped = 0
        for p in mat_paths:
            try:
                w, l = _extract_raw(Path(p))
                wins.append(w); labs.append(l)
            except Exception:
                skipped += 1
        if skipped:
            import warnings; warnings.warn(f"Skipped {skipped} files.")
        self.windows = torch.from_numpy(np.concatenate(wins, 0)).float()
        self.labels  = torch.from_numpy(np.concatenate(labs, 0)).long()

    @classmethod
    def from_tensors(cls, windows, labels):
        obj = cls.__new__(cls)
        obj.windows = windows; obj.labels = labels
        return obj

    def __len__(self): return len(self.labels)
    def __getitem__(self, i): return self.windows[i], self.labels[i]


def _balanced(ds: RawWindowDataset) -> RawWindowDataset:
    keep = class_balanced_indices(ds.labels, seed=SEED)
    return RawWindowDataset.from_tensors(ds.windows[keep], ds.labels[keep])


def _chunk_balance(wins: np.ndarray, labs: np.ndarray,
                   rng: np.random.RandomState):
    classes = np.unique(labs)
    if len(classes) < 2:
        return wins, labs
    min_n = min(int((labs == c).sum()) for c in classes)
    keep = np.concatenate([
        rng.choice(np.where(labs == c)[0], min_n, replace=False)
        for c in classes
    ])
    rng.shuffle(keep)
    return wins[keep], labs[keep]


# ── lazy full-dataset loader ──────────────────────────────────────────────────
class ChunkWindowDataset(IterableDataset):
    """Streams CHUNK_SIZE files at a time. No full preloading."""

    def __init__(self, mat_paths: list, norm: "ChannelNorm", seed: int = 0):
        self.paths = list(mat_paths)
        self.norm  = norm
        self.seed  = seed
        self._epoch = 0

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def __iter__(self):
        rng  = np.random.RandomState(self.seed + self._epoch)
        perm = rng.permutation(len(self.paths))
        mean = self.norm.mean.squeeze(0)   # (9,1)
        std  = self.norm.std.squeeze(0)    # (9,1)

        for start in range(0, len(perm), CHUNK_SIZE):
            chunk_paths = [self.paths[int(i)] for i in perm[start:start + CHUNK_SIZE]]
            wins, labs = [], []
            for p in chunk_paths:
                try:
                    w, l = _extract_raw(Path(p))
                    wins.append(w); labs.append(l)
                except Exception:
                    continue
            if not wins:
                continue

            wins = np.concatenate(wins)
            labs = np.concatenate(labs)
            wins, labs = _chunk_balance(wins, labs, rng)

            for w, l in zip(wins, labs):
                yield (torch.from_numpy(w) - mean) / std, torch.tensor(int(l), dtype=torch.long)


# ── per-channel normalizer ────────────────────────────────────────────────────
class ChannelNorm:
    def fit(self, windows: torch.Tensor):
        self.mean = windows.mean(dim=(0, 2), keepdim=True)               # (1,9,1)
        self.std  = windows.std(dim=(0, 2), keepdim=True).clamp(min=1e-6)

    def transform(self, windows: torch.Tensor) -> torch.Tensor:
        return (windows - self.mean) / self.std

    def save(self, path): torch.save({"mean": self.mean, "std": self.std}, path)
    def load(self, path): d = torch.load(path); self.mean = d["mean"]; self.std = d["std"]


# ── focal loss ───────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


# ── metrics ───────────────────────────────────────────────────────────────────
def _compute_metrics(preds: np.ndarray, y: np.ndarray):
    acc = float((preds == y).mean())
    f1s = []
    for c in range(NUM_CLASSES):
        tp = int(((preds == c) & (y == c)).sum())
        fp = int(((preds == c) & (y != c)).sum())
        fn = int(((preds != c) & (y == c)).sum())
        p  = tp / (tp + fp) if tp + fp > 0 else 0.0
        r  = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1s.append(2 * p * r / (p + r) if p + r > 0 else 0.0)
    return acc, f1s


def evaluate(model, norm, ds: RawWindowDataset, device):
    """Evaluate on preloaded dataset."""
    model.eval()
    all_preds = []
    X_norm = norm.transform(ds.windows)
    with torch.no_grad():
        for i in range(0, len(ds), BATCH_SIZE):
            all_preds.append(model(X_norm[i:i+BATCH_SIZE].to(device)).argmax(1).cpu())
    return _compute_metrics(torch.cat(all_preds).numpy(), ds.labels.numpy())


def evaluate_lazy(model, norm, paths: list, device):
    """Evaluate on full file list without preloading everything."""
    model.eval()
    all_preds, all_labels = [], []
    for p in paths:
        try:
            wins, labs = _extract_raw(Path(p))
            w_t = (torch.from_numpy(wins) - norm.mean) / norm.std
            with torch.no_grad():
                for i in range(0, len(labs), BATCH_SIZE):
                    out = model(w_t[i:i+BATCH_SIZE].to(device))
                    all_preds.append(out.argmax(1).cpu())
            all_labels.append(torch.from_numpy(labs))
        except Exception:
            continue
    return _compute_metrics(
        torch.cat(all_preds).numpy(),
        torch.cat(all_labels).numpy(),
    )


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(SEED); np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.RandomState(SEED)

    # paths
    all_paths = collect_mat_paths("data/brake_fault_dataset_circle_10s")
    if not FULL_DATASET:
        all_paths = subsample_mat_paths(all_paths, CNN_SWEEP_SPEEDS, CNN_SWEEP_STEER_MAGS)
    train_paths, val_paths, test_paths = stratified_run_split(all_paths, seed=SEED)
    print(f"{'Full' if FULL_DATASET else 'Subsampled'} dataset | "
          f"train={len(train_paths)}  val={len(val_paths)}  test={len(test_paths)} files")

    # wandb
    os.environ.setdefault("WANDB_ENTITY", "qwetttrerewq")
    os.environ.setdefault("WANDB_PROJECT", "fault-detection")
    run_tag = "full" if FULL_DATASET else "sub"
    wandb_run_name = f"circle_cnn_focal_{run_tag}_s{SEED}_{datetime.now():%Y%m%dT%H%M%S}"
    wandb.init(
        name=wandb_run_name,
        config={
            "model": "CnnMlpFaultIsolation",
            "label_mode": LABEL_MODE,
            "full_dataset": FULL_DATASET,
            "batch_size": BATCH_SIZE,
            "lr": LR, "weight_decay": WEIGHT_DECAY,
            "focal_gamma": FOCAL_GAMMA,
            "aug_noise": AUG_NOISE, "aug_scale": AUG_SCALE,
            "patience": PATIENCE, "lr_patience": LR_PATIENCE,
            "max_epochs": MAX_EPOCHS, "seed": SEED,
            "chunk_size": CHUNK_SIZE,
            "train_files": len(train_paths),
        }
    )

    # ChannelNorm: fit on random sample of train files
    print(f"Fitting ChannelNorm on {min(NORM_FIT_N, len(train_paths))} train files...")
    norm_paths = rng.choice(train_paths, size=min(NORM_FIT_N, len(train_paths)), replace=False).tolist()
    norm_ds = RawWindowDataset(norm_paths)
    norm = ChannelNorm()
    norm.fit(norm_ds.windows)
    del norm_ds

    # Val: preload small sample for epoch-level early stopping
    print(f"Preloading {min(VAL_PRELOAD_N, len(val_paths))} val files...")
    val_sample = rng.choice(val_paths, size=min(VAL_PRELOAD_N, len(val_paths)), replace=False).tolist()
    val_ds  = _balanced(RawWindowDataset(val_sample))
    val_ds_w = norm.transform(val_ds.windows)
    print(f"  val_sample windows={len(val_ds)}")

    # Train: lazy chunk streaming
    train_dataset = ChunkWindowDataset(train_paths, norm=norm, seed=SEED)
    loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=0, drop_last=True)

    model     = CnnMlpFaultIsolation(n_signals=len(SIGNALS), num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=LR_PATIENCE, min_lr=1e-5
    )
    criterion     = FocalLoss(gamma=FOCAL_GAMMA)
    val_criterion = nn.CrossEntropyLoss()

    best_val_loss   = float("inf")
    patience_counter = 0
    run_dir = Path("outputs") / f"cnn_run_{datetime.now():%Y%m%d_%H%M%S}_seed{SEED}"
    run_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, MAX_EPOCHS + 1):
        train_dataset.set_epoch(epoch)
        model.train()
        total_loss = 0.0; n_batches = 0

        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            # signal augmentation
            X_b = X_b + torch.randn_like(X_b) * AUG_NOISE
            X_b = X_b * (1.0 - AUG_SCALE + 2 * AUG_SCALE * torch.rand(X_b.shape[0], 1, 1, device=device))
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward(); optimizer.step()
            total_loss += loss.item(); n_batches += 1

        train_loss = total_loss / max(n_batches, 1)

        # validation (preloaded val sample)
        model.eval()
        val_loss_sum = 0.0; val_preds_list = []
        with torch.no_grad():
            for i in range(0, len(val_ds), BATCH_SIZE):
                xb = val_ds_w[i:i+BATCH_SIZE].to(device)
                yb = val_ds.labels[i:i+BATCH_SIZE].to(device)
                out = model(xb)
                val_loss_sum += val_criterion(out, yb).item() * len(yb)
                val_preds_list.append(out.argmax(1).cpu())
        val_loss = val_loss_sum / len(val_ds)
        val_acc  = (torch.cat(val_preds_list) == val_ds.labels).float().mean().item()

        scheduler.step(val_loss)
        print(f"Epoch {epoch:03d}/{MAX_EPOCHS} | train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")
        wandb.log({"train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc,
                   "lr": optimizer.param_groups[0]["lr"]}, step=epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({k: v.cpu().clone() for k, v in model.state_dict().items()},
                       run_dir / "model.pt")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    # test — full test set, evaluated lazily
    model.load_state_dict(torch.load(run_dir / "model.pt", weights_only=True))
    norm.save(run_dir / "norm.pt")

    print(f"\nEvaluating test set ({len(test_paths)} files, lazy)...")
    test_acc, test_f1 = evaluate_lazy(model, norm, test_paths, device)
    CLASS_NAMES = ["Normal", "bc_FL", "bc_FR", "bc_RL", "bc_RR"]
    macro_f1 = sum(test_f1) / len(test_f1)

    print(f"\nTEST  accuracy={test_acc:.4f}  macro_f1={macro_f1:.4f}")
    print(f"Per-class F1: {[round(v,3) for v in test_f1]}")

    wandb.summary.update({
        "test_accuracy": test_acc, "test_macro_f1": macro_f1,
        **{f"f1_{CLASS_NAMES[i]}": round(test_f1[i], 4) for i in range(NUM_CLASSES)},
        "stopped_epoch": epoch, "label_mode": LABEL_MODE, "full_dataset": FULL_DATASET,
    })
    wandb.finish()

    meta = {
        "model": "CnnMlpFaultIsolation",
        "label_mode": LABEL_MODE, "full_dataset": FULL_DATASET,
        "architecture": "9x500 -> Conv(64,128,256) -> MLP(128) -> 5",
        "signals": SIGNALS, "num_classes": NUM_CLASSES,
        "batch_size": BATCH_SIZE, "lr": LR,
        "weight_decay": WEIGHT_DECAY, "focal_gamma": FOCAL_GAMMA,
        "aug_noise": AUG_NOISE, "aug_scale": AUG_SCALE,
        "early_stopping_patience": PATIENCE,
        "train_files": len(train_paths), "test_files": len(test_paths),
        "test_accuracy": test_acc, "test_macro_f1": macro_f1,
        "test_per_class_f1": {CLASS_NAMES[i]: round(test_f1[i], 4) for i in range(NUM_CLASSES)},
    }
    (run_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved to {run_dir}")
