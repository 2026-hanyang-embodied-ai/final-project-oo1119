# dataset.py
from __future__ import annotations
from pathlib import Path
import re
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py

DEFAULT_MAT_DIRS = (
    Path("data/brake_fault_dataset_10s/mat"),
    Path("data/motor_fault_dataset_10s/mat"),
)

RESIDUAL_SIGNALS = [
    "res_vx", "res_vy", "res_r",
    "res_slip_fl", "res_slip_fr", "res_slip_rl", "res_slip_rr",
    "res_ax", "res_ay",
]

# Class label is encoded in the scenario-ID filename prefix
# (`<bf|af|mf>_<wheel>_...`), NOT in Fault_Flag (an unreliable detector flag).
# The normal/fault boundary is time-based: the per-file `FaultStart_s` marks
# when the fault begins.
#   0       = normal
#   1–4     = brake over-output (FL, FR, RL, RR)
#   5–8     = motor over-output (FL, FR, RL, RR)
FAULT_CLASS_MAP = {
    ("bf", "FL"): 1, ("bf", "FR"): 2, ("bf", "RL"): 3, ("bf", "RR"): 4,
    ("bc", "FL"): 1, ("bc", "FR"): 2, ("bc", "RL"): 3, ("bc", "RR"): 4,
    ("af", "FL"): 5, ("af", "FR"): 6, ("af", "RL"): 7, ("af", "RR"): 8,
    ("mf", "FL"): 5, ("mf", "FR"): 6, ("mf", "RL"): 7, ("mf", "RR"): 8,
    ("ac", "FL"): 5, ("ac", "FR"): 6, ("ac", "RL"): 7, ("ac", "RR"): 8,
}

_NAME_RE = re.compile(r"^(bf|af|mf|bc|ac)_(FL|FR|RL|RR)_", re.IGNORECASE)


def collect_mat_paths(mat_dirs: str | Path | list[str | Path] | tuple[str | Path, ...]) -> list[Path]:
    """Return sorted .mat files from one or more dataset directories."""
    dirs = [mat_dirs] if isinstance(mat_dirs, (str, Path)) else list(mat_dirs)
    paths: list[Path] = []
    for mat_dir in dirs:
        paths.extend(sorted(Path(mat_dir).glob("*.mat")))
    return sorted(paths, key=lambda p: str(p))


def _fault_class_from_name(path: Path) -> int:
    """Map a scenario-ID filename to its fault class (1–8)."""
    m = _NAME_RE.match(path.name)
    if m is None:
        raise ValueError(
            f"Cannot parse fault class from filename: {path.name!r}. "
            f"Expected '<bf|af|mf|bc|ac>_<FL|FR|RL|RR>_...' "
            f"(e.g. 'bf_FR_v010_accel_p5.mat')."
        )
    return FAULT_CLASS_MAP[(m.group(1).lower(), m.group(2).upper())]


class MatResidualDataset(Dataset):
    def __init__(
        self,
        mat_paths: list[str | Path],
        window_sec: float = 0.5,
        stride_sec: float = 0.1,
        warmup_sec: float = 1.0,
        ) -> None:
        all_features, all_labels = [], []
        skipped = 0
        for path in mat_paths:
            try:
                feats, labs = _extract_windows(Path(path), window_sec, stride_sec, warmup_sec)
            except Exception:
                skipped += 1
                continue
            all_features.append(feats)
            all_labels.append(labs)
        if skipped:
            import warnings
            warnings.warn(f"Skipped {skipped} unreadable .mat file(s).")
        self.features = torch.from_numpy(np.concatenate(all_features, axis=0))
        self.labels = torch.from_numpy(np.concatenate(all_labels, axis=0))

    @classmethod
    def from_tensors(cls, features: torch.Tensor, labels: torch.Tensor) -> "MatResidualDataset":
        """Build a dataset directly from pre-computed window tensors (e.g. a
        balanced subset), bypassing file I/O."""
        obj = cls.__new__(cls)
        obj.features = features
        obj.labels = labels
        return obj

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


def _h5_read(f: h5py.File, key: str) -> np.ndarray:
    """Read HDF5 dataset, resolving object references (MATLAB v7.3 #refs#)."""
    ds = f[key]
    if h5py.check_ref_dtype(ds.dtype):
        return np.asarray(f[ds[()].flat[0]]).ravel()
    return np.asarray(ds).ravel()


def _extract_windows(
    path: Path,
    window_sec: float,
    stride_sec: float,
    warmup_sec: float,
) -> tuple[np.ndarray, np.ndarray]:
    fault_class = _fault_class_from_name(path)
    with h5py.File(path, "r") as f:
        time = _h5_read(f, "data/time")
        fault_start = float(_h5_read(f, "data/FaultStart_s").flat[0])
        signals = np.stack([_h5_read(f, f"data/{s}") for s in RESIDUAL_SIGNALS], axis=0)

    mask = time >= warmup_sec
    time = time[mask]
    signals = signals[:, mask]
    sample_faulted = time >= fault_start          # per-sample fault period

    fs = int(round(1.0 / (time[1] - time[0])))
    win = int(round(window_sec * fs))
    stride = int(round(stride_sec * fs))
    n = signals.shape[1]

    feature_rows, label_rows = [], []
    for start in range(0, n - win + 1, stride):
        end = start + win
        seg = signals[:, start:end]               # (8, win)
        rms = np.sqrt(np.mean(seg ** 2, axis=1))
        mean = np.mean(seg, axis=1)
        var = np.var(seg, axis=1)
        feature_rows.append(np.concatenate([rms, mean, var]).astype(np.float32))  # (24,)
        # any-fault labeling: a window with ANY fault-period sample takes the
        # prefix class (biases toward early onset detection); else normal.
        faulted = bool(sample_faulted[start:end].any())
        label_rows.append(fault_class if faulted else 0)

    return (
        np.array(feature_rows, dtype=np.float32),
        np.array(label_rows, dtype=np.int64),
    )


def class_balanced_indices(labels: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """Indices that equalize the normal class against the faults.

    Target = size of the smallest fault class (labels 1-8) present. Normal
    (label 0) is randomly subsampled to the target; all fault windows are kept
    (minority; they carry the onset + developed trajectory). If no fault is
    present, all indices are returned.
    """
    g = torch.Generator().manual_seed(seed)
    classes = labels.unique().tolist()
    fault = [c for c in classes if c != 0]
    if not fault:
        return torch.arange(len(labels))
    target = min((labels == c).sum().item() for c in fault)

    keep = []
    for c in classes:
        idx = (labels == c).nonzero(as_tuple=True)[0]
        if c == 0 and len(idx) > target:
            perm = torch.randperm(len(idx), generator=g)[:target]
            idx = idx[perm]
        keep.append(idx)
    return torch.cat(keep).sort().values
