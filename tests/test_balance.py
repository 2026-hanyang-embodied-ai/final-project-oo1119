# tests/test_balance.py
import torch
from dataset import class_balanced_indices, MatResidualDataset


def test_subsamples_normal_keeps_all_fault():
    labels = torch.tensor([0] * 100 + [1] * 20 + [2] * 30, dtype=torch.long)
    idx = class_balanced_indices(labels, seed=0)
    kept = labels[idx]
    # target = smallest fault count = 20; normal subsampled to 20, faults all kept
    assert (kept == 0).sum().item() == 20
    assert (kept == 1).sum().item() == 20
    assert (kept == 2).sum().item() == 30


def test_no_fault_returns_all():
    labels = torch.zeros(50, dtype=torch.long)
    idx = class_balanced_indices(labels, seed=0)
    assert len(idx) == 50


def test_from_tensors_roundtrip():
    feats = torch.randn(7, 24)
    labs = torch.randint(0, 9, (7,))
    ds = MatResidualDataset.from_tensors(feats, labs)
    assert len(ds) == 7
    f, l = ds[3]
    assert f.shape == (24,)
    assert torch.equal(f, feats[3])
