# tests/test_dataset.py
import numpy as np
import pytest
import h5py
from dataset import MatResidualDataset

_SIGNALS = [
    "res_vx", "res_r", "res_slip_fl", "res_slip_fr",
    "res_slip_rl", "res_slip_rr", "res_ax", "res_ay", "res_vy",
]


def _make_mat(path, n_sec, fault_start_sec, fs=1000):
    """Synthetic scenario .mat at fs Hz.

    Class comes from the filename prefix. The fault boundary is time-based:
    the per-file ``FaultStart_s`` marks when the fault begins. Fault_Flag is
    deliberately omitted — labeling does not use it.
    """
    n = int(round(n_sec * fs))
    t = np.arange(n) / fs
    with h5py.File(path, "w") as f:
        f.create_dataset("data/time", data=t.reshape(1, -1))
        f.create_dataset("data/FaultStart_s", data=np.array([[fault_start_sec]]))
        for sig in _SIGNALS:
            f.create_dataset(f"data/{sig}", data=np.random.randn(1, n))
    return path


@pytest.fixture
def mat2s(tmp_path):
    """2 s bf_FR file, fault past the end (all-normal). Default warmup=1 s."""
    return _make_mat(tmp_path / "bf_FR_v010_steady.mat", n_sec=2, fault_start_sec=99.0)


def test_len(mat2s):
    ds = MatResidualDataset([mat2s])
    # warmup 1 s -> 1000 samples remain; 0.5 s window (500) at 0.1 s stride (100)
    # starts 0,100,...,500 -> 6 windows
    assert len(ds) == 6


def test_item_shapes(mat2s):
    ds = MatResidualDataset([mat2s])
    feat, label = ds[0]
    assert feat.shape == (24,)
    assert label.shape == ()


def test_any_fault_labeling(tmp_path):
    """A window with even one fault-period sample takes the fault class."""
    path = _make_mat(tmp_path / "bf_FR_v010_steady.mat", n_sec=3, fault_start_sec=2.0)
    ds = MatResidualDataset([path])
    labels = [ds[i][1].item() for i in range(len(ds))]
    # 16 windows total; any window touching t>=2.0 -> class 2 (bf_FR).
    # starts 0..1500 step100; fault if start+500>1000 -> start>=600 -> 10 windows
    assert labels.count(2) == 10
    assert labels.count(0) == 6


def test_label_uses_prefix_class(tmp_path):
    """The fault class is the filename prefix (bf_RR -> 4), never a flag value."""
    path = _make_mat(tmp_path / "bf_RR_v010_accel_p5.mat", n_sec=3, fault_start_sec=2.0)
    ds = MatResidualDataset([path])
    labels = {ds[i][1].item() for i in range(len(ds))}
    assert labels == {0, 4}


def test_af_prefix_maps_to_motor_fault_classes(tmp_path):
    """The motor dataset uses af_* filenames; they map to classes 5-8."""
    expected = {"FL": 5, "FR": 6, "RL": 7, "RR": 8}
    for wheel, cls in expected.items():
        path = _make_mat(tmp_path / f"af_{wheel}_v010_steady.mat", n_sec=3, fault_start_sec=2.0)
        ds = MatResidualDataset([path])
        labels = {ds[i][1].item() for i in range(len(ds))}
        assert labels == {0, cls}


def test_all_normal_when_fault_after_end(tmp_path):
    """Fault start beyond the run -> every window is label 0."""
    path = _make_mat(tmp_path / "bf_FL_v010_steady.mat", n_sec=3, fault_start_sec=99.0)
    ds = MatResidualDataset([path])
    labels = {ds[i][1].item() for i in range(len(ds))}
    assert labels == {0}


def test_unparseable_filename_raises(tmp_path):
    path = _make_mat(tmp_path / "mystery_run.mat", n_sec=2, fault_start_sec=99.0)
    with pytest.raises(ValueError):
        MatResidualDataset([path])


def test_warmup_excluded(tmp_path):
    path = _make_mat(tmp_path / "bf_RL_v010_accel_p5.mat", n_sec=2, fault_start_sec=99.0)
    ds = MatResidualDataset([path], warmup_sec=1.5)
    # warmup 1.5 s -> 500 samples remain -> exactly 1 window
    assert len(ds) == 1


def test_overlap_increases_window_count(tmp_path):
    path = _make_mat(tmp_path / "bf_FL_v010_steady.mat", n_sec=2, fault_start_sec=99.0)
    overlap = MatResidualDataset([path])                       # default stride 0.1 s
    tiled = MatResidualDataset([path], stride_sec=0.5)         # non-overlapping
    assert len(overlap) == 6
    assert len(tiled) == 2
    assert len(overlap) > len(tiled)


def test_multiple_files_concatenated(tmp_path):
    paths = [
        _make_mat(tmp_path / "bf_FL_v010_steady.mat", n_sec=2, fault_start_sec=99.0),
        _make_mat(tmp_path / "bf_FR_v010_steady.mat", n_sec=2, fault_start_sec=99.0),
    ]
    ds = MatResidualDataset(paths)
    assert len(ds) == 12  # 6 windows each


def test_circle_prefixes_map_to_classes(tmp_path):
    """Circle datasets use bc_*/ac_* filenames -> brake 1-4 / motor 5-8."""
    cases = {
        "bc_FL": 1, "bc_FR": 2, "bc_RL": 3, "bc_RR": 4,
        "ac_FL": 5, "ac_FR": 6, "ac_RL": 7, "ac_RR": 8,
    }
    for prefix, cls in cases.items():
        path = _make_mat(tmp_path / f"{prefix}_v010_stm010_accel_p5.mat",
                         n_sec=3, fault_start_sec=2.0)
        ds = MatResidualDataset([path])
        labels = {ds[i][1].item() for i in range(len(ds))}
        assert labels == {0, cls}
