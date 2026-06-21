# tests/test_split.py
from pathlib import Path
from split import stratified_run_split, _parse_run, subsample_mat_paths, _STEER_MAG_RE, SWEEP_SPEEDS, SWEEP_STEER_MAGS
from dataset import _fault_class_from_name

WHEELS = ["FL", "FR", "RL", "RR"]
MODES = ["steady", "accel_p5", "accel_p10", "decel_n5", "decel_n10"]


def _all_names():
    names = []
    for w in WHEELS:
        for m in MODES:
            for v in range(10, 131):  # 121 speeds
                names.append(Path(f"bf_{w}_v{v:03d}_{m}.mat"))
    return names


def test_no_run_in_two_splits():
    tr, va, te = stratified_run_split(_all_names(), seed=0)
    s_tr, s_va, s_te = set(tr), set(va), set(te)
    assert s_tr & s_va == set()
    assert s_tr & s_te == set()
    assert s_va & s_te == set()
    assert len(s_tr) + len(s_va) + len(s_te) == 2420


def test_every_class_mode_in_each_split():
    tr, va, te = stratified_run_split(_all_names(), seed=0)

    def keys(paths):
        out = set()
        for p in paths:
            parts = p.stem.split("_")
            cls = "_".join(parts[:2])          # bf_FL
            mode = "_".join(parts[3:])          # accel_p5
            out.add((cls, mode))
        return out

    assert len(keys(tr)) == 20  # 4 wheels x 5 modes
    assert keys(tr) == keys(va) == keys(te)


def test_ratios_approximately_respected():
    tr, va, te = stratified_run_split(_all_names(), ratios=(0.70, 0.15, 0.15), seed=0)
    assert abs(len(tr) / 2420 - 0.70) < 0.02
    assert abs(len(va) / 2420 - 0.15) < 0.02


def test_holdout_speeds_only_in_test():
    tr, va, te = stratified_run_split(_all_names(), seed=0, holdout_speeds=(130,))

    def speeds(paths):
        return {int(p.stem.split("_")[2][1:]) for p in paths}

    assert 130 not in speeds(tr)
    assert 130 not in speeds(va)
    assert 130 in speeds(te)


def test_parse_run_circle_and_straight():
    # circle: steering token st{m|p|z}{mag} is present
    assert _parse_run(Path("bc_FL_v010_stm010_accel_p5.mat")) == (10, "m", "accel_p5")
    assert _parse_run(Path("ac_RR_v130_stp300_steady.mat")) == (130, "p", "steady")
    assert _parse_run(Path("bc_FL_v012_stz000_decel_n5.mat")) == (12, "z", "decel_n5")
    # straight: no steering token -> sign is None (backward compatible)
    assert _parse_run(Path("bf_FL_v010_accel_p5.mat")) == (10, None, "accel_p5")
    assert _parse_run(Path("bf_FR_v010_steady.mat")) == (10, None, "steady")


def _circle_names():
    names = []
    for typ in ("bc", "ac"):
        for w in WHEELS:
            for mode in ("steady", "accel_p5", "decel_n5"):
                for stoken, smags in (("stm", (10, 150, 300)), ("stp", (10, 150, 300)), ("stz", (0,))):
                    speeds = (10, 25, 40, 55, 70, 85, 100, 115, 130) if stoken == "stz" else (10, 70, 130)
                    for smag in smags:
                        for v in speeds:
                            names.append(Path(f"{typ}_{w}_v{v:03d}_{stoken}{smag:03d}_{mode}.mat"))
    return names  # 2*4*3*(3*3 + 3*3 + 1*9) = 648 runs; 72 strata of 9 runs each


def test_subsample_none_returns_all():
    paths = [Path("bc_FL_v050_stm100_accel_p5.mat")]
    assert subsample_mat_paths(paths) == paths


def test_subsample_speeds_filters():
    paths = [Path(f"bc_FL_v{s:03d}_stm100_accel_p5.mat") for s in range(10, 131, 2)]
    result = subsample_mat_paths(paths, keep_speeds={10, 22})
    assert {_parse_run(p)[0] for p in result} == {10, 22}


def test_subsample_steer_mags_filters():
    paths = [Path(f"bc_FL_v050_stm{m:03d}_accel_p5.mat") for m in range(10, 301, 10)]
    result = subsample_mat_paths(paths, keep_steer_mags={10, 100, 300})
    mags = {int(_STEER_MAG_RE.search(p.stem).group(1)) for p in result}
    assert mags == {10, 100, 300}


def test_subsample_stz_always_passes_steer_filter():
    paths = [Path("bc_FL_v050_stz000_steady.mat")]
    result = subsample_mat_paths(paths, keep_steer_mags={10, 100})  # 0 not listed
    assert len(result) == 1


def test_subsample_straight_passes_steer_filter():
    paths = [Path("bf_FL_v010_accel_p5.mat")]
    result = subsample_mat_paths(paths, keep_steer_mags={10, 100})
    assert len(result) == 1


def test_sweep_constants_include_extremes():
    assert 10 in SWEEP_SPEEDS and 130 in SWEEP_SPEEDS
    assert 10 in SWEEP_STEER_MAGS and 300 in SWEEP_STEER_MAGS


def test_circle_stratifies_by_class_mode_sign():
    tr, va, te = stratified_run_split(_circle_names(), seed=0)
    # no run leaks across splits
    assert set(tr) & set(va) == set()
    assert set(tr) & set(te) == set()
    assert set(va) & set(te) == set()
    assert len(tr) + len(va) + len(te) == 648

    def keys(paths):
        out = set()
        for p in paths:
            cls = _fault_class_from_name(p)
            _, sign, mode = _parse_run(p)
            out.add((cls, mode, sign))
        return out

    # 8 classes x 3 modes x 3 signs = 72 strata, all present in every split
    assert len(keys(tr)) == 72
    assert keys(tr) == keys(va) == keys(te)
