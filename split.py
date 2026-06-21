# split.py
from __future__ import annotations
from pathlib import Path
import re
import random
from collections import defaultdict

from dataset import _fault_class_from_name

# stem: _v010_[stm010_]accel_p5  -- the st{m|p|z}{mag} steering token is optional
# (present only in the circle dataset). The [mpz] guard prevents matching "steady".
_RUN_RE = re.compile(r"_v(\d+)_(?:st([mpz])\d+_)?(.+)$")
_STEER_MAG_RE = re.compile(r"_st[mpz](\d+)_")

SWEEP_SPEEDS = set(range(10, 131, 12))            # {10,22,34,46,58,70,82,94,106,118,130}
SWEEP_STEER_MAGS = {10, 50, 100, 150, 200, 250, 300}  # 7 magnitudes, extremes kept


def _parse_run(path: Path) -> tuple[int, str | None, str]:
    """Return (speed_kph, steer_sign, mode) from a scenario-ID filename stem.

    steer_sign is one of 'm', 'p', 'z' for circle datasets, or None for
    straight-line datasets (backward compatible).
    """
    m = _RUN_RE.search(path.stem)
    if m is None:
        raise ValueError(f"Cannot parse speed/mode from {path.name!r}.")
    return int(m.group(1)), m.group(2), m.group(3)


def subsample_mat_paths(
    paths: list[Path],
    keep_speeds: set[int] | None = None,
    keep_steer_mags: set[int] | None = None,
) -> list[Path]:
    """Filter mat paths by speed and/or steer magnitude.

    Straight-line files (no steer token) pass the steer_mag filter unchanged.
    stz files (steer_mag == 0) always pass the steer_mag filter.
    """
    if keep_speeds is None and keep_steer_mags is None:
        return paths
    result = []
    for p in paths:
        try:
            speed, _, _ = _parse_run(p)
        except ValueError:
            result.append(p)
            continue
        if keep_speeds is not None and speed not in keep_speeds:
            continue
        if keep_steer_mags is not None:
            m = _STEER_MAG_RE.search(p.stem)
            if m is not None:
                mag = int(m.group(1))
                if mag != 0 and mag not in keep_steer_mags:
                    continue
        result.append(p)
    return result


def stratified_run_split(
    mat_paths: list[str | Path],
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 0,
    holdout_speeds: tuple[int, ...] | None = None,
) -> tuple[list[Path], list[Path], list[Path]]:
    """Assign whole runs to train/val/test, stratified by (class, mode, steer_sign).

    Speed and steer magnitude are distributed randomly within each stratum.
    If holdout_speeds is given, runs at those speeds go to test only
    (extrapolation stress test). No file appears in two splits.
    """
    rng = random.Random(seed)
    holdout = set(holdout_speeds or ())

    pool: list[Path] = []
    test_extra: list[Path] = []
    for p in (Path(x) for x in mat_paths):
        speed, _sign, _mode = _parse_run(p)
        (test_extra if speed in holdout else pool).append(p)

    groups: dict[tuple, list[Path]] = defaultdict(list)
    for p in pool:
        cls = _fault_class_from_name(p)
        _, sign, mode = _parse_run(p)
        groups[(cls, mode, sign)].append(p)

    train: list[Path] = []
    val: list[Path] = []
    test: list[Path] = []
    for key in sorted(groups, key=lambda k: (k[0], k[1], k[2] or "")):
        members = sorted(groups[key], key=lambda x: x.name)
        rng.shuffle(members)
        n = len(members)
        n_train = int(round(n * ratios[0]))
        n_val = int(round(n * ratios[1]))
        train += members[:n_train]
        val += members[n_train:n_train + n_val]
        test += members[n_train + n_val:]

    test += test_extra
    return train, val, test
