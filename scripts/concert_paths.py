"""Shared path resolution and import order for the reproduction scripts.

The four scripts in this directory reproduce one paper result each from a trained checkpoint.
Every input sits in two downloadable folders, and two environment variables point at them:

| variable                  | what it holds                                                  |
|---------------------------|----------------------------------------------------------------|
| `CONCERT_DATA_DIR`        | the `data/` folder: sections, benchmark assets, shipped results |
| `CONCERT_CHECKPOINT_ROOT` | the `checkpoints/` folder, one directory per trained model      |
| `CONCERT_WORKDIR`         | where the scripts write (optional)                             |

The defaults are the lab copies of the two folders, so the scripts run as-is inside the lab.
The layout of `data/` is defined once, in `benchmark_evaluation/release_paths.py`.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

MODELS = REPO_ROOT / "models"
EVAL_SRC = REPO_ROOT / "benchmark_evaluation"
EXPERIMENTS = REPO_ROOT / "experiments"

sys.path.insert(0, str(EVAL_SRC))
import release_paths as rp  # noqa: E402

DATA = rp.DATA
CHECKPOINT_ROOT = rp.CHECKPOINTS


def _p(env_var: str, default) -> Path:
    return Path(os.environ.get(env_var, str(default))).expanduser()


def workdir(name: str) -> Path:
    """Where a script writes. Never inside the repository, never on holylfs06."""
    root = rp.WORKDIR
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def prepend_src(*dirs: Path) -> None:
    """Put the released code at the front of sys.path, in the given order.

    Several evaluation modules insert their own directory at sys.path[0] when they
    are imported, and older copies of the same file names exist in the working
    trees. Importing with the release directories in front binds the release copy
    into sys.modules first; every later import then reuses it.
    """
    for d in dirs:
        s = str(d)
        while s in sys.path:
            sys.path.remove(s)
    for d in reversed(dirs):
        sys.path.insert(0, str(d))


def check(items, strict: bool = True) -> bool:
    """Print an existence table for (label, path) pairs. Fail early, not three steps later."""
    ok = True
    for label, path in items:
        present = Path(path).exists()
        ok &= present
        print(f"{'OK ' if present else 'MISSING':>8}  {label:<24}  {path}")
    if not ok and strict:
        raise SystemExit(
            "\nSome inputs are missing. Set the environment variables listed at the top of "
            "scripts/concert_paths.py, or see datasets/README.md for where the data comes from."
        )
    return ok


def describe_device():
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch {torch.__version__}  device {dev}"
          + (f"  ({torch.cuda.get_device_name(0)})" if dev == "cuda" else
             "   -- CPU ONLY: prediction steps will be unusably slow"))
    return dev


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}", flush=True)
