"""Energy-distance scoring aligned with CONCERT eval_utils.energy_distance."""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "benchmark_evaluation" / "release_paths.py").is_file()) / "benchmark_evaluation"))
import release_paths as _rp  # noqa: E402  data/, checkpoints/ and runs/ roots
import json
import os
import sys
from pathlib import Path
import numpy as np

EVAL_UTILS_DIR = Path(__file__).resolve().parent
DEFAULT_RAW_COUNTS_DIR = Path(os.environ.get("CONCERT_DATA_DIR", f"{_rp.DATA}")) / "spacseq" / "prepared" / "raw_counts"
RAW_COUNTS_NAME = "filtered_raw_counts.npy"
LIB_SIZES_NAME = "filtered_lib_sizes.npy"
RAW_META_NAME = "raw_counts_meta.json"


def resolve_raw_counts_dir(raw_counts_dir: Path | str | None = None) -> Path:
    if raw_counts_dir is not None:
        return Path(raw_counts_dir)
    env = os.environ.get("SPAC_RAW_COUNTS_DIR")
    if env:
        return Path(env)
    return DEFAULT_RAW_COUNTS_DIR


def hvg_mask_from_rank(adata, n_top: int = 2000) -> np.ndarray:
    if "hvg_rank" not in adata.var.columns:
        raise KeyError("adata.var must contain hvg_rank for aligned HVG scoring")
    return np.asarray(adata.var["hvg_rank"].values <= int(n_top), dtype=bool)


def log_cp10k_to_counts(log_x: np.ndarray, lib_size: np.ndarray) -> np.ndarray:
    x = np.asarray(log_x, dtype=np.float64)
    ls = np.asarray(lib_size, dtype=np.float64).reshape(-1, 1)
    cp10k = np.expm1(np.clip(x, 0.0, 14.5))  # avoid overflow when preds are mis-scaled
    return np.clip(cp10k * ls / 1e4, 0.0, None).astype(np.float32)


def _import_eval_utils():
    sys.path.insert(0, str(EVAL_UTILS_DIR))
    import eval_utils as eu
    return eu


def aligned_energy_distance(pred_log_cp10k, obs_raw, lib_size, gene_mask) -> float:
    eu = _import_eval_utils()
    pred_counts = log_cp10k_to_counts(pred_log_cp10k, lib_size)
    obs = np.asarray(obs_raw, dtype=np.float32)
    keep = np.asarray(gene_mask, dtype=bool)
    return float(eu.energy_distance(pred_counts, obs, keep, subsample=500, n_pcs=50, random_state=0))


def raw_counts_paths(raw_counts_dir: Path | str | None = None):
    base = resolve_raw_counts_dir(raw_counts_dir)
    return base / RAW_COUNTS_NAME, base / LIB_SIZES_NAME, base / RAW_META_NAME


def open_raw_counts(biolord_outdir: Path | None = None, raw_counts_dir: Path | str | None = None) -> np.memmap:
    del biolord_outdir  # kept for call-site compatibility; the cache location comes from raw_counts_dir
    counts_path, _, meta_path = raw_counts_paths(raw_counts_dir)
    if not counts_path.exists() or not meta_path.exists():
        base = resolve_raw_counts_dir(raw_counts_dir)
        raise FileNotFoundError(
            f"Missing raw count cache under {base}. Run build_filtered_raw_counts.py first."
        )
    meta = json.loads(meta_path.read_text())
    # The cache is written with np.lib.format.open_memmap, i.e. a real .npy file with a header
    # (128 bytes here). np.load honours the header, so rows start after it.
    arr = np.load(counts_path, mmap_mode="r")
    if tuple(arr.shape) != tuple(meta["shape"]):
        raise ValueError(f"{counts_path}: npy shape {arr.shape} != meta shape {tuple(meta['shape'])}")
    if arr.dtype != np.dtype(meta.get("dtype", "float32")):
        raise ValueError(f"{counts_path}: npy dtype {arr.dtype} != meta dtype {meta.get('dtype')}")
    return arr


def load_lib_sizes(biolord_outdir: Path | None = None, raw_counts_dir: Path | str | None = None) -> np.ndarray:
    del biolord_outdir
    _, lib_path, _ = raw_counts_paths(raw_counts_dir)
    if not lib_path.exists():
        raise FileNotFoundError(f"Missing {lib_path}. Run build_filtered_raw_counts.py first.")
    return np.load(lib_path)


def hvg_info_dict(hvg_mask: np.ndarray) -> dict:
    return {
        "n_hvgs": int(hvg_mask.sum()),
        "hvg_gene_indices": np.where(hvg_mask)[0].tolist(),
        "hvg_selection": "hvg_rank",
        "hvg_rank_max": 2000,
        "edist": "concert_eval_utils",
    }
