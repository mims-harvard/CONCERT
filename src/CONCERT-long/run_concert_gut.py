#!/usr/bin/env python3
"""
CONCERT (Gut) driver — spatial-aware perturbation over developmental "day" batches.

Features
--------
- YAML/JSON config file support (CLI flags only override what you pass)
- Structured logging, deterministic preprocessing
- Per-batch spatial scaling + batch one-hot appended to coordinates
- Grid inducing points tiled across batches (with batch one-hot block)
- Train or evaluate (counterfactual on a target day/perturbation)
- Saves outputs to .h5ad
- Optional Weights & Biases tracking (--wandb ...)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass, asdict, replace
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# Optional YAML support
try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

# Optional wandb
try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover
    wandb = None

from concert_batch_gut import CONCERT
from preprocess import normalize, geneSelection


# -----------------------------------------------------------------------------
# Logging / utils
# -----------------------------------------------------------------------------
def setup_logging(verbosity: int = 1) -> None:
    level = logging.WARNING if verbosity <= 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def auto_batch_size(n: int) -> int:
    if n <= 1024:
        return 128
    if n <= 2048:
        return 256
    return 512


def load_config_file(path: Optional[str]) -> dict:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    if p.suffix.lower() == ".json":
        return json.loads(p.read_text())
    if p.suffix.lower() in {".yml", ".yaml"}:
        if yaml is None:
            raise ImportError("PyYAML is not installed but a YAML config was provided.")
        return yaml.safe_load(p.read_text()) or {}
    # fallbacks
    try:
        if yaml is not None:
            return yaml.safe_load(p.read_text()) or {}
    except Exception:
        pass
    try:
        return json.loads(p.read_text())
    except Exception as e:  # pragma: no cover
        raise ValueError(f"Unsupported config format for {path}: {e}")


def strings_to_stable_index(values: np.ndarray) -> np.ndarray:
    """Deterministically map strings to contiguous integer codes (0..K-1), stable across runs."""
    hashed = np.array([sum(ord(c) for c in s) for s in values], dtype=int)
    uniq = np.unique(hashed)
    remap = {u: i for i, u in enumerate(uniq)}
    return np.array([remap[h] for h in hashed], dtype=int)


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
@dataclass
class RunConfigGut:
    # IO
    data_file: str = "data.h5"
    outdir: str = "./outputs"
    project_index: str = "x"  # used in filenames

    # Selection / normalization
    select_genes: int = 0

    # Training
    batch_size: str = "auto"
    maxiter: int = 5000
    train_size: float = 0.95
    patience: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-6

    # Architecture
    encoder_layers: Iterable[int] = (128, 64, 32)
    decoder_layers: Iterable[int] = (64, 128)
    GP_dim: int = 2
    Normal_dim: int = 8
    input_dim: int = 192  # per your original gut model

    # VAE / regularization
    noise: float = 0.25
    dropoutE: float = 0.0
    dropoutD: float = 0.0
    dynamicVAE: bool = True
    init_beta: float = 10.0
    min_beta: float = 5.0
    max_beta: float = 25.0
    KL_loss: float = 0.025
    num_samples: int = 1

    # GP / inducing points (batched by day)
    fix_inducing_points: bool = True
    grid_inducing_points: bool = True
    inducing_point_steps: int = 6
    inducing_point_nums: int | None = None
    fixed_gp_params: bool = False
    loc_range: float = 20.0
    kernel_scale: float = 20.0
    multi_kernel_mode: bool = False

    # Dispersion
    shared_dispersion: bool = True

    # Runtime / persistence
    model_file: str = "model.pt"
    device: str = "cuda"
    verbosity: int = 1

    # Eval / counterfactual
    stage: str = "train"  # {"train","eval"}
    pert_cells: str = "D73"
    target_cell_day: float = 13.0
    target_cell_perturbation: str = "0.0"

    # Weights & Biases
    wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_run: Optional[str] = None
    wandb_mode: str = "online"  # online|offline|disabled

    # Config file
    config: Optional[str] = None


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
def load_h5_dataset(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    X : (N, G) float32
    loc : (N, 2) float32
    region_str : (N,) str
    perturb_str : (N,) str
    day_str : (N,) str
    """
    with h5py.File(path, "r") as f:
        X = np.array(f["X"]).T.astype("float32")        # original stored as (G, N)
        loc = np.array(f["pos"]).T.astype("float32")    # to (N, 2+), we keep first 2
        loc = loc[:, :2]
        region_str = np.array(f["region"]).astype(str)
        perturb_str = np.array(f["perturbation"]).astype(str)
        day_str = np.array(f["day"]).astype(str)
    return X, loc, region_str, perturb_str, day_str


def map_days_to_numeric_and_batch(day_str: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    mapping = {"D0": (1.0, 0), "D12": (12.0, 1), "D30": (30.0, 2), "D73": (73.0, 3)}
    numeric, batch_ids = [], []
    for d in day_str:
        if d not in mapping:
            raise ValueError(f"Unknown day label '{d}'. Supported: {list(mapping)}")
        num, bi = mapping[d]
        numeric.append(num)
        batch_ids.append(bi)
    return np.asarray(numeric, dtype=float), np.asarray(batch_ids, dtype=int), {k: v for k, (_, v) in mapping.items()}


def per_batch_scale_positions(pos: np.ndarray, batch_onehot: np.ndarray, loc_range: float) -> np.ndarray:
    """Independently min-max scale positions within each batch to [0, loc_range]."""
    n_batch = batch_onehot.shape[1]
    pos_scaled = np.zeros_like(pos, dtype=np.float32)
    for i in range(n_batch):
        mask = batch_onehot[:, i] == 1.0
        scaler = MinMaxScaler()
        pos_scaled[mask, :] = scaler.fit_transform(pos[mask, :]) * float(loc_range)
    return pos_scaled.astype(np.float32)


def build_inducing_points_grid_tiled(n_batch: int, steps: int, loc_range: float) -> np.ndarray:
    """Grid (steps x steps) in [0, loc_range], tiled across batches with one-hot block appended."""
    eps = 1e-5
    grid_xy = np.mgrid[0:(1 + eps):(1.0 / steps), 0:(1 + eps):(1.0 / steps)].reshape(2, -1).T * float(loc_range)
    grid_xy = grid_xy.astype(np.float32)  # (M, 2)
    tiled_xy = np.tile(grid_xy, (n_batch, 1))  # (M * n_batch, 2)
    onehots = []
    for i in range(n_batch):
        oh = np.zeros((grid_xy.shape[0], n_batch), dtype=np.float32)
        oh[:, i] = 1.0
        onehots.append(oh)
    onehots = np.concatenate(onehots, axis=0)  # (M * n_batch, n_batch)
    inducing = np.concatenate([tiled_xy, onehots], axis=1)  # (M * n_batch, 2 + n_batch)
    return inducing.astype(np.float32)


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------
def run(cfg: RunConfigGut) -> None:
    logging.info("Config: %s", asdict(cfg))
    outdir = ensure_dir(cfg.outdir)

    # --- Optional Weights & Biases ---
    wb = None
    if cfg.wandb and cfg.wandb_mode != "disabled":
        if wandb is None:
            logging.warning("wandb not installed; continuing without wandb.")
        else:
            try:
                wb = wandb.init(
                    project=cfg.wandb_project or "concert",
                    name=cfg.wandb_run,
                    mode=cfg.wandb_mode,
                    config=asdict(cfg),
                )
            except Exception as e:  # pragma: no cover
                logging.warning("wandb init failed (%s); continuing without wandb.", e)
                wb = None

    # Load data
    X, loc, region_str, perturb_str, day_str = load_h5_dataset(cfg.data_file)

    # Encode attributes
    perturb_idx = strings_to_stable_index(perturb_str)
    day_numeric, day_batch_idx, day_batch_map = map_days_to_numeric_and_batch(day_str)

    # cell_atts = [perturbation_code, numeric_day]
    cell_atts = np.c_[perturb_idx, day_numeric].astype(int, copy=False)
    sample_indices = torch.arange(X.shape[0], dtype=torch.int)

    # Batch one-hot from day batch indices
    n_classes = int(len(np.unique(day_batch_idx)))
    batch_onehot = np.eye(n_classes, dtype=np.float32)[day_batch_idx]
    logging.info("Batches (day groups): %s", {k: int(v) for k, v in day_batch_map.items()})

    # Batch size
    bs = auto_batch_size(X.shape[0]) if cfg.batch_size == "auto" else int(cfg.batch_size)
    logging.info("Batch size: %s", bs)

    # Optional gene selection
    if cfg.select_genes and cfg.select_genes > 0:
        logging.info("Selecting top %d genes...", cfg.select_genes)
        important = geneSelection(X, n=cfg.select_genes, plot=False)
        X = X[:, important]
        np.savetxt(outdir / "selected_genes.txt", important, fmt="%d")

    # Per-batch scaling of positions to [0, loc_range], then append batch one-hots
    loc_scaled = per_batch_scale_positions(loc, batch_onehot, cfg.loc_range)
    loc_batched = np.concatenate([loc_scaled, batch_onehot], axis=1).astype(np.float32)
    logging.info("Shapes — X: %s, loc_batched: %s", X.shape, loc_batched.shape)

    # Inducing points
    if cfg.grid_inducing_points:
        inducing = build_inducing_points_grid_tiled(n_classes, cfg.inducing_point_steps, cfg.loc_range)
        logging.info("Inducing points (grid tiled): %s", inducing.shape)
    else:
        assert cfg.inducing_point_nums and cfg.inducing_point_nums > 0, \
            "inducing_point_nums must be set when grid_inducing_points=False"
        km = KMeans(n_clusters=cfg.inducing_point_nums, n_init=100).fit(loc_scaled)
        centers = km.cluster_centers_.astype(np.float32)
        oh = np.zeros((centers.shape[0], n_classes), dtype=np.float32)
        oh[:, 0] = 1.0
        inducing = np.concatenate([centers, oh], axis=1).astype(np.float32)
        logging.info("Inducing points (k-means): %s", inducing.shape)

    # AnnData + normalization
    adata = sc.AnnData(X, dtype="float32")
    adata = normalize(adata, size_factors=True, normalize_input=True, logtrans_input=True)

    # Model
    model = CONCERT(
        cell_atts=cell_atts,
        num_genes=adata.n_vars,
        input_dim=cfg.input_dim,
        GP_dim=cfg.GP_dim,
        Normal_dim=cfg.Normal_dim,
        n_batch=n_classes,
        encoder_layers=list(cfg.encoder_layers),
        decoder_layers=list(cfg.decoder_layers),
        noise=cfg.noise,
        encoder_dropout=cfg.dropoutE,
        decoder_dropout=cfg.dropoutD,
        shared_dispersion=cfg.shared_dispersion,
        fixed_inducing_points=cfg.fix_inducing_points,
        initial_inducing_points=inducing,
        fixed_gp_params=cfg.fixed_gp_params,
        kernel_scale=cfg.kernel_scale,
        multi_kernel_mode=cfg.multi_kernel_mode,
        N_train=adata.n_obs,
        KL_loss=cfg.KL_loss,
        dynamicVAE=cfg.dynamicVAE,
        init_beta=cfg.init_beta,
        min_beta=cfg.min_beta,
        max_beta=cfg.max_beta,
        dtype=torch.float32,
        device=cfg.device,
    )
    logging.info("Model initialized.")

    # Helper for wandb logging
    def _log(metrics: Dict, step: int) -> None:
        if wb is not None:
            try:
                wb.log(metrics, step=step)
            except Exception:
                pass

    # Train or Evaluate (counterfactual)
    if cfg.stage.lower() == "train":
        logging.info("Training...")
        model.train_model(
            pos=loc_batched,
            ncounts=adata.X,
            raw_counts=adata.raw.X,
            size_factors=adata.obs.size_factors,
            batch=batch_onehot,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            batch_size=bs,
            num_samples=cfg.num_samples,
            train_size=cfg.train_size,
            maxiter=cfg.maxiter,
            patience=cfg.patience,
            save_model=True,
            model_weights=str(outdir / cfg.model_file),
            log_fn=_log,  # <-- stream metrics to wandb if enabled
        )
        logging.info("Training finished.")
        return

    # ---- EVAL / COUNTERFACTUAL ----
    model.load_model(str(outdir / cfg.model_file) if os.path.isfile(outdir / cfg.model_file) else cfg.model_file)
    logging.info("Loaded weights.")

    # Use day string to select the subset of cells we’ll export after counterfactual
    sel_mask = (day_str == cfg.pert_cells)
    if not np.any(sel_mask):
        logging.warning("No cells matched pert_cells '%s'; proceeding without subset.", cfg.pert_cells)
    pert_ind = np.where(sel_mask)[0]

    # Map perturbation name → code for target perturbation
    pert_map = {perturb_str[i]: int(perturb_idx[i]) for i in range(len(perturb_str))}
    if cfg.target_cell_perturbation not in pert_map:
        raise ValueError(f"Unknown target perturbation '{cfg.target_cell_perturbation}'. "
                         f"Known: {sorted(set(perturb_str.tolist()))}")
    target_pert_code = int(pert_map[cfg.target_cell_perturbation])

    # Counterfactual: set selected cells to (target_day, target_perturbation)
    logging.info("Counterfactual → day=%.1f, perturbation=%s (code=%d)",
                 cfg.target_cell_day, cfg.target_cell_perturbation, target_pert_code)

    perturbed_counts, pert_atts = model.counterfactualPrediction(
        X=loc_batched,
        sample_index=sample_indices,
        cell_atts=cell_atts,
        batch_size=bs,
        n_samples=25,
        perturb_cell_id=pert_ind,
        target_cell_day=float(cfg.target_cell_day),
        target_cell_perturbation=target_pert_code,
    )

    # Keep only the subset cells in the output
    if pert_ind.size > 0:
        perturbed_counts = perturbed_counts[pert_ind, :]
        pert_atts = pert_atts[pert_ind, :]

    # Save as h5ad with obs=[perturbation, day]
    obs_df = pd.DataFrame(pert_atts, columns=["perturbation", "day"])
    ad_out = sc.AnnData(perturbed_counts, obs=obs_df)
    out_path = (
        Path(cfg.outdir)
        / f"res_gut_{cfg.project_index}_perturbed_counts.h5ad"
    )
    ad_out.write(out_path)
    logging.info("Wrote counterfactual counts to %s", out_path)


# -----------------------------------------------------------------------------
# CLI (config-first; CLI only overrides what you pass)
# -----------------------------------------------------------------------------
def str2bool(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    return v.lower() in ("1", "true", "yes", "y", "on")


def parse_args() -> RunConfigGut:
    p = argparse.ArgumentParser(
        description="CONCERT Gut runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config path only; no defaults for the rest to avoid clobbering YAML
    p.add_argument("--config", help="YAML/JSON config file")

    # IO / basic
    p.add_argument("--data_file")
    p.add_argument("--outdir")
    p.add_argument("--project_index")

    # Selection / training
    p.add_argument("--select_genes", type=int)
    p.add_argument("--batch_size")
    p.add_argument("--maxiter", type=int)
    p.add_argument("--train_size", type=float)
    p.add_argument("--patience", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--weight_decay", type=float)

    # Architecture / VAE
    p.add_argument("--encoder_layers", nargs="+", type=int)
    p.add_argument("--decoder_layers", nargs="+", type=int)
    p.add_argument("--GP_dim", type=int)
    p.add_argument("--Normal_dim", type=int)
    p.add_argument("--input_dim", type=int)
    p.add_argument("--noise", type=float)
    p.add_argument("--dropoutE", type=float)
    p.add_argument("--dropoutD", type=float)
    p.add_argument("--dynamicVAE", type=str2bool)
    p.add_argument("--init_beta", type=float)
    p.add_argument("--min_beta", type=float)
    p.add_argument("--max_beta", type=float)
    p.add_argument("--KL_loss", type=float)
    p.add_argument("--num_samples", type=int)

    # GP / inducing
    p.add_argument("--fix_inducing_points", type=str2bool)
    p.add_argument("--grid_inducing_points", type=str2bool)
    p.add_argument("--inducing_point_steps", type=int)
    p.add_argument("--inducing_point_nums", type=int)
    p.add_argument("--fixed_gp_params", type=str2bool)
    p.add_argument("--loc_range", type=float)
    p.add_argument("--kernel_scale", type=float)
    p.add_argument("--multi_kernel_mode", type=str2bool)
    p.add_argument("--shared_dispersion", type=str2bool)

    # Runtime / persistence
    p.add_argument("--model_file")
    p.add_argument("--device")
    p.add_argument("--verbosity", type=int)
    p.add_argument("--stage", choices=["train", "eval"])

    # Counterfactual
    p.add_argument("--pert_cells")
    p.add_argument("--target_cell_day", type=float)
    p.add_argument("--target_cell_perturbation")

    # Weights & Biases
    p.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    p.add_argument("--wandb_project")
    p.add_argument("--wandb_run")
    p.add_argument("--wandb_mode", choices=["online", "offline", "disabled"])

    a = p.parse_args()

    # 1) defaults
    cfg = RunConfigGut()

    # 2) merge config file
    file_cfg = load_config_file(getattr(a, "config", None)) if getattr(a, "config", None) else {}
    if file_cfg:
        for k in ("encoder_layers", "decoder_layers"):
            if k in file_cfg and isinstance(file_cfg[k], list):
                file_cfg[k] = tuple(file_cfg[k])
        cfg = replace(cfg, **{k: v for k, v in file_cfg.items() if hasattr(cfg, k)})

    # 3) apply ONLY explicit CLI overrides
    import sys as _sys
    specified = set()
    for action in p._actions:
        if not action.option_strings:
            continue
        if any(opt in _sys.argv for opt in action.option_strings):
            specified.add(action.dest)
    cli = {k: getattr(a, k) for k in specified if hasattr(cfg, k) and getattr(a, k) is not None}

    # post-process batch_size if given
    if "batch_size" in cli:
        try:
            cli["batch_size"] = int(cli["batch_size"])
        except (TypeError, ValueError):
            pass

    cfg = replace(cfg, **cli)
    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    setup_logging(cfg.verbosity)
    run(cfg)