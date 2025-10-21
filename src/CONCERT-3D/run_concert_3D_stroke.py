#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass, asdict, replace
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn.preprocessing import MinMaxScaler

# Optional YAML support
try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

# Optional W&B
try:
    import wandb  # type: ignore
    _WANDB_AVAILABLE = True
except Exception:
    wandb = None
    _WANDB_AVAILABLE = False

from concert_batch_3D_stroke import CONCERT
from preprocess import normalize, geneSelection


# -----------------------------------------------------------------------------
# Utils / logging
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


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
@dataclass
class RunConfigStroke3D:
    # IO
    data_file: str = "data.h5"
    outdir: str = "./outputs"
    project_index: str = "x"  # for filenames

    # Selection / normalization
    select_genes: int = 0

    # Training / runtime
    stage: str = "train"   # {"train","eval"}
    batch_size: str = "auto"
    maxiter: int = 5000
    train_size: float = 0.95
    patience: int = 200
    lr: float = 1e-4
    weight_decay: float = 1e-5

    # Architecture
    encoder_layers: Tuple[int, ...] = (128, 64)
    decoder_layers: Tuple[int, ...] = (64, 128)
    GP_dim: int = 2
    Normal_dim: int = 8
    input_dim: int = 256

    # VAE / regularization
    noise: float = 0.1
    dropoutE: float = 0.1
    dropoutD: float = 0.0
    dynamicVAE: bool = True
    init_beta: float = 10.0
    min_beta: float = 5.0
    max_beta: float = 25.0
    KL_loss: float = 0.025
    num_samples: int = 1

    # GP / inducing points
    fix_inducing_points: bool = True
    grid_inducing_points: bool = True
    inducing_point_steps: int = 6
    fixed_gp_params: bool = False
    loc_range: float = 60.0     # scales x & y
    z_range: float = 20.0       # scales z for data only
    kernel_scale: float = 60.0
    allow_batch_kernel_scale: bool = False

    # Dispersion
    shared_dispersion: bool = False

    # Persistence / device
    model_file: str = "model.pt"
    device: str = "cuda"
    verbosity: int = 1

    # Counterfactual
    pert_cells: str = "./pert_cells/cells.txt"  # 1-based indices within selected batch
    pert_batch: str = "Sham1"
    target_cell_perturbation: str = "ICA"

    # Outputs
    final_latent_file: str = "final_latent.txt"
    denoised_counts_file: str = "denoised_counts.txt"
    num_denoise_samples: int = 10000

    # Inducing grid behavior
    inducing_z_uses_loc_range: bool = True  # <<< important default to mirror old runner

    # W&B
    wandb_mode: Optional[str] = None      # "online" | "offline" | None
    wandb_project: Optional[str] = None
    wandb_run: Optional[str] = None

    # Config path
    config: Optional[str] = None


# -----------------------------------------------------------------------------
# Data loading & pre-processing helpers
# -----------------------------------------------------------------------------
def load_h5_dataset(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    X        : (N, G) float32
    pos3d    : (N, 3) float32 (PT + sham concatenated)
    batch    : (N,) str
    labels   : (N,) str (Celltype_coarse)
    barcodes : (N,) str
    """
    with h5py.File(path, "r") as f:
        X = np.array(f["X"])
        if X.shape[0] < X.shape[1]:  # (G,N) → (N,G)
            X = X.T
        X = X.astype("float32")

        loc_PT = np.array(f["3D_pos_PT"]).T.astype("float32")
        loc_sham = np.array(f["3D_pos_sham"]).T.astype("float32")
        pos3d = np.concatenate([loc_PT, loc_sham], axis=0)

        # tiny jitter on z to avoid exact duplicates
        if pos3d.shape[1] >= 3:
            pos3d[:, 2] = pos3d[:, 2] + np.random.normal(loc=0.0, scale=1e-3, size=pos3d.shape[0]).astype("float32")

        batch = np.array(f["Batch"]).astype(str)
        labels = np.array(f["Celltype_coarse"]).astype(str)
        barcodes = np.array(f["Barcode"]).astype(str)

    return X, pos3d, batch, labels, barcodes


def make_batch_onehot(batch_str: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Binary batch: PT-prefixed → 1, else 0, then one-hot with 2 classes."""
    numeric = np.array([1 if s.startswith("PT") else 0 for s in batch_str], dtype=int)
    return np.eye(2, dtype=np.float32)[numeric], numeric


def make_perturbation_from_labels(labels_str: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
    """Perturbation from Celltype_coarse: ICA → 1; others → 0."""
    numeric = np.array([1 if s == "ICA" else 0 for s in labels_str], dtype=int)
    mapping = {labels_str[i]: int(numeric[i]) for i in range(len(labels_str))}
    return numeric[:, None], mapping


def per_batch_scale_positions_3d(pos3d: np.ndarray, batch_onehot: np.ndarray, loc_range: float, z_range: float) -> np.ndarray:
    """Independently min-max scale positions within each batch to [0, loc_range] for x,y and [0, z_range] for z."""
    n_batch = batch_onehot.shape[1]
    out = np.zeros_like(pos3d, dtype=np.float32)
    for i in range(n_batch):
        mask = batch_onehot[:, i] == 1.0
        if not np.any(mask):
            continue
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(pos3d[mask, :3])
        scaled[:, :2] *= float(loc_range)
        scaled[:, 2] *= float(z_range)
        out[mask, :3] = scaled
    return out.astype(np.float32)


def build_inducing_grid3d_tiled(
    n_batch: int,
    steps: int,
    loc_range: float,
    z_range: float,
    inducing_z_uses_loc_range: bool = True,
) -> np.ndarray:
    """
    Build a (steps x steps x steps) 3D grid:
      x,y ∈ [0, loc_range], z ∈ [0, {loc_range | z_range}],
    then tile across batches and append an (M_total x n_batch) one-hot.
    """
    eps = 1e-5
    x, y, z = np.mgrid[
        0:(1 + eps):(1.0 / steps),
        0:(1 + eps):(1.0 / steps),
        0:(1 + eps):(1.0 / steps),
    ]
    grid = np.vstack((x.ravel(), y.ravel(), z.ravel())).T.astype(np.float32)
    grid[:, :2] *= float(loc_range)
    if inducing_z_uses_loc_range:
        # IMPORTANT: mirror the old runner behavior (z scaled by loc_range)
        grid[:, 2] *= float(loc_range)
    else:
        grid[:, 2] *= float(z_range)

    tiled_xyz = np.tile(grid, (n_batch, 1))
    onehots = []
    for i in range(n_batch):
        block = np.zeros((grid.shape[0], n_batch), dtype=np.float32)
        block[:, i] = 1.0
        onehots.append(block)
    onehots = np.concatenate(onehots, axis=0)
    inducing = np.concatenate([tiled_xyz, onehots], axis=1).astype(np.float32)
    return inducing


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------
def run(cfg: RunConfigStroke3D) -> None:
    logging.info("Config: %s", asdict(cfg))
    outdir = ensure_dir(cfg.outdir)

    # Load data
    X, pos3d, batch_str, labels_str, barcodes = load_h5_dataset(cfg.data_file)

    # Batch one-hot + perturbation from labels
    batch_oh, batch_vec = make_batch_onehot(batch_str)
    cell_atts, pert_map = make_perturbation_from_labels(labels_str)  # (N,1)

    # Report composition
    vals_b, cnt_b = np.unique(batch_str, return_counts=True)
    vals_l, cnt_l = np.unique(labels_str, return_counts=True)
    logging.info("Batch composition: %s", dict(zip(vals_b.tolist(), cnt_b.tolist())))
    logging.info("Label composition: %s", dict(zip(vals_l.tolist(), cnt_l.tolist())))

    # Batch size
    bs = auto_batch_size(X.shape[0]) if cfg.batch_size == "auto" else int(cfg.batch_size)
    logging.info("Batch size: %s", bs)

    # Optional gene selection
    if cfg.select_genes and cfg.select_genes > 0:
        logging.info("Selecting top %d genes...", cfg.select_genes)
        important = geneSelection(X, n=cfg.select_genes, plot=False)
        X = X[:, important]
        np.savetxt(outdir / "selected_genes.txt", important, fmt="%d")

    # Per-batch position scaling (data): x,y by loc_range; z by z_range
    pos3d_scaled = per_batch_scale_positions_3d(pos3d, batch_oh, cfg.loc_range, cfg.z_range)
    # Append batch one-hot → (N, 3 + n_batch)
    pos3d_batched = np.concatenate([pos3d_scaled, batch_oh], axis=1).astype(np.float32)

    # Inducing points (3D grid tiled across batches)
    n_batch = batch_oh.shape[1]
    if cfg.grid_inducing_points:
        inducing = build_inducing_grid3d_tiled(
            n_batch=n_batch,
            steps=cfg.inducing_point_steps,
            loc_range=cfg.loc_range,
            z_range=cfg.z_range,
            inducing_z_uses_loc_range=cfg.inducing_z_uses_loc_range,  # <<< key change
        )
        logging.info("Inducing points (3D grid tiled): %s  (z uses %s)",
                     inducing.shape,
                     "loc_range" if cfg.inducing_z_uses_loc_range else "z_range")
    else:
        raise NotImplementedError("For 3D stroke, please use grid_inducing_points=true.")

    # AnnData + normalization
    adata = sc.AnnData(X, dtype="float32")
    adata = normalize(adata, size_factors=True, normalize_input=True, logtrans_input=True)

    # W&B (optional)
    wb_run = None
    if _WANDB_AVAILABLE and cfg.wandb_mode:
        wb_kwargs = dict(
            project=cfg.wandb_project or "concert-3d",
            name=cfg.wandb_run,
            mode=cfg.wandb_mode,  # "online" or "offline"
            config=asdict(cfg),
        )
        wb_run = wandb.init(**{k: v for k, v in wb_kwargs.items() if v is not None})

    # Model
    model = CONCERT(
        cell_atts=cell_atts,
        num_genes=adata.n_vars,
        input_dim=cfg.input_dim,
        GP_dim=cfg.GP_dim,
        Normal_dim=cfg.Normal_dim,
        n_batch=n_batch,
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
        allow_batch_kernel_scale=cfg.allow_batch_kernel_scale,
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

    # Train or eval
    if cfg.stage.lower() == "train":
        logging.info("Training...")
        model.train_model(
            pos=pos3d_batched,
            ncounts=adata.X,
            raw_counts=adata.raw.X,
            size_factors=adata.obs.size_factors,
            batch=batch_oh,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            batch_size=bs,
            num_samples=cfg.num_samples,
            train_size=cfg.train_size,
            maxiter=cfg.maxiter,
            patience=cfg.patience,
            save_model=True,
            model_weights=str(outdir / cfg.model_file),
            wandb_run=wb_run,
        )
        logging.info("Training finished.")
        if wb_run is not None:
            wb_run.finish()
        return

    # ---- EVAL / COUNTERFACTUAL ----
    model_path = (outdir / cfg.model_file) if os.path.isfile(outdir / cfg.model_file) else Path(cfg.model_file)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found for eval: {model_path}")
    model.load_model(str(model_path))
    logging.info("Loaded weights: %s", model_path)

    # Load 1-based indices within the selected batch; map to absolute indices
    batch_mask = (batch_str == cfg.pert_batch)
    batch_indices = np.where(batch_mask)[0]
    if batch_indices.size == 0:
        raise ValueError(f"No cells found for batch '{cfg.pert_batch}'.")
    file_inds = np.loadtxt(cfg.pert_cells, dtype=int) - 1  # 0-based within that batch
    if file_inds.ndim == 0:
        file_inds = np.array([int(file_inds)], dtype=int)
    if np.any((file_inds < 0) | (file_inds >= batch_indices.size)):
        raise IndexError("pert_cells indices are out of range for the selected batch.")
    pert_ind = batch_indices[file_inds]
    np.savetxt(outdir / f"stroke_3d_{cfg.pert_batch}_pert_ind.txt", pert_ind, fmt="%i")

    # Counterfactual prediction
    target_code = 1 if cfg.target_cell_perturbation == "ICA" else 0
    logging.info("Counterfactual → perturbation=%s (code=%d)", cfg.target_cell_perturbation, target_code)
    perturbed_counts, perturbed_atts = model.counterfactualPrediction(
        X=pos3d_batched,
        sample_index=torch.arange(X.shape[0], dtype=torch.int),
        cell_atts=cell_atts,
        batch_size=bs,
        n_samples=25,
        perturb_cell_id=pert_ind,
        target_cell_perturbation=target_code,
    )

    # Save only Sham batches (to mirror original behavior)
    ad_out = sc.AnnData(perturbed_counts, obs=pd.DataFrame(batch_str, columns=["batch"]))
    ad_out = ad_out[ad_out.obs["batch"].str.startswith("Sham"), :]
    out_path = (
        Path(cfg.outdir)
        / f"res_stroke3d_perturb_{cfg.project_index}_counts.h5ad"
    )
    ad_out.write(out_path)
    logging.info("Wrote counterfactual counts to %s", out_path)

    if wb_run is not None:
        wb_run.finish()


# -----------------------------------------------------------------------------
# CLI (config-first; CLI only overrides what you pass)
# -----------------------------------------------------------------------------
def parse_args() -> RunConfigStroke3D:
    p = argparse.ArgumentParser(
        description="CONCERT 3D Stroke runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", help="YAML/JSON config file path.")

    # IO / basic
    p.add_argument("--data_file")
    p.add_argument("--outdir")
    p.add_argument("--project_index")

    # selection / training
    p.add_argument("--select_genes", type=int)
    p.add_argument("--stage", choices=["train", "eval"])
    p.add_argument("--batch_size")
    p.add_argument("--maxiter", type=int)
    p.add_argument("--train_size", type=float)
    p.add_argument("--patience", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--weight_decay", type=float)

    # arch / vae
    p.add_argument("--encoder_layers", nargs="+", type=int)
    p.add_argument("--decoder_layers", nargs="+", type=int)
    p.add_argument("--GP_dim", type=int)
    p.add_argument("--Normal_dim", type=int)
    p.add_argument("--input_dim", type=int)
    p.add_argument("--noise", type=float)
    p.add_argument("--dropoutE", type=float)
    p.add_argument("--dropoutD", type=float)

    def str2bool(v):
        if v is None:
            return None
        if isinstance(v, bool):
            return v
        return v.lower() in ("1", "true", "yes", "y", "on")

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
    p.add_argument("--fixed_gp_params", type=str2bool)
    p.add_argument("--loc_range", type=float)
    p.add_argument("--z_range", type=float)
    p.add_argument("--kernel_scale", type=float)
    p.add_argument("--allow_batch_kernel_scale", type=str2bool)
    p.add_argument("--shared_dispersion", type=str2bool)
    p.add_argument("--inducing_z_uses_loc_range", type=str2bool)

    # persistence / device
    p.add_argument("--model_file")
    p.add_argument("--device")
    p.add_argument("--verbosity", type=int)

    # counterfactual
    p.add_argument("--pert_cells")
    p.add_argument("--pert_batch")
    p.add_argument("--target_cell_perturbation")

    # W&B (explicit flags; no ambiguous --wandb)
    p.add_argument("--wandb_mode", choices=["online", "offline"])
    p.add_argument("--wandb_project")
    p.add_argument("--wandb_run")

    a = p.parse_args()

    # 1) defaults
    cfg = RunConfigStroke3D()

    # 2) merge config file
    file_cfg = load_config_file(getattr(a, "config", None)) if getattr(a, "config", None) else {}
    if file_cfg:
        for k in ("encoder_layers", "decoder_layers", "wandb_tags"):
            if k in file_cfg and isinstance(file_cfg[k], list):
                file_cfg[k] = tuple(file_cfg[k]) if k != "wandb_tags" else file_cfg[k]
        cfg = replace(cfg, **{k: v for k, v in file_cfg.items() if hasattr(cfg, k)})

    # 3) apply ONLY explicit CLI overrides (don’t clobber YAML with argparse defaults)
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