#!/usr/bin/env python3
"""
CONCERT (2D Stroke) driver — spatial counterfactual perturbation.

Features
--------
- YAML/JSON config file support (CLI flags only override what you pass)
- Structured logging, deterministic preprocessing
- Train or evaluate (counterfactual prediction)
- Saves outputs to .h5ad

HDF5 expectations (matching original script):
  X                 : (N x G) or (G x N); this script accepts both and converts to (N x G)
  Pos               : (2 x N) or (N x 2) → (N x 2)
  Batch             : (N,) str — batch names (e.g., "PT...")
  Celltype_coarse   : (N,) str — used both as labels and to derive perturbation (ICA vs not)
  Barcode           : (N,) str — unused here, but read for parity with original
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass, asdict, replace
from pathlib import Path
from typing import Optional, Tuple, Dict

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

from concert_batch_2D_stroke import CONCERT
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
class RunConfigStroke:
    # IO
    data_file: str = "data.h5"
    outdir: str = "./outputs"
    project_index: str = "x"  # for filenames

    # Selection / normalization
    select_genes: int = 0

    # Training
    stage: str = "eval"  # {"train","eval"}; eval will run counterfactual after loading weights
    batch_size: str = "auto"
    maxiter: int = 5000
    train_size: float = 0.95
    patience: int = 200
    lr: float = 1e-4
    weight_decay: float = 1e-6

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

    # GP / inducing points (batched by binary PT / non-PT)
    fix_inducing_points: bool = True
    grid_inducing_points: bool = True
    inducing_point_steps: int = 6
    inducing_point_nums: Optional[int] = None
    fixed_gp_params: bool = False
    loc_range: float = 20.0
    kernel_scale: float = 20.0
    allow_batch_kernel_scale: bool = True

    # Dispersion
    shared_dispersion: bool = False

    # Runtime / persistence
    model_file: str = "model.pt"
    device: str = "cuda"
    verbosity: int = 1

    # Counterfactual selection & targeting
    index: str = "patch"                        # {"random","patch"}; matches original behavior
    pert_cells: str = "./pert_cells/cells.txt"  # used when index == "patch" (1-based indices)
    pert_batch: str = "Sham1"                   # batch name to subset at the end
    target_cell_perturbation: str = "ICA"       # map -> {ICA:1, else:0}
    pert_cell_number: int = 20                  # number of cells to perturb/select

    # Config path
    config: Optional[str] = None


# -----------------------------------------------------------------------------
# Data loading & pre-processing helpers
# -----------------------------------------------------------------------------
def load_h5_dataset(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    X : (N, G) float32
    pos : (N, 2) float32
    batch_str : (N,) str
    labels_str : (N,) str   (Celltype_coarse)
    barcodes   : (N,) str
    """
    with h5py.File(path, "r") as f:
        X = np.array(f["X"])
        # accept both (G, N) and (N, G)
        if X.shape[0] < X.shape[1]:  # original file might be G x N
            X = X.T
        X = X.astype("float32")

        pos = np.array(f["Pos"]).T.astype("float32")
        pos = pos[:, :2]  # keep first two dims

        batch_str = np.array(f["Batch"]).astype(str)
        labels_str = np.array(f["Celltype_coarse"]).astype(str)
        barcodes = np.array(f["Barcode"]).astype(str)
    return X, pos, batch_str, labels_str, barcodes


def build_batch_onehot(batch_str: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Binary batch: PT-prefixed -> 1, else 0, then one-hot to 2 classes.
    Returns (onehot, numeric_vector).
    """
    numeric = np.array([1 if s.startswith("PT") else 0 for s in batch_str], dtype=int)
    onehot = np.eye(2, dtype=np.float32)[numeric]
    return onehot, numeric


def build_perturbation_from_labels(labels_str: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Derive perturbation from Celltype_coarse: ICA → 1; others → 0.
    Returns (vector (N,1) int, mapping dict for names encountered).
    """
    numeric = np.array([1 if s == "ICA" else 0 for s in labels_str], dtype=int)
    mapping = {labels_str[i]: int(numeric[i]) for i in range(len(labels_str))}
    return numeric[:, None], mapping


def per_batch_scale_positions(pos: np.ndarray, batch_onehot: np.ndarray, loc_range: float) -> np.ndarray:
    """Independently min-max scale positions within each batch to [0, loc_range]."""
    n_batch = batch_onehot.shape[1]
    out = np.zeros_like(pos, dtype=np.float32)
    for i in range(n_batch):
        mask = batch_onehot[:, i] == 1.0
        scaler = MinMaxScaler()
        out[mask, :] = scaler.fit_transform(pos[mask, :]) * float(loc_range)
    return out.astype(np.float32)


def build_inducing_grid_tiled(n_batch: int, steps: int, loc_range: float) -> np.ndarray:
    """
    Build a (steps x steps) grid in [0, loc_range] and tile across batches,
    appending an (M_total x n_batch) one-hot to indicate batch ownership.
    """
    eps = 1e-5
    grid_xy = np.mgrid[0:(1 + eps):(1.0 / steps), 0:(1 + eps):(1.0 / steps)].reshape(2, -1).T
    grid_xy = (grid_xy * float(loc_range)).astype(np.float32)
    tiled_xy = np.tile(grid_xy, (n_batch, 1))
    oh = []
    for i in range(n_batch):
        block = np.zeros((grid_xy.shape[0], n_batch), dtype=np.float32)
        block[:, i] = 1.0
        oh.append(block)
    onehots = np.concatenate(oh, axis=0)
    inducing = np.concatenate([tiled_xy, onehots], axis=1).astype(np.float32)
    return inducing


def sample_fixed_spots(spots: np.ndarray, center: np.ndarray, num_spots: int) -> Tuple[np.ndarray, np.ndarray]:
    """Select a fixed number of closest spots to `center`."""
    d = np.sqrt(((spots - center[None, :]) ** 2).sum(axis=1))
    order = np.argsort(d)[:num_spots]
    return spots[order], order


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------
def run(cfg: RunConfigStroke) -> None:
    logging.info("Config: %s", asdict(cfg))
    outdir = ensure_dir(cfg.outdir)

    # Load data
    X, pos, batch_str, labels_str, barcodes = load_h5_dataset(cfg.data_file)

    # Batch one-hot + perturbation from labels
    batch_oh, batch_vec = build_batch_onehot(batch_str)
    cell_atts, pert_map = build_perturbation_from_labels(labels_str)  # (N,1)

    # Report basic composition
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

    # Per-batch position scaling; append batch one-hot
    pos_scaled = per_batch_scale_positions(pos, batch_oh, cfg.loc_range)
    pos_batched = np.concatenate([pos_scaled, batch_oh], axis=1).astype(np.float32)

    # Inducing points
    n_batch = batch_oh.shape[1]
    if cfg.grid_inducing_points:
        inducing = build_inducing_grid_tiled(n_batch, cfg.inducing_point_steps, cfg.loc_range)
        logging.info("Inducing points (grid tiled): %s", inducing.shape)
    else:
        assert cfg.inducing_point_nums and cfg.inducing_point_nums > 0, \
            "inducing_point_nums must be set when grid_inducing_points=False"
        km = KMeans(n_clusters=cfg.inducing_point_nums, n_init=100).fit(pos_scaled)
        centers = km.cluster_centers_.astype(np.float32)
        oh = np.zeros((centers.shape[0], n_batch), dtype=np.float32)
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
            pos=pos_batched,
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
        )
        logging.info("Training finished.")
        return

    # ---- EVAL / COUNTERFACTUAL ----
    # Weights
    model_path = (outdir / cfg.model_file) if os.path.isfile(outdir / cfg.model_file) else Path(cfg.model_file)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found for eval: {model_path}")
    model.load_model(str(model_path))
    logging.info("Loaded weights: %s", model_path)

    # Select perturbed cell indices
    if cfg.index == "random":
        # choose from desired batch + optionally a label filter (as in original CTX); here we keep batch filter only
        pool = np.where(batch_str == cfg.pert_batch)[0]
        if pool.size < cfg.pert_cell_number:
            raise ValueError(f"Not enough cells in batch '{cfg.pert_batch}' to sample {cfg.pert_cell_number}.")
        pert_ind = np.random.choice(pool, cfg.pert_cell_number, replace=False)
        np.savetxt(outdir / f"stroke_pert_ind_random_{cfg.pert_cell_number}.txt", pert_ind, fmt="%i")
        out_index_tag = f"random_{cfg.pert_cell_number}"
    elif cfg.index == "patch":
        # load 1-based indices from file, realign to selected batch, then sample a compact patch around a center
        base_inds = np.loadtxt(cfg.pert_cells, dtype=int) - 1
        center_ind = int(base_inds[min(100, len(base_inds) - 1)])  # safety
        # restrict to chosen batch
        loc_batch = pos[batch_str == cfg.pert_batch, :]
        batch_ids = np.where(batch_str == cfg.pert_batch)[0]
        # take nearest patch within that batch
        center = loc_batch[center_ind % len(loc_batch)]
        _, patch_idx = sample_fixed_spots(loc_batch, center, cfg.pert_cell_number)
        pert_ind = batch_ids[patch_idx]
        np.savetxt(outdir / f"stroke_pert_ind_patch_{cfg.pert_cell_number}.txt", pert_ind, fmt="%i")
        out_index_tag = f"patch_{cfg.pert_cell_number}"
    else:
        raise ValueError(f"Unknown index mode '{cfg.index}'. Use 'random' or 'patch'.")

    # Counterfactual prediction
    logging.info("Counterfactual → perturbation=%s", cfg.target_cell_perturbation)
    target_code = int(pert_map.get(cfg.target_cell_perturbation, 0))
    perturbed_counts, perturbed_atts = model.counterfactualPrediction(
        X=pos_batched,
        sample_index=torch.arange(X.shape[0], dtype=torch.int),
        cell_atts=cell_atts,
        batch_size=bs,
        n_samples=25,
        perturb_cell_id=pert_ind,
        target_cell_perturbation=target_code,
    )

    # Subset outputs to the chosen batch (match original behavior)
    mask_batch = (batch_str == cfg.pert_batch)
    perturbed_counts = perturbed_counts[mask_batch, :]
    perturbed_atts = perturbed_atts[mask_batch, :]

    # Save
    obs_df = pd.DataFrame(perturbed_atts, columns=["perturbation"])
    ad_out = sc.AnnData(perturbed_counts, obs=obs_df)
    out_path = Path(cfg.outdir) / f"res_stroke2d_perturb_{cfg.pert_batch}_{out_index_tag}_counts.h5ad"
    ad_out.write(out_path)
    logging.info("Wrote counterfactual counts to %s", out_path)


# -----------------------------------------------------------------------------
# CLI (config-first; CLI only overrides what you pass)
# -----------------------------------------------------------------------------
def parse_args() -> RunConfigStroke:
    p = argparse.ArgumentParser(
        description="CONCERT 2D Stroke runner",
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
    p.add_argument("--inducing_point_nums", type=int)
    p.add_argument("--fixed_gp_params", type=str2bool)
    p.add_argument("--loc_range", type=float)
    p.add_argument("--kernel_scale", type=float)
    p.add_argument("--allow_batch_kernel_scale", type=str2bool)
    p.add_argument("--shared_dispersion", type=str2bool)

    # runtime / persistence
    p.add_argument("--model_file")
    p.add_argument("--device")
    p.add_argument("--verbosity", type=int)

    # counterfactual
    p.add_argument("--index")
    p.add_argument("--pert_cells")
    p.add_argument("--pert_batch")
    p.add_argument("--target_cell_perturbation")
    p.add_argument("--pert_cell_number", type=int)

    a = p.parse_args()

    # 1) defaults
    cfg = RunConfigStroke()

    # 2) merge config file
    file_cfg = load_config_file(getattr(a, "config", None)) if getattr(a, "config", None) else {}
    if file_cfg:
        for k in ("encoder_layers", "decoder_layers"):
            if k in file_cfg and isinstance(file_cfg[k], list):
                file_cfg[k] = tuple(file_cfg[k])
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
