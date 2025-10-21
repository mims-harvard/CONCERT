#!/usr/bin/env python3

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
from sklearn.preprocessing import MinMaxScaler

# Optional YAML
try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

# Optional wandb
try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover
    wandb = None

from concert_batch_2D_stroke import CONCERT
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


def str2bool(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("1", "true", "yes", "y", "on")


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


def sample_fixed_spots(spots: np.ndarray, center: np.ndarray, num_spots: int) -> Tuple[np.ndarray, np.ndarray]:
    """Select a fixed number of closest spots to `center`."""
    d = np.sqrt(((spots - center[None, :]) ** 2).sum(axis=1))
    order = np.argsort(d)[:num_spots]
    return spots[order], order


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
@dataclass
class RunConfig:
    # IO
    data_file: str = "data.h5"
    outdir: str = "./outputs"
    stage: str = "train"       # {"train","eval"}

    # Selection / normalization
    select_genes: int = 0

    # Training
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

    # GP / inducing
    fix_inducing_points: bool = True
    grid_inducing_points: bool = True
    inducing_point_steps: Optional[int] = 6
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
    index: str = "patch"                            # {"random","patch"}
    pert_cells: str = "./pert_cells/cells.txt"      # only used for "patch" (1-based indices)
    pert_batch: str = "Sham1"                       # batch name to subset for export
    target_cell_perturbation: str = "ICA"           # map -> {ICA:1, else:0}
    pert_cell_number: int = 20

    # Logging
    wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_run: Optional[str] = None

    # Config file
    config: Optional[str] = None


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
def load_h5_dataset(path: str):
    """
    Returns
    -------
    X : (N, G) float32   (accepts (G,N) too and transposes to (N,G))
    pos : (N, 2) float32
    batch_str : (N,) str
    labels_str : (N,) str   (Celltype_coarse)
    barcodes   : (N,) str
    """
    with h5py.File(path, "r") as f:
        X = np.array(f["X"])
        if X.shape[0] < X.shape[1]:  # tolerate G x N files (old variant was N x G)
            X = X.T
        X = X.astype("float32")

        pos = np.array(f["Pos"]).T.astype("float32")
        pos = pos[:, :2]  # keep first two dims

        batch_str = np.array(f["Batch"]).astype(str)
        labels_str = np.array(f["Celltype_coarse"]).astype(str)
        barcodes = np.array(f["Barcode"]).astype(str)
    return X, pos, batch_str, labels_str, barcodes


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------
def run(cfg: RunConfig) -> None:
    logging.info("Config: %s", asdict(cfg))
    outdir = ensure_dir(cfg.outdir)

    # W&B (optional)
    if cfg.wandb and wandb is not None:
        wandb.init(project=cfg.wandb_project or "CONCERT-stroke2d",
                   name=cfg.wandb_run or cfg.stage,
                   config=asdict(cfg))
        wandb.define_metric("epoch")
        for m in ["elbo", "nb_loss", "gp_kld", "gauss_kld", "beta"]:
            wandb.define_metric(m, step_metric="epoch")

    # Load dataset
    X, pos, batch_str, labels_str, barcodes = load_h5_dataset(cfg.data_file)

    # Batch one-hot: PT* → 1 else 0, two classes
    batch_vec = np.array([1 if s.startswith("PT") else 0 for s in batch_str], dtype=int)
    batch_oh = np.eye(2, dtype=np.float32)[batch_vec]  # (N,2)

    # Perturbation: ICA→1 else 0
    pert_vec = np.array([1 if s == "ICA" else 0 for s in labels_str], dtype=int)
    pert_map = {labels_str[i]: int(pert_vec[i]) for i in range(len(labels_str))}
    cell_atts = pert_vec[:, None].astype(int)

    logging.info("Batch composition: %s", dict(zip(*np.unique(batch_str, return_counts=True))))
    logging.info("Label composition: %s", dict(zip(*np.unique(labels_str, return_counts=True))))

    # Batch size
    bs = auto_batch_size(X.shape[0]) if cfg.batch_size == "auto" else int(cfg.batch_size)
    logging.info("Batch size: %s", bs)

    # Optional gene selection
    if cfg.select_genes and cfg.select_genes > 0:
        logging.info("Selecting top %d genes...", cfg.select_genes)
        important = geneSelection(X, n=cfg.select_genes, plot=False)
        X = X[:, important]
        np.savetxt(outdir / "selected_genes.txt", important, fmt="%d")

    # Per-batch scale positions to [0, loc_range], then append one-hot
    n_batch = batch_oh.shape[1]
    pos_scaled = np.zeros_like(pos, dtype=np.float32)
    for i in range(n_batch):
        mask = (batch_oh[:, i] == 1.0)
        scaler = MinMaxScaler()
        pos_scaled[mask, :] = scaler.fit_transform(pos[mask, :]) * float(cfg.loc_range)

    pos_batched = np.concatenate([pos_scaled, batch_oh], axis=1).astype(np.float32)

    # Inducing points (grid tiled across batches)
    steps = int(cfg.inducing_point_steps or 6)
    eps = 1e-5
    grid_xy = np.mgrid[0:(1 + eps):(1.0 / steps), 0:(1 + eps):(1.0 / steps)].reshape(2, -1).T
    grid_xy = (grid_xy * float(cfg.loc_range)).astype(np.float32)  # (M,2)
    tiled_xy = np.tile(grid_xy, (n_batch, 1))                      # (M*n_batch, 2)
    oh_blocks = []
    for i in range(n_batch):
        blk = np.zeros((grid_xy.shape[0], n_batch), dtype=np.float32)
        blk[:, i] = 1.0
        oh_blocks.append(blk)
    inducing = np.concatenate([tiled_xy, np.concatenate(oh_blocks, axis=0)], axis=1).astype(np.float32)

    # AnnData + normalization
    adata = sc.AnnData(X, dtype="float32")
    adata = normalize(adata, size_factors=True, normalize_input=True, logtrans_input=True)
    sample_indices = torch.arange(adata.n_obs, dtype=torch.int)
    
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
    model_path = Path(cfg.outdir) / cfg.model_file
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
            model_weights=str(model_path),
        )
        logging.info("Training finished.")
    else:
        # eval (counterfactual)
        load_from = model_path if model_path.exists() else Path(cfg.model_file)
        if not load_from.exists():
            raise FileNotFoundError(f"Model weights not found: {load_from}")
        model.load_model(str(load_from))
        logging.info("Loaded model: %s", load_from)

    # Always produce denoised counts (as in old script)
    logging.info("Denoising...")
    denoised = model.batching_denoise_counts(
        X=pos_batched,
        sample_index=sample_indices,
        cell_atts=cell_atts,
        batch_size=bs,
        n_samples=25,
    )
    sc.AnnData(denoised).write(Path(cfg.outdir) / "res_stroke2d_denoised_counts.h5ad")
    logging.info("Wrote denoised counts.")

    # ---- Counterfactual selection ----
    n = int(cfg.pert_cell_number)
    if cfg.index == "random":
        pool = np.where(batch_str == cfg.pert_batch)[0]
        # the original code further filtered by y == "CTX"; keep parity
        labels = np.array(labels_str)
        pool = pool[np.where(labels[pool] == "CTX")[0]]
        if pool.size < n:
            raise ValueError(f"Not enough cells in batch '{cfg.pert_batch}' (CTX) to sample {n}.")
        pert_ind = np.random.choice(pool, n, replace=False)
        np.savetxt(Path(cfg.outdir) / f"stroke_pert_ind_random_{n}.txt", pert_ind, fmt="%i")
        out_index_tag = f"random_{n}"
    elif cfg.index == "patch":
        base_inds = (np.loadtxt(cfg.pert_cells, dtype=int) - 1).astype(int)
        center_ind = int(base_inds[min(100, len(base_inds) - 1)])
        # realign to selected batch start as in old code
        left_anchor = np.where(batch_str == cfg.pert_batch)[0][0] - 1
        base_inds = base_inds + left_anchor
        # patch within that batch
        loc_batch = pos[batch_str == cfg.pert_batch, :]
        batch_ids = np.where(batch_str == cfg.pert_batch)[0]
        center = loc_batch[center_ind % len(loc_batch)]
        _, patch_idx = sample_fixed_spots(loc_batch, center, n)
        pert_ind = batch_ids[patch_idx]
        np.savetxt(Path(cfg.outdir) / f"stroke_pert_ind_patch_{n}.txt", pert_ind, fmt="%i")
        out_index_tag = f"patch_{n}"
    else:
        raise ValueError(f"Unknown index mode '{cfg.index}'. Use 'random' or 'patch'.")

    # ---- Counterfactual prediction ----
    logging.info("Counterfactual → perturbation=%s", cfg.target_cell_perturbation)
    target_code = int(pert_map.get(cfg.target_cell_perturbation, 0))
    cf_counts, cf_atts = model.counterfactualPrediction(
        X=pos_batched,
        sample_index=sample_indices,
        cell_atts=cell_atts,
        batch_size=bs,
        n_samples=25,
        perturb_cell_id=torch.tensor(pert_ind, dtype=torch.long),
        target_cell_perturbation=target_code,
    )

    # Subset to chosen batch for export (keep old behavior)
    mask_batch = (batch_str == cfg.pert_batch)
    cf_counts = cf_counts[mask_batch, :]
    cf_atts = cf_atts[mask_batch, :]
    ad_out = sc.AnnData(cf_counts, obs=pd.DataFrame(cf_atts, columns=["perturbation"]))
    out_path = Path(cfg.outdir) / f"res_stroke2d_perturb_{cfg.pert_batch}_{out_index_tag}_counts.h5ad"
    ad_out.write(out_path)
    logging.info("Wrote counterfactual counts to %s", out_path)

    if cfg.wandb and wandb is not None:
        wandb.finish()


# -----------------------------------------------------------------------------
# CLI (config-first; CLI only overrides what you pass)
# -----------------------------------------------------------------------------
def parse_args() -> RunConfig:
    p = argparse.ArgumentParser(
        description="CONCERT 2D Stroke runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", help="YAML/JSON config file path.")

    # IO / stage
    p.add_argument("--data_file")
    p.add_argument("--outdir")
    p.add_argument("--stage", choices=["train", "eval"])

    # selection / training
    p.add_argument("--select_genes", type=int)
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

    # dispersion / runtime
    p.add_argument("--shared_dispersion", type=str2bool)
    p.add_argument("--model_file")
    p.add_argument("--device")
    p.add_argument("--verbosity", type=int)

    # counterfactual
    p.add_argument("--index", choices=["random", "patch"])
    p.add_argument("--pert_cells")
    p.add_argument("--pert_batch")
    p.add_argument("--target_cell_perturbation")
    p.add_argument("--pert_cell_number", type=int)

    # wandb
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project")
    p.add_argument("--wandb_run")

    a = p.parse_args()

    cfg = RunConfig()
    # merge config file
    file_cfg = load_config_file(getattr(a, "config", None)) if getattr(a, "config", None) else {}
    if file_cfg:
        for k in ("encoder_layers", "decoder_layers"):
            if k in file_cfg and isinstance(file_cfg[k], list):
                file_cfg[k] = tuple(file_cfg[k])
        cfg = replace(cfg, **{k: v for k, v in file_cfg.items() if hasattr(cfg, k)})

    # apply ONLY explicit CLI overrides
    import sys as _sys
    specified = set()
    for action in p._actions:
        if not action.option_strings:
            continue
        if any(opt in _sys.argv for opt in action.option_strings):
            specified.add(action.dest)
    cli = {k: getattr(a, k) for k in specified if hasattr(cfg, k) and getattr(a, k) is not None}

    # batch_size post-process
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