#!/usr/bin/env python3
"""
Driver (SK) — Spatial-aware counterfactual perturbation using CONCERT

Highlights
----------
- YAML/JSON config file support (CLI flags only override what you pass)
- Structured logging and deterministic preprocessing
- Auto batch size, clean inducing-point creation (grid or k-means)
- Normalization via `preprocess.normalize`
- Saves counterfactual (perturbed) counts as .h5ad

Notes
-----
- Expects an HDF5 with keys:
  'X' (N x G), 'pos' (2 x N or N x 2; will be transposed to N x 2),
  'tissue' (str per cell), 'perturbation' (str per cell).
- The model class is imported from `concert_sk.CONCERT` and is assumed to
  accept the same arguments you used originally (e.g., input_dim=768).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass, asdict, replace
from pathlib import Path
from typing import Iterable, Optional, Tuple, Dict

import h5py
import numpy as np
import scanpy as sc
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# Optional YAML
try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

from concert_sk import CONCERT
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


def strings_to_index(values: np.ndarray) -> np.ndarray:
    """
    Deterministically map strings to [0..K-1] integer codes (order-stable).
    """
    hashed = np.array([sum(ord(c) for c in s) for s in values], dtype=int)
    uniq = np.unique(hashed)
    remap = {u: i for i, u in enumerate(uniq)}
    return np.array([remap[h] for h in hashed], dtype=int)


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
class RunConfigSK:
    # IO
    data_file: str = "data.h5"
    outdir: str = "./outputs"
    sample: str = "sample"
    project_index: str = "x"

    # Selection / normalization
    select_genes: int = 0

    # Training
    batch_size: str = "auto"
    maxiter: int = 5000
    train_size: float = 0.95
    patience: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-6

    # Architecture (as in original concert_sk)
    encoder_layers: Iterable[int] = (128, 64)
    decoder_layers: Iterable[int] = (128,)
    GP_dim: int = 2
    Normal_dim: int = 8
    input_dim: int = 768  # the sk variant uses a fixed input_dim in your original script

    # VAE / regularization
    noise: float = 0.0
    dropoutE: float = 0.0
    dropoutD: float = 0.0
    dynamicVAE: bool = True
    init_beta: float = 10.0
    min_beta: float = 4.0
    max_beta: float = 25.0
    KL_loss: float = 0.025
    num_samples: int = 1

    # GP / inducing points
    fix_inducing_points: bool = True
    grid_inducing_points: bool = True
    inducing_point_steps: int = 6        # grid steps per axis
    inducing_point_nums: int | None = None  # k-means cluster count (if grid_inducing_points=False)
    fixed_gp_params: bool = False
    loc_range: float = 20.0
    kernel_scale: float = 20.0

    # Runtime / persistence
    model_file: str = "model.pt"
    device: str = "cuda"
    verbosity: int = 1

    # Counterfactual
    pert_cells: str = "patch_jak2.txt"           # 1-based indices of cells to perturb
    target_cell_tissue: str = "tumor"
    target_cell_perturbation: str = "Jak2"

    # Config file (YAML/JSON)
    config: Optional[str] = None


# -----------------------------------------------------------------------------
# Data loading & attributes
# -----------------------------------------------------------------------------
def load_h5_dataset(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as f:
        X = np.array(f["X"], dtype=np.float32)
        pos = np.array(f["pos"], dtype=np.float32).T  # (N, 2)
        tissue_raw = np.array(f["tissue"], dtype=str)
        perturb_raw = np.array(f["perturbation"], dtype=str)
    return X, pos, tissue_raw, perturb_raw


def build_attributes(tissue_raw: np.ndarray, perturb_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, int], Dict[str, int]]:
    """
    Map raw strings to integer codes; return codes and name→code dicts.
    """
    tissue_idx = strings_to_index(tissue_raw)
    perturb_idx = strings_to_index(perturb_raw)
    tissue_dict = {name: int(code) for name, code in zip(tissue_raw, tissue_idx)}
    pert_dict = {name: int(code) for name, code in zip(perturb_raw, perturb_idx)}
    return tissue_idx, perturb_idx, tissue_dict, pert_dict


# -----------------------------------------------------------------------------
# Inducing points
# -----------------------------------------------------------------------------
def make_inducing_points(pos: np.ndarray, cfg: RunConfigSK) -> np.ndarray:
    if cfg.grid_inducing_points:
        steps = int(cfg.inducing_point_steps)
        grid_xy = np.mgrid[0:1:complex(steps), 0:1:complex(steps)].reshape(2, -1).T
        ip = (grid_xy * float(cfg.loc_range)).astype(np.float32)
        logging.info("Inducing points (grid): steps=%d -> %d points", steps, ip.shape[0])
        return ip
    # K-means
    assert cfg.inducing_point_nums and cfg.inducing_point_nums > 0, \
        "inducing_point_nums must be set when grid_inducing_points=False"
    km = KMeans(n_clusters=cfg.inducing_point_nums, n_init=100).fit(pos)
    ip = km.cluster_centers_.astype(np.float32)
    logging.info("Inducing points (k-means): k=%d", cfg.inducing_point_nums)
    return ip


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------
def run(cfg: RunConfigSK) -> None:
    logging.info("Config: %s", asdict(cfg))
    outdir = ensure_dir(cfg.outdir)

    # Load dataset
    X, pos, tissue_raw, perturb_raw = load_h5_dataset(cfg.data_file)

    # Attributes
    tissue_idx, perturb_idx, tissue_dict, pert_dict = build_attributes(tissue_raw, perturb_raw)
    cell_atts = np.c_[tissue_idx, perturb_idx].astype(int)
    sample_indices = torch.arange(X.shape[0], dtype=torch.int)

    # Show basic composition
    (vals_t, cnt_t) = np.unique(tissue_raw, return_counts=True)
    (vals_p, cnt_p) = np.unique(perturb_raw, return_counts=True)
    logging.info("Tissue composition: %s", dict(zip(vals_t.tolist(), cnt_t.tolist())))
    logging.info("Perturbation composition: %s", dict(zip(vals_p.tolist(), cnt_p.tolist())))

    # Batch size
    bs = auto_batch_size(X.shape[0]) if cfg.batch_size == "auto" else int(cfg.batch_size)
    logging.info("Batch size: %s", bs)

    # Optional gene selection
    if cfg.select_genes and cfg.select_genes > 0:
        logging.info("Selecting top %d genes...", cfg.select_genes)
        important = geneSelection(X, n=cfg.select_genes, plot=False)
        X = X[:, important]
        np.savetxt(outdir / "selected_genes.txt", important, fmt="%d")

    # Scale spatial coordinates to [0, loc_range]
    scaler = MinMaxScaler()
    pos_scaled = scaler.fit_transform(pos) * cfg.loc_range
    logging.info("Shapes — X: %s, pos: %s", X.shape, pos_scaled.shape)

    # Inducing points in 2D space
    initial_inducing_points = make_inducing_points(pos_scaled, cfg)

    # AnnData + normalization
    adata = sc.AnnData(X, dtype="float32")
    adata = normalize(adata, size_factors=True, normalize_input=True, logtrans_input=True)

    # Model (concert_sk variant)
    model = CONCERT(
        cell_atts=cell_atts,
        num_genes=adata.n_vars,
        input_dim=cfg.input_dim,
        GP_dim=cfg.GP_dim,
        Normal_dim=cfg.Normal_dim,
        encoder_layers=list(cfg.encoder_layers),
        decoder_layers=list(cfg.decoder_layers),
        noise=float(cfg.noise),
        encoder_dropout=float(cfg.dropoutE),
        decoder_dropout=float(cfg.dropoutD),
        fixed_inducing_points=bool(cfg.fix_inducing_points),
        initial_inducing_points=initial_inducing_points,
        fixed_gp_params=bool(cfg.fixed_gp_params),
        kernel_scale=float(cfg.kernel_scale),
        N_train=adata.n_obs,
        KL_loss=float(cfg.KL_loss),
        dynamicVAE=bool(cfg.dynamicVAE),
        init_beta=float(cfg.init_beta),
        min_beta=float(cfg.min_beta),
        max_beta=float(cfg.max_beta),
        dtype=torch.float32,
        device=cfg.device,
    )
    logging.info("Model initialized.")

    # Train or load
    if not os.path.isfile(cfg.model_file):
        logging.info("Training...")
        model.train_model(
            cell_atts=cell_atts,
            sample_indices=sample_indices,
            pos=pos_scaled,
            ncounts=adata.X,
            raw_counts=adata.raw.X,
            size_factors=adata.obs.size_factors,
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
    else:
        model.load_model(str(cfg.model_file))
        logging.info("Loaded existing weights: %s", cfg.model_file)

    # Counterfactual prediction
    project_index = f"{cfg.sample}_{cfg.project_index}"
    pert_ind = (np.loadtxt(cfg.pert_cells, dtype=int) - 1).astype(int)

    target_tissue_code = tissue_dict.get(cfg.target_cell_tissue, None)
    target_pert_code = pert_dict.get(cfg.target_cell_perturbation, None)
    if target_tissue_code is None:
        logging.warning("Target tissue '%s' not found; using max code.", cfg.target_cell_tissue)
        target_tissue_code = int(np.max(list(tissue_dict.values())))
    if target_pert_code is None:
        logging.warning("Target perturbation '%s' not found; using background (0).", cfg.target_cell_perturbation)
        target_pert_code = 0

    logging.info("Counterfactual → tissue=%s (code=%d), perturbation=%s (code=%d)",
                 cfg.target_cell_tissue, target_tissue_code,
                 cfg.target_cell_perturbation, target_pert_code)

    perturbed_counts = model.counterfactualPrediction(
        X=pos_scaled,
        sample_index=sample_indices,
        cell_atts=cell_atts,
        batch_size=bs,
        n_samples=25,
        perturb_cell_id=pert_ind,
        target_cell_tissue=target_tissue_code,
        target_cell_perturbation=target_pert_code,
    )

    ad = sc.AnnData(perturbed_counts)
    # mark perturbed cells (1/0)
    pert_set = set(pert_ind.tolist() if isinstance(pert_ind, np.ndarray) else list(pert_ind))
    ad.obs["perturbed"] = [1 if i in pert_set else 0 for i in range(ad.n_obs)]
    out_path = outdir / f"{project_index}_{cfg.target_cell_tissue}_{cfg.target_cell_perturbation}_perturbed_counts.h5ad"
    ad.write(out_path)
    logging.info("Wrote perturbed counts: %s", out_path)


# -----------------------------------------------------------------------------
# CLI (config-first, explicit overrides only)
# -----------------------------------------------------------------------------
def str2bool(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    return v.lower() in ("1", "true", "yes", "y", "on")
      
def parse_args() -> RunConfigSK:
    p = argparse.ArgumentParser(
        description="CONCERT (SK) runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # config path only; no defaults on the rest to avoid clobbering YAML
    p.add_argument("--config", help="YAML/JSON config file path.")

    # IO / basic
    p.add_argument("--data_file")
    p.add_argument("--outdir")
    p.add_argument("--sample")
    p.add_argument("--project_index")

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

    # runtime / persistence
    p.add_argument("--model_file")
    p.add_argument("--device")
    p.add_argument("--verbosity", type=int)

    # counterfactual
    p.add_argument("--pert_cells")
    p.add_argument("--target_cell_tissue")
    p.add_argument("--target_cell_perturbation")

    a = p.parse_args()

    # 1) defaults
    cfg = RunConfigSK()

    # 2) merge config file
    file_cfg = load_config_file(getattr(a, "config", None)) if getattr(a, "config", None) else {}
    if file_cfg:
        # coerce lists for dataclass Iterable fields
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

    # post-process batch_size if string/int
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
