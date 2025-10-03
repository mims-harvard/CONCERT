"""
Imputation driver for CONCERT — predict expression at new spatial coordinates.

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
import scanpy as sc
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# Optional YAML
try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

from concert_map import CONCERT
from preprocess import normalize, geneSelection


# --------------------------------------------------------------------------
# Logging & small utils
# --------------------------------------------------------------------------
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
    """Deterministically map strings to [0..K-1] integer codes (stable ordering)."""
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
    # fallback attempts
    try:
        if yaml is not None:
            return yaml.safe_load(p.read_text()) or {}
    except Exception:
        pass
    try:
        return json.loads(p.read_text())
    except Exception as e:  # pragma: no cover
        raise ValueError(f"Unsupported config format for {path}: {e}")


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
@dataclass
class ImputeConfig:
    # IO
    data_file: str = "data.h5"
    outdir: str = "./outputs"
    sample: str = "sample"
    project_index: str = "x"

    # Inducing points
    grid_inducing_points: bool = True
    inducing_point_steps: int = 6       # (steps x steps) grid
    inducing_point_nums: int | None = None  # for k-means
    inducing_point_file: Optional[str] = None  # reuse saved CSV of (M x 2)
    loc_range: float = 20.0
    kernel_scale: float = 20.0  # used only if your SVGP expects a scalar when not multi-kernel

    # Selection / normalization
    select_genes: int = 0

    # Imputation
    pert_cells: str = "patch_jak2.txt"  # file with NEW coordinates to impute (tab/space-delimited 2 cols)

    # Targeted imputation (optional)
    target_cell_tissue: str = "tumor"
    target_cell_perturbation: str = "Jak2"

    # Model/training minimal knobs (only needed if training happens here)
    device: str = "cuda"
    batch_size: str = "auto"
    maxiter: int = 5000
    train_size: float = 0.95
    patience: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-6
    num_samples: int = 1

    # Architecture to instantiate CONCERT (must match your training)
    encoder_layers: Iterable[int] = (128, 64)
    decoder_layers: Iterable[int] = (128,)
    encoder_dim: int = 256
    GP_dim: int = 2
    Normal_dim: int = 8
    noise: float = 0.1
    dropoutE: float = 0.1
    dropoutD: float = 0.0
    dynamicVAE: bool = True
    init_beta: float = 10.0
    min_beta: float = 5.0
    max_beta: float = 25.0
    KL_loss: float = 0.025

    # SVGP options (must match your model expectations)
    fix_inducing_points: bool = True
    fixed_gp_params: bool = False
    multi_kernel_mode: bool = True  # if your SVGP_Batch is multi-kernel
    shared_dispersion: bool = False  # only used if you also decode

    # Weights
    model_file: Optional[str] = None  # if present and exists, loads; else trains

    # Misc
    verbosity: int = 1
    config: Optional[str] = None  # YAML/JSON path


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def load_h5_dataset(path: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Returns
    -------
    X : (N, G) float32
    pos : (N, 2) float32
    tissue_raw : (N,) str or None (if missing)
    perturb_raw : (N,) str
    """
    with h5py.File(path, "r") as f:
        X = np.array(f["X"], dtype=np.float32)
        pos = np.array(f["pos"], dtype=np.float32).T  # to (N, 2)
        perturb_raw = np.array(f["perturbation"], dtype=str)

        tissue_raw = None
        if "tissue" in f:
            tissue_raw = np.array(f["tissue"], dtype=str)

    return X, pos, tissue_raw, perturb_raw


def build_attributes(tissue_raw: Optional[np.ndarray], perturb_raw: np.ndarray):
    """
    Map tissue & perturb strings to integer codes, plus name dicts.
    If tissue is missing, infer: {"None"->"normal"; known perturbations -> "tumor"}.
    """
    known_perts = ["Jak2", "Tgfbr2", "Ifngr2"]
    if tissue_raw is None:
        # infer tissue from perturbations
        tissue_raw = np.where(np.isin(perturb_raw, known_perts), "tumor", perturb_raw)
        tissue_raw = np.where(tissue_raw == "None", "normal", tissue_raw)

    tissue_idx = strings_to_index(tissue_raw)
    # perturb 0 = background; known perts get 1..K in stable order
    present = [p for p in known_perts if p in set(perturb_raw.tolist())]
    pert_map = {p: i + 1 for i, p in enumerate(present)}
    perturb_idx = np.vectorize(lambda s: pert_map.get(s, 0))(perturb_raw).astype(int)

    tissue_dict = {name: int(code) for name, code in zip(tissue_raw, tissue_idx)}
    return tissue_idx, tissue_dict, perturb_idx, pert_map


# --------------------------------------------------------------------------
# Inducing points
# --------------------------------------------------------------------------
def make_inducing_points(
    pos: np.ndarray,
    cfg: ImputeConfig,
) -> np.ndarray:
    """Create or load inducing points in the original (unbatched) 2D space."""
    if cfg.inducing_point_file and os.path.isfile(cfg.inducing_point_file):
        ip = np.loadtxt(cfg.inducing_point_file, delimiter=",")
        logging.info("Loaded inducing points from %s (shape %s)", cfg.inducing_point_file, ip.shape)
        return ip.astype(np.float32)

    if cfg.grid_inducing_points:
        # (steps x steps) grid in [0,1] then scale by loc_range
        grid_xy = np.mgrid[0:1:complex(cfg.inducing_point_steps),
                           0:1:complex(cfg.inducing_point_steps)].reshape(2, -1).T
        ip = (grid_xy * float(cfg.loc_range)).astype(np.float32)
        logging.info("Created grid inducing points: steps=%d -> %d points", cfg.inducing_point_steps, ip.shape[0])
        return ip

    # KMeans centroids
    assert cfg.inducing_point_nums and cfg.inducing_point_nums > 0, \
        "inducing_point_nums must be set when grid_inducing_points=False"
    km = KMeans(n_clusters=cfg.inducing_point_nums, n_init=100).fit(pos)
    ip = km.cluster_centers_.astype(np.float32)
    logging.info("Created k-means inducing points: k=%d", cfg.inducing_point_nums)
    return ip


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def run(cfg: ImputeConfig) -> None:
    logging.info("Config: %s", asdict(cfg))
    ensure_dir(cfg.outdir)

    # Load core data
    X, pos_raw, tissue_raw, perturb_raw = load_h5_dataset(cfg.data_file)

    # Scale the train locations to [0, loc_range], then scale NEW coords with same transform
    # NEW coords file (the locations where we want imputed counts)
    pert_loc_raw = np.loadtxt(cfg.pert_cells, dtype=float)
    if pert_loc_raw.ndim == 1:
        pert_loc_raw = pert_loc_raw[None, :]
    if pert_loc_raw.shape[1] != 2:
        raise ValueError(f"{cfg.pert_cells} must have 2 columns (x, y).")

    # Single scaler on combined set → split back (keeps relative placement)
    scaler = MinMaxScaler()
    pos_all = np.concatenate([pos_raw, pert_loc_raw], axis=0)
    pos_all = scaler.fit_transform(pos_all) * cfg.loc_range
    pos = pos_all[: pos_raw.shape[0]].astype(np.float32)
    pert_loc = pos_all[pos_raw.shape[0] :].astype(np.float32)

    # Attributes
    tissue_idx, tissue_dict, perturb_idx, pert_map = build_attributes(tissue_raw, perturb_raw)
    cell_atts = np.c_[tissue_idx, perturb_idx].astype(int)
    sample_indices = torch.arange(X.shape[0], dtype=torch.int)

    # Name maps for pretty outputs
    code_to_name = {0: "background"}
    code_to_name.update({code: name for name, code in pert_map.items()})

    # Batch one-hot from perturbation codes (used by the kernel)
    n_batch = int(len(np.unique(perturb_idx)))
    batch = np.eye(n_batch, dtype=np.float32)[perturb_idx]

    # Build inducing points (2D)
    inducing_points = make_inducing_points(pos, cfg)

    # Prepare AnnData + normalization, optional gene selection
    if cfg.select_genes and cfg.select_genes > 0:
        logging.info("Selecting top %d genes...", cfg.select_genes)
        important = geneSelection(X, n=cfg.select_genes, plot=False)
        X = X[:, important]
        np.savetxt(Path(cfg.outdir) / "selected_genes.txt", important, fmt="%d")

    adata = sc.AnnData(X, dtype="float32")
    adata = normalize(adata, size_factors=True, normalize_input=True, logtrans_input=True)

    # Model (keep consistent with your training-time hyperparams)
    spatial_dims = 2
    kernel_scale = (np.full((n_batch, spatial_dims), cfg.kernel_scale, dtype=np.float32)
                    if cfg.multi_kernel_mode else float(cfg.kernel_scale))

    model = CONCERT(
        cell_atts=cell_atts,
        num_genes=adata.n_vars,
        encoder_dim=cfg.encoder_dim,
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
        initial_inducing_points=inducing_points,
        fixed_gp_params=cfg.fixed_gp_params,
        kernel_scale=kernel_scale,
        multi_kernel_mode=cfg.multi_kernel_mode,
        N_train=adata.n_obs,
        KL_loss=cfg.KL_loss,
        dynamicVAE=cfg.dynamicVAE,
        init_beta=cfg.init_beta,
        min_beta=cfg.min_beta,
        max_beta=cfg.max_beta,
        mask_cutoff=np.full(pos.shape[0], 0.5, dtype=np.float32),
        dtype=torch.float32,
        device=cfg.device,
    )
    logging.info("Model initialized.")

    # Train (if no weights) or load existing
    if cfg.model_file and os.path.isfile(cfg.model_file):
        model.load_model(cfg.model_file)
        logging.info("Loaded weights: %s", cfg.model_file)
    else:
        bs = auto_batch_size(X.shape[0]) if cfg.batch_size == "auto" else int(cfg.batch_size)
        logging.info("Training (batch_size=%s)...", bs)
        model.train_model(
            pos=np.concatenate([pos, batch], axis=1).astype(np.float32),  # pos_batched if your kernel expects batch cols
            ncounts=adata.X,
            raw_counts=adata.raw.X,
            size_factors=adata.obs.size_factors,
            batch=batch,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            batch_size=bs,
            num_samples=cfg.num_samples,
            train_size=cfg.train_size,
            maxiter=cfg.maxiter,
            patience=cfg.patience,
            save_model=True,
            model_weights=str(Path(cfg.outdir) / (cfg.model_file or "model.pt")),
        )
        logging.info("Training complete.")

    # ---- Impute at NEW coordinates (baseline) ----
    bs = auto_batch_size(X.shape[0]) if cfg.batch_size == "auto" else int(cfg.batch_size)
    logging.info("Imputing counts at %d new coordinates...", pert_loc.shape[0])
    _, imputed_counts = model.batching_predict_samples(
        X_test=pert_loc,               # (M, 2) scaled
        X_train=pos,                   # (N, 2) scaled
        Y_sample_index=sample_indices,
        Y_cell_atts=cell_atts,
        batch_size=bs,
        n_samples=25,
    )
    sc.AnnData(imputed_counts).write(
        Path(cfg.outdir) / f"{cfg.sample}_{cfg.project_index}_imputed_counts.h5ad"
    )
    logging.info("Wrote baseline imputed counts.")

    # ---- Targeted (counterfactual) imputation, if supported by the model ----
    if hasattr(model, "batching_predict_samples_target"):
        # build lookup for names → integer codes
        tissue_code = None
        if isinstance(cfg.target_cell_tissue, str):
            # match exact name if present
            if cfg.target_cell_tissue in tissue_dict:
                tissue_code = int(tissue_dict[cfg.target_cell_tissue])
            else:
                logging.warning("Target tissue '%s' not found; using max code.", cfg.target_cell_tissue)
                tissue_code = int(np.max(list(tissue_dict.values())))
        else:
            tissue_code = int(cfg.target_cell_tissue)

        if cfg.target_cell_perturbation in pert_map:
            pert_code = int(pert_map[cfg.target_cell_perturbation])
        else:
            logging.warning("Target perturbation '%s' not found; using background (0).",
                            cfg.target_cell_perturbation)
            pert_code = 0

        logging.info("Imputing counterfactual: tissue=%s, perturbation=%s",
                     cfg.target_cell_tissue, cfg.target_cell_perturbation)
        _, imputed_pert_counts = model.batching_predict_samples_target(
            target=pert_code,
            tissue=tissue_code,
            X_test=pert_loc,
            X_train=pos,
            Y_sample_index=sample_indices,
            Y_cell_atts=cell_atts,
            batch_size=bs,
            n_samples=25,
        )
        sc.AnnData(imputed_pert_counts).write(
            Path(cfg.outdir) / f"{cfg.sample}_{cfg.project_index}_{cfg.target_cell_perturbation}_imputedPert_counts.h5ad"
        )
        logging.info("Wrote targeted imputed counts.")
    else:
        logging.info(
            "Model has no `batching_predict_samples_target`; skipping targeted imputation. "
            "Add that method to CONCERT to enable targeted impute at new coords."
        )


# --------------------------------------------------------------------------
# CLI (config-first; CLI only overrides what you pass)
# --------------------------------------------------------------------------
def parse_args() -> ImputeConfig:
    p = argparse.ArgumentParser(
        description="CONCERT imputation runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Only set types; NO defaults—so config.yaml won't get clobbered
    p.add_argument("--config", help="YAML/JSON config file path.")

    # IO / basic
    p.add_argument("--data_file")
    p.add_argument("--outdir")
    p.add_argument("--sample")
    p.add_argument("--project_index")

    # Inducing / scaling
    def str2bool(v):
        if v is None:
            return None
        if isinstance(v, bool):
            return v
        return v.lower() in ("1", "true", "yes", "y", "on")

    p.add_argument("--grid_inducing_points", type=str2bool)
    p.add_argument("--inducing_point_steps", type=int)
    p.add_argument("--inducing_point_nums", type=int)
    p.add_argument("--inducing_point_file")
    p.add_argument("--loc_range", type=float)
    p.add_argument("--kernel_scale", type=float)

    # Selection / impute
    p.add_argument("--select_genes", type=int)
    p.add_argument("--pert_cells")
    p.add_argument("--target_cell_tissue")
    p.add_argument("--target_cell_perturbation")

    # Train / model (minimal knobs)
    p.add_argument("--device")
    p.add_argument("--batch_size")
    p.add_argument("--maxiter", type=int)
    p.add_argument("--train_size", type=float)
    p.add_argument("--patience", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--weight_decay", type=float)
    p.add_argument("--num_samples", type=int)

    # Architecture
    p.add_argument("--encoder_layers", nargs="+", type=int)
    p.add_argument("--decoder_layers", nargs="+", type=int)
    p.add_argument("--encoder_dim", type=int)
    p.add_argument("--GP_dim", type=int)
    p.add_argument("--Normal_dim", type=int)
    p.add_argument("--noise", type=float)
    p.add_argument("--dropoutE", type=float)
    p.add_argument("--dropoutD", type=float)
    p.add_argument("--dynamicVAE", type=str2bool)
    p.add_argument("--init_beta", type=float)
    p.add_argument("--min_beta", type=float)
    p.add_argument("--max_beta", type=float)
    p.add_argument("--KL_loss", type=float)
    p.add_argument("--fix_inducing_points", type=str2bool)
    p.add_argument("--fixed_gp_params", type=str2bool)
    p.add_argument("--multi_kernel_mode", type=str2bool)
    p.add_argument("--shared_dispersion", type=str2bool)

    p.add_argument("--model_file")
    p.add_argument("--verbosity", type=int)

    a = p.parse_args()

    # 1) defaults
    cfg = ImputeConfig()

    # 2) merge config file
    file_cfg = load_config_file(getattr(a, "config", None)) if getattr(a, "config", None) else {}
    if file_cfg:
        # coerce list→tuple for dataclass Iterable fields
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
