#!/usr/bin/env python3
"""
Driver script for CONCERT — Counterfactual spatial perturbation prediction with a GP-VAE backbone.

Features
--------
- YAML/JSON config file support (CLI flags override config file values)
- Weights & Biases (wandb) logging for training and validation metrics
- Research-friendly structure, logging, and utilities
"""

from __future__ import annotations
import sys
import argparse
import json
import logging
import time
from dataclasses import dataclass, asdict, replace
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# Optional YAML support
try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

from concert_map import CONCERT
from preprocess import normalize, geneSelection


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
def setup_logging(verbosity: int = 1) -> None:
    level = logging.WARNING if verbosity <= 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )

# -----------------------------------------------------------------------------
# --- Reporting (kernel scales & cutoffs) --------------------------------
# -----------------------------------------------------------------------------
def _report_and_save_final(
    *,
    model,
    cell_atts: np.ndarray,                # (N, 2) → [tissue_code, perturb_code]
    perturb_name_map: Dict[int, str],     # e.g., {0: "background", 1: "Jak2", ...}
    outdir: Path,
    sample: str,
    project_index: str,
) -> None:
    """Log final kernel scales and mean cutoffs per perturbation, and dump per-cell cutoffs to CSV."""
    try:
        with torch.no_grad():
            # Kernel scale (detach to avoid printing a Parameter object)
            scale = getattr(model.svgp.kernel, "scale", None)
            if scale is not None:
                scale_np = scale.detach().float().cpu().numpy()
                if scale_np.ndim == 1:
                    logging.info("[final] Kernel scale (shared): %s", np.array2string(scale_np, precision=4))
                else:
                    logging.info("[final] Kernel scales by perturbation:")
                    for code in range(scale_np.shape[0]):
                        name = perturb_name_map.get(code, f"pert={code}")
                        logging.info("  • %-16s | scale=%s", name, np.array2string(scale_np[code], precision=4))
            else:
                logging.info("[final] Kernel has no `scale` attribute; skipping kernel report.")
    except Exception as e:
        logging.info("[final] Could not read kernel scales (%s); skipping kernel report.", e)

    # Mean cutoff per perturbation
    try:
        cut = model.mask_cutoff.detach().float().cpu().numpy()
        perts = cell_atts[:, 1].astype(int)
        uniq = sorted(np.unique(perts).tolist())
        logging.info("[final] Mean cutoff per perturbation:")
        for code in uniq:
            name = perturb_name_map.get(code, f"pert={code}")
            m = float(np.mean(cut[perts == code])) if np.any(perts == code) else float("nan")
            logging.info("  • %-16s | mean_cutoff=%.4f", name, m)
    except Exception as e:
        logging.info("[final] Could not compute mean cutoffs (%s).", e)

    # Save per-sample cutoffs to CSV
    try:
        df = pd.DataFrame(
            {
                "index": np.arange(cell_atts.shape[0], dtype=int),
                "tissue_code": cell_atts[:, 0].astype(int),
                "perturbation_code": cell_atts[:, 1].astype(int),
                "perturbation_name": [perturb_name_map.get(int(c), f"pert={int(c)}") for c in cell_atts[:, 1]],
                "cutoff": cut,
            }
        )
        csv_path = outdir / f"{sample}_{project_index}_cutoffs.csv"
        df.to_csv(csv_path, index=False)
        logging.info("Saved per-sample cutoffs to %s", csv_path)
    except Exception as e:
        logging.warning("Failed to save per-sample cutoffs CSV (%s).", e)

# -----------------------------------------------------------------------------
# Dataclasses / Config
# -----------------------------------------------------------------------------
@dataclass
class RunConfig:
    # IO
    data_file: str = "data.h5"
    outdir: str = "./outputs"
    sample: str = "sample"
    project_index: str = "x"
    model_file: str = "model.pt"
    report_every: int = 50

    # Config file (YAML/JSON)
    config: Optional[str] = None

    # Train / Eval
    stage: str = "train"  # {train, eval}
    select_genes: int = 0
    train_size: float = 0.95
    maxiter: int = 5000
    patience: int = 200
    lr: float = 1e-4
    weight_decay: float = 1e-6
    batch_size: str = "auto"  # {auto or int}
    num_samples: int = 1

    # Architecture / Latents
    encoder_layers: Iterable[int] = (128, 64)
    decoder_layers: Iterable[int] = (128,)
    encoder_dim: int = 256
    GP_dim: int = 2
    Normal_dim: int = 8

    # Regularization / VAE control
    noise: float = 0.0
    dropoutE: float = 0.0
    dropoutD: float = 0.0
    shared_dispersion: bool = False
    dynamicVAE: bool = True
    init_beta: float = 10.0
    min_beta: float = 5.0
    max_beta: float = 25.0
    KL_loss: float = 0.025

    # GP / Inducing points
    fix_inducing_points: bool = True
    grid_inducing_points: bool = True
    inducing_point_steps: int = 6
    inducing_point_nums: int | None = None
    fixed_gp_params: bool = False
    multi_kernel_mode: bool = True
    kernel_scale: float = 10.0
    loc_range: float = 20.0

    # Runtime
    device: str = "cuda"
    seed: int | None = None
    verbosity: int = 1

    # Counterfactual
    pert_cells: str = "patch_jak2.txt"  # 1-based indices file
    target_cell_tissue: str = "tumor"   # only used for naming by default
    target_cell_perturbation: str = "Jak2"

    # Weights & Biases
    wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_run: Optional[str] = None
    wandb_mode: str = "online"  # online|offline|disabled


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
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


def distance_quantile(loc: np.ndarray, q: float = 0.1) -> float:
    """Euclidean distance quantile for a set of locations."""
    d = pdist(loc, metric="euclidean")
    return float(np.quantile(d, q))


def strings_to_index(values: np.ndarray) -> np.ndarray:
    """Deterministically map strings to [0..K-1] integer codes (order-stable)."""
    hashed = np.array([sum(ord(c) for c in s) for s in values], dtype=int)
    uniq = np.unique(hashed)
    remap = {u: i for i, u in enumerate(uniq)}
    return np.array([remap[h] for h in hashed], dtype=int)


def build_inducing_points(
    pos_batched: np.ndarray,
    n_batch: int,
    steps: int,
    loc_range: float,
    grid: bool,
    k_clusters: int | None,
) -> np.ndarray:
    """Return inducing points with appended one-hot batch column block (first batch active)."""
    if grid:
        grid_xy = np.mgrid[0:1:complex(steps), 0:1:complex(steps)].reshape(2, -1).T * loc_range
        onehot = np.zeros((grid_xy.shape[0], n_batch), dtype=np.float32)
        onehot[:, 0] = 1.0
        return np.concatenate([grid_xy, onehot], axis=1).astype(np.float32)
    assert k_clusters is not None and k_clusters > 0, "inducing_point_nums must be > 0 when grid_inducing_points=False"
    km = KMeans(n_clusters=k_clusters, n_init=100).fit(pos_batched)
    centers = km.cluster_centers_
    onehot = np.zeros((centers.shape[0], n_batch), dtype=np.float32)
    onehot[:, 0] = 1.0
    return np.concatenate([centers, onehot], axis=1).astype(np.float32)


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

def build_attributes(perturb_raw: np.ndarray):
    """
    Map tissue & perturbation strings to integer codes.
    • Tissue: {"None"→"normal", {Jak2,Tgfbr2,Ifngr2,KP}→"tumor", "periphery"->"periphery"}
    • Perturbation: {Jak2,Tgfbr2,Ifngr2}→1..K, background→0
    """
    known_types = ["Jak2", "Tgfbr2", "Ifngr2", "KP", "normal", "periphery"]
    known_tumor = ["Jak2", "Tgfbr2", "Ifngr2", "KP"]
    known_normal = "normal"
    known_periphery = "periphery"

    #initialize tissue_raw for all cells as "normal"
    tissue_raw = np.full_like(perturb_raw, known_normal, dtype=object)
    tissue_raw = np.where(np.isin(perturb_raw, known_tumor), "tumor", tissue_raw)
    tissue_raw = np.where(perturb_raw == known_periphery, "periphery", tissue_raw)
    tissue_idx = strings_to_index(tissue_raw)

    present = [p for p in known_types if p in set(perturb_raw.tolist())]
    pert_map = {p: i + 1 for i, p in enumerate(present)}
    perturb_idx = np.vectorize(lambda s: pert_map.get(s, 0))(perturb_raw).astype(int)

    tissue_dict = {t: i for t, i in zip(tissue_raw, tissue_idx)}
    return tissue_idx, tissue_dict, perturb_idx, pert_map

def load_config_file(path: Optional[str]) -> dict:
    """Load a YAML or JSON config into a dict. Returns {} if path is None."""
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    if p.suffix.lower() in {".json"}:
        return json.loads(p.read_text())
    if p.suffix.lower() in {".yml", ".yaml"}:
        if yaml is None:
            raise ImportError("PyYAML is not installed but a YAML config was provided.")
        return yaml.safe_load(p.read_text()) or {}
    # Fallback: try YAML then JSON
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
# Orchestration
# -----------------------------------------------------------------------------
def run(cfg: RunConfig) -> None:
    logging.info("Config: %s", asdict(cfg))

    # Optional Weights & Biases
    wb = None
    if cfg.wandb and cfg.wandb_mode != "disabled":
        try:
            import wandb  # type: ignore

            wb = wandb
            wandb.init(
                project=cfg.wandb_project or "concert",
                name=cfg.wandb_run,
                mode=cfg.wandb_mode,
                config=asdict(cfg),
            )
        except Exception as e:  # pragma: no cover
            logging.warning("wandb init failed (%s); continuing without wandb.", e)
            wb = None

    # Seed
    if cfg.seed is not None:
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

    # IO
    outdir = ensure_dir(cfg.outdir)

    # Load
    X, pos_raw, _, perturb_raw = load_h5_dataset(cfg.data_file)
    tissue_idx, tissue_dict, perturb_idx, pert_map = build_attributes(perturb_raw)

    cell_atts = np.c_[tissue_idx, perturb_idx].astype(int)
    sample_indices = torch.arange(X.shape[0], dtype=torch.int)

    pert_name = {0: "background"}
    pert_name.update({code: name for name, code in pert_map.items()})

    # Batch one-hot from perturbation codes
    n_batch = int(len(np.unique(perturb_idx)))
    batch = np.eye(n_batch, dtype=np.float32)[perturb_idx]

    # Scale spatial & append batch columns for kernel conditioning
    scaler = MinMaxScaler()
    pos_scaled = scaler.fit_transform(pos_raw) * cfg.loc_range
    cutoff = np.full(pos_scaled.shape[0], 0.5, dtype=np.float32)  # learnable init (vector)
    pos_batched = np.concatenate([pos_scaled, batch], axis=1).astype(np.float32)

    # Inducing points
    inducing = build_inducing_points(
        pos_batched=pos_batched,
        n_batch=n_batch,
        steps=cfg.inducing_point_steps,
        loc_range=cfg.loc_range,
        grid=cfg.grid_inducing_points,
        k_clusters=cfg.inducing_point_nums,
    )

    # Optional gene selection
    if cfg.select_genes and cfg.select_genes > 0:
        logging.info("Selecting top %d genes...", cfg.select_genes)
        important = geneSelection(X, n=cfg.select_genes, plot=False)
        X = X[:, important]
        np.savetxt(outdir / "selected_genes.txt", important, fmt="%d")

    # AnnData normalization (size factors + log)
    adata = sc.AnnData(X, dtype="float32")
    adata = normalize(adata, size_factors=True, normalize_input=True, logtrans_input=True)

    # Kernel scale per batch (n_batch, spatial_dims)
    spatial_dims = pos_batched.shape[1] - n_batch
    kernel_scale = np.full((n_batch, spatial_dims), cfg.kernel_scale, dtype=np.float32)

    # Model
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
        initial_inducing_points=inducing,
        fixed_gp_params=cfg.fixed_gp_params,
        kernel_scale=kernel_scale,
        multi_kernel_mode=cfg.multi_kernel_mode,
        N_train=adata.n_obs,
        KL_loss=cfg.KL_loss,
        dynamicVAE=cfg.dynamicVAE,
        init_beta=cfg.init_beta,
        min_beta=cfg.min_beta,
        max_beta=cfg.max_beta,
        mask_cutoff=cutoff,
        dtype=torch.float32,
        device=cfg.device,
    )
    logging.info("Model initialized: %s", model.__class__.__name__)

    # Wandb logger helper
    def _log(metrics: Dict, step: int) -> None:
        if wb is not None:
            try:
                wb.log(metrics, step=step)
            except Exception:
                pass

    # Stage
    if cfg.stage.lower() == "train":
        bs = auto_batch_size(X.shape[0]) if cfg.batch_size == "auto" else int(cfg.batch_size)
        t0 = time.time()
        model.train_model(
            pos=pos_batched,
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
            model_weights=str(outdir / cfg.model_file),
            log_fn=_log,
            report_every=cfg.report_every,
            perturb_name_map=pert_name,
        )
        logging.info("Training done in %.1fs", time.time() - t0)

        _report_and_save_final(
        model=model,
        cell_atts=cell_atts,
        perturb_name_map=pert_name,
        outdir=outdir,
        sample=cfg.sample,
        project_index=cfg.project_index,
    )
        return

    # Inference / export
    model.load_model(str(outdir / cfg.model_file))
    logging.info("Loaded weights: %s", outdir / cfg.model_file)
    _report_and_save_final(
        model=model,
        cell_atts=cell_atts,
        perturb_name_map=pert_name,
        outdir=outdir,
        sample=cfg.sample,
        project_index=cfg.project_index,
    )

    # Counterfactuals
    pert_idx = (np.loadtxt(cfg.pert_cells, dtype=int) - 1).astype(int)
    if hasattr(model, "counterfactualPrediction"):
        # build lookup for names → integer codes
        tissue_code = None
        if isinstance(cfg.target_cell_tissue, str):
            # match exact name if present
            if cfg.target_cell_tissue in tissue_dict:
                tissue_code = int(tissue_dict[cfg.target_cell_tissue])
            else: # error
                raise ValueError(f"Target tissue '{cfg.target_cell_tissue}' not found in data; available: {list(tissue_dict.keys())}")
                
        else:
            tissue_code = int(cfg.target_cell_tissue)

        if cfg.target_cell_perturbation in pert_map:
            pert_code = int(pert_map[cfg.target_cell_perturbation])
        else: # error
            raise ValueError(f"Target perturbation '{cfg.target_cell_perturbation}' not found in data; available: {list(pert_map.keys())}")

    perturbed_counts, pert_cell_att = model.counterfactualPrediction(
        X=pos_batched,
        sample_index=sample_indices,
        cell_atts=cell_atts.copy(),
        batch_size=512,
        n_samples=25,
        perturb_cell_id=torch.tensor(pert_idx),
        target_cell_tissue=tissue_code,
        target_cell_perturbation=pert_code,
    )

    sc.AnnData(perturbed_counts, obs=pd.DataFrame(pert_cell_att, columns=["tissue", "perturbation"])) \
      .write(outdir / f"{cfg.sample}_{cfg.project_index}_{cfg.target_cell_tissue}_{cfg.target_cell_perturbation}_perturbed_counts.h5ad")
    logging.info(
        "Wrote counterfactuals: %s",
        outdir / f"{cfg.sample}_{cfg.project_index}_{cfg.target_cell_tissue}_{cfg.target_cell_perturbation}_perturbed_counts.h5ad",
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def str2bool(v):
    if v is None: return None
    if isinstance(v, bool): return v
    return v.lower() in ("1","true","yes","y","on")
def parse_args() -> RunConfig:
    p = argparse.ArgumentParser(description="CONCERT runner",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # IO
    p.add_argument("--data_file")
    p.add_argument("--outdir")
    p.add_argument("--sample")
    p.add_argument("--project_index")
    p.add_argument("--model_file")
    p.add_argument("--config", help="YAML/JSON config file path. CLI overrides file.")

    # Stage
    p.add_argument("--stage", choices=["train", "eval"])

    # Selection / training
    p.add_argument("--select_genes", type=int)
    p.add_argument("--train_size", type=float)
    p.add_argument("--maxiter", type=int)
    p.add_argument("--patience", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--weight_decay", type=float)
    p.add_argument("--batch_size")              # allow "auto" or int as str
    p.add_argument("--num_samples", type=int)

    # Architecture
    p.add_argument("--encoder_layers", nargs="+", type=int)
    p.add_argument("--decoder_layers", nargs="+", type=int)
    p.add_argument("--encoder_dim", type=int)
    p.add_argument("--GP_dim", type=int)
    p.add_argument("--Normal_dim", type=int)

    # Regularization / VAE control
    p.add_argument("--noise", type=float)
    p.add_argument("--dropoutE", type=float)
    p.add_argument("--dropoutD", type=float)
    p.add_argument("--shared_dispersion", type=str2bool)
    p.add_argument("--dynamicVAE", type=str2bool)
    p.add_argument("--init_beta", type=float)
    p.add_argument("--min_beta", type=float)
    p.add_argument("--max_beta", type=float)
    p.add_argument("--KL_loss", type=float)

    # GP / inducing
    p.add_argument("--fix_inducing_points", type=str2bool)
    p.add_argument("--grid_inducing_points", type=str2bool)
    p.add_argument("--inducing_point_steps", type=int)
    p.add_argument("--inducing_point_nums", type=int)
    p.add_argument("--fixed_gp_params", type=str2bool)
    p.add_argument("--multi_kernel_mode", type=str2bool)
    p.add_argument("--kernel_scale", type=float)
    p.add_argument("--loc_range", type=float)

    # Runtime
    p.add_argument("--device")
    p.add_argument("--seed", type=int)
    p.add_argument("--verbosity", type=int)

    # Counterfactual
    p.add_argument("--pert_cells")
    p.add_argument("--target_cell_tissue")
    p.add_argument("--target_cell_perturbation")

    # Weights & Biases
    p.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    p.add_argument("--wandb_project")
    p.add_argument("--wandb_run")
    p.add_argument("--wandb_mode", choices=["online", "offline", "disabled"])

    a = p.parse_args()

    # 1) Start with code defaults
    cfg = RunConfig()

    # 2) Merge config file (if provided)
    file_cfg = load_config_file(getattr(a, "config", None)) if getattr(a, "config", None) else {}
    if file_cfg:
        # coerce lists/tuples if needed
        if "encoder_layers" in file_cfg and isinstance(file_cfg["encoder_layers"], list):
            file_cfg["encoder_layers"] = tuple(file_cfg["encoder_layers"])
        if "decoder_layers" in file_cfg and isinstance(file_cfg["decoder_layers"], list):
            file_cfg["decoder_layers"] = tuple(file_cfg["decoder_layers"])
        cfg = replace(cfg, **{k: v for k, v in file_cfg.items() if hasattr(cfg, k)})

    # 3) Override ONLY with CLI flags the user actually typed
    specified = set()
    for action in p._actions:
        # skip positionals (we don't use them here)
        if not action.option_strings:
            continue
        if any(opt in sys.argv for opt in action.option_strings):
            specified.add(action.dest)

    cli_explicit = {k: getattr(a, k) for k in specified if hasattr(cfg, k) and getattr(a, k) is not None}

    # post-process a few that came as strings
    if "batch_size" in cli_explicit:
        try:
            cli_explicit["batch_size"] = int(cli_explicit["batch_size"])
        except (TypeError, ValueError):
            pass  # keep "auto" or whatever was provided

    cfg = replace(cfg, **cli_explicit)
    return cfg

if __name__ == "__main__":
    cfg = parse_args()
    setup_logging(cfg.verbosity)
    run(cfg)
