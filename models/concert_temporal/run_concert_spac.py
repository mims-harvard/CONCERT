#!/usr/bin/env python3
"""
Driver script for CONCERT-SPAC -- CONCERT for Visium HD spatial CRISPR screen (SPAC-seq).

Extends concert_map with the Perturbation Action Denoiser (PAD) to remove the
shared guide-delivery / Cas9 batch effect during pretraining.

Features
--------
- PAD: binary is_perturbed embedding + MMD + adversarial GRL loss (--use_pad)
- YAML/JSON config file support (CLI flags override config file values)
- Weights & Biases logging for training and validation metrics
"""

from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "benchmark_evaluation" / "release_paths.py").is_file()) / "benchmark_evaluation"))
import release_paths as _rp  # noqa: E402  data/, checkpoints/ and runs/ roots
import os
import sys
import argparse
import json
import logging
import time
from dataclasses import dataclass, asdict, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# Optional YAML support
try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

from concert_spac import CONCERT
from preprocess import normalize, geneSelection


# Logging
def setup_logging(verbosity: int = 1) -> None:
    level = logging.WARNING if verbosity <= 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )

# Reporting (kernel scales & cutoffs)
def _report_and_save_final(
    *,
    model,
    cell_atts: np.ndarray,                # (N, 2) -> [tissue_code, perturb_code]
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
                        logging.info("  * %-16s | scale=%s", name, np.array2string(scale_np[code], precision=4))
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
            logging.info("  * %-16s | mean_cutoff=%.4f", name, m)
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

# Dataclasses / Config
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
    holdout_unassigned_frac: float = 0.0   # fraction of unassigned cells to hold out (saved for eval)
    holdout_unassigned_file: str = ""      # explicit unassigned indices to hold out (post-exclusion concat space)
    holdout_file: str = ""                 # benchmark holdout: coord-keyed (sid,array_coords) unassigned source + masked KO
    # Deconvolution / imputation stages ("deconv" / "impute_patch")
    pseudobulk_file: str = ""              # pseudo-bulk spot h5ad (deconv input)
    deconv_perts: str = ""                # comma KO-gene list; empty -> top-K by patch size
    deconv_top_k: int = 10
    deconv_roi_radius: float = 1500.0     # raw-coord (micron) margin around each patch
    deconv_n_samples: int = 25
    mask_index_file: str = ""             # patch-removal cellindex npz (impute_patch GT)
    raw_slide_file: str = ""              # full unmasked slide h5ad (impute_patch GT)
    maxiter: int = 5000
    patience: int = 200
    earlystop_min_delta: float = 0.0     # min val-ELBO improvement to reset patience
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

    # Perturbation Action Denoiser (PAD) -- now downgraded to soft regularizer
    # (explicit action correction is the primary mechanism)
    use_pad: bool = False          # muted for now; re-enable after action correction is validated
    pad_action_dim: int = 64       # binary action embedding dim
    pad_lambda_mmd: float = 0.1    # MMD loss weight
    pad_lambda_adv: float = 0.01   # adversarial GRL loss weight

    # HVG selection
    # Genes with hvg_rank <= hvg_max_rank are used; ~10k genes in SPAC-seq data
    hvg_max_rank: int = 10000

    # Multi-sample joint training
    # Comma-separated list of h5ad file paths.  Empty string -> single-sample mode.
    # Example: "raw_output_1-1.h5ad,raw_output_1-2.h5ad,raw_output_1-3.h5ad,raw_output_1-4.h5ad"
    multi_sample_files: str = ""

    # Control-anchored explicit action correction
    # 3-term decoder: log mu = f_basal + 1[has_guide]*mu_action + 1[is_real_KO]*f_pert
    use_action_correction: bool = True
    detach_mu_action_for_ko: bool = False
    mu_action_anchor_lambda: float = 0.0

    # latent disentanglement (drop pert from encoder)
    # When use_basal_encoder_only=True, the VAE encoder sees only [basal,
    # tissue] from LORD; the perturbation embedding feeds only the gene-space
    # f_pert term. This structurally guarantees z is pert-blind and makes
    # f_pert the unique counterfactual mechanism (clean & interpretable).
    use_basal_encoder_only: bool = False   # opt-in: requires retraining from scratch
    use_f_pert: bool = False               # add per-pert gene-space additive log-fold term
    f_pert_init_std: float = 0.01
    f_pert_clamp: float = 5.0
    # Coordinate-conditioned footprint field (constant = legacy / anti-flat gate baseline)
    fpert_field: str = "constant"          # 'constant'|'rff'|'svgp'|'hybrid'|'mb_rff'|'gp_pp'
    fpert_rff_features: int = 128
    fpert_rff_bandwidth: float = 0.15
    fpert_hidden: int = 256
    fpert_svgp_inducing: int = 32
    fpert_svgp_ls_init: float = 5.0
    fpert_coord_scale: float = 1.0
    fpert_mb_bandwidths: str = ""          # comma list, e.g. "0.05,0.15,0.5" (mb_rff)
    basal_bayes: bool = True               # M2: Bayes-linear basal head (generally on)
    # deterministic RFF basal field + scale-aware disentanglement
    basal_field: str = "svgp"              # 'svgp' (legacy) | 'rff' (collapse-proof basal)
    basal_rff_features: int = 128
    basal_rff_bandwidth: float = 0.15
    basal_rff_hidden: int = 128
    basal_coord_scale: float = 20.0
    disentangle_radius: float = 0.0        # >0 -> scale-aware Moran at footprint radius
    use_consistency_loss: bool = False     # batch-level MMD on log mu_basal between KO and ctrl
    consistency_lambda: float = 1.0

    # inductive expression encoder + collapse fixes
    use_expr_encoder: bool = False         # amortized basal encoder on expression (inductive)
    expr_hidden_dims: tuple = (512, 256)
    expr_dropout: float = 0.1
    use_basal_consistency: bool = False    # MMD on basal_latent (KO vs ctrl)
    basal_consistency_lambda: float = 1.0
    beta_warmup_epochs: int = 0            # linear KL warmup 0 -> beta_target
    beta_target: float = 1.0
    free_bits: float = 0.0                 # per-dim KL floor (nats) on Gaussian block
    gp_free_bits: float = 0.0              # per-dim KL floor (nats) on GP block
    use_pert_correction: bool = False      # default-off leakage fallback

    # NNGP local field + precomputed spatial graph
    use_nngp: bool = False
    nngp_lambda: float = 1.0
    graph_file: Optional[str] = None
    # paracrine spillover + niche x pert interaction
    use_paracrine: bool = False
    paracrine_l1: float = 0.0
    use_niche_interaction: bool = False
    niche_rank: int = 8
    use_niche_film: bool = False
    niche_film_dim: int = 8
    niche_film_k: int = 25
    niche_film_gcap: float = 3.0
    niche_film_l2: float = 1e-3
    niche_film_rank: int = 8
    niche_film_dispersion: bool = False
    # Continuous-time conditioning (gut DSS temporal case study)
    use_time_conditioning: bool = False
    time_fourier_bands: int = 6
    time_emb_dim: int = 16
    time_rank: int = 8
    time_l2: float = 1e-4
    time_col: str = "day_norm"     # per-cell obs column with normalized day t in [0,1]
    # Time-varying inflammation composition (membership field)
    use_membership_field: bool = False
    membership_hidden: int = 64
    membership_bce_lambda: float = 1.0
    membership_moran_lambda: float = 0.0
    use_time_intensity: bool = True   # legacy time x pert intensity residual; off for composition model
    # timekernel: shared Tier1+niche basal, a(t)*Phi(l(t))*f_pert
    # Master switch. When on, the forcing logic (in parse_args) sets a consistent
    # combo: use_shared_basal defaults True, use_time_conditioning forced True,
    # use_action_correction / use_inflammation_field / use_membership_field forced
    # False (timekernel replaces the field/intensity mechanisms; DSS has no NTC).
    use_timekernel: bool = False
    use_shared_basal: Optional[bool] = None   # basal = shared Tier1 embed + linear(arch_desc) (no free per-cell latent); None -> follows use_timekernel
    # Directory of per-slice day-invariant architectural-descriptor sidecars
    # (<Slice_ID>.npy + manifest.json). Required when use_timekernel is on.
    arch_desc_dir: str = f"{_rp.DATA}/dss_colitis/architecture"
    # Baseline anchor a(t=0)->0: day 0 is pre-injury, so basal alone must reconstruct it.
    # This identifies the SIGN of f_pert (without it the model stores inflammation in basal
    # and has f_pert subtract it -- the inverted mechanism found by the shift-sign check).
    # PRIMARY = by-construction offset; the soft lambda alone loses to the NB term.
    amp_anchor_offset: float = 0.0
    amp_anchor_lambda: float = 0.0
    amp_anchor_t: float = 0.0   # day_norm of the healthy baseline (day 0 / 21)
    marker_sign_positive: bool = False
    time_in_field_mlp: bool = False
    # continuous inflammation field (supersedes binary membership field)
    use_inflammation_field: bool = False   # continuous s=softplus field, MSE-supervised, no gate_state
    field_mse_lambda: float = 1.0          # weight of MSE(s, observed continuous inflammation score)
    # Freeze the inflammation field's time input at this NORMALIZED day (t/21); None = field
    # sees the true t. s(x,t) is unidentifiable off each cell's own day (one day per slice), so
    # pinning it makes s architecture-only and leaves time to Phi. Also ablates an existing ckpt.
    field_time_fixed_t: Optional[float] = None
    field_moran_lambda: float = 0.0        # optional spatial-smoothness (Moran's I) reward on s
    # LOTO-D9 temporal evaluation (stage=loto_eval)
    loto_holdout_files: str = ""   # comma-sep held-out (day-9) h5ads = ground truth
    loto_target_day: float = 9.0   # day to predict (held out from training)
    loto_day_denom: float = 21.0   # normalization denominator (day_norm = day/denom)
    # composition_eval: run the descriptive case-study deliverables (C1 per-mouse trajectories,
    # C2 spatial Moran, C3 heterogeneity) instead of the make-or-break trajectory gate.
    case_study_deliverables: bool = False
    # composition_eval on the LOMO ckpt, pointed at the held-out mice's slices: ground-truth
    # unseen-individual reconstruction vs a population-mean baseline (deliverable 2).
    heldout_mouse_eval: bool = False
    dump_percell: bool = False     # in heldout_mouse_eval, save per-cell coords/obs/pred@t0-t21 npz
    dump_pathways: bool = False    # in heldout_mouse_eval, save per-cell per-PATHWAY counterfactual
                                   # scores at t0/t3/t9/t21 + observed diagonal -> pathway_dump.npz
                                   # (independent of dump_percell; reads pathway_gene_sets.json)
    # Comma list of the TRAINING run's --multi_sample_files, in training order. When an eval
    # runs on a SUBSET of slices, load_model front-truncates slide_bias, pairing each eval slice
    # with an unrelated training slide's bias. Supplying this list re-maps each eval slice to its
    # OWN trained row (unseen slices get the training mean). Empty = legacy truncation.
    slide_bias_ref_files: str = ""
    loto_n_samples: int = 10       # decode MC samples for counterfactual pseudobulk
    loto_batch_size: int = 512
    # multiplicative per-cell scalar gate on the global pert effect
    use_pert_gate: bool = False
    gate_hidden: int = 64
    # spatial-basal disentanglement + mini-bulk LFC supervision
    spatial_basal_only: bool = False
    lfc_lambda: float = 0.0
    disentangle_lambda: float = 0.0
    moran_lambda: float = 0.0
    deamortize_gp: bool = False
    additive_decoder: bool = False
    gp_field_aux_lambda: float = 0.0
    basal_moran_output_lambda: float = 0.0
    # Stage 1 GP guardrail: soft band prior on the SVGP kernel lengthscale
    ls_prior_lambda: float = 0.0
    ls_prior_min: float = 3.0
    ls_prior_max: float = 8.0

    # SPAC-seq cell taxonomy labels
    # Comma-separated; extend if your data uses different naming.
    # "multiple" belongs in excluded_label: it represents co-captured cells with
    # distinct KOs -- their expression is a mixture of multiple perturbation effects
    # and cannot be cleanly attributed to any single target gene.
    unassigned_label: str = "unassigned,Unassigned,No_guide,no_guide,unknown"
    control_label: str = "Non_targeting,non_targeting,NonTargeting,control,Control,CTRL,ctrl,NT"
    excluded_label: str = "multiple,Multiple"  # cells dropped entirely before training

    # Perturbation frequency filter
    # Perturbation classes with fewer than this many cells are excluded from
    # training (their cells are dropped).  Prevents near-empty embeddings for
    # singleton or very-rare guides.  0 = no filtering.
    min_cells_per_pert: int = 10

    # Negative-control label shuffle (ablation)
    # If > 0, permute the perturbation labels of the top-N real-KO classes
    # (by post-filter cell count) AMONG those cells only. Marginal class counts
    # are preserved but the cell<->label link is destroyed, so the model has no
    # real perturbation signal to learn for those N genes. 0 = disabled.
    shuffle_top_perts: int = 0

    # GP / Inducing points
    fix_inducing_points: bool = True
    grid_inducing_points: bool = True
    inducing_point_steps: int = 6
    inducing_point_nums: int | None = None
    # If set (grid mode), derive grid resolution from data density instead of a fixed
    # inducing_point_steps: steps = round(sqrt(N / cells_per_inducing)), so the number of
    # inducing points scales with the cell count (M = steps^2 ~= N / cells_per_inducing).
    cells_per_inducing: int | None = None
    inducing_steps_min: int = 6
    inducing_steps_max: int = 32
    # Multi-slide: tile each slide's coords into a disjoint region so the GP finds NO
    # cross-slide neighbors (slide locations are not aligned). Kernel params stay shared;
    # only the geometry is separated. Tile origin spacing = loc_range * slide_tile_gap_factor
    # (must keep coords float32-safe, so do not set absurdly large).
    slide_tile_gap_factor: float = 2.5
    fixed_gp_params: bool = False
    multi_kernel_mode: bool = True
    kernel_scale: float = 10.0
    loc_range: float = 20.0

    # Runtime
    device: str = "cuda"
    seed: int | None = None
    verbosity: int = 1
    resume: bool = False  # if True, load existing model_file weights before training

    # Counterfactual
    pert_cells: str = "patch_jak2.txt"  # 1-based indices file
    target_cell_tissue: str = "tumor"   # only used for naming by default
    target_cell_perturbation: str = "Jak2"

    # Weights & Biases
    wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_run: Optional[str] = None
    wandb_mode: str = "online"  # online|offline|disabled


# Utilities
def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _remap_slide_bias(model, ckpt_path: str, multi_paths, cfg) -> None:
    """Row-align the checkpoint's per-slide bias to THIS eval's slice list.

    `load_model` front-truncates a 52-slide checkpoint down to an N-slide eval model, which
    silently pairs eval slice i with TRAINING slide i -- a different tissue whenever the eval
    runs on a subset (held-out-mouse eval, imputation dump). The bias is a per-gene additive
    log-shift, so a mismatched row rescales predicted expression (observed 2-4x pseudobulk
    inflation on the DSS imputation dump).

    With --slide_bias_ref_files (the training file list, training order) each eval slice gets
    its OWN trained row, matched by filename. Slices the checkpoint never saw (a genuinely
    held-out mouse) get the training mean row -- neutral, instead of an arbitrary other slide.
    No-op when the flag is empty or the model has no slide_bias.
    """
    ref = [p.strip() for p in str(getattr(cfg, "slide_bias_ref_files", "") or "").split(",") if p.strip()]
    if not ref or getattr(model, "slide_bias", None) is None:
        return
    sb = torch.load(ckpt_path, map_location="cpu").get("slide_bias")
    if sb is None:
        return
    if len(ref) != sb.shape[0]:
        # The reference list IS the checkpoint's row order; a length mismatch means the wrong
        # list was passed (e.g. the full manifest against a leave-mice-out ckpt) and every row
        # would be off by the excluded slices. Fail loudly rather than mis-map silently.
        raise ValueError(f"--slide_bias_ref_files lists {len(ref)} slices but the checkpoint's "
                         f"slide_bias has {sb.shape[0]} rows -- pass the TRAINING file list.")
    ref_idx = {Path(p).name: i for i, p in enumerate(ref)}
    mean_row = sb.mean(0)
    rows, n_seen = [], 0
    for p in multi_paths:
        i = ref_idx.get(Path(p).name)
        if i is not None and i < sb.shape[0]:
            rows.append(sb[i]); n_seen += 1
        else:
            rows.append(mean_row)
    new = torch.stack(rows).to(model.slide_bias.device, dtype=model.slide_bias.dtype)
    if new.shape != model.slide_bias.shape:
        raise RuntimeError(f"slide_bias remap shape {tuple(new.shape)} != model "
                           f"{tuple(model.slide_bias.shape)}")
    with torch.no_grad():
        model.slide_bias.copy_(new)
    logging.info("[slide_bias] remapped %d/%d eval slices to their own trained rows "
                 "(%d unseen -> training mean of %d rows)",
                 n_seen, len(multi_paths), len(multi_paths) - n_seen, sb.shape[0])


def auto_batch_size(n: int) -> int:
    """Return a sensible batch size for H100 (80 GB) training."""
    if n <= 1024:
        return 256
    if n <= 8192:
        return 512
    return 2048  # H100 can comfortably handle this even for 10k-gene decoder


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


def _derive_grid_steps(n_cells: int, cfg) -> tuple:
    """Grid steps from data density if cells_per_inducing set, else fixed inducing_point_steps.
    Returns (steps_clipped, steps_preclip)."""
    if cfg.cells_per_inducing and cfg.cells_per_inducing > 0:
        derived = int(round((n_cells / cfg.cells_per_inducing) ** 0.5))
        return int(np.clip(derived, cfg.inducing_steps_min, cfg.inducing_steps_max)), derived
    return cfg.inducing_point_steps, cfg.inducing_point_steps


def tile_multislide_coords(pos_raw, sample_id, loc_range, gap_factor):
    """Per-slide MinMax-scale to [0, loc_range], then offset each slide into a disjoint tile
    (origin spacing = loc_range * gap_factor) so the GP finds NO cross-slide neighbors --
    slide locations are not aligned, so they must not share a coordinate frame. The kernel
    (params shared) still computes distance only on the first 2 cols, now tiled.
    Returns tiled (N,2) coords, a layout list of (sid,row,col,n_cells), and the tile spacing."""
    sids = np.unique(sample_id)
    ncol = int(np.ceil(np.sqrt(len(sids))))
    spacing = float(loc_range) * float(gap_factor)
    pos_scaled = np.zeros((pos_raw.shape[0], 2), dtype=np.float32)
    layout = []
    for k, sid in enumerate(sids):
        m = sample_id == sid
        sc = MinMaxScaler().fit_transform(pos_raw[m]) * loc_range
        r, c = divmod(k, ncol)
        sc[:, 0] += c * spacing
        sc[:, 1] += r * spacing
        pos_scaled[m] = sc.astype(np.float32)
        layout.append((int(sid), int(r), int(c), int(m.sum())))
    return pos_scaled, layout, spacing


def build_tiled_inducing(pos_scaled, sample_id, n_batch, cfg):
    """Build a density-scaled inducing grid INSIDE each slide's tile (pos_scaled already tiled)
    and concatenate. Per-slide grids sit only where that slide's cells are, so a cell loads on
    its own tile's inducing points (no cross-slide pooling). One-hot col -> shared kernel."""
    pts = []
    for sid in np.unique(sample_id):
        m = sample_id == sid
        p = pos_scaled[m]
        steps, _ = _derive_grid_steps(int(m.sum()), cfg)
        xs = np.linspace(p[:, 0].min(), p[:, 0].max(), steps)
        ys = np.linspace(p[:, 1].min(), p[:, 1].max(), steps)
        gx, gy = np.meshgrid(xs, ys)
        pts.append(np.stack([gx.ravel(), gy.ravel()], axis=1))
    grid = np.concatenate(pts, axis=0).astype(np.float32)
    onehot = np.zeros((grid.shape[0], n_batch), dtype=np.float32)
    onehot[:, 0] = 1.0
    return np.concatenate([grid, onehot], axis=1).astype(np.float32)


def load_h5_dataset(
    path: str,
    unassigned_labels: Optional[frozenset] = None,
    control_labels: Optional[frozenset] = None,
    hvg_max_rank: int = 10000,
    force_gene_order: Optional[np.ndarray] = None,
    arch_dir: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray],
           Optional[np.ndarray], Optional[np.ndarray]]:
    """Load data from either a CONCERT .h5 file or an AnnData .h5ad file.

    Auto-detects format by file extension.

    Returns
    -------
    X           : (N, G) float32       -- raw counts (HVG-subset for .h5ad)
    pos         : (N, 2) float32       -- spatial coordinates [col, row]
    tissue_raw  : (N,) str or None
    perturb_raw : (N,) str
    has_guide   : (N,) bool            -- True for control + gene_KO
    is_real_KO  : (N,) bool            -- True for gene_KO only
    gene_names  : (G,) str or None     -- HVG gene names (None for legacy .h5)
    tier1_raw   : (N,) str or None     -- Tier1 lineage label (timekernel; None for legacy .h5)
    arch_raw    : (N, D) float32 or None -- raw architectural descriptor (timekernel; None for legacy .h5)
    """
    if path.endswith(".h5ad"):
        return _load_h5ad_dataset(path, unassigned_labels=unassigned_labels,
                                   control_labels=control_labels, hvg_max_rank=hvg_max_rank,
                                   force_gene_order=force_gene_order, arch_dir=arch_dir)
    # Legacy .h5: no 3-class taxonomy -- treat all cells as gene_KO (backward compat)
    # No Tier1/arch descriptor for legacy .h5 files (timekernel not supported for this format).
    X, pos, tissue_raw, perturb_raw = _load_h5_dataset(path)
    has_guide  = np.ones(len(perturb_raw), dtype=bool)
    is_real_KO = np.ones(len(perturb_raw), dtype=bool)
    return X, pos, tissue_raw, perturb_raw, has_guide, is_real_KO, None, None, None


def _load_h5_dataset(path: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Original CONCERT .h5 loader."""
    with h5py.File(path, "r") as f:
        X = np.array(f["X"], dtype=np.float32)
        pos = np.array(f["pos"], dtype=np.float32).T  # to (N, 2)
        perturb_raw = np.array(f["perturbation"], dtype=str)
        tissue_raw = None
        if "tissue" in f:
            tissue_raw = np.array(f["tissue"], dtype=str)
    return X, pos, tissue_raw, perturb_raw


def _read_obs_str_column(f_obs, col: str) -> np.ndarray:
    """Read an obs column from an open HDF5 obs group -> numpy str array.
    Handles categorical (codes + categories), plain string, and plain numeric storage.
    """
    import h5py as _h5py
    obj = f_obs[col]
    # Categorical: stored as a Group with 'codes' and 'categories' datasets
    if isinstance(obj, _h5py.Group) and "codes" in obj and "categories" in obj:
        codes = np.array(obj["codes"][:], dtype=int)
        cats  = np.array([
            c.decode() if isinstance(c, bytes) else str(c)
            for c in obj["categories"][:]
        ])
        return cats[codes]
    # Plain dataset (string or numeric)
    raw = np.array(obj[:])
    if raw.dtype.kind in ("S", "O", "U"):
        return np.array([
            v.decode() if isinstance(v, bytes) else str(v)
            for v in raw
        ])
    # Numeric (e.g. filter_pass as 0/1 integers)
    return raw.astype(str)


def _load_arch_sidecar(f_obs, path: str, arch_dir: str, n_obs_full: int) -> np.ndarray:
    """Load this slice's day-invariant architectural descriptor sidecar.

    Contract (arch-descriptor design Sec. 5): `<arch_dir>/<Slice_ID>.npy` is float32
    (n_cells, arch_dim) with rows in the source h5ad's ON-DISK cell order -- i.e. the
    PRE-filter order. The caller must therefore apply exactly the same row filters to it
    as to tier1_raw/perturb_raw. A silent misalignment here is fatal (every cell gets
    another cell's architecture), so both the row count and the declared width are asserted
    against the manifest rather than trusted.
    """
    arch_dir_p = Path(arch_dir)
    manifest_p = arch_dir_p / "manifest.json"
    if not manifest_p.is_file():
        raise FileNotFoundError(
            f"arch descriptor manifest not found: {manifest_p}. use_timekernel needs the "
            f"descriptor sidecars (see docs/2026-07-16-arch-descriptor-design.md Sec. 5).")
    with open(manifest_p) as fh:
        manifest = json.load(fh)
    # precompute_arch_descriptor.py writes the column list as "columns"; accept the
    # "column_names" spelling too so either producer version aligns.
    _cols = manifest.get("column_names") or manifest.get("columns")
    if _cols is None:
        raise KeyError(
            f"{manifest_p}: no 'columns'/'column_names' key; cannot verify descriptor width. "
            f"Keys present: {sorted(manifest)}")
    expected_dim = len(_cols)

    slice_ids = _read_obs_str_column(f_obs, "Slice_ID")
    uniq = set(slice_ids.tolist())
    if len(uniq) != 1:
        raise ValueError(
            f"{path}: expected exactly one Slice_ID per file, found {len(uniq)} ({sorted(uniq)[:5]}); "
            f"the arch sidecar is keyed by Slice_ID and cannot be aligned.")
    slice_id = slice_ids[0]

    npy_p = arch_dir_p / f"{slice_id}.npy"
    if not npy_p.is_file():
        raise FileNotFoundError(f"arch descriptor sidecar missing for Slice_ID={slice_id}: {npy_p}")
    arch = np.load(npy_p).astype(np.float32)

    if arch.ndim != 2:
        raise ValueError(f"{npy_p}: expected 2-D (n_cells, arch_dim), got shape {arch.shape}")
    # Row count MUST match the PRE-filter n_obs -- the sidecar is in on-disk cell order.
    if arch.shape[0] != n_obs_full:
        raise ValueError(
            f"{npy_p}: sidecar has {arch.shape[0]} rows but {path} has {n_obs_full} cells on disk "
            f"(pre-filter). The descriptor is not row-aligned to the h5ad; refusing to guess.")
    if arch.shape[1] != expected_dim:
        raise ValueError(
            f"{npy_p}: arch_dim {arch.shape[1]} != manifest column_names width {expected_dim}")
    if not np.isfinite(arch).all():
        raise ValueError(f"{npy_p}: descriptor contains non-finite values")
    return arch


def _load_h5ad_dataset(
    path: str,
    unassigned_labels: Optional[frozenset] = None,
    control_labels: Optional[frozenset] = None,
    hvg_max_rank: int = 10000,
    force_hvg_mask: Optional[np.ndarray] = None,
    force_gene_order: Optional[np.ndarray] = None,
    arch_dir: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray],
           Optional[np.ndarray], Optional[np.ndarray]]:
    """Load SPAC-seq .h5ad OOM-safe: X stays sparse until after HVG + filter_pass masking.

    Strategy
    --------
    1. Read filter_pass obs column -> boolean row mask.
    2. Read highly_variable var column -> boolean col mask (HVGs).
    3. Read X as sparse CSR (never densify the full 632k x 20k matrix).
    4. Apply row mask -> apply col mask (both stay sparse).
    5. Densify only the reduced (N_pass x G_hvg) submatrix.

    Returns
    -------
    X           : (N_pass, G_hvg) float32 dense
    pos         : (N_pass, 2)     float32 -- [array_col, array_row]
    tissue_raw  : None            (no tissue labels in SPAC-seq h5ad)
    perturb_raw : (N_pass,)       str
    has_guide   : (N_pass,)       bool -- True for control + gene_KO cells
    is_real_KO  : (N_pass,)       bool -- True for gene_KO only
    gene_names  : (G_hvg,)        str  -- HVG gene names (or None)
    tier1_raw   : (N_pass,)       str or None -- Tier1 lineage label (timekernel)
    arch_raw    : (N_pass, D)     float32 or None -- day-invariant architectural descriptor,
                                  RAW (unstandardised); None when arch_dir is not given.
                                  Occupies the slot Leiden_neigh used to (arch-descriptor Sec. 4).
    """
    import scipy.sparse as sp_scipy

    if unassigned_labels is None:
        unassigned_labels = frozenset({
            "unassigned", "Unassigned", "No_guide", "no_guide", "unknown",
        })
    if control_labels is None:
        # "multiple" = multi-guide cells: has_guide=True, is_real_KO=False
        control_labels = frozenset({
            "Non_targeting", "non_targeting", "NonTargeting",
            "control", "Control", "CTRL", "ctrl", "NT",
            "multiple", "Multiple",
        })

    logging.info("[load] Opening h5ad (sparse, lazy): %s", path)
    with h5py.File(path, "r") as f:

        # perturbation labels
        perturb_raw_full = _read_obs_str_column(f["obs"], "perturbation")
        n_obs_full = len(perturb_raw_full)

        # --- tissue labels (optional; present for perturbmap, absent for spac) -
        tissue_raw_full = None
        if "tissue" in f["obs"]:
            tissue_raw_full = _read_obs_str_column(f["obs"], "tissue")

        # Tier1 lineage (timekernel basal identity)
        tier1_raw_full = None
        if "Tier1" in f["obs"]:
            tier1_raw_full = _read_obs_str_column(f["obs"], "Tier1")
        # --- day-invariant architectural descriptor sidecar (replaces Leiden_neigh, which
        #    was a day proxy and let the time-invariant basal memorise day-9 inflammation).
        #    Rows are in ON-DISK order here and are filtered below exactly like tier1_raw.
        arch_raw_full = None
        if arch_dir:
            arch_raw_full = _load_arch_sidecar(f["obs"], path, arch_dir, n_obs_full)

        # spatial coordinates
        pos_full = np.array(f["obsm"]["array_coords"][:], dtype=np.float32)

        # filter_pass row mask
        filter_pass_mask = None
        if "filter_pass" in f["obs"]:
            fp_str = _read_obs_str_column(f["obs"], "filter_pass")
            filter_pass_mask = np.array([
                s.lower() in {"1", "true", "pass", "yes"}
                for s in fp_str
            ], dtype=bool)
            logging.info("[load] filter_pass: %d / %d spots pass QC",
                         filter_pass_mask.sum(), n_obs_full)
        else:
            logging.warning("[load] No 'filter_pass' in obs; keeping all %d spots.", n_obs_full)

        # HVG column mask from var
        # If force_hvg_mask is given (multi-sample mode), use it directly.
        # Otherwise: var['hvg_rank'] (1-based, lower = more variable) -> top-N;
        #   fallback: var['highly_variable'] bool; last-resort: all genes.
        gene_names_full = None
        if "var" in f and "_index" in f["var"]:
            gene_names_full = np.array([
                g.decode() if isinstance(g, bytes) else str(g)
                for g in f["var"]["_index"][:]
            ])
        if force_hvg_mask is not None:
            hvg_mask = force_hvg_mask
            logging.info("[load] HVG mask (pre-computed, shared): %d / %d genes selected",
                         hvg_mask.sum(), len(hvg_mask))
        elif "var" in f and "hvg_rank" in f["var"]:
            ranks    = np.array(f["var"]["hvg_rank"][:], dtype=np.int32)
            hvg_mask = ranks <= hvg_max_rank
            logging.info("[load] hvg_rank <= %d -> %d / %d genes selected",
                         hvg_max_rank, hvg_mask.sum(), len(hvg_mask))
        elif "var" in f and "highly_variable" in f["var"]:
            hv = f["var"]["highly_variable"]
            if isinstance(hv, h5py.Group) and "codes" in hv:
                hv_codes = np.array(hv["codes"][:], dtype=int)
                hv_cats  = np.array(hv["categories"][:])
                hvg_mask = np.array([bool(hv_cats[c]) for c in hv_codes], dtype=bool)
            else:
                hvg_mask = np.array(hv[:], dtype=bool)
            logging.info("[load] highly_variable: %d / %d genes selected",
                         hvg_mask.sum(), len(hvg_mask))
        else:
            hvg_mask = None
            logging.warning("[load] No hvg_rank or highly_variable in var -- using all %d genes!",
                            len(gene_names_full) if gene_names_full is not None else 0)

        # sparse count matrix
        x_grp   = f["X"]
        data    = np.array(x_grp["data"][:],    dtype=np.float32)
        indices = np.array(x_grp["indices"][:], dtype=np.int32)
        indptr  = np.array(x_grp["indptr"][:],  dtype=np.int32)
        n_obs   = len(indptr) - 1
        n_vars  = (len(gene_names_full) if gene_names_full is not None
                   else (int(indices.max()) + 1 if len(indices) > 0 else 0))
        X_sp = sp_scipy.csr_matrix((data, indices, indptr), shape=(n_obs, n_vars))

    # Apply filter_pass row mask (stays sparse)
    if filter_pass_mask is not None:
        X_sp          = X_sp[filter_pass_mask]
        perturb_raw   = perturb_raw_full[filter_pass_mask]
        pos           = pos_full[filter_pass_mask]
        tissue_raw    = (tissue_raw_full[filter_pass_mask]
                         if tissue_raw_full is not None else None)
        tier1_raw     = (tier1_raw_full[filter_pass_mask]
                         if tier1_raw_full is not None else None)
        # Same row mask as tier1_raw -- the descriptor is (n_obs_full, D) in on-disk order.
        arch_raw      = (arch_raw_full[filter_pass_mask]
                         if arch_raw_full is not None else None)
    else:
        perturb_raw   = perturb_raw_full
        pos           = pos_full
        tissue_raw    = tissue_raw_full
        tier1_raw     = tier1_raw_full
        arch_raw      = arch_raw_full

    # Column selection
    # force_gene_order (eval on a model trained with a DIFFERENT HVG set, e.g. a
    # multi-slide model whose avg-rank genes != this slide's per-slide HVG rank):
    # subset AND reorder to the model's exact training gene list so the decoder
    # output and the model input align gene-for-gene. Otherwise use the HVG mask.
    if force_gene_order is not None:
        if gene_names_full is None:
            raise ValueError("force_gene_order requires gene names in var['_index']")
        name_to_col = {g: i for i, g in enumerate(gene_names_full)}
        missing = [g for g in force_gene_order if g not in name_to_col]
        if missing:
            raise ValueError(f"{len(missing)}/{len(force_gene_order)} model genes absent from "
                             f"{path} var (e.g. {missing[:5]}); cannot align eval to this checkpoint")
        cols = np.array([name_to_col[g] for g in force_gene_order], dtype=int)
        X_sp       = X_sp[:, cols]
        gene_names = np.asarray(force_gene_order)
        logging.info("[load] force_gene_order: subset+reordered to %d model genes", len(gene_names))
    elif hvg_mask is not None:
        X_sp       = X_sp[:, hvg_mask]
        gene_names = gene_names_full[hvg_mask] if gene_names_full is not None else None
    else:
        gene_names = gene_names_full

    # Densify only the reduced submatrix
    X = X_sp.toarray().astype(np.float32)
    logging.info("[load] Final matrix: %d spots x %d genes", X.shape[0], X.shape[1])

    # Derive 3-class indicators
    # has_guide : 1 for control + gene_KO (any guide delivered)
    # is_real_KO: 1 for gene_KO only (not unassigned, not non-targeting control)
    has_guide  = np.array([p not in unassigned_labels for p in perturb_raw], dtype=bool)
    is_real_KO = np.array([
        p not in unassigned_labels and p not in control_labels
        for p in perturb_raw
    ], dtype=bool)

    logging.info(
        "[load] 3-class: unassigned=%d  control=%d  gene_KO=%d",
        (~has_guide).sum(),
        (has_guide & ~is_real_KO).sum(),
        is_real_KO.sum(),
    )

    return X, pos, tissue_raw, perturb_raw, has_guide, is_real_KO, gene_names, tier1_raw, arch_raw


def load_h5ad_multi(
    paths: List[str],
    unassigned_labels: Optional[frozenset] = None,
    control_labels: Optional[frozenset] = None,
    hvg_max_rank: int = 10000,
    arch_dir: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray],
           np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Load multiple SPAC-seq .h5ad files and concatenate for joint training.

    Strategy
    --------
    1. Pass 1 (metadata only): read hvg_rank arrays from all samples to build a
       shared gene set using **average-rank aggregation**.
       * For each gene, compute the mean hvg_rank across samples.
       * Select the top `hvg_max_rank` genes by mean rank (lowest = most variable).
       * This yields ~hvg_max_rank genes that are consistently highly variable,
         rather than the strict intersection (~3.7k with 4 samples).
    2. Pass 2: load each sample with `force_hvg_mask=shared_hvg_mask` (one read
       per file, no double-load).
    3. Concatenate (X, pos, perturb_raw, has_guide, is_real_KO) and return a
       per-cell sample_id array.

    Returns the same 7-tuple as `load_h5_dataset`, plus an extra `sample_id`, plus the
    per-cell Tier1 labels and the raw architectural descriptor (timekernel; None if any
    sample lacks them).
    Caller unpacks as: X, pos, _, perturb_raw, has_guide, is_real_KO, gene_names, sample_id,
    tier1_raw, arch_raw

    Parameters
    ----------
    paths : list of str -- paths to h5ad files (in order; index = sample_id)
    """
    if unassigned_labels is None:
        unassigned_labels = frozenset({
            "unassigned", "Unassigned", "No_guide", "no_guide", "unknown",
        })
    if control_labels is None:
        control_labels = frozenset({
            "Non_targeting", "non_targeting", "NonTargeting",
            "control", "Control", "CTRL", "ctrl", "NT",
            "multiple", "Multiple",
        })

    logging.info("[multi-load] Loading %d samples with shared HVG set (avg-rank) ...", len(paths))

    # Pass 1: collect hvg_rank arrays (var only, no X)
    all_ranks: List[Optional[np.ndarray]] = []
    gene_names_ref: Optional[np.ndarray] = None   # gene list from first sample
    in_panel_ref: Optional[np.ndarray] = None     # benchmark KO-panel genes to force-keep

    for path in paths:
        with h5py.File(path, "r") as f:
            if gene_names_ref is None and "var" in f and "_index" in f["var"]:
                gene_names_ref = np.array([
                    g.decode() if isinstance(g, bytes) else str(g)
                    for g in f["var"]["_index"][:]
                ])
            if in_panel_ref is None and "var" in f and "in_panel_1053" in f["var"]:
                in_panel_ref = np.asarray(f["var"]["in_panel_1053"][:], dtype=bool)
            if "var" in f and "hvg_rank" in f["var"]:
                ranks = np.array(f["var"]["hvg_rank"][:], dtype=np.float32)
                all_ranks.append(ranks)
            else:
                all_ranks.append(None)
                logging.warning("[multi-load] %s missing hvg_rank!", Path(path).stem)

    # Build shared HVG mask via average-rank aggregation
    if all(r is not None for r in all_ranks):
        rank_matrix  = np.stack(all_ranks, axis=0)          # (n_samples, G)
        avg_rank     = rank_matrix.mean(axis=0)              # (G,)
        sorted_idx   = np.argsort(avg_rank)                  # ascending: low rank = more variable
        n_select     = min(hvg_max_rank, len(avg_rank))
        shared_hvg_mask = np.zeros(len(avg_rank), dtype=bool)
        shared_hvg_mask[sorted_idx[:n_select]] = True

        # Diagnostics: how many shared genes are in each sample's top-N individually
        for i, r in enumerate(all_ranks):
            n_overlap = int((r[shared_hvg_mask] <= hvg_max_rank).sum())
            logging.info("[multi-load]   Sample %d: %d/%d shared genes in own top-%d",
                         i, n_overlap, shared_hvg_mask.sum(), hvg_max_rank)
        logging.info("[multi-load] Shared HVG mask (avg-rank top-%d): %d genes selected",
                     hvg_max_rank, shared_hvg_mask.sum())
    else:
        # Fallback: use first sample's per-sample mask (or all genes)
        shared_hvg_mask = None
        logging.warning("[multi-load] Cannot build avg-rank mask -- falling back to per-sample ranks!")

    # Force-include the benchmark KO panel (in_panel_1053) so the model predicts EVERY
    # panel gene the eval scores. Without this the variance-only top-N HVG selection only
    # overlaps ~793/1053 panel genes, so CONCERT is scored on a different gene set than the
    # competitors (who score all 1053) -- the concert-vs-competitor gene-set mismatch.
    if shared_hvg_mask is not None and in_panel_ref is not None:
        n_before = int(shared_hvg_mask.sum())
        shared_hvg_mask = shared_hvg_mask | in_panel_ref
        logging.info("[multi-load] force-included in_panel_1053: %d -> %d genes (+%d panel-only)",
                     n_before, int(shared_hvg_mask.sum()), int(shared_hvg_mask.sum()) - n_before)

    gene_names_out = (gene_names_ref[shared_hvg_mask]
                      if (gene_names_ref is not None and shared_hvg_mask is not None)
                      else gene_names_ref)

    # --- Pass 2: load each sample with the shared mask (single read per file) -
    Xs, poss, perts_list, hg_list, rk_list, sid_list, tis_list = [], [], [], [], [], [], []
    tier1_list, arch_list = [], []

    for s_idx, path in enumerate(paths):
        X_s, pos_s, tis_s, pert_s, hg_s, rk_s, _, tier1_s, arch_s = _load_h5ad_dataset(
            path,
            unassigned_labels=unassigned_labels,
            control_labels=control_labels,
            hvg_max_rank=hvg_max_rank,
            force_hvg_mask=shared_hvg_mask,
            arch_dir=arch_dir,
        )
        Xs.append(X_s)
        poss.append(pos_s)
        perts_list.append(pert_s)
        hg_list.append(hg_s)
        rk_list.append(rk_s)
        tis_list.append(tis_s)
        tier1_list.append(tier1_s)
        arch_list.append(arch_s)
        sid_list.append(np.full(len(pert_s), s_idx, dtype=np.int32))
        logging.info("[multi-load] Sample %d (%s): %d cells x %d genes",
                     s_idx, Path(path).stem, X_s.shape[0], X_s.shape[1])

    X_all          = np.concatenate(Xs,         axis=0)
    pos_all        = np.concatenate(poss,        axis=0)
    perturb_all    = np.concatenate(perts_list,  axis=0)
    has_guide_all  = np.concatenate(hg_list,     axis=0)
    is_real_KO_all = np.concatenate(rk_list,     axis=0)
    sample_id_all  = np.concatenate(sid_list,    axis=0)
    # tissue axis: keep only if EVERY sample carries it (perturbmap); else None (spac)
    if all(t is not None for t in tis_list):
        tissue_all = np.concatenate(tis_list, axis=0)
        logging.info("[multi-load] tissue axis present on all %d samples: %d classes %s",
                     len(paths), len(set(tissue_all.tolist())), sorted(set(tissue_all.tolist())))
    else:
        tissue_all = None

    # Tier1 / arch-descriptor axes (timekernel basal identity): keep only if EVERY sample
    # carries them; None otherwise. (Required-presence enforcement for use_timekernel
    # happens in run(), which is the only place that knows whether timekernel is on --
    # this loader is also used when timekernel is off, so it must not hard-fail here.)
    if all(t is not None for t in tier1_list):
        tier1_all = np.concatenate(tier1_list, axis=0)
        logging.info("[multi-load] Tier1 axis present on all %d samples: %d classes %s",
                     len(paths), len(set(tier1_all.tolist())), sorted(set(tier1_all.tolist())))
    else:
        tier1_all = None
        logging.warning("[multi-load] Tier1 axis missing from >=1 sample -- unavailable "
                        "for timekernel (%d/%d samples had it)",
                        sum(t is not None for t in tier1_list), len(paths))
    if all(a is not None for a in arch_list):
        widths = {a.shape[1] for a in arch_list}
        if len(widths) != 1:
            raise ValueError(f"[multi-load] arch descriptor width differs across slices: {widths}")
        arch_all = np.concatenate(arch_list, axis=0).astype(np.float32)
        logging.info("[multi-load] arch descriptor present on all %d samples: %s",
                     len(paths), (arch_all.shape,))
    else:
        arch_all = None
        logging.warning("[multi-load] arch descriptor missing from >=1 sample -- unavailable "
                        "for timekernel (%d/%d samples had it)",
                        sum(a is not None for a in arch_list), len(paths))

    logging.info("[multi-load] Total: %d cells x %d genes across %d samples",
                 X_all.shape[0], X_all.shape[1], len(paths))
    return (X_all, pos_all, tissue_all, perturb_all, has_guide_all, is_real_KO_all, gene_names_out,
            sample_id_all, tier1_all, arch_all)


def build_attributes(perturb_raw: np.ndarray):
    """
    Legacy attribute builder (old tumor-model data with Jak2/Tgfbr2/KP labels).
    Kept for backward compatibility with .h5 files.
    For SPAC-seq .h5ad data, use build_attributes_spac() instead.
    """
    known_types = ["Jak2", "Tgfbr2", "Ifngr2", "KP", "normal", "periphery"]
    known_tumor = ["Jak2", "Tgfbr2", "Ifngr2", "KP"]
    known_normal = "normal"
    known_periphery = "periphery"

    tissue_raw = np.full_like(perturb_raw, known_normal, dtype=object)
    tissue_raw = np.where(np.isin(perturb_raw, known_tumor), "tumor", tissue_raw)
    tissue_raw = np.where(perturb_raw == known_periphery, "periphery", tissue_raw)
    tissue_idx = strings_to_index(tissue_raw)

    present = [p for p in known_types if p in set(perturb_raw.tolist())]
    pert_map = {p: i + 1 for i, p in enumerate(present)}
    perturb_idx = np.vectorize(lambda s: pert_map.get(s, 0))(perturb_raw).astype(int)

    tissue_dict = {t: i for t, i in zip(tissue_raw, tissue_idx)}
    return tissue_idx, tissue_dict, perturb_idx, pert_map


BASAL_PERT_KEY = "__basal__"   # reserved perturbation code 0: basal/control (no f_pert)


def build_attributes_spac(
    perturb_raw: np.ndarray,
    tissue_raw: Optional[np.ndarray] = None,
    unassigned_labels: Optional[frozenset] = None,
    control_labels: Optional[frozenset] = None,
) -> Tuple[np.ndarray, dict, np.ndarray, dict]:
    """Generic attribute builder for SPAC-seq / perturbmap .h5ad data.

    tissue_idx : from `tissue_raw` if provided (e.g. perturbmap tumor/normal/periphery ->
                 codes for tissue-axis disentanglement in LORD); else all zeros
                 (spac has no tissue annotation).
    perturb_idx: REAL perturbations only get a learned embedding. Non-perturbation
                 cells -- unassigned (basal) and control/NTC -- are mapped to a single
                 RESERVED code 0 (`BASAL_PERT_KEY`); real KO labels are sorted -> 1..K.
                 This keeps the perturbation vocabulary == #real KO (+1 basal slot), so
                 basal/NTC never get a phantom perturbation footprint. f_pert is gated by
                 is_real_KO (concert_spac.py) and mu_action is a global gene bias, so the
                 basal slot's embedding is never applied to reconstruction.

    Returns
    -------
    tissue_idx  : (N,) int    -- tissue codes (or all zeros)
    tissue_dict : dict        -- {tissue_label -> code}
    perturb_idx : (N,) int    -- integer codes for each cell's perturbation (0 = basal slot)
    pert_map    : dict[str->int] -- label -> integer code mapping (incl. BASAL_PERT_KEY->0)
    """
    if tissue_raw is not None:
        unique_tissues = sorted(set(np.asarray(tissue_raw).tolist()))
        tissue_dict = {t: i for i, t in enumerate(unique_tissues)}
        tissue_idx  = np.array([tissue_dict[t] for t in np.asarray(tissue_raw)], dtype=int)
        logging.info("[attrs] %d tissue classes: %s", len(unique_tissues), tissue_dict)
    else:
        tissue_idx  = np.zeros(len(perturb_raw), dtype=int)
        tissue_dict = {"tissue_0": 0}

    non_pert = frozenset(unassigned_labels or ()) | frozenset(control_labels or ())
    real_perts = sorted(p for p in set(perturb_raw.tolist()) if p not in non_pert)
    # code 0 reserved for basal/control; real KO genes -> 1..K
    pert_map = {BASAL_PERT_KEY: 0}
    for i, p in enumerate(real_perts, start=1):
        pert_map[p] = i
    perturb_idx = np.array([pert_map.get(p, 0) for p in perturb_raw], dtype=int)

    logging.info("[attrs] %d real perturbations (+1 basal slot); %d cells in basal slot",
                 len(real_perts), int((perturb_idx == 0).sum()))
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


# Orchestration
# DSS inflammation marker panel (Cadinu et al., Cell 2024) -- same panel used to derive the
# perturbation label in process_dryad_merfish_dss.py. Used here as a CONTINUOUS gene-score target.
_DSS_MARKERS = ['Isg15', 'Ifit1', 'Tnf', 'Cxcl1', 'Nos2', 'Il11', 'Il1b', 'Mmp3', 'Mmp10',
                'Mmp13', 'Plau', 'Igfbp5', 'Col18a1', 'Mmp2', 'S100a8', 'S100a9', 'Osm',
                'Cxcl10', 'Cxcl5', 'Ly6a']


def _marker_idx_on_model_axis(gene_names) -> list:
    """Positions of the DSS marker panel on the MODEL's gene axis (the HVG-selected
    `gene_names`), NOT on adata.raw's full panel -- the decoder emits one column per
    model gene, so `shift` must be indexed in that space.
    """
    if gene_names is None:
        raise ValueError("marker_sign_positive=True needs gene names; none were loaded "
                         "(the h5ad must carry var['gene'] / var_names)")
    names = [str(g) for g in gene_names]
    pos = {g: i for i, g in enumerate(names)}
    idx = [pos[g] for g in _DSS_MARKERS if g in pos]
    missing = [g for g in _DSS_MARKERS if g not in pos]
    if len(idx) == 0:
        raise ValueError("marker_sign_positive=True but none of the %d DSS marker genes "
                         "are on the model gene axis" % len(_DSS_MARKERS))
    logging.info("[marker-sign] constraining %d/%d DSS marker genes >= 0 on the model axis"
                 "%s", len(idx), len(_DSS_MARKERS),
                 (" (absent: %s)" % ", ".join(missing)) if missing else "")
    return idx


def _compute_inflammation_score(adata) -> np.ndarray:
    """Per-cell continuous inflammation gene-score s_hat_i in [0,1] from the DSS marker panel.

    Honest gene state (not a threshold label): for each present marker gene, take
    log1p(raw_count / size_factor), z-score across cells, average over the panel, then
    [0,1]-scale via robust 1st/99th percentiles. Computed on the raw-count matrix aligned
    to the training cell order (adata.raw.X). Non-negative to match the softplus field range.
    """
    raw = adata.raw.X if adata.raw is not None else adata.X
    raw = raw.toarray() if hasattr(raw, "toarray") else np.asarray(raw)
    raw = raw.astype(np.float32)
    var_names = list(adata.raw.var_names if adata.raw is not None else adata.var_names)
    sf = adata.obs["size_factors"].to_numpy(dtype=np.float32)[:, None]
    idx = [var_names.index(g) for g in _DSS_MARKERS if g in var_names]
    if len(idx) == 0:
        logging.warning("[infl] no DSS marker genes present; inflammation score = zeros")
        return np.zeros(raw.shape[0], dtype=np.float32)
    sub = np.log1p(raw[:, idx] / np.clip(sf, 1e-8, None))          # (N, n_markers) normalized log-expr
    z = (sub - sub.mean(0, keepdims=True)) / (sub.std(0, keepdims=True) + 1e-8)
    score = z.mean(1)                                             # (N,) mean z-scored marker program
    lo, hi = np.percentile(score, [1.0, 99.0])
    s01 = np.clip((score - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    logging.info("[infl] %d/%d marker genes present", len(idx), len(_DSS_MARKERS))
    return s01.astype(np.float32)


def run(cfg: RunConfig, rank: int = 0, world_size: int = 1) -> None:
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

    # Parse SPAC-seq taxonomy label sets
    unassigned_labels = frozenset(
        l.strip() for l in cfg.unassigned_label.split(",") if l.strip()
    )
    control_labels = frozenset(
        l.strip() for l in cfg.control_label.split(",") if l.strip()
    )
    excluded_labels = frozenset(
        l.strip() for l in cfg.excluded_label.split(",") if l.strip()
    )

    # Load (single-sample or multi-sample)
    multi_paths = [p.strip() for p in cfg.multi_sample_files.split(",") if p.strip()]
    sample_id_all = None  # only set in multi-sample mode

    if multi_paths:
        # joint training over multiple samples
        logging.info("[run] Multi-sample mode: %d samples", len(multi_paths))
        X, pos_raw, tissue_raw, perturb_raw, has_guide, is_real_KO, gene_names, sample_id_all, \
            tier1_raw, arch_raw = load_h5ad_multi(
                multi_paths,
                unassigned_labels=unassigned_labels,
                control_labels=control_labels,
                hvg_max_rank=cfg.hvg_max_rank,
                arch_dir=(cfg.arch_desc_dir if getattr(cfg, "use_timekernel", False) else None),
            )
        # tissue axis is threaded through when every slide carries it (perturbmap:
        # tumor/normal/periphery); None for spac (no tissue annotation).
        if tissue_raw is not None:
            logging.info("[run] Multi-sample tissue axis active (%d classes) -> basal path",
                         len(set(tissue_raw.tolist())))
    else:
        # Single-sample mode (default)
        X, pos_raw, tissue_raw, perturb_raw, has_guide, is_real_KO, gene_names, \
            tier1_raw, arch_raw = load_h5_dataset(
                cfg.data_file,
                unassigned_labels=unassigned_labels,
                control_labels=control_labels,
                hvg_max_rank=cfg.hvg_max_rank,
                arch_dir=(cfg.arch_desc_dir if getattr(cfg, "use_timekernel", False) else None),
            )

    # timekernel: GLOBAL Tier1 codes + the architectural descriptor
    # Tier1 codes are built once, over the FULL unfiltered per-cell label universe (all
    # slides), so a given lineage label maps to the SAME integer code regardless of which
    # downstream filters (excluded_labels, holdout, min_cells_per_pert, ...) happen to run
    # for a given config. The descriptor (arch_desc_all) is already per-cell and needs no
    # vocabulary, but it is threaded through every subsequent row-filter alongside
    # tier1_idx_all (see each `if tier1_idx_all is not None:` block below) so both stay
    # row-aligned with X all the way to the train_model() call (Task 2).
    if getattr(cfg, "use_timekernel", False):
        if tier1_raw is None or arch_raw is None:
            raise ValueError(
                "use_timekernel requires a 'Tier1' obs column in every slice's obs and an "
                "architectural-descriptor sidecar per slice under --arch_desc_dir; at least "
                "one is missing.")
        tier1_map = {lab: i for i, lab in enumerate(sorted(set(tier1_raw.tolist())))}
        tier1_idx_all = np.array([tier1_map[s] for s in tier1_raw], dtype=int)
        arch_desc_all = np.asarray(arch_raw, dtype=np.float32)
        n_tier1 = int(tier1_idx_all.max()) + 1
        arch_dim = int(arch_desc_all.shape[1])
        logging.info("[timekernel] Tier1 classes=%d %s", n_tier1,
                     sorted(tier1_map, key=tier1_map.get))
        logging.info("[timekernel] arch descriptor: %d cells x %d dims (raw)",
                     arch_desc_all.shape[0], arch_dim)
    else:
        tier1_idx_all = None
        arch_desc_all = None
        n_tier1 = 0
        arch_dim = 20

    # Filter 1: excluded labels (e.g. "multiple" -- mixed-KO spots)
    # These cells have ambiguous perturbation identity and are dropped entirely.
    if excluded_labels:
        keep_excl = np.array([p not in excluded_labels for p in perturb_raw], dtype=bool)
        n_excl = (~keep_excl).sum()
        if n_excl > 0:
            logging.info("[filter] Excluded %d cells with labels %s", n_excl, sorted(excluded_labels))
            X          = X[keep_excl]
            pos_raw    = pos_raw[keep_excl]
            perturb_raw = perturb_raw[keep_excl]
            has_guide  = has_guide[keep_excl]
            is_real_KO = is_real_KO[keep_excl]
            if tissue_raw is not None:
                tissue_raw = tissue_raw[keep_excl]
            if sample_id_all is not None:
                sample_id_all = sample_id_all[keep_excl]
            if tier1_idx_all is not None:
                tier1_idx_all = tier1_idx_all[keep_excl]
                arch_desc_all = arch_desc_all[keep_excl]

    # Holdout unassigned cells (for unseen-cell eval, scenarios 2 & 3)
    if cfg.holdout_unassigned_frac > 0.0:
        rng_ho = np.random.default_rng(cfg.seed if cfg.seed is not None else 0)
        una_idx = np.where(~has_guide)[0]   # unassigned = no guide
        n_hold  = max(1, int(len(una_idx) * cfg.holdout_unassigned_frac))
        hold_idx = rng_ho.choice(una_idx, n_hold, replace=False)
        hold_mask = np.zeros(len(X), dtype=bool); hold_mask[hold_idx] = True
        np.save(str(outdir / "holdout_unassigned_indices.npy"), hold_idx)
        logging.info("[holdout] Held out %d / %d unassigned cells (%.1f%%). "
                     "Saved to holdout_unassigned_indices.npy",
                     n_hold, len(una_idx), 100 * cfg.holdout_unassigned_frac)
        keep_ho = ~hold_mask
        X           = X[keep_ho]; pos_raw     = pos_raw[keep_ho]
        perturb_raw = perturb_raw[keep_ho];  has_guide   = has_guide[keep_ho]
        is_real_KO  = is_real_KO[keep_ho]
        if tissue_raw is not None:
            tissue_raw = tissue_raw[keep_ho]
        if sample_id_all is not None:
            sample_id_all = sample_id_all[keep_ho]
        if tier1_idx_all is not None:
            tier1_idx_all = tier1_idx_all[keep_ho]
            arch_desc_all = arch_desc_all[keep_ho]

    elif cfg.holdout_unassigned_file:
        hold_idx = np.load(cfg.holdout_unassigned_file).astype(int)
        una_mask_all = ~has_guide                       # genuine unassigned only
        valid = hold_idx[(hold_idx >= 0) & (hold_idx < len(X))]
        valid = valid[una_mask_all[valid]]
        hold_mask = np.zeros(len(X), dtype=bool); hold_mask[valid] = True
        np.save(str(outdir / "holdout_unassigned_indices.npy"), valid)
        logging.info("[holdout-file] Held out %d unassigned cells from %s "
                     "(%d listed, KO/NTC filtered out)",
                     len(valid), cfg.holdout_unassigned_file, len(hold_idx))
        keep_ho = ~hold_mask
        X           = X[keep_ho]; pos_raw     = pos_raw[keep_ho]
        perturb_raw = perturb_raw[keep_ho];  has_guide   = has_guide[keep_ho]
        is_real_KO  = is_real_KO[keep_ho]
        if tissue_raw is not None:
            tissue_raw = tissue_raw[keep_ho]
        if sample_id_all is not None:
            sample_id_all = sample_id_all[keep_ho]
        if tier1_idx_all is not None:
            tier1_idx_all = tier1_idx_all[keep_ho]
            arch_desc_all = arch_desc_all[keep_ho]

    elif cfg.holdout_file:
        import json as _json
        hv = _json.load(open(cfg.holdout_file))
        sid_arr = (sample_id_all if sample_id_all is not None
                   else np.zeros(len(X), dtype=np.int32))
        kx = np.rint(np.asarray(pos_raw)[:, 0]).astype(np.int64)
        ky = np.rint(np.asarray(pos_raw)[:, 1]).astype(np.int64)
        keys = list(zip(sid_arr.tolist(), kx.tolist(), ky.tolist()))
        una_all = ~has_guide

        def _want(coord_map):
            w = set()
            for sid_str, coords in coord_map.items():
                sid = int(sid_str)
                for xy in coords:
                    w.add((sid, int(xy[0]), int(xy[1])))
            return w

        w_una = _want(hv.get("unassigned", {}))
        w_ko = _want(hv.get("ko", {}))
        m_una = np.fromiter((k in w_una for k in keys), dtype=bool, count=len(keys)) & una_all
        m_ko = np.fromiter((k in w_ko for k in keys), dtype=bool, count=len(keys))
        if w_ko:
            leaked = int((m_ko & una_all).sum())
            if leaked:
                logging.warning("[holdout] %d ko-coord cells were unassigned (unexpected)", leaked)
        hold_mask = m_una | m_ko
        np.save(str(outdir / "holdout_unassigned_indices.npy"), np.where(m_una)[0])
        np.save(str(outdir / "holdout_ko_indices.npy"), np.where(m_ko)[0])
        logging.info("[holdout] %s: held out %d unassigned + %d KO = %d cells "
                     "(coords listed: %d una, %d ko)",
                     os.path.basename(cfg.holdout_file), int(m_una.sum()),
                     int(m_ko.sum()), int(hold_mask.sum()), len(w_una), len(w_ko))
        keep_ho = ~hold_mask
        X           = X[keep_ho]; pos_raw     = pos_raw[keep_ho]
        perturb_raw = perturb_raw[keep_ho];  has_guide   = has_guide[keep_ho]
        is_real_KO  = is_real_KO[keep_ho]
        if tissue_raw is not None:
            tissue_raw = tissue_raw[keep_ho]
        if sample_id_all is not None:
            sample_id_all = sample_id_all[keep_ho]
        if tier1_idx_all is not None:
            tier1_idx_all = tier1_idx_all[keep_ho]
            arch_desc_all = arch_desc_all[keep_ho]

    # Filter 2: low-frequency perturbations
    # Drop cells whose perturbation class has fewer than min_cells_per_pert cells.
    # Prevents near-empty embeddings for singleton or very-rare guides.
    if cfg.min_cells_per_pert > 0:
        from collections import Counter
        pert_counts = Counter(perturb_raw.tolist())
        keep_freq = np.array([pert_counts[p] >= cfg.min_cells_per_pert for p in perturb_raw], dtype=bool)
        n_rare = (~keep_freq).sum()
        n_rare_perts = len([p for p, c in pert_counts.items() if c < cfg.min_cells_per_pert])
        if n_rare > 0:
            logging.info("[filter] Dropped %d cells from %d rare perturbations "
                         "(< %d cells each)", n_rare, n_rare_perts, cfg.min_cells_per_pert)
            X          = X[keep_freq]
            pos_raw    = pos_raw[keep_freq]
            perturb_raw = perturb_raw[keep_freq]
            has_guide  = has_guide[keep_freq]
            is_real_KO = is_real_KO[keep_freq]
            if tissue_raw is not None:
                tissue_raw = tissue_raw[keep_freq]
            if sample_id_all is not None:
                sample_id_all = sample_id_all[keep_freq]
            if tier1_idx_all is not None:
                tier1_idx_all = tier1_idx_all[keep_freq]
                arch_desc_all = arch_desc_all[keep_freq]

    logging.info("[filter] After filtering: %d cells, %d unique perturbations",
                 len(perturb_raw), len(set(perturb_raw.tolist())))

    # Negative-control: shuffle labels of the top-N real-KO classes
    # Permute perturbation labels AMONG the cells of the top-N real-KO genes only.
    # Marginal counts per gene are preserved (so embeddings/filters are unchanged),
    # but each cell's label is randomly reassigned within the pool -> no learnable
    # perturbation signal for those genes. is_real_KO/has_guide stay valid because
    # we only permute among real-KO labels (no class boundary is crossed).
    if getattr(cfg, "shuffle_top_perts", 0) and cfg.shuffle_top_perts > 0:
        from collections import Counter as _Counter
        ko_mask_arr = np.asarray(is_real_KO, dtype=bool).reshape(-1)
        ko_counts = _Counter(np.asarray(perturb_raw)[ko_mask_arr].tolist())
        top_genes = [g for g, _ in ko_counts.most_common(cfg.shuffle_top_perts)]
        cell_in_top = ko_mask_arr & np.isin(perturb_raw, top_genes)
        idx_top = np.where(cell_in_top)[0]
        _seed = cfg.seed if cfg.seed is not None else 42
        _rng = np.random.default_rng(_seed)
        perm = _rng.permutation(idx_top)
        perturb_raw = np.asarray(perturb_raw).copy()
        perturb_raw[idx_top] = perturb_raw[perm]
        logging.warning("[shuffle_top_perts=%d] SHUFFLED labels of %d cells across top-%d real-KO genes "
                        "(negative control; seed=%d). Genes: %s",
                        cfg.shuffle_top_perts, idx_top.size, cfg.shuffle_top_perts, _seed,
                        ",".join(top_genes))
        np.savetxt(outdir / "shuffled_top_perts.txt", np.array(top_genes, dtype=object), fmt="%s")

    # Use generic SPAC-seq attribute builder for .h5ad, legacy builder for .h5
    if cfg.data_file.endswith(".h5ad") or multi_paths:
        tissue_idx, tissue_dict, perturb_idx, pert_map = build_attributes_spac(
            perturb_raw, tissue_raw=tissue_raw,
            unassigned_labels=unassigned_labels, control_labels=control_labels)
    else:
        tissue_idx, tissue_dict, perturb_idx, pert_map = build_attributes(perturb_raw)

    with open(outdir / "pert_map.json", "w") as _fh:
        json.dump({
            "pert_map": {str(k): int(v) for k, v in pert_map.items()},
            "unassigned_labels": sorted(unassigned_labels),
            "control_labels": sorted(control_labels),
            "excluded_labels": sorted(excluded_labels),
            "tissue_dict": {str(k): int(v) for k, v in tissue_dict.items()},
        }, _fh, indent=2)
    logging.info("Saved pert_map.json (%d perts incl basal slot) -> %s",
                 len(pert_map), outdir / "pert_map.json")

    # Save gene names if available (useful for downstream analysis)
    if gene_names is not None:
        np.savetxt(outdir / "hvg_gene_names.txt", gene_names, fmt="%s")
        logging.info("Saved %d HVG gene names -> %s", len(gene_names), outdir / "hvg_gene_names.txt")

    cell_atts = np.c_[tissue_idx, perturb_idx].astype(int)
    sample_indices = torch.arange(X.shape[0], dtype=torch.int)

    pert_name = {code: name for name, code in pert_map.items()}

    # Batch one-hot from perturbation codes
    n_batch = int(len(np.unique(perturb_idx)))
    batch = np.eye(n_batch, dtype=np.float32)[perturb_idx]

    # Scale spatial & append batch columns for kernel conditioning.
    if sample_id_all is not None:
        # Multi-slide: tile each slide into a disjoint region so the GP has NO cross-slide
        # neighbors (slides are unaligned 2D sections). Kernel params remain shared.
        pos_scaled, tile_layout, tile_spacing = tile_multislide_coords(
            pos_raw, sample_id_all, cfg.loc_range, cfg.slide_tile_gap_factor)
        logging.info(
            "[multi-tile] %d slides tiled (spacing=%.1f, gap_factor=%.2f, coord max=%.1f); "
            "layout (sid,row,col,n)=%s",
            len(tile_layout), tile_spacing, cfg.slide_tile_gap_factor,
            float(pos_scaled.max()), tile_layout)
    else:
        pos_scaled = (MinMaxScaler().fit_transform(pos_raw) * cfg.loc_range).astype(np.float32)
    cutoff = np.full(pos_scaled.shape[0], 0.5, dtype=np.float32)  # learnable init (vector)
    pos_batched = np.concatenate([pos_scaled, batch], axis=1).astype(np.float32)

    # Inducing points.
    if sample_id_all is not None and cfg.grid_inducing_points:
        # Per-slide tiled grids (density-scaled per slide), concatenated.
        inducing = build_tiled_inducing(pos_scaled, sample_id_all, n_batch, cfg)
        logging.info("[inducing] multi-slide tiled grids: M=%d total inducing points "
                     "(cells_per_inducing=%s)", inducing.shape[0], cfg.cells_per_inducing)
    else:
        # Single-slide: optional density scaling (cells_per_inducing) else fixed steps.
        induc_steps, derived = _derive_grid_steps(int(pos_batched.shape[0]), cfg)
        if cfg.grid_inducing_points and cfg.cells_per_inducing and cfg.cells_per_inducing > 0:
            logging.info("[inducing] density mode: N=%d, cells_per_inducing=%d -> steps=%d "
                         "(M=%d; pre-clip=%d, clip=[%d,%d])",
                         int(pos_batched.shape[0]), cfg.cells_per_inducing, induc_steps,
                         induc_steps * induc_steps, derived, cfg.inducing_steps_min, cfg.inducing_steps_max)
        else:
            logging.info("[inducing] fixed grid: steps=%d (M=%d inducing points)",
                         induc_steps, induc_steps * induc_steps)
        inducing = build_inducing_points(
            pos_batched=pos_batched,
            n_batch=n_batch,
            steps=induc_steps,
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
    # filter_min_counts=False: disable explicit filtering inside normalize().
    # However, sc.pp.normalize_per_cell may still silently drop cells with 0
    # total counts (a scanpy internals quirk).  After normalize(), we recover
    # which cells survived via obs_names (0-based int strings) and re-index
    # all companion arrays so every array has the same N_eff cells.
    N_before = X.shape[0]
    adata = sc.AnnData(X, dtype="float32")
    # Preserve HVG gene symbols on var_names so downstream stages (e.g. deconv)
    # can match the model's genes against external files (pseudo-bulk). Without
    # this, var_names default to integer strings and every gene is treated as
    # "absent", zero-filling the ROI matrix and crashing normalize().
    if gene_names is not None and len(gene_names) == adata.n_vars:
        adata.var_names = [str(g) for g in gene_names]
    adata = normalize(adata, filter_min_counts=False, size_factors=True, normalize_input=True, logtrans_input=True)
    N_after = adata.n_obs

    if N_after < N_before:
        kept_idx = adata.obs_names.astype(int).values
        keep_mask = np.zeros(N_before, dtype=bool)
        keep_mask[kept_idx] = True
        logging.warning(
            "[normalize] %d cells dropped (0-count rows after HVG subset); "
            "re-indexing all companion arrays.", N_before - N_after
        )
        pos_batched = pos_batched[keep_mask]
        cell_atts   = cell_atts[keep_mask]
        batch       = batch[keep_mask]
        cutoff      = cutoff[keep_mask]
        has_guide   = has_guide[keep_mask]
        is_real_KO  = is_real_KO[keep_mask]
        if sample_id_all is not None:
            sample_id_all = sample_id_all[keep_mask]
        if tier1_idx_all is not None:
            tier1_idx_all = tier1_idx_all[keep_mask]
            arch_desc_all = arch_desc_all[keep_mask]
        logging.info("[normalize] N_cells after re-index: %d", N_after)

    # timekernel: standardise the architectural descriptor
    # The raw columns have wildly different scales (one-hot 0/1 vs distances in microns),
    # so an unstandardised Linear would be dominated by the distance columns. z-score per
    # column with statistics POOLED over the training cells across all slices, then persist
    # them next to the checkpoint so every later stage (composition_eval, counterfactual
    # re-decode) applies the SAME transform instead of re-deriving stats from its own
    # subset -- which would silently shift the descriptor the embeddings were fit on.
    if arch_desc_all is not None:
        scaler_path = outdir / "arch_scaler.npz"
        if cfg.stage.lower() == "train":
            arch_mean = arch_desc_all.mean(axis=0)
            arch_std = arch_desc_all.std(axis=0)
            # Constant columns (e.g. a scaffold lineage absent from every slice) get std=1
            # so they map to exactly 0 rather than blowing up.
            n_const = int((arch_std < 1e-6).sum())
            arch_std = np.where(arch_std < 1e-6, 1.0, arch_std)
            np.savez(scaler_path, mean=arch_mean.astype(np.float32),
                     std=arch_std.astype(np.float32))
            logging.info("[timekernel] Fit arch scaler on %d training cells (%d constant "
                         "columns held at 0) -> %s", arch_desc_all.shape[0], n_const, scaler_path)
        else:
            if not scaler_path.is_file():
                raise FileNotFoundError(
                    f"stage={cfg.stage} needs the TRAINING arch scaler at {scaler_path} so the "
                    f"descriptor matches what the shared-basal embeddings were fit on. It is "
                    f"written by the train stage; re-deriving it here would silently shift the "
                    f"descriptor.")
            _sc = np.load(scaler_path)
            arch_mean, arch_std = _sc["mean"], _sc["std"]
            if arch_mean.shape[0] != arch_desc_all.shape[1]:
                raise ValueError(
                    f"arch scaler width {arch_mean.shape[0]} != descriptor width "
                    f"{arch_desc_all.shape[1]} ({scaler_path})")
            logging.info("[timekernel] Loaded training arch scaler from %s", scaler_path)
        arch_desc_all = ((arch_desc_all - arch_mean) / arch_std).astype(np.float32)
        logging.info("[timekernel] arch descriptor standardised: mean|.|=%.3f, max|.|=%.3f",
                     float(np.abs(arch_desc_all).mean()), float(np.abs(arch_desc_all).max()))

    # Kernel scale per batch (n_batch, spatial_dims)
    spatial_dims = pos_batched.shape[1] - n_batch
    kernel_scale = np.full((n_batch, spatial_dims), cfg.kernel_scale, dtype=np.float32)

    # empirical mu_action anchor = logmean(NTC) - logmean(unassigned).
    # Use RAW counts / size_factors (non-negative) -- adata.X is standardized
    # (normalize_input=True) so its mean can be <=0 -> log -> NaN. adata.raw.X is the
    # HVG-subset raw counts (n_vars-dim -> matches mu_action).
    mu_action_anchor = None
    if cfg.use_action_correction and cfg.mu_action_anchor_lambda > 0:
        raw = adata.raw.X
        raw = raw.toarray() if hasattr(raw, "toarray") else np.asarray(raw)
        sf = adata.obs["size_factors"].to_numpy(dtype=np.float32)[:, None]
        Xn = raw / np.clip(sf, 1e-8, None)                 # normalized counts >= 0
        ntc_m = (has_guide & ~is_real_KO)
        una_m = ~has_guide
        if int(ntc_m.sum()) >= 10 and int(una_m.sum()) >= 10:
            eps = 1e-3
            a = (np.log(Xn[ntc_m].mean(0) + eps)
                 - np.log(Xn[una_m].mean(0) + eps))
            a = np.nan_to_num(a, nan=0.0, posinf=5.0, neginf=-5.0)
            mu_action_anchor = np.clip(a, -5.0, 5.0).astype(np.float32)
            logging.info(
                "mu_action anchor: ||a||=%.3f (NTC=%d, unassigned=%d, lambda=%.3g)",
                float(np.linalg.norm(mu_action_anchor)),
                int(ntc_m.sum()), int(una_m.sum()), cfg.mu_action_anchor_lambda)
        else:
            logging.warning("Too few NTC/unassigned cells for mu_action anchor; disabled")

    # Model
    model = CONCERT(
        cell_atts=cell_atts,
        num_genes=adata.n_vars,
        encoder_dim=cfg.encoder_dim,
        GP_dim=cfg.GP_dim,
        Normal_dim=cfg.Normal_dim,
        n_batch=n_batch,
        n_slides=int(len(np.unique(sample_id_all))) if sample_id_all is not None else 1,
        fpert_tile_spacing=(cfg.loc_range * cfg.slide_tile_gap_factor) if sample_id_all is not None else 0.0,
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
        use_pad=cfg.use_pad,
        pad_action_dim=cfg.pad_action_dim,
        pad_lambda_mmd=cfg.pad_lambda_mmd,
        pad_lambda_adv=cfg.pad_lambda_adv,
        use_action_correction=cfg.use_action_correction,
        detach_mu_action_for_ko=cfg.detach_mu_action_for_ko,
        mu_action_anchor=mu_action_anchor,
        mu_action_anchor_lambda=cfg.mu_action_anchor_lambda,
        use_basal_encoder_only=cfg.use_basal_encoder_only,
        use_f_pert=cfg.use_f_pert,
        f_pert_init_std=cfg.f_pert_init_std,
        f_pert_clamp=cfg.f_pert_clamp,
        fpert_field=cfg.fpert_field,
        fpert_rff_features=cfg.fpert_rff_features,
        fpert_rff_bandwidth=cfg.fpert_rff_bandwidth,
        fpert_hidden=cfg.fpert_hidden,
        fpert_svgp_inducing=cfg.fpert_svgp_inducing,
        fpert_svgp_ls_init=cfg.fpert_svgp_ls_init,
        fpert_coord_scale=cfg.fpert_coord_scale,
        fpert_mb_bandwidths=(
            [float(x) for x in cfg.fpert_mb_bandwidths.split(",") if x.strip()]
            if cfg.fpert_mb_bandwidths else None),
        basal_bayes=cfg.basal_bayes,
        basal_field=cfg.basal_field,
        basal_rff_features=cfg.basal_rff_features,
        basal_rff_bandwidth=cfg.basal_rff_bandwidth,
        basal_rff_hidden=cfg.basal_rff_hidden,
        basal_coord_scale=cfg.basal_coord_scale,
        disentangle_radius=cfg.disentangle_radius,
        use_consistency_loss=cfg.use_consistency_loss,
        consistency_lambda=cfg.consistency_lambda,
        use_expr_encoder=cfg.use_expr_encoder,
        expr_hidden_dims=list(cfg.expr_hidden_dims),
        expr_dropout=cfg.expr_dropout,
        use_basal_consistency=cfg.use_basal_consistency,
        basal_consistency_lambda=cfg.basal_consistency_lambda,
        beta_warmup_epochs=cfg.beta_warmup_epochs,
        beta_target=cfg.beta_target,
        free_bits=cfg.free_bits,
        gp_free_bits=cfg.gp_free_bits,
        use_pert_correction=cfg.use_pert_correction,
        use_nngp=cfg.use_nngp,
        nngp_lambda=cfg.nngp_lambda,
        use_paracrine=cfg.use_paracrine,
        paracrine_l1=cfg.paracrine_l1,
        use_niche_interaction=cfg.use_niche_interaction,
        niche_rank=cfg.niche_rank,
        use_niche_film=cfg.use_niche_film,
        niche_film_dim=cfg.niche_film_dim,
        niche_film_k=cfg.niche_film_k,
        niche_film_gcap=cfg.niche_film_gcap,
        niche_film_l2=cfg.niche_film_l2,
        niche_film_rank=cfg.niche_film_rank,
        niche_film_dispersion=cfg.niche_film_dispersion,
        use_time_conditioning=cfg.use_time_conditioning,
        time_fourier_bands=cfg.time_fourier_bands,
        time_emb_dim=cfg.time_emb_dim,
        time_rank=cfg.time_rank,
        time_l2=cfg.time_l2,
        use_membership_field=cfg.use_membership_field,
        membership_hidden=cfg.membership_hidden,
        membership_bce_lambda=cfg.membership_bce_lambda,
        membership_moran_lambda=cfg.membership_moran_lambda,
        use_time_intensity=cfg.use_time_intensity,
        use_inflammation_field=cfg.use_inflammation_field,
        field_mse_lambda=cfg.field_mse_lambda,
        field_time_fixed_t=cfg.field_time_fixed_t,
        field_moran_lambda=cfg.field_moran_lambda,
        use_pert_gate=cfg.use_pert_gate,
        gate_hidden=cfg.gate_hidden,
        spatial_basal_only=cfg.spatial_basal_only,
        lfc_lambda=cfg.lfc_lambda,
        disentangle_lambda=cfg.disentangle_lambda,
        moran_lambda=cfg.moran_lambda,
        deamortize_gp=cfg.deamortize_gp,
        additive_decoder=cfg.additive_decoder,
        gp_field_aux_lambda=cfg.gp_field_aux_lambda,
        basal_moran_output_lambda=cfg.basal_moran_output_lambda,
        ls_prior_lambda=cfg.ls_prior_lambda,
        ls_prior_min=cfg.ls_prior_min,
        ls_prior_max=cfg.ls_prior_max,
        use_timekernel=cfg.use_timekernel,
        use_shared_basal=cfg.use_shared_basal,
        n_tier1=n_tier1,
        arch_dim=arch_dim,
        amp_anchor_offset=cfg.amp_anchor_offset,
        amp_anchor_lambda=cfg.amp_anchor_lambda,
        amp_anchor_t=cfg.amp_anchor_t,
        marker_sign_positive=cfg.marker_sign_positive,
        marker_gene_idx=(_marker_idx_on_model_axis(gene_names)
                         if cfg.marker_sign_positive else None),
        time_in_field_mlp=cfg.time_in_field_mlp,
    )
    logging.info("Model initialized: %s | action_correction=%s | num_perts=%d",
                 model.__class__.__name__, cfg.use_action_correction, n_batch)

    # attach precomputed spatial graph (NNGP + paracrine)
    if cfg.use_nngp or cfg.graph_file:
        from spatial_graph import SpatialGraph
        if not cfg.graph_file:
            raise ValueError("use_nngp=True requires --graph_file")
        graph = SpatialGraph.load(cfg.graph_file, device=cfg.device)
        n_graph = int(graph.knn_idx.shape[0])
        n_cells = int(cell_atts.shape[0])
        if n_graph != n_cells:
            raise ValueError(
                f"[graph] alignment mismatch: graph N={n_graph} != data N={n_cells}. "
                f"Rebuild the graph on the SAME post-normalize cell order.")
        model.attach_graph(graph)
        logging.info("[graph] attached SpatialGraph N=%d from %s", n_graph, cfg.graph_file)

    # DDP wrapping
    ddp_model = None
    if world_size > 1:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                        find_unused_parameters=True)
        logging.info("DDP enabled: rank=%d / world_size=%d", rank, world_size)

    # Wandb logger helper (rank 0 only)
    def _log(metrics: Dict, step: int) -> None:
        if wb is not None and rank == 0:
            try:
                wb.log(metrics, step=step)
            except Exception:
                pass

    # Stage
    if cfg.stage.lower() == "train":
        ckpt_path = str(outdir / cfg.model_file)
        if cfg.resume and os.path.exists(ckpt_path):
            model.load_model(ckpt_path)
            logging.info("Resumed from checkpoint: %s", ckpt_path)
        bs = auto_batch_size(X.shape[0]) if cfg.batch_size == "auto" else int(cfg.batch_size)

        # precompute mini-bulk KO-vs-NTC LFC targets
        # niche grid MUST match eval --niche_grid (default 6), which bins each slide's own
        # MinMax*LOC_RANGE coordinates.
        niche_id_arr = None
        if cfg.lfc_lambda > 0:
            from eval_utils import build_niche_grid, pseudobulk_lfc as _pblfc
            import scipy.sparse as _sp
            # Niches are defined within each slide. Multi-slide coordinates are tiled, so fold
            # them back to the per-slide [0, loc_range] frame and offset the bin id by the tile.
            _xy = pos_batched[:, :2].astype(np.float64)
            _tile = np.zeros(len(_xy), dtype=np.int64)
            if sample_id_all is not None:
                _S = float(cfg.loc_range) * float(cfg.slide_tile_gap_factor)
                _cell = np.floor(_xy / _S)
                _xy = _xy - _S * _cell
                _tile = (_cell[:, 1] * 1000 + _cell[:, 0]).astype(np.int64)
            _local, _nb = build_niche_grid(_xy, grid=6, loc_range=float(cfg.loc_range))
            _, _tile_idx = np.unique(_tile, return_inverse=True)
            _bin = (_tile_idx * _nb + _local).astype(int)
            logging.info("[lfc] niche grid: %d tiles x %d bins", int(_tile_idx.max()) + 1, _nb)
            _raw = adata.raw.X
            _raw = _raw.toarray() if _sp.issparse(_raw) else np.asarray(_raw)
            _pcode = cell_atts[:, 1].astype(int)
            _hg = has_guide.astype(bool); _rk = is_real_KO.astype(bool)
            _ntc_mask = _hg & (~_rk)
            tgts = {}
            if _ntc_mask.sum() >= 30:
                # --- datasets WITH NTC (spac): per-niche KO-vs-NTC. mu_action absorbs the
                #    NTC-vs-basal guide shift, so this LFC isolates f_pert. ---
                for b in np.unique(_bin):
                    ntc_b = np.where(_ntc_mask & (_bin == b))[0]
                    if len(ntc_b) < 30:                       # eval ntc_floor
                        continue
                    for gcode in np.unique(_pcode[_rk & (_bin == b)]):
                        ko_b = np.where(_rk & (_pcode == gcode) & (_bin == b))[0]
                        if len(ko_b) < 25:                    # eval min_cells_per_pert
                            continue
                        lfc = _pblfc(_raw[ko_b], _raw[ntc_b])
                        tgts[(int(b), int(gcode))] = torch.tensor(lfc, dtype=torch.float32, device=cfg.device)
                logging.info("[lfc] precomputed %d (niche,pert) KO-vs-NTC targets over %d genes",
                             len(tgts), adata.n_vars)
            else:
                # --- datasets WITHOUT NTC (perturbmap): mu_action is OFF, so f_pert IS the
                #    full KO-vs-unperturbed log shift. Reference = unperturbed BASAL cells,
                #    tissue-matched to the KO cells (KO spots are tumor) so tissue structure
                #    (already carried by the tissue-conditioned basal field) does not leak into
                #    the target. KO counts are too sparse (~350 across 4 slides) for a per-niche
                #    floor, so compute ONE global LFC per KO gene and broadcast it to every niche
                #    bin containing that gene (keys the forward (niche,pert) lookup). Without
                #    this, lfc_targets={} and f_pert receives no gradient (frozen at init). ---
                _tcol = cell_atts[:, 0].astype(int)
                _tumor_code = tissue_dict.get("tumor")
                _basal = ~_hg
                ref_mask = _basal & (_tcol == _tumor_code) if _tumor_code is not None else _basal
                if ref_mask.sum() < 30:                       # too few tumor-basal -> all basal
                    ref_mask = _basal
                ref_idx = np.where(ref_mask)[0]
                n_perts = 0
                for gcode in np.unique(_pcode[_rk]):
                    ko_all = np.where(_rk & (_pcode == gcode))[0]
                    if len(ko_all) < 10 or len(ref_idx) < 30:
                        continue
                    t = torch.tensor(_pblfc(_raw[ko_all], _raw[ref_idx]),
                                     dtype=torch.float32, device=cfg.device)
                    n_perts += 1
                    for b in np.unique(_bin[ko_all]):
                        tgts[(int(b), int(gcode))] = t
                logging.info("[lfc] NO NTC -> %d global KO-vs-basal(tumor) targets broadcast to "
                             "%d (niche,pert) keys over %d genes (ref=%d basal cells)",
                             n_perts, len(tgts), adata.n_vars, len(ref_idx))
            model.lfc_targets = tgts
            niche_id_arr = _bin

        # per-cell normalized day for continuous-time conditioning
        # Each slide-file = one timepoint, so reconstruct day from sample_id_all (the
        # per-cell file index, already masked in lockstep through every filter) + a
        # per-file day lookup. Avoids threading `day` through all the row-filter blocks.
        time_t_arr = None
        if cfg.use_time_conditioning:
            if sample_id_all is not None and multi_paths:
                import anndata as _ad_time
                day_by_file = np.array([
                    float(_ad_time.read_h5ad(pth, backed="r").obs[cfg.time_col].iloc[0])
                    for pth in multi_paths
                ], dtype=np.float32)
                time_t_arr = day_by_file[np.asarray(sample_id_all).astype(int)]
                logging.info("[time] continuous-time ON: %d files, day_norm range [%.3f, %.3f], "
                             "per-cell N=%d", len(multi_paths), float(day_by_file.min()),
                             float(day_by_file.max()), len(time_t_arr))
            else:
                raise ValueError("use_time_conditioning requires multi-sample mode "
                                 "(--multi_sample_files); single-file temporal not wired.")

        # per-cell continuous inflammation gene-score target s_hat_i
        # The "honest gene state" (not a label): [0,1]-scaled mean of z-scored DSS marker-panel
        # log-expression, computed from raw counts on the SAME (post-normalize) cell order as
        # the training matrices. Required when use_inflammation_field is on.
        infl_score_arr = None
        if getattr(cfg, "use_inflammation_field", False):
            infl_score_arr = _compute_inflammation_score(adata)
            logging.info("[infl] continuous inflammation score: N=%d, range [%.3f, %.3f], mean %.3f",
                         len(infl_score_arr), float(infl_score_arr.min()),
                         float(infl_score_arr.max()), float(infl_score_arr.mean()))

        t0 = time.time()
        model.train_model(
            pos=pos_batched,
            ncounts=adata.X,
            raw_counts=adata.raw.X,
            size_factors=adata.obs["size_factors"].to_numpy(dtype=np.float32),
            batch=batch,
            has_guide=has_guide.astype(np.float32),
            is_real_KO=is_real_KO.astype(np.float32),
            niche_id=niche_id_arr,
            sample_id=sample_id_all,
            time_t=time_t_arr,
            infl_score=infl_score_arr,
            tier1_id=tier1_idx_all,
            arch_desc=arch_desc_all,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            batch_size=bs,
            num_samples=cfg.num_samples,
            train_size=cfg.train_size,
            maxiter=cfg.maxiter,
            patience=cfg.patience,
            earlystop_min_delta=cfg.earlystop_min_delta,
            save_model=True,
            model_weights=str(outdir / cfg.model_file),
            log_fn=_log,
            report_every=cfg.report_every,
            perturb_name_map=pert_name,
            ddp_model=ddp_model,
            rank=rank,
            world_size=world_size,
        )
        logging.info("Training done in %.1fs", time.time() - t0)

        if rank == 0:
            _report_and_save_final(
                model=model,
                cell_atts=cell_atts,
                perturb_name_map=pert_name,
                outdir=outdir,
                sample=cfg.sample,
                project_index=cfg.project_index,
            )
            # Timekernel: persist the training Tier1 string->code map so any downstream
            # inference (e.g. the field-trajectory harness) can map its per-cell labels to
            # the SAME integer codes the trained embeddings use. The niche descriptor is a
            # per-cell vector with no vocabulary, so Tier1 is the only map.
            if getattr(cfg, "use_timekernel", False):
                try:
                    lm_path = outdir / "label_maps.json"
                    with open(lm_path, "w") as _fh:
                        json.dump({"tier1": tier1_map}, _fh, indent=2)
                    logging.info("[timekernel] Saved label maps to %s", lm_path)
                except Exception as e:
                    logging.warning("[timekernel] Failed to save label_maps.json (%s).", e)
        return

    # Deconvolution / patch-imputation stages (load ckpt, run, exit)
    if cfg.stage.lower() in ("deconv", "impute_patch"):
        model.load_model(str(outdir / cfg.model_file))
        logging.info("Loaded weights for %s: %s", cfg.stage, outdir / cfg.model_file)
        if cfg.stage.lower() == "deconv":
            from deconv_impute import run_deconvolution
            run_deconvolution(model, adata, pos_raw, pert_map, cfg, outdir)
        else:
            from deconv_impute import run_impute_patch
            run_impute_patch(model, adata, pos_raw, pos_batched, cell_atts,
                             sample_indices, pert_map, cfg, outdir)
        return

    # LOTO-D9 temporal evaluation (load ckpt, predict held-out day, exit)
    if cfg.stage.lower() == "loto_eval":
        model.load_model(str(outdir / cfg.model_file))
        logging.info("Loaded weights for loto_eval: %s", outdir / cfg.model_file)
        if not cfg.loto_holdout_files.strip():
            raise ValueError("stage=loto_eval requires --loto_holdout_files (held-out day h5ads)")
        holdout = [p.strip() for p in cfg.loto_holdout_files.split(",") if p.strip()]
        cfg.loto_holdout_files = holdout   # module expects a list
        from evaluate_left_out_day9 import run_loto_eval
        run_loto_eval(
            model=model, adata=adata, cell_atts=cell_atts, pos_batched=pos_batched,
            sample_indices=sample_indices, sample_id_all=sample_id_all,
            multi_paths=multi_paths, gene_names=gene_names, pert_map=pert_map,
            cfg=cfg, outdir=outdir,
        )
        return

    # Descriptive-reconstruction validation for the composition model
    if cfg.stage.lower() == "composition_eval":
        # heldout_mouse_eval reconstructs the model on the held-out mice's slices, whose cells
        # are DISJOINT from training -> the per-cell params (mask_cutoff, basal lookup) can't be
        # loaded by position; skip them and keep the fresh init (both are inert in this decode).
        _skip_pc = bool(getattr(cfg, "heldout_mouse_eval", False))
        model.load_model(str(outdir / cfg.model_file), skip_per_cell=_skip_pc)
        logging.info("Loaded weights for composition_eval: %s (skip_per_cell=%s)",
                     outdir / cfg.model_file, _skip_pc)
        _remap_slide_bias(model, str(outdir / cfg.model_file), multi_paths, cfg)
        if getattr(cfg, "heldout_mouse_eval", False):
            # Deliverable 2: LOMO ground-truth reconstruction on the held-out mice's slices.
            from case_study_eval import run_heldout_mouse_eval
            run_heldout_mouse_eval(
                model=model, adata=adata, cell_atts=cell_atts, pos_batched=pos_batched,
                sample_indices=sample_indices, sample_id_all=sample_id_all,
                multi_paths=multi_paths, gene_names=gene_names, pert_map=pert_map,
                cfg=cfg, outdir=outdir,
                tier1_id=tier1_idx_all, arch_desc=arch_desc_all,
            )
            return
        if getattr(cfg, "case_study_deliverables", False):
            # Descriptive case-study deliverables C1-C3 (pure inference on the final ckpt).
            # Same leakage-free re-decode + code-threading contract as the gate below.
            from case_study_eval import run_case_study_deliverables
            run_case_study_deliverables(
                model=model, adata=adata, cell_atts=cell_atts, pos_batched=pos_batched,
                sample_indices=sample_indices, sample_id_all=sample_id_all,
                multi_paths=multi_paths, gene_names=gene_names, pert_map=pert_map,
                cfg=cfg, outdir=outdir,
                tier1_id=tier1_idx_all, arch_desc=arch_desc_all,
            )
            return
        if getattr(cfg, "use_inflammation_field", False) or getattr(cfg, "use_timekernel", False):
            # timekernel: the make-or-break forward/backward re-decode trajectory test.
            # (timekernel forces use_inflammation_field=False but still needs the codes-aware
            #  trajectory path -- falling to run_composition_eval, which passes no tier1/niche
            #  codes, would crash the shared-basal decoder.)
            from case_study_eval import run_field_trajectory_test
            run_field_trajectory_test(
                model=model, adata=adata, cell_atts=cell_atts, pos_batched=pos_batched,
                sample_indices=sample_indices, sample_id_all=sample_id_all,
                multi_paths=multi_paths, gene_names=gene_names, pert_map=pert_map,
                cfg=cfg, outdir=outdir,
                # Timekernel per-cell Tier1/niche codes, row-aligned to adata through
                # every filter (built at L1105-1108, subset alongside X). These ARE the
                # training codes -- the shared basal + niche-FiLM embeddings are keyed by
                # them, so the CF re-decode is leakage-free (basal from (tier1,niche) only).
                tier1_id=tier1_idx_all, arch_desc=arch_desc_all,
                # The field's training target, recomputed here (pure fn of adata; the
                # training branch's copy is out of scope on the eval-only path) so the
                # a(t)-by-source-day diagnostic can compare s_i against what supervised it.
                infl_score_all=_compute_inflammation_score(adata),
            )
        else:
            from case_study_eval import run_composition_eval
            run_composition_eval(
                model=model, adata=adata, cell_atts=cell_atts, pos_batched=pos_batched,
                sample_indices=sample_indices, sample_id_all=sample_id_all,
                multi_paths=multi_paths, gene_names=gene_names, pert_map=pert_map,
                cfg=cfg, outdir=outdir,
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
        # build lookup for names -> integer codes
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

    # under use_expr_encoder the basal latent is amortized from the cell's
    # own normalized expression -- thread Y so counterfactuals use the trained encoder
    # rather than the untrained per-cell lookup (seen-cell pert-swap scenario).
    import scipy.sparse as sp_scipy
    Y_expr = adata.X
    Y_expr = Y_expr.toarray() if sp_scipy.issparse(Y_expr) else np.asarray(Y_expr)
    Y_expr = Y_expr.astype(np.float32)

    perturbed_counts, pert_cell_att = model.counterfactualPrediction(
        X=pos_batched,
        sample_index=sample_indices,
        cell_atts=cell_atts.copy(),
        batch_size=512,
        n_samples=25,
        perturb_cell_id=torch.tensor(pert_idx),
        target_cell_tissue=tissue_code,
        target_cell_perturbation=pert_code,
        Y=Y_expr,
    )

    sc.AnnData(perturbed_counts, obs=pd.DataFrame(pert_cell_att, columns=["tissue", "perturbation"])) \
      .write(outdir / f"{cfg.sample}_{cfg.project_index}_{cfg.target_cell_tissue}_{cfg.target_cell_perturbation}_perturbed_counts.h5ad")
    logging.info(
        "Wrote counterfactuals: %s",
        outdir / f"{cfg.sample}_{cfg.project_index}_{cfg.target_cell_tissue}_{cfg.target_cell_perturbation}_perturbed_counts.h5ad",
    )


# CLI
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
    p.add_argument("--stage", choices=["train", "eval", "deconv", "impute_patch", "loto_eval", "composition_eval"])
    p.add_argument("--pseudobulk_file", type=str, default=None)
    p.add_argument("--deconv_perts", type=str, default=None)
    p.add_argument("--deconv_top_k", type=int, default=None)
    p.add_argument("--deconv_roi_radius", type=float, default=None)
    p.add_argument("--deconv_n_samples", type=int, default=None)
    p.add_argument("--mask_index_file", type=str, default=None)
    p.add_argument("--raw_slide_file", type=str, default=None)

    # Selection / training
    p.add_argument("--select_genes", type=int)
    p.add_argument("--train_size", type=float)
    p.add_argument("--holdout_unassigned_frac", type=float)
    p.add_argument("--holdout_unassigned_file", type=str, default=None)
    p.add_argument("--holdout_file", type=str, default=None)
    p.add_argument("--maxiter", type=int)
    p.add_argument("--patience", type=int)
    p.add_argument("--earlystop_min_delta", type=float)
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
    p.add_argument("--cells_per_inducing", type=int)
    p.add_argument("--inducing_steps_min", type=int)
    p.add_argument("--inducing_steps_max", type=int)
    p.add_argument("--slide_tile_gap_factor", type=float)
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

    # Perturbation Action Denoiser (PAD)
    p.add_argument("--use_pad", type=str2bool, help="Enable PAD (default True for spac)")
    p.add_argument("--pad_action_dim", type=int, help="Binary action embedding dim (default 64)")
    p.add_argument("--pad_lambda_mmd", type=float, help="MMD loss weight (downgraded default 0.1)")
    p.add_argument("--pad_lambda_adv", type=float, help="Adversarial GRL loss weight (downgraded default 0.01)")

    # HVG selection
    p.add_argument("--hvg_max_rank", type=int,
                   help="Select genes with hvg_rank <= this threshold (default 10000)")

    # Multi-sample joint training
    p.add_argument("--multi_sample_files", type=str,
                   help="Comma-separated list of h5ad paths for joint training. "
                        "Overrides --data_file when set.")

    # control-anchored explicit action correction
    p.add_argument("--use_action_correction", type=str2bool,
                   help="Enable 3-term decoder with explicit mu_action (default True)")

    # latent disentanglement (drop pert from encoder, add f_pert)
    p.add_argument("--use_basal_encoder_only", type=str2bool,
                   help="If True, drop pert embedding from encoder input -> z becomes pert-blind. "
                        "Requires retraining from scratch (changes encoder input dim).")
    p.add_argument("--use_f_pert", type=str2bool,
                   help="Add per-perturbation gene-space additive log-fold term (counterfactual mechanism)")
    p.add_argument("--f_pert_init_std", type=float, help="Init std of f_pert linear weights")
    p.add_argument("--f_pert_clamp", type=float, help="Symmetric clamp on f_pert log-space output")
    p.add_argument("--fpert_field", type=str, default=None,
                   choices=["constant", "rff", "svgp", "hybrid", "mb_rff", "gp_pp"],
                   help="Coordinate-conditioned footprint field variant (constant = legacy, anti-flat gate)")
    p.add_argument("--fpert_rff_features", type=int, default=None, help="RFF feature count (rff/hybrid)")
    p.add_argument("--fpert_rff_bandwidth", type=float, default=None, help="RFF Gaussian bandwidth ~1/lengthscale")
    p.add_argument("--fpert_hidden", type=int, default=None, help="FpertField MLP hidden width")
    p.add_argument("--fpert_svgp_inducing", type=int, default=None, help="SVGP inducing point count (rounded to a grid)")
    p.add_argument("--fpert_svgp_ls_init", type=float, default=None, help="SVGP initial lengthscale")
    p.add_argument("--fpert_coord_scale", type=float, default=None,
                   help="Divide coords by this before the field (set to LOC_RANGE so coords ~[0,1])")
    p.add_argument("--fpert_mb_bandwidths", type=str, default=None,
                   help="comma list of RFF bandwidths for mb_rff, e.g. 0.05,0.15,0.5")
    p.add_argument("--basal_bayes", type=str2bool, default=None,
                   help="M2: Bayes-linear basal head (default True)")
    p.add_argument("--basal_field", type=str, default=None, choices=["svgp", "rff"],
                   help="Basal spatial latent: 'svgp' (legacy per-pert GP) or 'rff' (collapse-proof RFF field)")
    p.add_argument("--basal_rff_features", type=int, default=None, help="RFF feature count for the basal field")
    p.add_argument("--basal_rff_bandwidth", type=float, default=None, help="RFF Gaussian bandwidth for the basal field")
    p.add_argument("--basal_rff_hidden", type=int, default=None, help="Basal RFF MLP hidden width")
    p.add_argument("--basal_coord_scale", type=float, default=None, help="Divide coords by this before the basal RFF field")
    p.add_argument("--disentangle_radius", type=float, default=None,
                   help="Footprint radius for the scale-aware Moran penalty on gs_mu (>0 enables it)")
    p.add_argument("--use_consistency_loss", type=str2bool,
                   help="Batch-level MMD between KO and control mu_basal -- forces f_pert to absorb pert variance")
    p.add_argument("--consistency_lambda", type=float, help="Weight for the consistency MMD term")

    # inductive expression encoder + collapse fixes
    p.add_argument("--use_expr_encoder", type=str2bool, default=None,
                   help="Amortized basal encoder on expression (inductive; cures posterior collapse)")
    p.add_argument("--expr_hidden_dims", nargs="+", type=int, default=None)
    p.add_argument("--expr_dropout", type=float, default=None)
    p.add_argument("--use_basal_consistency", type=str2bool, default=None,
                   help="MMD on basal_latent between KO and control cells")
    p.add_argument("--basal_consistency_lambda", type=float, default=None)
    p.add_argument("--beta_warmup_epochs", type=int, default=None)
    p.add_argument("--beta_target", type=float, default=None)
    p.add_argument("--free_bits", type=float, default=None)
    p.add_argument("--gp_free_bits", type=float, default=None,
                   help="Per-dim KL free-bits floor (nats) on the GP block. "
                        ">0 keeps GP latent dims from collapsing to the SVGP prior.")
    p.add_argument("--use_pert_correction", type=str2bool, default=None)

    # NNGP local field + precomputed spatial graph
    p.add_argument("--use_nngp", type=str2bool, default=None)
    p.add_argument("--nngp_lambda", type=float, default=None)
    p.add_argument("--graph_file", default=None)
    p.add_argument("--use_paracrine", type=str2bool, default=None)
    p.add_argument("--paracrine_l1", type=float, default=None)
    p.add_argument("--use_niche_interaction", type=str2bool, default=None)
    p.add_argument("--niche_rank", type=int, default=None)
    p.add_argument("--use_niche_film", type=str2bool, default=None)
    p.add_argument("--niche_film_dim", type=int, default=None)
    p.add_argument("--niche_film_k", type=int, default=None)
    p.add_argument("--niche_film_gcap", type=float, default=None)
    p.add_argument("--niche_film_l2", type=float, default=None)
    p.add_argument("--niche_film_rank", type=int, default=None)
    p.add_argument("--niche_film_dispersion", type=str2bool, default=None)
    # timekernel: shared Tier1+arch-descriptor basal, a(t)*Phi(l(t))*f_pert
    p.add_argument("--use_timekernel", type=str2bool, default=None,
                   help="Master switch: shared Tier1+arch-descriptor basal, a(t)*Phi(l(t))*f_pert. "
                        "Forces use_time_conditioning=True, use_action_correction=False, "
                        "use_inflammation_field=False, use_membership_field=False; "
                        "use_shared_basal defaults True.")
    p.add_argument("--use_shared_basal", type=str2bool, default=None,
                   help="Basal = shared Tier1 embedding + Linear(arch_desc), no free per-cell "
                        "latent (defaults to use_timekernel).")
    p.add_argument("--arch_desc_dir", type=str, default=None,
                   help="Directory of per-slice day-invariant architectural-descriptor sidecars "
                        "(<Slice_ID>.npy + manifest.json). Required when --use_timekernel is on; "
                        "replaces the Leiden_neigh niche code, which was a day proxy.")
    p.add_argument("--amp_anchor_offset", type=float, default=None,
                   help="By-construction baseline anchor: a(t)=softplus(g(t)-g(t_ref)-offset), so "
                        "a(t_ref)=softplus(-offset)~0 and day 0 (pre-injury) must be reconstructed "
                        "by basal alone. Identifies the SIGN of f_pert. 0 = off (reproduces the "
                        "inverted-mechanism model); 2.0 = a(0)~0.13 with the gradient still alive.")
    p.add_argument("--amp_anchor_lambda", type=float, default=None,
                   help="Soft fallback: weight of the ELBO penalty lambda*a(t_ref)^2. Inert when "
                        "--amp_anchor_offset is on. NOT sufficient alone -- at lambda=10 the NB "
                        "term out-pulled it and a(0) rose (job 31812173).")
    p.add_argument("--amp_anchor_t", type=float, default=None,
                   help="Normalised time of the healthy baseline for the anchor (default 0.0 = day 0).")
    p.add_argument("--time_in_field_mlp", type=str2bool, default=None,
                   help="Concatenate the time embedding into the fpert field's MLP input, so t "
                        "can reshape the spatial program per (location, time). Without it t "
                        "reaches Phi only via the scalar lengthscale l(t) -- measured near-inert "
                        "(job 31897911: a(t)*shift moved 0.023 over the whole DSS course and "
                        "day-9 cells never resolved). Timekernel only.")
    p.add_argument("--marker_sign_positive", type=str2bool, default=None,
                   help="Force the DSS marker panel's component of the pert shift >= 0 (softplus on "
                        "those gene columns only; the other ~925 genes stay free/signed, since "
                        "resolution genuinely down-regulates genes). The panel is the "
                        "inflammation-UP set by definition, so this is definitional, not a "
                        "prediction. Identifies the sign IN f_pert -- constraining the non-negative "
                        "multipliers a(t)/s_i cannot (they scale a signed shift, never flip it).")
    # continuous-time conditioning (temporal gut DSS)
    p.add_argument("--use_time_conditioning", type=str2bool, default=None,
                   help="Add Fourier time embed + low-rank time x pert coupling so the "
                        "perturbation fold-change evolves over the DSS course.")
    p.add_argument("--time_fourier_bands", type=int, default=None)
    p.add_argument("--time_emb_dim", type=int, default=None)
    p.add_argument("--time_rank", type=int, default=None)
    p.add_argument("--time_l2", type=float, default=None)
    p.add_argument("--time_col", type=str, default=None,
                   help="obs column holding per-cell normalized day t in [0,1] (default day_norm).")
    # Time-varying inflammation composition (membership field)
    p.add_argument("--use_membership_field", type=str2bool, default=None,
                   help="Soft spatiotemporal inflammation-membership field pi in [0,1] gating f_pert "
                        "(composition can evolve over time). Supervised by the per-cell label.")
    p.add_argument("--membership_hidden", type=int, default=None)
    p.add_argument("--membership_bce_lambda", type=float, default=None)
    p.add_argument("--membership_moran_lambda", type=float, default=None)
    p.add_argument("--use_time_intensity", type=str2bool, default=None,
                   help="Keep the legacy low-rank time x pert intensity residual. Off for the "
                        "composition model (all temporal dependence lives in pi).")
    # continuous inflammation field
    p.add_argument("--use_inflammation_field", type=str2bool, default=None,
                   help="Continuous non-negative spatiotemporal inflammation field s=softplus(MLP("
                        "[z_niche, coords, time_emb])) scaling f_pert. MSE-supervised by the observed "
                        "continuous inflammation gene-score; input DROPS gate_state (leak prevention). "
                        "Supersedes --use_membership_field (mutually exclusive).")
    p.add_argument("--field_time_fixed_t", type=float, default=None,
                   help="Freeze the inflammation field's time input at this NORMALIZED day "
                        "(t/21, e.g. 0.0 for day 0). Makes s architecture-only so time flows "
                        "only through Phi. Omit for the current (t-dependent) behaviour.")
    p.add_argument("--field_mse_lambda", type=float, default=None,
                   help="Weight of MSE(s, observed continuous inflammation gene-score).")
    p.add_argument("--field_moran_lambda", type=float, default=None,
                   help="Optional weight rewarding spatial contiguity (Moran's I) of s. 0 = off.")
    # NOTE: --use_consistency_loss / --consistency_lambda already defined above (~1888) -- reuse them.
    # LOTO-D9 temporal evaluation (stage=loto_eval)
    p.add_argument("--loto_holdout_files", type=str, default=None,
                   help="comma-sep held-out (day-9) h5ads used as ground truth for stage=loto_eval.")
    p.add_argument("--loto_target_day", type=float, default=None)
    p.add_argument("--loto_day_denom", type=float, default=None)
    p.add_argument("--case_study_deliverables", type=str2bool, default=None,
                   help="In composition_eval, run the descriptive case-study deliverables "
                        "(C1 per-mouse trajectories, C2 spatial Moran, C3 heterogeneity) on the "
                        "final timekernel ckpt instead of the make-or-break trajectory gate.")
    p.add_argument("--heldout_mouse_eval", type=str2bool, default=None,
                   help="In composition_eval, run the leave-one-mouse-out ground-truth "
                        "reconstruction (deliverable 2). Point --multi_sample_files at the "
                        "held-out mice's slices and load the LOMO ckpt.")
    p.add_argument("--dump_percell", type=str2bool, default=None,
                   help="In heldout_mouse_eval, also write percell_dump.npz (coords, obs, and the "
                        "leakage-free re-decode of every held-out cell at t0/t3/t9/t21) for the "
                        "viz 1-3 spatial figures.")
    p.add_argument("--dump_pathways", type=str2bool, default=None,
                   help="In heldout_mouse_eval, also write pathway_dump.npz: per-cell per-PATHWAY "
                        "counterfactual score at t0/t3/t9/t21 + observed diagonal, for curated "
                        "inflammation/repair programs in pathway_gene_sets.json. Independent of "
                        "--dump_percell.")
    p.add_argument("--slide_bias_ref_files", type=str, default=None,
                   help="Comma list of the TRAINING run's --multi_sample_files in training order. "
                        "When an eval runs on a subset of slices, this re-maps slide_bias rows by "
                        "filename so each eval slice gets its OWN trained bias (unseen slices get "
                        "the training mean) instead of load_model's front-truncation.")
    p.add_argument("--loto_n_samples", type=int, default=None)
    p.add_argument("--loto_batch_size", type=int, default=None)
    p.add_argument("--use_pert_gate", type=str2bool, default=None)
    p.add_argument("--detach_mu_action_for_ko", type=str2bool, default=None)
    p.add_argument("--mu_action_anchor_lambda", type=float, default=None)
    p.add_argument("--gate_hidden", type=int, default=None)
    # spatial-basal disentanglement + mini-bulk LFC supervision
    p.add_argument("--spatial_basal_only", type=str2bool, default=None,
                   help="If True, NTC/KO basal z comes from the spatial field (no own-expr leak); "
                        "perturbation effect forced into f_pert. Requires retraining.")
    p.add_argument("--lfc_lambda", type=float, default=None,
                   help="Weight of mini-bulk KO-vs-NTC LFC supervision on f_pert (0 = off).")
    p.add_argument("--disentangle_lambda", type=float, default=None,
                   help="Weight for HSIC(gs, coords) latent-disentanglement penalty (Change B). 0 = off.")
    p.add_argument("--moran_lambda", type=float, default=None,
                   help="Weight for Moran's-I basal-latent spatial-autocorrelation penalty (Change C). 0 = off.")
    p.add_argument("--deamortize_gp", type=lambda x: str(x).lower() in ("1", "true", "yes"), default=None,
                   help="Lever 2: use a free per-cell GP latent target (recon-driven) instead of the encoder-amortized gp_mu.")
    p.add_argument("--additive_decoder", type=lambda x: str(x).lower() in ("1", "true", "yes"), default=None,
                   help="Option 2: gated additive decoder -- reconstruct log mu as dec_gp(z_gp)+dec_basal(z_basal) "
                        "via two separate paths so the coord-driven GP latent has an un-shared gradient route to "
                        "reconstruction (pair with moran_lambda + small Normal_dim to force spatial signal through z_gp).")
    p.add_argument("--gp_field_aux_lambda", type=float, default=None,
                   help="Route 2: weight of auxiliary loss forcing the coord-driven SVGP field to reconstruct the "
                        "spatially-smoothed expression (un-starves the encoder gp_mu + inducing points). 0 = off.")
    p.add_argument("--basal_moran_output_lambda", type=float, default=None,
                   help="RC2 spatial residualization: weight of Moran's-I penalty on the ADDITIVE basal OUTPUT "
                        "contribution (log_basal, gene space). Forces the smooth spatial component out of the basal "
                        "path and into the coord-driven GP path. Requires --additive_decoder True. 0 = off.")
    p.add_argument("--ls_prior_lambda", type=float, default=None,
                   help="Weight of GP kernel lengthscale band prior (0 = off). Stage 1 guardrail "
                        "against runaway kernel scale that collapses the SVGP.")
    p.add_argument("--ls_prior_min", type=float, default=None,
                   help="Lower edge of allowed kernel lengthscale band (coordinate units).")
    p.add_argument("--ls_prior_max", type=float, default=None,
                   help="Upper edge of allowed kernel lengthscale band (coordinate units).")

    # SPAC-seq cell taxonomy labels (comma-separated)
    p.add_argument("--unassigned_label", type=str,
                   help="Comma-separated labels for unassigned (no-guide) cells")
    p.add_argument("--control_label", type=str,
                   help="Comma-separated labels for non-targeting control cells")
    p.add_argument("--excluded_label", type=str,
                   help="Comma-separated labels for cells excluded entirely (e.g. 'multiple')")
    p.add_argument("--shuffle_top_perts", type=int,
                   help="Negative control: permute labels of the top-N real-KO classes "
                        "among their own cells (destroys pert signal for those N genes). 0=off.")
    p.add_argument("--min_cells_per_pert", type=int,
                   help="Drop perturbation classes with fewer than this many cells (default 10)")

    # Resume from checkpoint
    p.add_argument("--resume", type=str2bool,
                   help="Load existing model_file weights before training (resume from checkpoint)")

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

    if cfg.use_timekernel:
        if cfg.use_shared_basal is None:
            cfg.use_shared_basal = True
        if not cfg.use_time_conditioning:
            logging.warning("[timekernel] forcing use_time_conditioning=True "
                            "(the a(t) amplitude / time coupling needs time).")
            cfg.use_time_conditioning = True
        if cfg.use_action_correction:
            logging.warning("[timekernel] forcing use_action_correction=False "
                            "(DSS has no NTC / action-correction anchor).")
            cfg.use_action_correction = False
        if cfg.use_membership_field:
            logging.warning("[timekernel] forcing use_membership_field=False "
                            "(timekernel replaces the membership field).")
            cfg.use_membership_field = False
    elif cfg.use_shared_basal is None:
        cfg.use_shared_basal = False

    if cfg.time_in_field_mlp and not cfg.use_timekernel:
        logging.warning("[time-field] --time_in_field_mlp has NO effect without "
                        "--use_timekernel (the field is only time-wired in the timekernel "
                        "path); ignoring it.")
        cfg.time_in_field_mlp = False

    if cfg.marker_sign_positive and not cfg.use_timekernel:
        # The constraint is applied only inside the timekernel branch of the shift path;
        # anywhere else it would be a silent no-op, and a silently-ignored sign constraint
        # is exactly the kind of thing that gets read as evidence later.
        logging.warning("[marker-sign] --marker_sign_positive has NO effect without "
                        "--use_timekernel (the constraint lives in the timekernel shift "
                        "path); ignoring it.")
        cfg.marker_sign_positive = False

    return cfg

if __name__ == "__main__":
    # DDP bootstrap
    # torchrun sets LOCAL_RANK / RANK / WORLD_SIZE.  Fall back to single-GPU.
    _local_rank  = int(os.environ.get("LOCAL_RANK", 0))
    _rank        = int(os.environ.get("RANK", 0))
    _world_size  = int(os.environ.get("WORLD_SIZE", 1))

    if _world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(_local_rank)

    cfg = parse_args()

    # Override device for this rank
    cfg = replace(cfg, device=f"cuda:{_local_rank}")

    # Only rank 0 writes to log files / stdout so other ranks stay quiet
    if _rank == 0:
        setup_logging(cfg.verbosity)
    else:
        logging.disable(logging.CRITICAL)

    run(cfg, rank=_rank, world_size=_world_size)

    if _world_size > 1:
        dist.destroy_process_group()
