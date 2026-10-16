#!/usr/bin/env python3
"""
profile_train.py -- Phase 1.4: Training profiler for CONCERT-spac.

Runs a short training session (10 epochs, small batch) with PyTorch profiler
to identify the top bottlenecks. Output is a Chrome trace JSON and a text
summary of the slowest ops.

Usage:
    python profile_train.py \
        --data_file /path/to/raw_output_1-1.h5ad \
        --n_epochs 10 \
        --batch_size 512 \
        --outdir ./profile_output

Interpretation:
    - If "Linear" ops dominate -> decoder over 10k genes is the bottleneck
      (expected; bf16 autocast gives ~2x speedup here)
    - If "csr_matrix.toarray" appears -> OOM fix for sparse loading not applied
    - If "KMeans.fit" dominates -> inducing point construction is slow
      (won't repeat after first epoch)
    - If "approximate_posterior_params" dominates -> SVGP itself is bottleneck
      (rare; M=36 should be fast)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.profiler import profile, record_function, ProfilerActivity, schedule


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile CONCERT-spac training")
    parser.add_argument("--data_file", required=True)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--outdir", default="./profile_output")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Reuse the run_concert_spac machinery for data loading / model init
    sys.argv = [
        "run_concert_spac.py",
        "--config",     args.config,
        "--data_file",  args.data_file,
        "--outdir",     str(outdir / "model"),
        "--sample",     "profile",
        "--stage",      "train",
        "--maxiter",    str(args.n_epochs),
        "--batch_size", str(args.batch_size),
        "--patience",   "9999",
    ]

    from run_concert_spac import parse_args, run, setup_logging
    import logging
    setup_logging(1)

    cfg = parse_args()

    # Load data
    from run_concert_spac import (
        load_h5_dataset, build_attributes_spac, ensure_dir,
        build_inducing_points, auto_batch_size,
    )
    import scanpy as sc
    from sklearn.preprocessing import MinMaxScaler
    from preprocess import normalize
    import frozendict  # type: ignore  # may not be available -- OK

    unassigned_labels = frozenset(l.strip() for l in cfg.unassigned_label.split(",") if l.strip())
    control_labels    = frozenset(l.strip() for l in cfg.control_label.split(",") if l.strip())

    logging.info("Loading data for profiling...")
    t0 = time.time()
    X, pos_raw, _, perturb_raw, has_guide, is_real_KO, gene_names = load_h5_dataset(
        args.data_file,
        unassigned_labels=unassigned_labels,
        control_labels=control_labels,
    )
    logging.info("Data loaded in %.1fs: %s", time.time() - t0, X.shape)

    tissue_idx, tissue_dict, perturb_idx, pert_map = build_attributes_spac(perturb_raw)
    n_batch = int(len(np.unique(perturb_idx)))
    batch = np.eye(n_batch, dtype=np.float32)[perturb_idx]

    scaler = MinMaxScaler()
    pos_scaled = scaler.fit_transform(pos_raw) * cfg.loc_range
    pos_batched = np.concatenate([pos_scaled, batch], axis=1).astype(np.float32)

    inducing = build_inducing_points(
        pos_batched=pos_batched, n_batch=n_batch,
        steps=cfg.inducing_point_steps, loc_range=cfg.loc_range,
        grid=cfg.grid_inducing_points, k_clusters=cfg.inducing_point_nums,
    )

    adata = sc.AnnData(X, dtype="float32")
    adata = normalize(adata, size_factors=True, normalize_input=True, logtrans_input=True)

    spatial_dims = pos_batched.shape[1] - n_batch
    kernel_scale = np.full((n_batch, spatial_dims), cfg.kernel_scale, dtype=np.float32)
    cutoff = np.full(pos_scaled.shape[0], 0.5, dtype=np.float32)
    cell_atts = np.c_[tissue_idx, perturb_idx].astype(int)

    from concert_spac import CONCERT
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
        use_pad=cfg.use_pad,
        pad_action_dim=cfg.pad_action_dim,
        pad_lambda_mmd=cfg.pad_lambda_mmd,
        pad_lambda_adv=cfg.pad_lambda_adv,
        use_action_correction=cfg.use_action_correction,
    )
    logging.info("Model initialized.")

    # Profile
    trace_path = str(outdir / "trace")
    logging.info("Starting profiler. Trace -> %s", trace_path)

    prof_schedule = schedule(wait=1, warmup=1, active=5, repeat=1)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=prof_schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_path),
        record_shapes=True,
        with_stack=False,
        profile_memory=True,
    ) as prof:
        model.train_model(
            pos=pos_batched,
            ncounts=adata.X,
            raw_counts=adata.raw.X,
            size_factors=adata.obs.size_factors,
            batch=batch,
            has_guide=has_guide.astype(np.float32),
            is_real_KO=is_real_KO.astype(np.float32),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            batch_size=args.batch_size,
            maxiter=args.n_epochs,
            patience=9999,
            save_model=False,
            train_size=1.0,   # no val set during profiling
        )
        prof.step()

    # Print summary
    summary = prof.key_averages().table(
        sort_by="cuda_time_total", row_limit=30
    )
    print(summary)
    (outdir / "profile_summary.txt").write_text(summary)
    logging.info("Profile summary saved -> %s", outdir / "profile_summary.txt")
    logging.info("Chrome trace -> %s (open in chrome://tracing)", trace_path)


if __name__ == "__main__":
    main()
