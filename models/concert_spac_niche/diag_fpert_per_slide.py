"""Load a multi-slide checkpoint and assert the gp_pp footprint is non-zero on EVERY slide.

Decisive check for C1 (fpert per-slide-local coords): before C1, only slide 0 (untiled
origin) got a footprint; slides 1+ fell outside the [0,1] Z-grid and produced ~0 residual.
This evaluates the FpertField alone on each slide's tiled coords and fails loudly if any
slide collapses or is >100x weaker than the strongest.

Usage:
  python diag_fpert_per_slide.py --model_file <OUT>/model.pt \
      --multi_sample_files "<target_first>,<slide2>,..." [--hvg_max_rank 2000]
"""
import argparse
import logging
import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(message)s")

LOC_RANGE = 20.0        # RunConfig.loc_range
GAP_FACTOR = 2.5        # RunConfig.slide_tile_gap_factor
COORD_SCALE = 20.0      # --fpert_coord_scale used in training

p = argparse.ArgumentParser()
p.add_argument("--model_file", required=True)
p.add_argument("--multi_sample_files", required=True)   # comma-sep, target slide first
p.add_argument("--hvg_max_rank", type=int, default=2000)
args = p.parse_args()

import run_concert_spac as R
from concert_spac import FpertField

paths = [s for s in args.multi_sample_files.split(",") if s.strip()]
# load_h5ad_multi returns: X, pos_raw, _, perturb_raw, has_guide, is_real_KO, gene_names, sample_id
_, pos_raw, _, _, _, _, _, sample_id = R.load_h5ad_multi(paths, hvg_max_rank=args.hvg_max_rank)
pos_scaled, layout, spacing = R.tile_multislide_coords(pos_raw, sample_id, LOC_RANGE, GAP_FACTOR)
logging.info("tiled %d slides, spacing=%.1f", len(np.unique(sample_id)), spacing)

# save_model writes a bare state_dict
sd = torch.load(args.model_file, map_location="cpu")
if not any(k.startswith("fpert_field.") for k in sd):
    raise SystemExit("checkpoint has no fpert_field.* keys — not a gp_pp/rff fpert model")
enc_dim = sd["fpert_field.ls_head.weight"].shape[1]     # gp_pp has ls_head (encoder_dim -> 1)
n_genes = sd["fpert_field.mlp.2.weight"].shape[0]
m_ind = sd["fpert_field.Z"].shape[0]                    # perfect square (e.g. 16)

f = FpertField(encoder_dim=enc_dim, num_genes=n_genes, field="gp_pp",
               svgp_inducing=m_ind, coord_scale=COORD_SCALE,
               tile_spacing=LOC_RANGE * GAP_FACTOR)
f.load_state_dict({k[len("fpert_field."):]: v for k, v in sd.items()
                   if k.startswith("fpert_field.")}, strict=False)
f.eval()

emb = torch.zeros(1, enc_dim)   # fixed probe embedding (footprint shape is coord-driven)
mags = {}
for sid in np.unique(sample_id):
    m = sample_id == sid
    c = torch.tensor(pos_scaled[m][:512], dtype=torch.float32)
    e = emb.expand(c.shape[0], -1)
    with torch.no_grad():
        resid = f(e, c)                                 # (n, G)
    mags[int(sid)] = float(resid.abs().mean())
    logging.info("slide %s  mean|footprint| = %.4e", sid, mags[int(sid)])

vals = np.array(list(mags.values()))
assert (vals > 1e-4).all(), f"footprint collapsed on some slide: {mags}"
assert vals.min() * 100 > vals.max(), f"footprint magnitude uneven across slides (>100x): {mags}"
print("PASS per-slide footprint:", mags)
