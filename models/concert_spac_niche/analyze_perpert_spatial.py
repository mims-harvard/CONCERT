"""Paper-figure analysis: show the multi-slide CONCERT-spac model learns a DISTINCT
spatial perturbation effect for each perturbation.

Produces (into --fig_dir):
  fig1_embeddings.png      per-pert embedding cosine-sim + norms  (embeddings are distinct)
  fig2_lengthscale.png     per-pert learned footprint lengthscale ls=softplus(ls_head(emb))
  fig3_core_program.png    core f_pert(emb) gene-program: heatmap + Jak2-vs-Tgfbr2 scatter +
                           #genes-moved bar + |shift| distributions  (DIFFERENT programs)
  fig4_footprint_slides.png  learned spatial field magnitude rendered on EVERY slide's real
                             coords, one row per perturbation  (the kernel on tissue space)
and perpert_summary.json with all numbers.

Reuses the training data loader so coordinate preprocessing is identical to training.

Usage:
  python analyze_perpert_spatial.py --model_file <OUT>/model.pt \
      --multi_sample_files "s0,s1,s2,s3" --gene_names <OUT>/hvg_gene_names.txt \
      --fig_dir <docs>/figures/2026-07-02-perpert-spatial
"""
import argparse, json, logging, os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(message)s")

LOC_RANGE = 20.0
GAP_FACTOR = 2.5
COORD_SCALE = 20.0

p = argparse.ArgumentParser()
p.add_argument("--model_file", required=True)
p.add_argument("--multi_sample_files", required=True)
p.add_argument("--gene_names", required=True)
p.add_argument("--fig_dir", required=True)
p.add_argument("--hvg_max_rank", type=int, default=2000)
args = p.parse_args()
os.makedirs(args.fig_dir, exist_ok=True)

import run_concert_spac as R
from concert_spac import FpertField

# perturbation codes are sorted-unique label strings -> 0..K-1 (run_concert_spac.py:799)
# confirmed by cutoffs CSV: 0=Jak2, 1=Tgfbr2, 2=unperturbed
NAMES = ["Jak2", "Tgfbr2", "unperturbed"]
COLORS = {"Jak2": "#2563eb", "Tgfbr2": "#dc2626", "unperturbed": "#6b7280"}

paths = [s for s in args.multi_sample_files.split(",") if s.strip()]
_, pos_raw, _, _, _, _, _, sample_id = R.load_h5ad_multi(paths, hvg_max_rank=args.hvg_max_rank)
pos_scaled, layout, spacing = R.tile_multislide_coords(pos_raw, sample_id, LOC_RANGE, GAP_FACTOR)
logging.info("tiled %d slides, spacing=%.1f", len(np.unique(sample_id)), spacing)

sd = torch.load(args.model_file, map_location="cpu")
emb_table = sd["lord_encoder.s_encoder.perturbation.weight"]          # (K, E)
K, enc_dim = emb_table.shape
n_genes = sd["fpert_field.mlp.2.weight"].shape[0]
m_ind = sd["fpert_field.Z"].shape[0]
gene_names = [l.strip() for l in open(args.gene_names)][:n_genes]
if K != len(NAMES):
    NAMES = [f"pert{k}" for k in range(K)]
    COLORS = {n: c for n, c in zip(NAMES, ["#2563eb", "#dc2626", "#6b7280", "#059669"])}

f = FpertField(encoder_dim=enc_dim, num_genes=n_genes, field="gp_pp",
               svgp_inducing=m_ind, coord_scale=COORD_SCALE,
               tile_spacing=LOC_RANGE * GAP_FACTOR)
f.load_state_dict({k[len("fpert_field."):]: v for k, v in sd.items()
                   if k.startswith("fpert_field.")}, strict=False)
f.eval()

# ---- per-pert scalars ---------------------------------------------------------
fw = sd["f_pert.weight"]                       # (G, E)
core = (emb_table @ fw.t()).detach()           # (K, G) spatially-constant gene program
lsW, lsB = sd["fpert_field.ls_head.weight"], sd["fpert_field.ls_head.bias"]
ls = F.softplus(emb_table @ lsW.t() + lsB).clamp(min=1e-2).squeeze(-1).detach()  # (K,)

summary = {"perturbations": NAMES, "encoder_dim": int(enc_dim), "n_genes": int(n_genes),
           "lengthscale": {}, "core": {}, "embedding": {}}
cos_emb = F.cosine_similarity(emb_table.unsqueeze(1), emb_table.unsqueeze(0), dim=-1).detach().numpy()
for i, n in enumerate(NAMES):
    summary["lengthscale"][n] = float(ls[i])
    summary["core"][n] = {"l2": float(core[i].norm()), "mean_abs": float(core[i].abs().mean()),
                          "n_genes_gt0.5": int((core[i].abs() > 0.5).sum())}
    summary["embedding"][n] = {"l2_norm": float(emb_table[i].norm())}

# ============================ FIG 1: embeddings ================================
fig, ax = plt.subplots(1, 2, figsize=(9, 4))
im = ax[0].imshow(cos_emb, vmin=-1, vmax=1, cmap="RdBu_r")
ax[0].set_xticks(range(K)); ax[0].set_yticks(range(K))
ax[0].set_xticklabels(NAMES, rotation=30, ha="right"); ax[0].set_yticklabels(NAMES)
for i in range(K):
    for j in range(K):
        ax[0].text(j, i, f"{cos_emb[i,j]:.2f}", ha="center", va="center",
                   color="white" if abs(cos_emb[i, j]) > 0.5 else "black", fontsize=10)
ax[0].set_title("Perturbation embedding\ncosine similarity")
fig.colorbar(im, ax=ax[0], fraction=0.046)
norms = [float(emb_table[i].norm()) for i in range(K)]
ax[1].bar(NAMES, norms, color=[COLORS[n] for n in NAMES])
ax[1].set_ylabel("embedding L2 norm"); ax[1].set_title("Embedding magnitude")
ax[1].set_xticklabels(NAMES, rotation=30, ha="right")
fig.tight_layout(); fig.savefig(f"{args.fig_dir}/fig1_embeddings.png", dpi=150); plt.close(fig)

# ============================ FIG 2: lengthscale ==============================
fig, ax = plt.subplots(figsize=(4.5, 4))
bars = ax.bar(NAMES, [float(ls[i]) for i in range(K)], color=[COLORS[n] for n in NAMES])
ax.axhspan(3, 18, color="green", alpha=0.08, label="ls_prior band [3,18]")
ax.set_ylabel("footprint lengthscale  ls = softplus(ls_head(emb))")
ax.set_title("Per-perturbation learned footprint radius")
for b, i in zip(bars, range(K)):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.05, f"{float(ls[i]):.2f}",
            ha="center", fontsize=10)
ax.legend(); ax.set_xticklabels(NAMES, rotation=30, ha="right")
fig.tight_layout(); fig.savefig(f"{args.fig_dir}/fig2_lengthscale.png", dpi=150); plt.close(fig)

# ============================ FIG 3: core gene program ========================
core_np = core.numpy()
var_across = core_np.var(axis=0)
top = np.argsort(var_across)[::-1][:40]
fig = plt.figure(figsize=(13, 4.2))
# (a) heatmap of top variable genes
axa = fig.add_subplot(1, 3, 1)
hm = core_np[:, top]
vmax = np.abs(hm).max()
im = axa.imshow(hm, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
axa.set_yticks(range(K)); axa.set_yticklabels(NAMES)
axa.set_xticks(range(0, len(top), 4))
axa.set_xticklabels([gene_names[top[j]] for j in range(0, len(top), 4)], rotation=90, fontsize=6)
axa.set_title("Core response f_pert(emb): top-40 variable genes")
fig.colorbar(im, ax=axa, fraction=0.046)
# (b) Jak2 vs Tgfbr2 per-gene shift scatter
axb = fig.add_subplot(1, 3, 2)
axb.scatter(core_np[0], core_np[1], s=4, alpha=0.4, color="#111")
lim = np.abs(core_np[:2]).max() * 1.05
axb.plot([-lim, lim], [-lim, lim], "r--", lw=0.8)
axb.axhline(0, color="gray", lw=0.5); axb.axvline(0, color="gray", lw=0.5)
axb.set_xlabel(f"{NAMES[0]} gene shift"); axb.set_ylabel(f"{NAMES[1]} gene shift")
r = float(np.corrcoef(core_np[0], core_np[1])[0, 1])
axb.set_title(f"Per-gene program: {NAMES[0]} vs {NAMES[1]}\n(off-diagonal = distinct; r={r:.2f})")
# (c) # genes moved bar
axc = fig.add_subplot(1, 3, 3)
ng = [int((np.abs(core_np[i]) > 0.5).sum()) for i in range(K)]
axc.bar(NAMES, ng, color=[COLORS[n] for n in NAMES])
axc.set_ylabel("# genes with |shift| > 0.5")
axc.set_title("Program breadth")
for i, v in enumerate(ng):
    axc.text(i, v + 5, str(v), ha="center", fontsize=10)
axc.set_xticklabels(NAMES, rotation=30, ha="right")
fig.tight_layout(); fig.savefig(f"{args.fig_dir}/fig3_core_program.png", dpi=150); plt.close(fig)

# ============================ FIG 4: field on slide space =====================
sids = sorted(np.unique(sample_id).tolist())
fig, axes = plt.subplots(K, len(sids), figsize=(3.0 * len(sids), 2.9 * K), squeeze=False)
field_stats = {}
for i, n in enumerate(NAMES):
    e_row = emb_table[i:i + 1]
    field_stats[n] = {}
    # global color scale per pert across slides
    permag_all = {}
    for sid in sids:
        m = sample_id == sid
        c = torch.tensor(pos_scaled[m], dtype=torch.float32)
        with torch.no_grad():
            resid = f(e_row.expand(c.shape[0], -1), c)          # (n,G)
        permag_all[sid] = resid.norm(dim=1).numpy()
    vmin = min(v.min() for v in permag_all.values())
    vmax = max(v.max() for v in permag_all.values())
    for j, sid in enumerate(sids):
        m = sample_id == sid
        xy = pos_raw[m]
        sc = axes[i][j].scatter(xy[:, 0], xy[:, 1], c=permag_all[sid], s=6,
                                cmap="viridis", vmin=vmin, vmax=vmax)
        axes[i][j].set_aspect("equal"); axes[i][j].invert_yaxis()
        axes[i][j].set_xticks([]); axes[i][j].set_yticks([])
        if i == 0:
            axes[i][j].set_title(f"slide {sid}")
        if j == 0:
            axes[i][j].set_ylabel(f"{n}\nls={float(ls[i]):.2f}", fontsize=10)
        field_stats[n][f"slide{sid}"] = {"mean": float(permag_all[sid].mean()),
                                         "spatial_std": float(permag_all[sid].std())}
    fig.colorbar(sc, ax=axes[i, :].tolist(), fraction=0.02, pad=0.01,
                 label="||field||")
fig.suptitle("Learned per-perturbation spatial field magnitude on tissue coordinates", y=1.0)
fig.savefig(f"{args.fig_dir}/fig4_footprint_slides.png", dpi=150, bbox_inches="tight"); plt.close(fig)
summary["field_on_slides"] = field_stats

with open(f"{args.fig_dir}/perpert_summary.json", "w") as fh:
    json.dump(summary, fh, indent=2)
logging.info("Wrote figures + perpert_summary.json to %s", args.fig_dir)
for n in NAMES:
    s = summary["core"][n]
    logging.info("  %-12s ls=%.3f  core_l2=%.1f  genes>0.5=%d",
                 n, summary["lengthscale"][n], s["l2"], s["n_genes_gt0.5"])
print("DONE")
