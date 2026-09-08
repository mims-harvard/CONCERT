"""Patch-level footprint eval for perturbmap s0 (no niche grid).

perturbmap s0 has 1 Jak2 patch + 2 Tgfbr2 patches. There is NO NTC class here (all
non-KO cells collapsed to "unperturbed"), so we use the WITH-noise footprint: for each
patch compare the model's predicted (KO target vs unperturbed target = μ_action+f_pert)
against the observed KO-vs-unperturbed pseudobulk LFC, by Spearman over HVG genes.
Runs all 4 field variants (constant/rff/svgp/hybrid) and prints a table.
"""
import numpy as np, torch, logging, sys
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from make_umap import load_data, build_model
import eval_utils as eu

logging.basicConfig(level=logging.INFO, format="%(message)s")
DATA = "/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/spaVAE/datasets/perturbmap_s0_spacfmt.h5ad"
BASE = "/n/netscratch/mzitnik_lab/Lab/xiang/CONCERT/outputs"
VARIANTS = {
    "constant": dict(fpert_field="constant"),
    "rff":      dict(fpert_field="rff", fpert_rff_features=128, fpert_rff_bandwidth=0.15, fpert_coord_scale=20.0),
    "svgp":     dict(fpert_field="svgp", fpert_svgp_inducing=16, fpert_svgp_ls_init=5.0, fpert_coord_scale=20.0),
    "hybrid":   dict(fpert_field="hybrid", fpert_rff_features=128, fpert_rff_bandwidth=0.15, fpert_coord_scale=20.0),
}
N_PATCH = {"Jak2": 1, "Tgfbr2": 2}

# --- data (shared across variants; load once) ---
(adata, pos_batched, cell_atts, batch_oh, pert_name, cutoff_t,
 perturb_raw, has_guide, is_real_KO, n_batch, perturb_idx) = load_data(
    DATA, hvg_max_rank=2000, min_cells_per_pert=20)
name2code = {v: k for k, v in pert_name.items()}
perturb_raw = np.asarray(perturb_raw)
coords = pos_batched[:, :2].astype(float)
Yexpr = np.asarray(adata.X.todense() if hasattr(adata.X, "todense") else adata.X, dtype=np.float32)
Yobs = adata.raw.X if adata.raw is not None else adata.X
Yobs = np.asarray(Yobs.todense() if hasattr(Yobs, "todense") else Yobs, dtype=np.float32)
N = adata.n_obs
ctrl_mask = perturb_raw == "unperturbed"
logging.info("N=%d  unperturbed=%d  Jak2=%d  Tgfbr2=%d  codes=%s",
             N, ctrl_mask.sum(), (perturb_raw=="Jak2").sum(), (perturb_raw=="Tgfbr2").sum(), pert_name)

# --- define patches (spatial KMeans per KO gene) ---
patches = []  # (label, gene, code, idx_array)
for gene, k in N_PATCH.items():
    gidx = np.where(perturb_raw == gene)[0]
    if len(gidx) == 0:
        logging.warning("no %s cells", gene); continue
    if k == 1 or len(gidx) < k:
        labs = np.zeros(len(gidx), dtype=int)
    else:
        labs = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(coords[gidx])
    for c in np.unique(labs):
        pidx = gidx[labs == c]
        patches.append((f"{gene}_p{c}", gene, name2code[gene], pidx))
        logging.info("patch %s: %d cells, centroid=%s", f"{gene}_p{c}", len(pidx),
                     np.round(coords[pidx].mean(0), 1))

ctrl_code = name2code["unperturbed"]

def eval_variant(v, extra):
    ck = f"{BASE}/fpert_pmap_s0_{v}/model_pmap_s0_fpert_{v}.pt"
    model, device = build_model(adata, pos_batched, cell_atts, cutoff_t, n_batch,
                                use_basal_encoder_only=True, spatial_basal_only=True,
                                use_pert_gate=True, gate_hidden=64,
                                use_expr_encoder=True, expr_hidden_dims=[512, 256],
                                use_niche_interaction=True, niche_rank=4,
                                ind_steps=4, f_pert_clamp=5.0, **extra)
    model.load_model(ck, allow_cell_size_mismatch=True)
    model.eval()
    si = torch.arange(N)
    allids = np.arange(N)
    out = {}
    for gene in N_PATCH:
        gcode = name2code[gene]
        # KO target: force μ_action + f_pert ON even if the KO maps to code 0 (Jak2=0).
        pred_g, _ = model.counterfactualPrediction(
            X=pos_batched, sample_index=si, cell_atts=cell_atts, perturb_cell_id=allids,
            target_cell_perturbation=gcode, target_has_guide=True, target_is_real_KO=True,
            n_samples=5, batch_size=512, Y=Yexpr)
        # basal baseline: force shift OFF (no μ_action, no f_pert) regardless of ctrl code.
        pred_b, _ = model.counterfactualPrediction(
            X=pos_batched, sample_index=si, cell_atts=cell_atts, perturb_cell_id=allids,
            target_cell_perturbation=ctrl_code, target_has_guide=False, target_is_real_KO=False,
            n_samples=5, batch_size=512, Y=Yexpr)
        out[gene] = (np.asarray(pred_g), np.asarray(pred_b))
    res = {}
    for lab, gene, gcode, pidx in patches:
        pred_g, pred_b = out[gene]
        pred_lfc = eu.pseudobulk_lfc(pred_g[pidx], pred_b[pidx])         # predicted μ_action+f_pert
        obs_lfc  = eu.pseudobulk_lfc(Yobs[pidx], Yobs[ctrl_mask])        # observed KO vs unperturbed
        rho = spearmanr(pred_lfc, obs_lfc).correlation
        res[lab] = rho
    del model; torch.cuda.empty_cache()
    return res

rows = {}
for v, extra in VARIANTS.items():
    try:
        rows[v] = eval_variant(v, extra)
        logging.info("[%s] %s", v, {k: round(x, 4) for k, x in rows[v].items()})
    except Exception as e:
        import traceback; logging.error("[%s] FAILED %s", v, e); traceback.print_exc()

# --- table ---
labels = [p[0] for p in patches]
print("\n===== perturbmap s0 patch footprint Spearman (predicted vs observed KO-vs-unperturbed, with-noise) =====")
print("variant   | " + " | ".join(f"{l:>11s}" for l in labels) + " |   mean")
for v in VARIANTS:
    if v not in rows: print(f"{v:9s} | FAILED"); continue
    vals = [rows[v].get(l, float('nan')) for l in labels]
    m = np.nanmean(vals)
    print(f"{v:9s} | " + " | ".join(f"{x:11.4f}" for x in vals) + f" | {m:7.4f}")
print("\nGate: spatial variants (rff/svgp/hybrid) should beat 'constant' if the footprint is real.")
