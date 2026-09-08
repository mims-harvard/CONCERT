"""D4 (LFC) patch-footprint eval for the P0 BASAL-field gate (rff vs svgp basal) on perturbmap.

Uses USER-DEFINED patches (concert_data/.../patches/slide{N}.npz) instead of KMeans:
  role==1 = source CONTROL cells where we make the counterfactual prediction;
  target_bin = niche holding the REAL-KO ground-truth footprint (same niche as src for
  patches 0/1/4/8, NEIGHBORING niche for 2/3/6/7); target_ko = Jak2|Tgfbr2.

Per patch:
  predicted footprint LFC = pseudobulk_lfc( model_KO(src) , model_basal(src) )
  observed footprint LFC  = pseudobulk_lfc( Yobs[realKO in target_bin] , Yobs[control in target_bin] )
  score = Spearman(pred_lfc, obs_lfc) over the 2000 HVG genes.

Patch rows map to model rows by cell_index (verified: perturbation_canon positional match=1.0,
load_data keeps all 1903 cells in native order; cell_index is a permutation of arange(N)).
Loops basal_field in {svgp, rff} so one run gives the rff-vs-svgp comparison table.
"""
import argparse, json, logging, os
from pathlib import Path
import numpy as np, torch
from scipy.stats import spearmanr
from make_umap import load_data, build_model
import eval_utils as eu

logging.basicConfig(level=logging.INFO, format="%(message)s")

DATA = "/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/spaVAE/datasets/perturbmap_s0_spacfmt.h5ad"
BASE = "/n/netscratch/mzitnik_lab/Lab/xiang/CONCERT/outputs"
GENES = ["Jak2", "Tgfbr2"]
MIN_SUPPORT = 5   # min real-KO and min control cells in target_bin to trust a patch's obs_lfc


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--patch_file",
                   default="/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/concert_data/exp/perturb_map/patches/slide0.npz")
    p.add_argument("--data", default=DATA)
    p.add_argument("--fields", nargs="+", default=["svgp", "rff"],
                   help="arm labels; substituted into --ckpt_template as {field} and used as basal_field")
    p.add_argument("--ckpt_template", default=f"{BASE}/basal_pmap_s0_{{field}}/model_pmap_s0_basal_{{field}}.pt",
                   help="checkpoint path; {field} is replaced per arm (P1: point at p1_* ckpts)")
    p.add_argument("--basal_field", default="rff", help="basal field of the loaded ckpts (P1 holds at rff)")
    p.add_argument("--fpert_field", default="rff", help="perturbation field: rff|gp_pp|mb_rff (MUST match training)")
    p.add_argument("--basal_bayes", type=lambda s: str(s).lower() == "true", default=False)
    p.add_argument("--fpert_mb_bandwidths", default="", help="comma list for mb_rff")
    p.add_argument("--fpert_svgp_inducing", type=int, default=16,
                   help="gp_pp grid = round(sqrt(N))^2; MUST match training (16 -> 4x4).")
    p.add_argument("--fpert_svgp_ls_init", type=float, default=5.0,
                   help="gp_pp initial lengthscale; MUST match training.")
    p.add_argument("--niche_rank", type=int, default=4)
    p.add_argument("--outdir", default=f"{BASE}/eval_basal_patches/pmap_s0")
    p.add_argument("--n_samples", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=512)
    # multi-slide models: action correction OFF (no mu_action) + persisted vocab
    p.add_argument("--use_action_correction", type=lambda s: str(s).lower() == "true", default=True)
    p.add_argument("--pert_map_json", default="",
                   help="path to training pert_map.json (default: <ckpt dir>/pert_map.json). "
                        "When present, the perturbation vocab is taken from it (basal=code 0) "
                        "instead of rebuilt from --data (avoids eval-vocab mis-indexing).")
    return p.parse_args()


def predict_fields(field, adata, pos_batched, cell_atts, cutoff_t, n_batch,
                   name2code, Yexpr, n_samples, batch_size, args):
    """Build the model for one arm, return {gene: (pred_KO[N,G], pred_basal[N,G])}.

    The arm's basal field is fixed by --basal_field (P1 holds rff); ``field`` labels the arm
    and is substituted into --ckpt_template. The perturbation field + Bayes head come from args.
    """
    ck = args.ckpt_template.format(field=field)
    extra = {}
    if args.fpert_field == "mb_rff" and args.fpert_mb_bandwidths:
        extra["fpert_mb_bandwidths"] = [float(x) for x in args.fpert_mb_bandwidths.split(",") if x.strip()]
    if args.fpert_field == "gp_pp":
        extra["fpert_svgp_inducing"] = args.fpert_svgp_inducing
        extra["fpert_svgp_ls_init"] = args.fpert_svgp_ls_init
    model, device = build_model(
        adata, pos_batched, cell_atts, cutoff_t, n_batch,
        use_basal_encoder_only=True, spatial_basal_only=True,
        use_pert_gate=True, gate_hidden=64,
        use_expr_encoder=True, expr_hidden_dims=[512, 256],
        use_niche_interaction=True, niche_rank=args.niche_rank,
        ind_steps=4, f_pert_clamp=5.0,
        use_action_correction=args.use_action_correction,
        basal_field=args.basal_field, basal_bayes=args.basal_bayes,
        fpert_field=args.fpert_field, fpert_rff_features=128, fpert_rff_bandwidth=0.15,
        fpert_coord_scale=20.0, **extra)
    model.load_model(ck, allow_cell_size_mismatch=True)
    model.eval()
    N = adata.n_obs
    si = torch.arange(N); allids = np.arange(N)
    out = {}
    for gene in GENES:
        gcode = name2code[gene]
        pred_g, _ = model.counterfactualPrediction(
            X=pos_batched, sample_index=si, cell_atts=cell_atts, perturb_cell_id=allids,
            target_cell_perturbation=gcode, target_has_guide=True, target_is_real_KO=True,
            n_samples=n_samples, batch_size=batch_size, Y=Yexpr)
        # basal reference = reserved basal slot (code 0 in the persisted vocab; falls
        # back to a legacy "unperturbed" key for old single-slide vocabs).
        basal_code = name2code.get("__basal__", name2code.get("unperturbed", 0))
        pred_b, _ = model.counterfactualPrediction(
            X=pos_batched, sample_index=si, cell_atts=cell_atts, perturb_cell_id=allids,
            target_cell_perturbation=basal_code, target_has_guide=False,
            target_is_real_KO=False, n_samples=n_samples, batch_size=batch_size, Y=Yexpr)
        out[gene] = (np.asarray(pred_g), np.asarray(pred_b))
    del model; torch.cuda.empty_cache()
    return out


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Load the model's EXACT training gene order (persisted next to the checkpoint). The
    # multi-slide model's avg-rank HVG set differs from this slide's per-slide HVG rank
    # (only ~39% overlap, ~1% same-position), so without this the decoder output and the
    # model input are gene-misaligned → predicted LFC uncorrelated with observed (rho~0).
    ckdir = Path(args.ckpt_template.format(field=args.fields[0])).parent
    gpath = ckdir / "hvg_gene_names.txt"
    force_go = None
    if gpath.exists():
        force_go = np.array([l.strip() for l in open(gpath) if l.strip()])
        logging.info("[genes] forcing model gene order from %s (%d genes)", gpath, len(force_go))
    else:
        logging.warning("[genes] %s not found — using per-slide HVG (only valid if the model was "
                        "trained on THIS slide's HVG set)", gpath)
    (adata, pos_batched, cell_atts, batch_oh, pert_name, cutoff_t,
     perturb_raw, has_guide, is_real_KO, n_batch, perturb_idx) = load_data(
        args.data, hvg_max_rank=2000, min_cells_per_pert=20, force_gene_order=force_go)
    N = adata.n_obs

    # ── Vocab: use the TRAINED pert_map.json (single source of truth) instead of the
    #    one load_data rebuilds — the multi-slide model reserves code 0 for basal and
    #    numbers real KO 1..K; rebuilding from a single slide mis-indexes the embedding
    #    (and can miss KO genes absent from this slide, e.g. Ifngr2). Keeps load_data's
    #    row order (patch cell_index is a permutation of these rows). ──
    pmj = args.pert_map_json or str(Path(args.ckpt_template.format(field=args.fields[0])).parent / "pert_map.json")
    if os.path.exists(pmj):
        pm = json.load(open(pmj))["pert_map"]
        pert_map = {k: int(v) for k, v in pm.items()}
        n_batch = max(pert_map.values()) + 1
        perturb_idx = np.array([pert_map.get(p, 0) for p in perturb_raw], dtype=int)
        cell_atts = np.c_[cell_atts[:, 0], perturb_idx].astype(int)   # keep tissue col 0
        batch_oh = np.eye(n_batch, dtype=np.float32)[perturb_idx]
        pos_batched = np.concatenate([pos_batched[:, :2], batch_oh], axis=1).astype(np.float32)
        pert_name = {v: k for k, v in pert_map.items()}
        logging.info("[vocab] using persisted pert_map (%s): n_batch=%d %s", pmj, n_batch, pert_map)
    else:
        logging.warning("[vocab] pert_map.json NOT found at %s — falling back to load_data vocab "
                        "(only valid for single-slide models trained without basal-code reservation)", pmj)
    name2code = {v: k for k, v in pert_name.items()}
    Yexpr = np.asarray(adata.X.todense() if hasattr(adata.X, "todense") else adata.X, dtype=np.float32)
    Yobs = adata.raw.X if adata.raw is not None else adata.X
    Yobs = np.asarray(Yobs.todense() if hasattr(Yobs, "todense") else Yobs, dtype=np.float32)
    logging.info("N=%d pert_name=%s", N, pert_name)

    # ── patches (model-row space via cell_index permutation) ──────────────────
    P = np.load(args.patch_file, allow_pickle=True)
    ci = P["cell_index"].astype(int)            # patch-row -> model-row
    role = P["role"]; pid = P["patch_id"]; tbin = P["target_bin"]
    tko = P["target_ko"].astype(str); binid = P["bin_id"]; canon = P["perturbation_canon"].astype(str)
    assert np.array_equal(np.sort(ci), np.arange(N)), "cell_index not a permutation of model rows"

    patches = []
    for pp in np.unique(pid[pid >= 0]):
        m = pid == pp
        b = int(np.unique(tbin[m])[0]); g = str(np.unique(tko[m])[0])
        src_rows = np.where(m & (role == 1))[0]
        src_bins = np.unique(binid[src_rows])
        ko_rows = np.where((binid == b) & (canon == g))[0]
        ct_rows = np.where((binid == b) & (canon == "control"))[0]
        niche = "same" if b in src_bins else "neb"
        patches.append(dict(pid=int(pp), gene=g, target_bin=b, niche=niche,
                            src=ci[src_rows], ko=ci[ko_rows], ctrl=ci[ct_rows]))

    results = {}
    for field in args.fields:
        preds = predict_fields(field, adata, pos_batched, cell_atts, cutoff_t, n_batch,
                               name2code, Yexpr, args.n_samples, args.batch_size, args)
        rows = []
        for pt in patches:
            pred_g, pred_b = preds[pt["gene"]]
            pred_lfc = eu.pseudobulk_lfc(pred_g[pt["src"]], pred_b[pt["src"]])
            n_ko, n_ct = len(pt["ko"]), len(pt["ctrl"])
            if n_ko >= MIN_SUPPORT and n_ct >= MIN_SUPPORT:
                obs_lfc = eu.pseudobulk_lfc(Yobs[pt["ko"]], Yobs[pt["ctrl"]])
                rho = float(spearmanr(pred_lfc, obs_lfc).correlation)
                lowsup = False
            else:
                rho = float("nan"); lowsup = True
            rows.append(dict(pid=pt["pid"], gene=pt["gene"], target_bin=pt["target_bin"],
                             niche=pt["niche"], n_src=len(pt["src"]), n_ko=n_ko, n_ctrl=n_ct,
                             rho=rho, low_support=lowsup))
        results[field] = rows
        valid = [r["rho"] for r in rows if not r["low_support"]]
        same = [r["rho"] for r in rows if r["niche"] == "same" and not r["low_support"]]
        neb = [r["rho"] for r in rows if r["niche"] == "neb" and not r["low_support"]]
        logging.info("[%s] mean=%.4f same=%.4f neb=%.4f (n_valid=%d)",
                     field, np.nanmean(valid), np.nanmean(same) if same else float("nan"),
                     np.nanmean(neb) if neb else float("nan"), len(valid))

    out = dict(patch_file=args.patch_file, min_support=MIN_SUPPORT, results=results)
    with open(os.path.join(args.outdir, "basal_patch_lfc.json"), "w") as f:
        json.dump(out, f, indent=2)

    # ── table ─────────────────────────────────────────────────────────────────
    print("\n===== perturbmap basal-field gate: per-patch footprint Spearman (pred vs observed KO-vs-ctrl@target_bin) =====")
    hdr = "patch  gene    tbin niche  nsrc nko nct |" + "".join(f" {f:>8s}" for f in args.fields)
    print(hdr)
    for i, pt in enumerate(patches):
        r0 = results[args.fields[0]][i]
        cells = f"{r0['pid']:<5d} {r0['gene']:<7s} {r0['target_bin']:<4d} {r0['niche']:<4s} " \
                f"{r0['n_src']:>4d} {r0['n_ko']:>3d} {r0['n_ctrl']:>3d} |"
        vals = "".join(f" {results[f][i]['rho']:8.4f}" for f in args.fields)
        print(cells + vals)
    print("-" * len(hdr))
    for tag, sel in [("mean(all valid)", lambda r: True),
                     ("mean(same)", lambda r: r["niche"] == "same"),
                     ("mean(neb)", lambda r: r["niche"] == "neb")]:
        line = f"{tag:<31s}|"
        for f in args.fields:
            vv = [r["rho"] for r in results[f] if sel(r) and not r["low_support"]]
            line += f" {np.nanmean(vv) if vv else float('nan'):8.4f}"
        print(line)
    print("\nGate: rff basal should be >= svgp basal (footprint fidelity not degraded by replacing the SVGP basal).")


if __name__ == "__main__":
    main()
