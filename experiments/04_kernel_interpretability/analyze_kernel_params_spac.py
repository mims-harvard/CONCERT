"""Per-perturbation learned kernel parameters of the CONCERT-spac SPAC-seq *seen*
(within-slide) model, and whether they carry spatial/biological signal.

What is actually learned per perturbation in ms_spac1_niche_seen (fpert_field=gp_pp,
basal_field=rff, use_niche_film=True) -- both are read out from the 256-d LORD
perturbation embedding e_p = lord_encoder.s_encoder.perturbation.weight[p]:

  1. footprint lengthscale   ls(p)      = ls_min + (ls_max-ls_min) * sigmoid(ls_head(e_p))
                                        = 3 + 15*sigmoid(...)          (concert_spac.py:222)
     used as exp(-||c - Z||^2 / (2 ls^2)) with c = coords / coord_scale, Z = 4x4 grid
  2. niche-FiLM gain cap     g_scale(p) = clamp(softplus(g_scale_head(e_p)), max=g_cap)
                                          (concert_spac.py:1103) -> gamma = 1 + g_scale*tanh(...)

NOTE mask_cutoff is *not* learned in this configuration: it only gates the SVGP kernel and
this run used --basal_field rff, so self.svgp is None and the cutoff keeps its 0.5 init.
The script verifies that and reports it rather than pretending it is a fitted parameter.

Spatial covariates are computed from the same inputs/preprocessing as training
(obsm/array_coords + obs/filter_pass per slide, then tile_multislide_coords), metadata only.

Usage (concert_env):
  python analyze_kernel_params_spac.py \
      --model_file  $CONCERT_CHECKPOINT_ROOT/spacseq_kernel_analysis/model.pt \
      --pert_map    $CONCERT_CHECKPOINT_ROOT/spacseq_kernel_analysis/pert_map.json \
      --cutoffs_csv $CONCERT_CHECKPOINT_ROOT/spacseq_kernel_analysis/ms_spac1_spac_cutoffs.csv \
      --outdir <dir>
"""
import argparse, json, logging, os
import numpy as np
import pandas as pd
import torch
import h5py
from scipy.stats import spearmanr, pearsonr
from scipy.spatial import cKDTree
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN

logging.basicConfig(level=logging.INFO, format="%(message)s")

# training recipe constants (sub_train_spac_niche.sh)
LOC_RANGE = 20.0
GAP_FACTOR = 2.5
COORD_SCALE = 20.0
LS_MIN, LS_MAX = 3.0, 18.0
G_CAP = 3.0

import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "benchmark_evaluation"))
import release_paths as rp  # noqa: E402

SP = str(rp.SPACSEQ_RAW)
DEFAULT_FILES = ",".join(f"{SP}/raw_output_1-{i}.h5ad" for i in range(1, 6))

ap = argparse.ArgumentParser()
ap.add_argument("--model_file", required=True)
ap.add_argument("--pert_map", required=True)
ap.add_argument("--cutoffs_csv", default="")
ap.add_argument("--multi_sample_files", default=DEFAULT_FILES)
ap.add_argument("--outdir", required=True)
ap.add_argument("--dbscan_eps", type=float, default=2.0, help="patch linkage radius, array-coord bins")
ap.add_argument("--dbscan_min", type=int, default=3)
ap.add_argument("--density_radius", type=float, default=10.0, help="local-density radius, array-coord bins")
ap.add_argument("--max_cells_per_group", type=int, default=30000, help="subsample cap per (pert, slide)")
ap.add_argument("--seed", type=int, default=0)
# gp_pp lengthscale band. Defaults reproduce the legacy runs, where ls_prior_min/max were
# reused unconverted (loc units). Runs trained with --fpert_ls_min/--fpert_ls_max carry the
# band in NORMALIZED units (coords/coord_scale), e.g. [0.15, 0.90] == [3, 18] loc units.
ap.add_argument("--ls_min", type=float, default=3.0)
ap.add_argument("--ls_max", type=float, default=18.0)
args = ap.parse_args()
LS_MIN, LS_MAX = args.ls_min, args.ls_max
os.makedirs(args.outdir, exist_ok=True)
rng = np.random.default_rng(args.seed)


# ---------------------------------------------------------------- 1. model side
def read_obs_str(f_obs, col):
    """Mirror of run_concert_spac._read_obs_str_column (categorical or plain)."""
    obj = f_obs[col]
    if isinstance(obj, h5py.Group) and "codes" in obj and "categories" in obj:
        codes = np.array(obj["codes"][:], dtype=int)
        cats = np.array(obj["categories"][:]).astype(str)
        return cats[codes]
    raw = np.array(obj[:])
    if raw.dtype.kind in "SO":
        return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in raw])
    return raw.astype(str)


sd = torch.load(args.model_file, map_location="cpu", weights_only=False)
if isinstance(sd, dict) and "model_state_dict" in sd:
    sd = sd["model_state_dict"]

emb = sd["lord_encoder.s_encoder.perturbation.weight"].float()        # (K, 256)
K, E = emb.shape
logging.info("perturbation embedding table: %s", tuple(emb.shape))

ls_w = sd["fpert_field.ls_head.weight"].float()                       # (1, 256)
ls_b = sd["fpert_field.ls_head.bias"].float()                         # (1,)
gs_w = sd["g_scale_head.weight"].float()
gs_b = sd["g_scale_head.bias"].float()

raw_ls = emb @ ls_w.t() + ls_b                                        # (K,1)
ls = (LS_MIN + (LS_MAX - LS_MIN) * torch.sigmoid(raw_ls)).squeeze(-1).numpy()
raw_gs = emb @ gs_w.t() + gs_b
g_scale = torch.clamp(torch.nn.functional.softplus(raw_gs), max=G_CAP).squeeze(-1).numpy()

# did the readouts actually train off their zero init?
diag = {
    "ls_head_weight_absmax": float(ls_w.abs().max()),
    "ls_head_weight_l2": float(ls_w.norm()),
    "ls_head_bias": float(ls_b.item()),
    "g_scale_head_weight_absmax": float(gs_w.abs().max()),
    "g_scale_head_weight_l2": float(gs_w.norm()),
    "g_scale_head_bias": float(gs_b.item()),
    "ls_min_max_band": [LS_MIN, LS_MAX],
    "ls_observed": [float(ls.min()), float(np.median(ls)), float(ls.max())],
    "g_scale_observed": [float(g_scale.min()), float(np.median(g_scale)), float(g_scale.max())],
    "n_perturbations": int(K),
}
if "mask_cutoff" in sd:
    mc = sd["mask_cutoff"].float().numpy()
    diag["mask_cutoff"] = {
        "n": int(mc.size), "min": float(mc.min()), "median": float(np.median(mc)),
        "max": float(mc.max()), "n_unique": int(np.unique(mc).size),
        "is_learned": bool(np.unique(mc).size > 1),
    }
if "fpert_field.Z" in sd:
    diag["fpert_Z_shape"] = list(sd["fpert_field.Z"].shape)
logging.info("diagnostics: %s", json.dumps(diag, indent=2))

pert_map = json.load(open(args.pert_map))
pert_map = pert_map.get("pert_map", pert_map)
code2name = {int(v): k for k, v in pert_map.items()}
names = np.array([code2name.get(i, f"pert={i}") for i in range(K)])

model_df = pd.DataFrame({
    "perturbation": names,
    "pert_code": np.arange(K),
    "ls_footprint": ls,
    "g_scale_niche": g_scale,
    "emb_l2": emb.norm(dim=1).numpy(),
})

# training cell counts straight from the per-cell dump (authoritative, training row order)
if args.cutoffs_csv and os.path.exists(args.cutoffs_csv):
    cut = pd.read_csv(args.cutoffs_csv, usecols=["perturbation_code", "perturbation_name", "cutoff"])
    n_train = cut.groupby("perturbation_code").size().rename("n_train_cells")
    model_df = model_df.merge(n_train, left_on="pert_code", right_index=True, how="left")
    diag["cutoff_csv_unique_values"] = sorted(cut["cutoff"].unique().tolist())[:10]
    logging.info("cutoffs csv: %d rows, %d unique cutoff values",
                 len(cut), cut["cutoff"].nunique())


# ------------------------------------------------------- 2. spatial covariates
paths = [p for p in args.multi_sample_files.split(",") if p.strip()]
pos_list, pert_list, sid_list = [], [], []
for k, p in enumerate(paths):
    with h5py.File(p, "r") as f:
        pr = read_obs_str(f["obs"], "perturbation")
        pos = np.array(f["obsm"]["array_coords"][:], dtype=np.float32)
        if "filter_pass" in f["obs"]:
            fp = read_obs_str(f["obs"], "filter_pass")
            keep = np.array([s.lower() in {"1", "true", "pass", "yes"} for s in fp], dtype=bool)
        else:
            keep = np.ones(len(pr), dtype=bool)
    logging.info("[%s] %d spots, %d pass filter_pass", os.path.basename(p), len(pr), keep.sum())
    pos_list.append(pos[keep]); pert_list.append(pr[keep])
    sid_list.append(np.full(int(keep.sum()), k, dtype=np.int32))

pos_raw = np.concatenate(pos_list); perturb = np.concatenate(pert_list)
sample_id = np.concatenate(sid_list)
logging.info("total cells after filter_pass: %d", pos_raw.shape[0])

# tiled coords, identical to training (run_concert_spac.tile_multislide_coords)
sids = np.unique(sample_id)
ncol = int(np.ceil(np.sqrt(len(sids))))
spacing = LOC_RANGE * GAP_FACTOR
pos_tiled = np.zeros((pos_raw.shape[0], 2), dtype=np.float32)
for k, sid in enumerate(sids):
    m = sample_id == sid
    sc = MinMaxScaler().fit_transform(pos_raw[m]) * LOC_RANGE
    r, c = divmod(k, ncol)
    sc[:, 0] += c * spacing
    sc[:, 1] += r * spacing
    pos_tiled[m] = sc.astype(np.float32)

# slide bounding-box areas in raw array-coord units (for the CSR expectation)
slide_area, slide_n = {}, {}
for sid in sids:
    m = sample_id == sid
    p = pos_raw[m]
    slide_area[sid] = float((p[:, 0].max() - p[:, 0].min()) * (p[:, 1].max() - p[:, 1].min()))
    slide_n[sid] = int(m.sum())

GRID = 20.0     # occupancy bin, array-coord units
rows = []
for name in np.unique(perturb):
    mask_p = perturb == name
    per_slide = []
    for sid in sids:
        m = mask_p & (sample_id == sid)
        n = int(m.sum())
        if n < 3:
            continue
        pts = pos_raw[m]
        if n > args.max_cells_per_group:
            sel = rng.choice(n, args.max_cells_per_group, replace=False)
            pts = pts[sel]
            n_used = args.max_cells_per_group
        else:
            n_used = n
        tree = cKDTree(pts)
        d, _ = tree.query(pts, k=2)
        nnd = d[:, 1]
        # local same-perturbation density within density_radius (excluding self)
        cnt = np.array(tree.query_ball_point(pts, args.density_radius, return_length=True)) - 1.0
        # nearest-neighbour index vs complete spatial randomness on the slide bbox
        lam = n / max(slide_area[sid], 1e-9)
        exp_nnd = 0.5 / np.sqrt(max(lam, 1e-12))
        nni = float(np.mean(nnd) / exp_nnd) if exp_nnd > 0 else np.nan
        # patches
        lab = DBSCAN(eps=args.dbscan_eps, min_samples=args.dbscan_min).fit_predict(pts)
        core = lab[lab >= 0]
        sizes = np.bincount(core) if core.size else np.array([0])
        n_patch = int(sizes.size) if core.size else 0
        # extent
        cen = pts.mean(0)
        rg = float(np.sqrt(((pts - cen) ** 2).sum(1).mean()))
        occ = np.unique(np.floor(pts / GRID).astype(int), axis=0).shape[0]
        per_slide.append(dict(
            sid=int(sid), n_cells=n, n_used=n_used,
            nnd_mean=float(np.mean(nnd)), nnd_median=float(np.median(nnd)),
            nni=nni, local_density=float(np.mean(cnt)),
            n_patches=n_patch,
            mean_patch_size=float(sizes.mean()) if core.size else 0.0,
            max_patch_frac=float(sizes.max() / n_used) if core.size else 0.0,
            singleton_frac=float(np.mean(lab < 0)),
            radius_gyration=rg, occupancy_bins=int(occ),
            occupancy_frac=float(occ * GRID * GRID / max(slide_area[sid], 1e-9)),
            slide_frac=float(n / slide_n[sid]),
        ))
    if not per_slide:
        continue
    ps = pd.DataFrame(per_slide)
    w = ps["n_cells"].to_numpy(dtype=float)
    def wmean(col):
        return float(np.average(ps[col].to_numpy(dtype=float), weights=w))
    rows.append(dict(
        perturbation=name,
        n_cells_total=int(ps["n_cells"].sum()),
        n_slides_present=int(len(ps)),
        nnd_mean=wmean("nnd_mean"), nnd_median=wmean("nnd_median"),
        nni=wmean("nni"), local_density=wmean("local_density"),
        n_patches=float(ps["n_patches"].sum()),
        n_patches_per_slide=float(ps["n_patches"].mean()),
        mean_patch_size=wmean("mean_patch_size"),
        max_patch_frac=wmean("max_patch_frac"),
        singleton_frac=wmean("singleton_frac"),
        radius_gyration=wmean("radius_gyration"),
        occupancy_bins=float(ps["occupancy_bins"].sum()),
        occupancy_frac=wmean("occupancy_frac"),
        slide_frac=wmean("slide_frac"),
    ))
spatial_df = pd.DataFrame(rows)
logging.info("spatial covariates for %d perturbation labels", len(spatial_df))


# --------------------------------------------------------------- 3. join + test
df = model_df.merge(spatial_df, on="perturbation", how="left")
df = df.sort_values("perturbation").reset_index(drop=True)
df.to_csv(os.path.join(args.outdir, "per_perturbation_kernel_params.csv"), index=False)

COVARS = ["n_cells_total", "n_train_cells", "n_slides_present", "nnd_mean", "nni",
          "local_density", "n_patches", "n_patches_per_slide", "mean_patch_size",
          "max_patch_frac", "singleton_frac", "radius_gyration", "occupancy_bins",
          "occupancy_frac", "slide_frac", "emb_l2"]
TARGETS = ["ls_footprint", "g_scale_niche"]

res = []
sub = df[df["n_cells_total"].notna()]
for t in TARGETS:
    for c in COVARS:
        if c not in sub.columns:
            continue
        d = sub[[t, c]].dropna()
        if len(d) < 10 or d[c].nunique() < 3 or d[t].nunique() < 3:
            continue
        rho, pr_ = spearmanr(d[t], d[c])
        r, pp_ = pearsonr(d[t], d[c])
        res.append(dict(target=t, covariate=c, n=len(d), spearman_rho=rho,
                        spearman_p=pr_, pearson_r=r, pearson_p=pp_))
cor = pd.DataFrame(res)
if len(cor):
    # Benjamini-Hochberg within each target
    cor = cor.sort_values(["target", "spearman_p"])
    out = []
    for t, g in cor.groupby("target"):
        g = g.sort_values("spearman_p").copy()
        m = len(g)
        g["q_bh"] = np.minimum.accumulate(
            (g["spearman_p"].to_numpy() * m / np.arange(1, m + 1))[::-1])[::-1]
        out.append(g)
    cor = pd.concat(out)
    cor = cor.sort_values(["target", "spearman_p"])
cor.to_csv(os.path.join(args.outdir, "kernel_param_correlations.csv"), index=False)

diag["n_perts_with_spatial"] = int(len(sub))
json.dump(diag, open(os.path.join(args.outdir, "kernel_param_diagnostics.json"), "w"), indent=2)

print("\n=== learned per-perturbation parameters ===")
print(df[["perturbation", "pert_code", "ls_footprint", "g_scale_niche", "n_cells_total",
          "nni", "n_patches_per_slide", "local_density"]].describe().to_string())
print("\n=== Spearman: learned parameter vs spatial covariate ===")
if len(cor):
    print(cor.to_string(index=False))
else:
    print("no testable covariate pairs")
print(f"\nwrote -> {args.outdir}")
