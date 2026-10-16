"""Convert a perturbmap slide (0..3) -> spac-format .h5ad for CONCERT-spac.

The ONE prep script for every perturbmap slide, including slide 0 (`--slide 0`).
Run all four via sub_prep_perturbmap_multi.sh (SLURM array 0-3).

Two ORTHOGONAL axes (perturbmap has NO NTC; KP/periphery are tissue regions, not guides):
  perturbation in {Jak2, Tgfbr2, Ifngr2, unperturbed}  -- only the 3 real KO genes are guides;
                 unassigned/KP/periphery all -> "unperturbed" (basal).
  tissue       in {tumor, normal, periphery}    -- region axis (feeds basal path):
                 raw=="periphery"->periphery; raw=="KP"->tumor (KP = unperturbed tumor genotype,
                 forced to override any mislabeled obs.tissue); else original obs.tissue.
No cells are excluded.

Usage: python prep_perturbmap_slide.py --slide 0|1|2|3
Output: perturbmap_s<slide>_spacfmt.h5ad
"""
import argparse
import numpy as np
import scanpy as sc
import anndata as ad
import collections

_p = argparse.ArgumentParser()
_p.add_argument("--slide", required=True)   # "0".."3"
_args = _p.parse_args()
SLIDE = str(_args.slide)

SRC = "/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/spaVAE/datasets/perturb_map_data_all_fullgene.h5ad"
OUT = f"/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/spaVAE/datasets/perturbmap_s{SLIDE}_spacfmt.h5ad"
KO = {"Jak2", "Tgfbr2", "Ifngr2"}   # the 3 real CRISPR-KO genes (perturbmap has no NTC)

a = ad.read_h5ad(SRC)
a = a[a.obs["slide"].astype(str) == SLIDE].copy()
a.var_names_make_unique()
print(f"slide {SLIDE}: {a.shape[0]} cells x {a.shape[1]} genes")
assert a.shape[0] > 0, f"no cells for slide {SLIDE!r}; check obs['slide'] values in {SRC}"

canon = a.obs["perturbation_canon"].astype(str).values
raw   = a.obs["perturbation"].astype(str).values
tissue0 = a.obs["tissue"].astype(str).values

# perturbation axis: the 3 real KO genes are the only guides; everything else = basal
pert = np.where(np.isin(canon, list(KO)), canon, "unperturbed")
# tissue axis (region), independent of perturbation. KP = unperturbed tumor: force -> tumor
# (source obs.tissue mislabels 22 KP cells as normal); periphery its own region.
tissue = tissue0.astype(object).copy()
tissue[raw == "KP"] = "tumor"
tissue[raw == "periphery"] = "periphery"
tissue = tissue.astype(str)

# HVG on normalized copy; keep raw counts in X
counts = a.X.copy()
norm = a.copy()
sc.pp.normalize_total(norm, target_sum=1e4)
sc.pp.log1p(norm)
sc.pp.highly_variable_genes(norm, flavor="seurat", n_top_genes=2000)
disp = np.asarray(norm.var["dispersions_norm"].values, dtype=float)
disp = np.where(np.isfinite(disp), disp, -np.inf)
order = np.argsort(-disp)
hvg_rank = np.empty(len(disp), dtype=np.int32)
hvg_rank[order] = np.arange(1, len(disp) + 1)

a.X = counts
a.obs["perturbation"] = pert
a.obs["tissue"] = tissue
a.obs["filter_pass"] = np.array(["1"] * a.shape[0])
a.var["hvg_rank"] = hvg_rank
coords = np.asarray(a.obsm["spatial"], dtype=np.float32)
a.obsm["array_coords"] = coords
a.obsm["spatial"] = coords

print("perturbation:", dict(collections.Counter(pert)))
print("tissue:", dict(collections.Counter(tissue)))
a.write_h5ad(OUT)
print("wrote", OUT)

# verify via driver loader + attribute builder
import run_concert_spac as R
una = frozenset({"unperturbed"})
ctrl = frozenset({"__NONE__"})  # no NTC in perturbmap
X, pos, tissue_raw, perturb_raw, has_guide, is_real_KO, gene_names = R.load_h5_dataset(
    OUT, unassigned_labels=una, control_labels=ctrl, hvg_max_rank=2000)
ti, td, pi, pm = R.build_attributes_spac(perturb_raw, tissue_raw=tissue_raw,
                                         unassigned_labels=una, control_labels=ctrl)
print("\n[verify] X", X.shape, "genes", len(gene_names))
print("[verify] tissue_dict:", td)
print("[verify] pert_map:", pm)
print("[verify] is_real_KO sum:", int(is_real_KO.sum()), "has_guide sum:", int(has_guide.sum()))
# Per-slide composition may vary (a slide can lack a tissue class or a KO gene); WARN not assert.
if len(td) != 3:
    print(f"[verify][WARN] slide {SLIDE}: {len(td)} tissue classes (expected <=3: tumor/normal/periphery)")
real_pm = {k for k in pm if k != R.BASAL_PERT_KEY}
if not real_pm.issubset({"Jak2", "Tgfbr2", "Ifngr2"}):
    print(f"[verify][WARN] slide {SLIDE}: unexpected real pert classes: {real_pm}")
# structural invariants that must always hold:
assert pm.get(R.BASAL_PERT_KEY) == 0, "basal must be reserved perturbation code 0"
assert int((pi == 0).sum()) == int((~is_real_KO).sum()), "basal-slot cells must be exactly non-KO"
ko_cells = np.isin(perturb_raw, ["Jak2", "Tgfbr2", "Ifngr2"]).sum()
assert int(is_real_KO.sum()) == ko_cells, "is_real_KO mismatch"
assert int(has_guide.sum()) == ko_cells, "has_guide should be KO-only (no NTC in perturbmap)"
print(f"[verify] OK -- slide {SLIDE}: real perts {real_pm}, tissue classes {set(td)}, "
      f"{ko_cells} KO / {int((~is_real_KO).sum())} basal, KO-only guides")
