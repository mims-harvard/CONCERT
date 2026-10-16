"""Count the cells behind the Mean target baseline and compare with the KO cells CONCERT trains on."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad
ROOT = Path("/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/spac_seq_eval")
PREP = ROOT / "multi_1-5/prepared"; B4 = PREP / "benchmark_v4"
SLIDES = ["1-1", "1-2", "1-3", "1-4", "1-5"]
ms = json.loads((B4 / "train_manifest_seen.json").read_text())
mu = json.loads((B4 / "train_manifest_unseen_union.json").read_text())
for name, m in (("seen", ms), ("unseen", mu)):
    s = m["per_slide"]["1-1"]
    print(name, "n_train_total", m.get("n_train_total"), "per-slide keys", {k: (len(v) if isinstance(v, list) else v) for k, v in s.items()})
obs = {}
for sl in SLIDES:
    a = ad.read_h5ad(PREP / f"slide_{sl}.h5ad", backed="r")
    obs[sl] = (a.obs["perturbation_canon"].astype(str).to_numpy(), a.obs["bin_id"].to_numpy().astype(int))
    a.file.close()
    print(sl, "n_obs", len(obs[sl][0]), flush=True)

def train_set(m, sl):
    s = m["per_slide"][sl]
    return set(s["train_idx"]) if "train_idx" in s else None

tr_s = {sl: train_set(ms, sl) for sl in SLIDES}
tr_u = {sl: train_set(mu, sl) for sl in SLIDES}

# UNSEEN: pool = KO cells on donor slides (rescore_naive_baselines_unseen_v4_genefix.donor_target_means)
panel = json.loads((B4 / "eval_panel.json").read_text())
rows = []
for u in panel["eval_units"]:
    g, T = u["gene"], u["target_slide"]
    pool = {sl: np.where(obs[sl][0] == g)[0] for sl in u["donor_slides"] if sl != T}
    n_pool = sum(len(v) for v in pool.values())
    n_pool_tr = sum(len(set(v.tolist()) & tr_u[sl]) for sl, v in pool.items()) if tr_u[T] is not None else np.nan
    n_other = sum(int((obs[sl][0] == g).sum()) for sl in SLIDES if sl != T and sl not in u["donor_slides"])
    n_T = int((obs[T][0] == g).sum())
    rows.append(dict(gene=g, target=T, donors=",".join(u["donor_slides"]), n_meantarget_pool=n_pool,
                     n_pool_in_concert_train=n_pool_tr, n_ko_nondonor_slides_unused=n_other, n_ko_target_slide=n_T))
du = pd.DataFrame(rows); du.to_csv("unseen_mean_target_pool.csv", index=False)
print("\nUNSEEN pool size:", du.n_meantarget_pool.describe().round(1).to_dict())
print("UNSEEN CONCERT-train KO cells from same pool:", du.n_pool_in_concert_train.describe().round(1).to_dict())

# SEEN: pool = KO cells on ALL 5 slides minus this unit's (gene, slide, niche) cells
rec = pd.read_csv(ROOT / "v4_aggregate_sampled/spac_seen_lfc_dist_per_record.csv")
rec = rec[rec.method_label == "CONCERT"][["gene", "slide", "niche_key"]].drop_duplicates()
rows = []
for r in rec.itertuples(index=False):
    g, sl, nk = r.gene, str(r.slide), int(r.niche_key)
    tot = {s: np.where(obs[s][0] == g)[0] for s in SLIDES}
    unit = np.where((obs[sl][0] == g) & (obs[sl][1] == nk))[0]
    n_tot = sum(len(v) for v in tot.values())
    same_slide_other = len(tot[sl]) - len(unit)
    n_tr = sum(len(set(v.tolist()) & tr_s[s]) for s, v in tot.items()) if tr_s[sl] is not None else np.nan
    unit_tr = len(set(unit.tolist()) & tr_s[sl]) if tr_s[sl] is not None else np.nan
    rows.append(dict(gene=g, slide=sl, niche=nk, n_unit_gt=len(unit), n_meantarget_pool=n_tot - len(unit),
                     n_pool_same_slide=same_slide_other, n_pool_other_slides=n_tot - len(tot[sl]),
                     n_ko_in_concert_train_all=n_tr, n_unit_gt_in_concert_train=unit_tr))
ds = pd.DataFrame(rows); ds.to_csv("seen_mean_target_pool.csv", index=False)
for c in ["n_unit_gt", "n_meantarget_pool", "n_pool_same_slide", "n_pool_other_slides", "n_ko_in_concert_train_all", "n_unit_gt_in_concert_train"]:
    print("SEEN", c, ds[c].describe().round(1).to_dict())
