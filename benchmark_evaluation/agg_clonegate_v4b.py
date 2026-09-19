#!/usr/bin/env python3
"""Aggregate clonegate v4b eval variants (paired, over the intersecting unit list)."""
import json, glob, sys, os
import numpy as np
from scipy.stats import wilcoxon

ROOT = "/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/spac_seq_eval/multi_1-5_ho_v4_clonegate"

def load_unseen(tag):
    out = {}
    for d in sorted(glob.glob(f"{ROOT}/eval_{tag}/concert_clonegate/unseen/unseen_pert/*")):
        f = os.path.join(d, "lfc_scores.json")
        if not os.path.exists(f):
            continue
        L2 = json.load(open(f))["L2"]
        for g, recs in L2.items():
            for r in recs:
                out[f"{os.path.basename(d)}|{g}|{r.get('niche',-1)}"] = r
    return out

def load_seen(tag):
    out = {}
    for f in sorted(glob.glob(f"{ROOT}/eval_{tag}/concert_clonegate/seen/patch/slide_*/lfc_scores.json")):
        sl = f.split("/")[-2]
        for g, recs in json.load(open(f))["L2"].items():
            for r in recs:
                out[f"{sl}|{g}|{r.get('niche',-1)}"] = r
    return out

def report(name, tags, loader):
    sets = {t: loader(t) for t in tags}
    keys = sorted(set.intersection(*[set(v) for v in sets.values()]))
    print(f"\n## {name}  (n={len(keys)} paired units; per-variant n={[len(sets[t]) for t in tags]})")
    print(f"{'variant':<18} {'rho':>8} {'lfc_dist':>9} {'e_dist':>8}")
    arr = {}
    for t in tags:
        rho = np.array([sets[t][k]["spearman"] for k in keys])
        lfc = np.array([sets[t][k]["lfc_dist"] for k in keys])
        ed  = np.array([sets[t][k]["e_dist"] for k in keys])
        arr[t] = (rho, lfc, ed)
        print(f"{t:<18} {rho.mean():8.4f} {lfc.mean():9.4f} {ed.mean():8.4f}")
    base = tags[0]
    for t in tags[1:]:
        d = arr[t][0] - arr[base][0]
        nz = d[d != 0]
        p = wilcoxon(nz).pvalue if len(nz) else float("nan")
        print(f"  {t} vs {base}: rho delta {d.mean():+.4f}, better {int((d>0).sum())}/{len(d)}, p={p:.3g}")
    return arr, keys

if __name__ == "__main__":
    report("SEEN arm", ["mean", "poisson", "poisson_crn", "poisson_obsref", "mean_obsref"], load_seen)
    arr, keys = report("UNSEEN arm poisson seed repeats",
                       ["poisson", "poisson_s1", "poisson_s2", "poisson_s3", "poisson_s4"], load_unseen)
    rhos = np.array([arr[t][0].mean() for t in ["poisson","poisson_s1","poisson_s2","poisson_s3","poisson_s4"]])
    eds  = np.array([arr[t][2].mean() for t in ["poisson","poisson_s1","poisson_s2","poisson_s3","poisson_s4"]])
    print(f"\n  unseen poisson rho across 5 draws: mean {rhos.mean():.4f} sd {rhos.std(ddof=1):.4f} "
          f"min {rhos.min():.4f} max {rhos.max():.4f}")
    print(f"  unseen poisson e_dist across 5 draws: mean {eds.mean():.4f} sd {eds.std(ddof=1):.4f}")
