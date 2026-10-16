"""Summarise the Mean target cell-count sweep against CONCERT and the published Mean target (paired units)."""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmark_evaluation"))
import release_paths as rp  # noqa: E402
# sweep outputs (run_mean_target_audit.sh) and the reported tables they are compared against
D = rp.WORKDIR / "mean_target_audit"; AGG = rp.SPACSEQ_RESULTS
KEY = ["gene", "slide", "niche_key"]
out = []
for arm in ("unseen", "seen"):
    f = D / f"mean_target_ncells_{arm}.csv"
    if not f.is_file():
        continue
    sw = pd.read_csv(f, dtype={"slide": str, "niche_key": str})
    rec = pd.read_csv(AGG / f"spac_{arm}_lfc_dist_per_record.csv", dtype={"slide": str, "niche_key": str})
    con = rec[rec.method_label == "CONCERT"][KEY + ["lfc_spearman", "lfc_dist"]]
    pub = pd.read_csv(AGG / f"spac_{arm}_naive_baselines_our_rule.csv", dtype={"slide": str, "niche_key": str})
    pub = pub[pub.method == "mean_target"][KEY + ["lfc_spearman", "lfc_dist"]]
    chk = sw[sw.config == "all"].merge(pub, on=KEY, suffixes=("", "_pub"))
    print(f"[{arm}] validation 'all' vs published Mean target: n={len(chk)} max|d rho|="
          f"{(chk.lfc_spearman - chk.lfc_spearman_pub).abs().max():.4g} max|d rmse|={(chk.lfc_dist - chk.lfc_dist_pub).abs().max():.4g}")
    g = sw.groupby(["config"] + KEY, as_index=False)[["lfc_spearman", "lfc_dist", "n_cells"]].mean()
    for cfg, d in g.groupby("config"):
        m = d.merge(con, on=KEY, suffixes=("_mt", "_c"))
        row = dict(arm=arm, config=cfg, n_units=len(m), median_cells=float(d.n_cells.median()),
                   mt_rho=m.lfc_spearman_mt.mean(), concert_rho=m.lfc_spearman_c.mean(),
                   mt_rmse=m.lfc_dist_mt.mean(), concert_rmse=m.lfc_dist_c.mean(),
                   concert_wins_rho=int((m.lfc_spearman_c > m.lfc_spearman_mt).sum()),
                   concert_wins_rmse=int((m.lfc_dist_c < m.lfc_dist_mt).sum()))
        row["p_rho"] = wilcoxon(m.lfc_spearman_c, m.lfc_spearman_mt).pvalue if len(m) > 5 else np.nan
        row["p_rmse"] = wilcoxon(m.lfc_dist_c, m.lfc_dist_mt).pvalue if len(m) > 5 else np.nan
        out.append(row)
res = pd.DataFrame(out)
order = {c: i for i, c in enumerate(["n5", "n10", "n20", "n50", "n100", "n200", "n500", "manifest", "all"])}
res = res.sort_values(["arm", "config"], key=lambda s: s.map(order) if s.name == "config" else s)
res.to_csv(D / "mean_target_ncells_summary.csv", index=False)
pd.set_option("display.width", 250)
print(res.round(4).to_string(index=False))
