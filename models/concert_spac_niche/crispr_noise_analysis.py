"""Test + visualize CRISPR delivery noise (= mu_action) across the 3 datasets.

CRISPR noise = systematic transcriptional shift of guide-bearing NON-TARGETING (control/NTC)
cells relative to no-guide (unassigned) basal cells -- i.e. an effect of guide delivery/selection
that is NOT a gene knockout. The CONCERT model removes it via mu_action so f_pert = pure KO effect.

Metric: pseudobulk log2-CPM LFC(control vs unassigned), genes filtered to expressed.
Null  : split the unassigned pool into two random halves -> LFC (reliability floor ~0).
If noise_LFC ~ null -> no CRISPR noise (modeling mu_action unnecessary, e.g. claim for perturbmap).
If noise_LFC >> null -> real CRISPR noise that must be disentangled (spac).

Outputs: a 3-dataset figure + a quantitative summary table + npz of the LFC vectors.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "benchmark_evaluation" / "release_paths.py").is_file()) / "benchmark_evaluation"))
import release_paths as _rp  # noqa: E402  data/, checkpoints/ and runs/ roots
import numpy as np, anndata as ad, scipy.sparse as sp, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTDIR = f"{_rp.WORKDIR}/scoring/figures/crispr_noise"
os.makedirs(OUTDIR, exist_ok=True)
RNG = np.random.RandomState(0)
MAX_UNA = 30000   # subsample unassigned for memory / balanced null
CPM_FLOOR = 1.0   # expressed-gene filter: pseudobulk CPM > 1 in at least one group
DE_THR = 1.0      # |log2FC| > 1 (2-fold) = "DE"

DATASETS = [
    ("spac_1-1",   f"{_rp.DATA}/spacseq/raw/raw_output_1-1.h5ad", "perturbation"),
    ("spac_2-2",   f"{_rp.DATA}/spacseq/raw/raw_output_2-2.h5ad", "perturbation"),
    ("perturbmap", f"{_rp.DATA}perturbmap/raw/perturb_map_data_all_fullgene.h5ad", "perturbation_canon"),
]

def log2cpm(counts_sum):
    s = counts_sum.sum()
    cpm = counts_sum / s * 1e6 if s > 0 else counts_sum
    return np.log2(cpm + 1.0)

def group_sum(X, idx):
    sub = X[idx]
    return np.asarray(sub.sum(axis=0)).ravel()

results = {}
for name, path, col in DATASETS:
    print(f"\n=== {name} ===")
    a = ad.read_h5ad(path)
    lab = a.obs[col].astype(str).values
    X = a.X
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    X = X.tocsr()
    una_idx = np.where(lab == "unassigned")[0]
    ctl_idx = np.where(lab == "control")[0]
    print(f"  unassigned={len(una_idx)}  control={len(ctl_idx)}  genes={X.shape[1]}")
    if len(ctl_idx) < 10 or len(una_idx) < 20:
        print(f"  SKIP metric (control too sparse: {len(ctl_idx)})");
        results[name] = dict(skip=True, n_ctrl=len(ctl_idx), n_una=len(una_idx));
        del a; continue
    # subsample unassigned for balanced null + memory
    if len(una_idx) > MAX_UNA:
        una_idx = RNG.choice(una_idx, MAX_UNA, replace=False)
    ctl_sum = group_sum(X, ctl_idx)
    una_sum = group_sum(X, una_idx)
    ctl_cpm, una_cpm = log2cpm(ctl_sum), log2cpm(una_sum)
    expressed = (ctl_cpm > np.log2(CPM_FLOOR + 1)) | (una_cpm > np.log2(CPM_FLOOR + 1))
    noise_lfc = (ctl_cpm - una_cpm)[expressed]
    # null: split unassigned into halves, repeat for a band; keep one representative + median stats
    null_lfcs = []
    for r in range(20):
        perm = RNG.permutation(len(una_idx)); h = len(perm) // 2
        a_sum = group_sum(X, una_idx[perm[:h]]); b_sum = group_sum(X, una_idx[perm[h:]])
        nl = (log2cpm(a_sum) - log2cpm(b_sum))[expressed]
        null_lfcs.append(nl)
    null_lfc = null_lfcs[0]
    null_med = np.median([np.median(np.abs(x)) for x in null_lfcs])
    null_de  = np.median([int(np.sum(np.abs(x) > DE_THR)) for x in null_lfcs])
    noise_med = float(np.median(np.abs(noise_lfc)))
    noise_de  = int(np.sum(np.abs(noise_lfc) > DE_THR))
    results[name] = dict(skip=False, n_ctrl=len(ctl_idx), n_una=len(una_idx),
                         ctl_cpm=ctl_cpm[expressed], una_cpm=una_cpm[expressed],
                         noise_lfc=noise_lfc, null_lfc=null_lfc,
                         noise_med=noise_med, null_med=null_med,
                         noise_de=noise_de, null_de=null_de, n_expr=int(expressed.sum()))
    print(f"  expressed genes={int(expressed.sum())}")
    print(f"  median|noise LFC|={noise_med:.4f}   median|null LFC|={null_med:.4f}   ratio={noise_med/max(null_med,1e-9):.2f}")
    print(f"  DE genes (|LFC|>{DE_THR}): noise={noise_de}  null~{null_de:.0f}")
    np.savez(f"{OUTDIR}/{name}_noise.npz", noise_lfc=noise_lfc, null_lfc=null_lfc,
             ctl_cpm=ctl_cpm[expressed], una_cpm=una_cpm[expressed])
    del a, X

# figure: per-dataset MA scatter (top) + |LFC| hist real vs null (bottom)
usable = [n for n, _, _ in DATASETS if not results[n].get("skip")]
nc = len(DATASETS)
fig, axes = plt.subplots(2, nc, figsize=(4.2 * nc, 8))
for j, (name, _, _) in enumerate(DATASETS):
    r = results[name]
    ax0, ax1 = axes[0, j], axes[1, j]
    if r.get("skip"):
        for ax in (ax0, ax1):
            ax.text(0.5, 0.5, f"{name}\ncontrol too sparse\n(n={r['n_ctrl']})",
                    ha="center", va="center", transform=ax.transAxes); ax.set_xticks([]); ax.set_yticks([])
        continue
    # top: control vs unassigned pseudobulk (diagonal = no noise)
    lo = min(r["una_cpm"].min(), r["ctl_cpm"].min()); hi = max(r["una_cpm"].max(), r["ctl_cpm"].max())
    de = np.abs(r["noise_lfc"]) > DE_THR
    ax0.scatter(r["una_cpm"][~de], r["ctl_cpm"][~de], s=3, c="#999999", alpha=0.4, rasterized=True)
    ax0.scatter(r["una_cpm"][de],  r["ctl_cpm"][de],  s=6, c="#d62728", alpha=0.7, rasterized=True, label=f"DE (|LFC|>1) n={r['noise_de']}")
    ax0.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax0.set_xlabel("unassigned log2CPM (basal)"); ax0.set_ylabel("control/NTC log2CPM")
    ax0.set_title(f"{name}\nctrl={r['n_ctrl']} una={r['n_una']}  DE={r['noise_de']} genes")
    ax0.legend(loc="upper left", fontsize=8, markerscale=2)
    # bottom: |LFC| distribution real vs null
    bins = np.linspace(0, max(np.abs(r["noise_lfc"]).max(), 1.0), 40)
    ax1.hist(np.abs(r["null_lfc"]),  bins=bins, color="#1f77b4", alpha=0.5, density=True, label=f"null (split-una) med={r['null_med']:.3f}")
    ax1.hist(np.abs(r["noise_lfc"]), bins=bins, color="#d62728", alpha=0.5, density=True, label=f"NTC-vs-basal med={r['noise_med']:.3f}")
    ax1.axvline(DE_THR, color="k", ls=":", lw=1)
    ax1.set_xlabel("|pseudobulk log2 fold-change|"); ax1.set_ylabel("density")
    ax1.set_title(f"noise/null ratio = {r['noise_med']/max(r['null_med'],1e-9):.1f}x")
    ax1.legend(fontsize=8)
fig.suptitle("CRISPR delivery noise (NTC vs basal) -- must disentangle where noise >> null", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
for ext in ("png", "pdf"):
    fig.savefig(f"{OUTDIR}/crispr_noise_3datasets.{ext}", dpi=150, bbox_inches="tight")
print(f"\nsaved figure -> {OUTDIR}/crispr_noise_3datasets.png")

# summary table
print("\n================ CRISPR NOISE SUMMARY ================")
print(f"{'dataset':12s} | {'n_ctrl':>7s} | {'n_una':>7s} | {'med|noise|':>10s} | {'med|null|':>9s} | {'ratio':>6s} | {'DE_noise':>8s} | {'DE_null':>7s}")
for name, _, _ in DATASETS:
    r = results[name]
    if r.get("skip"):
        print(f"{name:12s} | {r['n_ctrl']:7d} | {r['n_una']:7d} |  (control too sparse -- not evaluable)"); continue
    ratio = r["noise_med"] / max(r["null_med"], 1e-9)
    print(f"{name:12s} | {r['n_ctrl']:7d} | {r['n_una']:7d} | {r['noise_med']:10.4f} | {r['null_med']:9.4f} | {ratio:5.1f}x | {r['noise_de']:8d} | {r['null_de']:7.0f}")
print("\nInterpretation: ratio~1 & DE_noise~DE_null -> NO CRISPR noise (mu_action unnecessary).")
print("                ratio>>1 & DE_noise>>DE_null -> real noise that the model must remove.")
print("DONE")
