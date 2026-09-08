"""Pure functions for the CONCERT-spac spatial-counterfactual evaluation harness.

No model/IO dependencies — these are the unit-tested core used by
``eval_spatial_counterfactual.py``: niche-grid construction, NTC-referenced
pseudobulk LFC, LFC-agreement scoring, perturbation-panel selection, a
spatial-kNN baseline, and contiguous held-out patch construction.
"""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.decomposition import PCA


def build_niche_grid(coords: np.ndarray, grid: int = 6,
                     loc_range: float = 20.0) -> tuple[np.ndarray, int]:
    """Assign each (x,y) coord to a regular grid bin in [0,loc_range]^2.
    Returns (bin_id per cell, total n_bins=grid*grid)."""
    c = np.clip(np.asarray(coords, dtype=np.float64)[:, :2], 0, loc_range - 1e-9)
    ix = np.minimum((c[:, 0] / loc_range * grid).astype(int), grid - 1)
    iy = np.minimum((c[:, 1] / loc_range * grid).astype(int), grid - 1)
    return (iy * grid + ix).astype(int), grid * grid


def pseudobulk_lfc(pert: np.ndarray, ntc: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Per-gene log fold change of pseudobulk means: log(mean_pert+eps) - log(mean_ntc+eps).
    Inputs are (cells, genes) on the model's expression scale. Returns (genes,)."""
    mp = np.asarray(pert, dtype=np.float64).mean(axis=0)
    mn = np.asarray(ntc, dtype=np.float64).mean(axis=0)
    return np.log(mp + eps) - np.log(mn + eps)


def score_lfc_agreement(model_lfc: np.ndarray, obs_lfc: np.ndarray, top_k: int = 50,
                        sig_thresh: float = 0.25) -> dict:
    """Agreement between model-predicted and observed per-gene LFC.
    - spearman: rank correlation over all genes
    - lfc_dist: RMSE between LFC vectors (magnitude, not rank; lower is better)
    - sign_agreement: fraction of observed-significant genes with matching sign
    - topk_overlap: |top-k |obs| genes ∩ top-k |model| genes| / k"""
    m = np.asarray(model_lfc, dtype=np.float64)
    o = np.asarray(obs_lfc, dtype=np.float64)
    rho = float(spearmanr(m, o).correlation)
    fin = np.isfinite(m) & np.isfinite(o)
    lfc_dist = float(np.sqrt(np.mean((m[fin] - o[fin]) ** 2))) if fin.any() else float("nan")
    sig = np.abs(o) >= sig_thresh
    sign_agr = float(np.mean(np.sign(m[sig]) == np.sign(o[sig]))) if sig.any() else float("nan")
    k = int(min(top_k, m.size))
    tm = set(np.argsort(-np.abs(m))[:k].tolist())
    to = set(np.argsort(-np.abs(o))[:k].tolist())
    return {"spearman": rho, "lfc_dist": lfc_dist, "sign_agreement": sign_agr,
            "topk_overlap": len(tm & to) / k, "n_sig": int(sig.sum())}


def energy_distance(X: np.ndarray, Y: np.ndarray, gene_mask: np.ndarray,
                    subsample: int = 500, n_pcs: int = 50,
                    random_state: int = 0) -> float:
    """Energy distance between two cell populations (population-level, not pseudobulk).

    Captures distributional similarity (shape, variance) not just means.
    Use for: comparing predicted vs observed KO cell distributions per niche.
    Lower = more similar distributions.

    Steps: subset HVGs → log1p normalize → subsample → joint PCA (50 PCs) → e-distance.
    E(X,Y) = 2·mean||x-y|| - mean||x-x'|| - mean||y-y'||
    """
    X = np.log1p(np.asarray(X, dtype=np.float64)[:, gene_mask])
    Y = np.log1p(np.asarray(Y, dtype=np.float64)[:, gene_mask])
    rng = np.random.default_rng(random_state)
    if len(X) > subsample:
        X = X[rng.choice(len(X), subsample, replace=False)]
    if len(Y) > subsample:
        Y = Y[rng.choice(len(Y), subsample, replace=False)]
    n_comp = min(n_pcs, X.shape[1], X.shape[0] + Y.shape[0] - 1)
    combined = np.vstack([X, Y])
    pca = PCA(n_components=n_comp, random_state=random_state)
    proj = pca.fit_transform(combined)
    Xp, Yp = proj[:len(X)], proj[len(X):]
    xy = cdist(Xp, Yp).mean()
    # U-statistic: exclude diagonal (i==j, distance=0) so we average over
    # distinct pairs only.  cdist.mean() is the V-statistic which includes
    # n diagonal zeros, underestimating xx/yy by factor (n-1)/n — a 5-10%
    # downward bias for small niches (n≈10–20) that inflates the result.
    n, m = len(Xp), len(Yp)
    xx_d = cdist(Xp, Xp)
    xx = xx_d[np.triu_indices(n, k=1)].mean() if n > 1 else 0.0
    yy_d = cdist(Yp, Yp)
    yy = yy_d[np.triu_indices(m, k=1)].mean() if m > 1 else 0.0
    return float(2 * xy - xx - yy)


EXCLUDE_LABELS = {"control", "unassigned", "Unassigned", "multiple", "Multiple"}


def sample_global_source_indices(src_mask, max_src_cells: int = 500, seed: int = 42) -> np.ndarray:
    """Sample unassigned/source cells from the whole slide (never niche-restricted).

    Non-spatial methods must not use unassigned cells from the same niche as the
    perturbed evaluation target — that leaks spatial context into the input."""
    mask = np.asarray(src_mask, dtype=bool)
    global_src_idx = np.where(mask)[0]
    if max_src_cells > 0 and len(global_src_idx) > max_src_cells:
        rng = np.random.default_rng(seed)
        global_src_idx = rng.choice(global_src_idx, max_src_cells, replace=False)
    return global_src_idx


def select_perturbation_panel(labels, bin_id, n_evaluable_niches, min_cells=200,
                              min_cells_per_niche=20, min_niche_frac=0.0, max_panel=30):
    """Select top gene-KO perturbations by total cell count.

    Keep g iff total cells >= min_cells AND it has >= min_cells_per_niche cells in
    at least one spatial niche. If min_niche_frac > 0, also require presence in
    >= ceil(min_niche_frac * n_evaluable_niches) niches (legacy spatial-spread filter).
    NTC/unassigned/multiple excluded. Returns up to max_panel labels, sorted by
    descending cell count."""
    labels = np.asarray(labels)
    bin_id = np.asarray(bin_id, dtype=int)
    n_bins = int(bin_id.max()) + 1 if len(bin_id) else 1
    if min_niche_frac > 0:
        need_niches = int(np.ceil(min_niche_frac * n_evaluable_niches))
    else:
        need_niches = 1
    cand = {}
    for g in set(labels.tolist()) - EXCLUDE_LABELS:
        m = labels == g
        if int(m.sum()) < min_cells:
            continue
        per_niche = np.bincount(bin_id[m], minlength=n_bins)
        if int((per_niche >= min_cells_per_niche).sum()) >= need_niches:
            cand[g] = int(m.sum())
    return [g for g, _ in sorted(cand.items(), key=lambda kv: -kv[1])[:max_panel]]


def spatial_knn_mean(query_xy, coords, expr, k=20):
    """Baseline: mean expression of the k spatially-nearest reference cells per query coord."""
    from scipy.spatial import cKDTree
    k = min(k, len(coords))
    tree = cKDTree(np.asarray(coords)[:, :2])
    _, idx = tree.query(np.asarray(query_xy)[:, :2], k=k)
    idx = np.atleast_2d(idx)
    return np.asarray(expr)[idx].mean(axis=1)


def select_hvgs(counts, n_top=2000, n_bins=20):
    """Seurat-style highly-variable-gene selection on size-factor + log1p normalized
    counts. Bins genes by mean, z-scores dispersion (var/mean) within each bin to
    remove the mean-variance trend, returns a boolean mask of the top n_top genes.

    counts: (cells, genes) raw counts. Operating on normalized data isolates genes
    carrying real cell-state/perturbation signal rather than high-mean housekeeping."""
    counts = np.asarray(counts, dtype=np.float64)
    lib = counts.sum(axis=1, keepdims=True)
    sf = lib / (np.median(lib[lib > 0]) + 1e-12)
    sf[sf == 0] = 1.0
    logn = np.log1p(counts / sf)
    mean = logn.mean(axis=0)
    var = logn.var(axis=0)
    n_top = int(min(n_top, logn.shape[1]))
    with np.errstate(invalid="ignore", divide="ignore"):
        disp = np.where(mean > 0, var / mean, 0.0)
    # bin by mean, z-score dispersion within each bin (mean-variance detrending).
    # clamp n_bins so each bin keeps enough genes for a meaningful within-bin z-score.
    n_bins = max(1, min(n_bins, logn.shape[1] // 5))
    order = np.argsort(mean)
    norm_disp = np.zeros_like(disp)
    edges = np.linspace(0, len(order), n_bins + 1).astype(int)
    for i in range(n_bins):
        idx = order[edges[i]:edges[i + 1]]
        if len(idx) == 0:
            continue
        d = disp[idx]
        sd = d.std()
        norm_disp[idx] = (d - d.mean()) / sd if sd > 1e-12 else 0.0
    mask = np.zeros(logn.shape[1], dtype=bool)
    mask[np.argsort(-norm_disp)[:n_top]] = True
    return mask


def pseudobulk_profile_score(pred, obs, bin_id, n_bins, gene_mask=None, min_cells=30):
    """Sparsity-robust imputation metric: per-niche pseudobulk MEAN profile (over
    gene_mask genes), Spearman across genes between predicted and observed, averaged
    over niches with >= min_cells. Returns (mean_spearman, n_scored_niches)."""
    pred = np.asarray(pred, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    bin_id = np.asarray(bin_id)
    cols = np.where(gene_mask)[0] if gene_mask is not None else np.arange(pred.shape[1])
    rhos = []
    for b in range(n_bins):
        m = bin_id == b
        if m.sum() < min_cells:
            continue
        mp = pred[m][:, cols].mean(axis=0)
        mo = obs[m][:, cols].mean(axis=0)
        rho = spearmanr(mp, mo).correlation
        if np.isfinite(rho):
            rhos.append(rho)
    return (float(np.mean(rhos)) if rhos else float("nan")), len(rhos)


def per_cell_spearman_mean(pred, obs, gene_mask=None):
    """Mean over cells of per-cell Spearman across genes (ranks genes within a cell).
    Restrict to gene_mask (e.g. HVGs) to avoid zero-tie domination on sparse data."""
    pred = np.asarray(pred, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    if gene_mask is not None:
        pred, obs = pred[:, gene_mask], obs[:, gene_mask]
    if pred.shape[1] < 3:
        return float("nan")
    rp = pred.argsort(axis=1).argsort(axis=1).astype(np.float64)
    ro = obs.argsort(axis=1).argsort(axis=1).astype(np.float64)
    rp -= rp.mean(axis=1, keepdims=True)
    ro -= ro.mean(axis=1, keepdims=True)
    num = (rp * ro).sum(axis=1)
    den = np.sqrt((rp ** 2).sum(axis=1) * (ro ** 2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        c = num / den
    return float(np.nanmean(c))


def split_half_lfc_reliability(g_expr, ntc_expr, gene_mask=None, n_seeds=5, base_seed=0):
    """Split-half reliability of the OBSERVED per-niche NTC-referenced pseudobulk LFC.

    Randomly halve the g-cell and the NTC-cell populations, compute the LFC on each
    half independently, and Spearman-correlate the two halves across (gene_mask) genes.
    Averaged over n_seeds. Returns (mean_split_half_rho, spearman_brown_corrected).

    This BOUNDS the maximum achievable model<->observed agreement. If the observed LFC
    is noise-dominated (the two halves of the same data do not even agree), then no
    model can correlate with it — a low model score is a property of the ground truth,
    not the model. Spearman-Brown (`2r/(1+r)`) projects the half-sample reliability up
    to the full-population size that the model is actually scored against."""
    g_expr = np.asarray(g_expr, dtype=np.float64)
    ntc_expr = np.asarray(ntc_expr, dtype=np.float64)
    cols = np.where(gene_mask)[0] if gene_mask is not None else np.arange(g_expr.shape[1])
    if g_expr.shape[0] < 4 or ntc_expr.shape[0] < 4 or len(cols) < 3:
        return float("nan"), float("nan")
    rhos = []
    for s in range(n_seeds):
        rng = np.random.RandomState(base_seed + s)
        gp = rng.permutation(g_expr.shape[0])
        npp = rng.permutation(ntc_expr.shape[0])
        g1, g2 = gp[:len(gp) // 2], gp[len(gp) // 2:]
        n1, n2 = npp[:len(npp) // 2], npp[len(npp) // 2:]
        lfc1 = pseudobulk_lfc(g_expr[g1], ntc_expr[n1])[cols]
        lfc2 = pseudobulk_lfc(g_expr[g2], ntc_expr[n2])[cols]
        rho = spearmanr(lfc1, lfc2).correlation
        if np.isfinite(rho):
            rhos.append(rho)
    if not rhos:
        return float("nan"), float("nan")
    r = float(np.mean(rhos))
    sb = float(2 * r / (1 + r)) if r > -1 + 1e-9 else float("nan")
    return r, sb


def cell_state_consistency_test(Y_obs, perturb_raw, bin_id, panel, ntc_label="control",
                                gene_mask=None, n_pairs=500, n_seeds=5, base_seed=0):
    """Single-cell perturbation signal-detection test (no model required).

    For each panel gene g and each niche b, samples `n_pairs` SAME-state pairs
    (g,g) and `n_pairs` CROSS-state pairs (g, NTC) from within the same niche.
    Computes per-pair Spearman across (gene_mask) genes, then reports the mean
    difference and a Mann-Whitney U p-value.

    If same-state pairs are NOT significantly more similar than cross-state pairs,
    the perturbation effect is undetectable at the single-cell level and no model
    can learn it. Returns dict: per g -> {n_niches, same_mean, cross_mean,
    effect_size, mw_pvalue_median}.
    """
    from scipy.stats import mannwhitneyu
    Y = np.asarray(Y_obs, dtype=np.float64)
    cols = np.where(gene_mask)[0] if gene_mask is not None else np.arange(Y.shape[1])
    perturb_raw = np.asarray(perturb_raw)
    bin_id = np.asarray(bin_id)
    n_bins = int(bin_id.max()) + 1

    def _pair_spearman(a, b):
        # per-pair Spearman across cols for two 1-D gene vectors
        ra = a[cols].argsort().argsort().astype(np.float64)
        rb = b[cols].argsort().argsort().astype(np.float64)
        ra -= ra.mean(); rb -= rb.mean()
        d2 = np.sqrt((ra**2).sum() * (rb**2).sum())
        return float((ra * rb).sum() / d2) if d2 > 1e-12 else 0.0

    results = {}
    for g in panel:
        niche_same, niche_cross, niche_pvals = [], [], []
        for b in range(n_bins):
            idx_g = np.where((perturb_raw == g) & (bin_id == b))[0]
            idx_n = np.where((perturb_raw == ntc_label) & (bin_id == b))[0]
            if len(idx_g) < 4 or len(idx_n) < 2:
                continue
            same_rhos, cross_rhos = [], []
            for s in range(n_seeds):
                rng = np.random.RandomState(base_seed + s)
                # same-state: two g cells
                for _ in range(n_pairs // n_seeds):
                    i, j = rng.choice(len(idx_g), 2, replace=False)
                    same_rhos.append(_pair_spearman(Y[idx_g[i]], Y[idx_g[j]]))
                # cross-state: one g cell, one NTC cell
                for _ in range(n_pairs // n_seeds):
                    i = rng.randint(len(idx_g))
                    j = rng.randint(len(idx_n))
                    cross_rhos.append(_pair_spearman(Y[idx_g[i]], Y[idx_n[j]]))
            if not same_rhos or not cross_rhos:
                continue
            _, p = mannwhitneyu(same_rhos, cross_rhos, alternative="greater")
            niche_same.append(float(np.mean(same_rhos)))
            niche_cross.append(float(np.mean(cross_rhos)))
            niche_pvals.append(float(p))
        if not niche_same:
            continue
        results[g] = dict(
            n_niches=len(niche_same),
            same_mean=float(np.mean(niche_same)),
            cross_mean=float(np.mean(niche_cross)),
            effect_size=float(np.mean(niche_same)) - float(np.mean(niche_cross)),
            mw_pvalue_median=float(np.median(niche_pvals)),
        )
    return results


def make_spatial_patches(coords, k=4, loc_range=20.0, area_frac=0.12, seed=0):
    """Boolean mask for k square contiguous patches summing to ~area_frac of the tissue."""
    rng = np.random.RandomState(seed)
    c = np.asarray(coords)[:, :2]
    side = loc_range * np.sqrt(area_frac / k)         # k square patches summing to area_frac
    mask = np.zeros(c.shape[0], dtype=bool)
    for _ in range(k):
        cx, cy = rng.uniform(side / 2, loc_range - side / 2, size=2)
        in_patch = (np.abs(c[:, 0] - cx) <= side / 2) & (np.abs(c[:, 1] - cy) <= side / 2)
        mask |= in_patch
    return mask
