"""Per-method expression preprocessing and scoring-space conversion.

Prepared slides store:
  - ``layers['counts']`` / raw cache  -> integer raw counts
  - ``adata.X``                       -> log-CP10K (CP10K + log1p) for generic scoring

Tutorial input formats (``adata_for_method``):

  biolord, scGEN, GEARS, CellAgentChat
      log-CP10K from prepared ``adata.X``.

  CelComen
      raw counts -> CPM (1e6) + log1p.

  Steamboat
      raw counts -> ``prep_adatas(norm=True, log1p=True)`` (scanpy default target_sum=1e4).

Scoring expects **predictions in log-CP10K** matching prepared ``adata.X``. The
reported metrics are LFC Spearman rho and LFC RMSE between predicted and observed
pseudobulk log fold changes.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import scanpy as sc

LOG_CP10K_METHODS = frozenset({"biolord", "gears", "scgen", "cellagentchat"})
LOG_CP10K_CLIP = 14.5


def with_raw_counts(adata: ad.AnnData, copy_data: bool = True) -> ad.AnnData:
    """Return AnnData with ``.X`` set to raw counts."""
    out = adata.copy() if copy_data else adata
    if "counts" not in out.layers:
        raise ValueError("prepared adata missing layers['counts'] -- cannot recover raw counts")
    out.X = out.layers["counts"].copy()
    return out


def celcomen_logcpm(adata: ad.AnnData, copy_data: bool = True) -> ad.AnnData:
    """CelComen tutorial: CPM (1e6) + log1p from raw counts."""
    out = with_raw_counts(adata, copy_data=copy_data)
    sc.pp.normalize_total(out, target_sum=1e6)
    sc.pp.log1p(out)
    out.uns.pop("log1p", None)
    return out


def celcomen_preds_to_logcp10k(log_cpm: np.ndarray) -> np.ndarray:
    """Map CelComen SimComen output (log-CPM) to prepared-slide log-CP10K."""
    x = np.asarray(log_cpm, dtype=np.float64)
    # Already converted (log-CP10K is typically lower than log-CPM for same counts)
    if x.size and float(np.max(x)) < 12 and float(np.median(x)) < 1.5:
        return np.clip(x, 0.0, LOG_CP10K_CLIP)
    cpm = np.expm1(np.clip(x, 0.0, None))
    return np.log1p(cpm / 100.0)


def raw_counts_to_logcp10k(counts: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Raw integer counts -> log-CP10K (per-cell library size normalization)."""
    x = np.clip(np.asarray(counts, dtype=np.float64), 0.0, None)
    ls = x.sum(axis=1, keepdims=True) + eps
    return np.log1p(x / ls * 1e4)


def _looks_like_raw_counts(x: np.ndarray) -> bool:
    """Heuristic: matrix is raw counts, not log-normalized expression."""
    if x.size == 0:
        return False
    med = float(np.median(x))
    mx = float(x.max())
    return mx > 30 and med >= 0.5 and mx / (med + 1e-9) > 20


def preds_to_scoring_space(method: str, pred: np.ndarray) -> np.ndarray:
    """Convert method predictions to log-CP10K for LFC scoring."""
    m = method.lower()
    x = np.asarray(pred, dtype=np.float64)
    if x.size == 0:
        return x

    if m == "concert":
        # CONCERT counterfactualPrediction returns decoder MEAN expression in count
        # scale (mu = exp(log mu)). It is sparse (median ~0), so _looks_like_raw_counts
        # (which requires median >= 0.5) misclassifies it as already-log-CP10K, leaving
        # count-scale values. Always normalize from counts.
        x = raw_counts_to_logcp10k(x)
    elif _looks_like_raw_counts(x):
        x = raw_counts_to_logcp10k(x)
    elif m == "celcomen":
        x = celcomen_preds_to_logcp10k(x)
    else:
        # biolord, scgen, gears, cellagentchat, steamboat: already log-CP10K (or close)
        x = np.clip(x, 0.0, None)

    return np.clip(x, 0.0, LOG_CP10K_CLIP)


def normalize_pred_cache(method: str, pred_g_cache: dict[str, np.ndarray | None]) -> dict[str, np.ndarray | None]:
    """Apply ``preds_to_scoring_space`` to every finite prediction matrix."""
    out: dict[str, np.ndarray | None] = {}
    for g, pred in pred_g_cache.items():
        if pred is None:
            out[g] = None
        else:
            out[g] = preds_to_scoring_space(method, pred)
    return out


def adata_for_method(adata: ad.AnnData, method: str, copy_data: bool = True) -> ad.AnnData:
    """Expression matrix in the method's tutorial input format."""
    m = method.lower()
    if m == "celcomen":
        return celcomen_logcpm(adata, copy_data=copy_data)
    if m == "steamboat":
        return with_raw_counts(adata, copy_data=copy_data)
    if m in LOG_CP10K_METHODS:
        return adata.copy() if copy_data else adata
    raise ValueError(f"unknown method: {method}")


def steamboat_prep_adatas(adatas: list[ad.AnnData], sf, n_neighs: int, mask_var=False):
    """Steamboat tutorial: ``prep_adatas`` on raw counts with norm + log1p."""
    raw = [with_raw_counts(a) for a in adatas]
    prep = sf.prep_adatas(raw, norm=True, log1p=True, n_neighs=n_neighs)
    return prep, raw
