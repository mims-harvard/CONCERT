#!/usr/bin/env python
"""Should the sham prediction be calibrated against the true PT?

The two quantities currently plotted are NOT the same thing. The prediction is a DELTA (lesion
decode minus basal decode on the sham brain); the observed comparator is an ABSOLUTE level on the
PT brain. Putting them on one axis needs a twin axis precisely because of that mismatch -- which
is a sign the comparator is wrong, not that a scale factor is missing.

The commensurable observed quantity is an observed delta:
    observed_delta = mean(PT lesion spots) - mean(SHAM spots at the same imposed location)
same gene, same place, healthy vs struck. This reports both, plus the per-gene scale factor
observed_delta / predicted_delta -- which IS the model's amplitude deficit. If we calibrate, that
number disappears into the fit, so it is reported here rather than silently applied.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "benchmark_evaluation" / "release_paths.py").is_file()) / "benchmark_evaluation"))
import release_paths as _rp  # noqa: E402  data/, checkpoints/ and runs/ roots
import json, sys, numpy as np, anndata as ad, h5py, pandas as pd
from pathlib import Path
NS = Path(f"{_rp.WORKDIR}")
DEL = Path(f"{_rp.DATA}/external/CONCERT/delivery/stroke-registration-benchmark")
tag = sys.argv[1] if len(sys.argv) > 1 else "_crA"
OUT = NS / f"outputs/stroke_3d{tag}"
SL = NS / "stroke3d/slides_cr"
GEN = ["Gm42418", "Scgb1a1", "Spp1", "Vim", "Lgals1", "Timp1"]

def cpm(X):
    s = X.sum(1, keepdims=True); s[s == 0] = 1.0
    return np.log1p(X / s * 1e4)

def stack(p):
    ps = sorted(SL.glob(f"stroke_paste2_{p}*.h5ad"))
    return ad.concat([ad.read_h5ad(x) for x in ps], index_unique=None)

side = json.loads((OUT / "sham_cells.json").read_text())
rows = np.concatenate([np.arange(o, o+c) for o, c, t in zip(
    np.concatenate([[0], np.cumsum(side["counts"])])[:-1], side["counts"], side["is_target"]) if t])
rd = lambda t: cpm(np.asarray(ad.read_h5ad(next(OUT.glob(
    f"stroke_3d{tag}_{t}_*_perturbed_counts.h5ad"))).X[rows], np.float32))
Lc, Lr, Lo = rd("oncore"), rd("onrim"), rd("off")
gp = np.loadtxt(OUT / "hvg_gene_names.txt", dtype=str)
sd = OUT / "shape_sets"
A_sh = stack("Sham"); n = A_sh.n_obs
mk = lambda f: np.isin(np.arange(n), np.loadtxt(sd / f, dtype=np.int64))
p_in, p_core, p_rim = mk("shape_lesion_local.txt"), mk("shape_core_local.txt"), mk("shape_rim_local.txt")
p_core, p_rim = p_core & p_in, p_rim & p_in
D = np.zeros_like(Lo); D[p_core] = Lc[p_core]-Lo[p_core]; D[p_rim] = Lr[p_rim]-Lo[p_rim]
Lsh = cpm(np.asarray(A_sh.X.todense(), np.float32))          # observed sham (healthy baseline)
A_pt = stack("PT"); Lpt = cpm(np.asarray(A_pt.X.todense(), np.float32))
gs = np.asarray(A_pt.var_names)
with h5py.File(DEL/"benchmark_artifacts/data/Mouse_brain_stroke_all_data.h5", "r") as f:
    bc = np.array([b.decode() for b in f["Barcode"][:]]); rg = np.array([b.decode() for b in f["Celltype_coarse"][:]])
rp = pd.Series(rg, index=bc).loc[np.asarray(A_pt.obs_names)].to_numpy()
o_core, o_rim = rp == "ICA", rp == "PIA"

print(f"{'gene':9s} {'obs_delta_core':>15s} {'pred_core':>10s} {'scale':>7s} | "
      f"{'obs_delta_rim':>14s} {'pred_rim':>9s} {'scale':>7s}")
res = {}
for g in GEN:
    ip, io = int(np.where(gp == g)[0][0]), int(np.where(gs == g)[0][0])
    # observed delta = struck tissue vs the SAME place on the healthy brain
    od_c = float(Lpt[o_core, io].mean() - Lsh[p_core, io].mean())
    od_r = float(Lpt[o_rim, io].mean() - Lsh[p_rim, io].mean())
    pc, pr = float(D[p_core, ip].mean()), float(D[p_rim, ip].mean())
    sc = od_c/pc if abs(pc) > 1e-6 else float("nan")
    sr = od_r/pr if abs(pr) > 1e-6 else float("nan")
    res[g] = dict(obs_delta_core=od_c, obs_delta_rim=od_r, pred_core=pc, pred_rim=pr,
                  scale_core=sc, scale_rim=sr)
    print(f"{g:9s} {od_c:15.3f} {pc:10.3f} {sc:7.2f} | {od_r:14.3f} {pr:9.3f} {sr:7.2f}")
(OUT / "diag_amplitude.json").write_text(json.dumps(res, indent=2))
