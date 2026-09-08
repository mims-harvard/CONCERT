#!/usr/bin/env python
"""Global 1-based indices of the REAL lesion spots on PT1 (perturbation == 'stroke').

Leakage test: if switching these OFF (--target_cell_perturbation __basal__) leaves the injury
genes just as high, the lesion is memorised by the coordinate-only basal RFF field and f_pert
has nothing left to learn.
"""
from pathlib import Path
import numpy as np, h5py

SL = Path("/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/stroke3d/slides")
ORDER = ["PT1", "PT2", "PT3", "PT4", "Sham1", "Sham2", "Sham3", "Sham4"]
OUT = Path("/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/CONCERT/outputs/stroke_2dpara/pert_sets")

off, sel = 0, None
for s in ORDER:
    with h5py.File(SL / f"stroke_paste2_{s}.h5ad", "r") as f:
        n = f["obs"]["_index"].shape[0]
        if s == "PT1":
            cats = [c.decode() if isinstance(c, bytes) else str(c)
                    for c in f["obs"]["perturbation"]["categories"][:]]
            codes = f["obs"]["perturbation"]["codes"][:]
            lab = np.array(cats)[codes]
            print("PT1 perturbation categories:", cats, "counts:",
                  {c: int((lab == c).sum()) for c in cats})
            sel = np.nonzero(lab == "stroke")[0] + off + 1     # 1-based global
            tis = f["obs"]["tissue"]
            tcats = np.array([c.decode() if isinstance(c, bytes) else str(c)
                              for c in tis["categories"][:]])[tis["codes"][:]]
            u, c = np.unique(tis and tcats[lab == "stroke"], return_counts=True)
            print("  tissue of the stroke spots:", dict(zip(u.tolist(), c.tolist())))
    off += n
np.savetxt(OUT / "real_lesion_PT1.txt", sel, fmt="%d")
print(f"wrote {len(sel)} indices -> {OUT/'real_lesion_PT1.txt'}  (range {sel.min()}..{sel.max()})")
