"""Formal inflammation gene sets = MSigDB C2 Reactome v2025.1, mapped to the 942-gene
mouse MERFISH panel (human->mouse by symbol case, standard for 1:1 orthologs).

We SELECT a curated list of specific inflammation Reactome pathways (dropping the giant
umbrella terms -- Cytokine signaling 216, Innate immune 102, Signaling by interleukins 162 --
which overlap everything and wash out the heatmap). Each kept set has >=8 panel genes.
Writes reactome_inflammation_gene_sets.json {short_label: [panel genes]}.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "benchmark_evaluation" / "release_paths.py").is_file()) / "benchmark_evaluation"))
import release_paths as _rp  # noqa: E402  data/, checkpoints/ and runs/ roots
import json
from pathlib import Path

GMT = f"{_rp.DATA}/dss_colitis/c2.cp.reactome.v2025.1.Hs.symbols.gmt"
PANEL = Path(f"{_rp.WORKDIR}/outputs/"
             "dss_timekernel_markersign_timefield_fixedt/hvg_gene_names.txt")
OUT = Path(__file__).resolve().parent / "reactome_inflammation_gene_sets.json"

panel = [l.strip() for l in PANEL.read_text().splitlines() if l.strip()]
panel_upper = {g.upper(): g for g in panel}          # UPPER(human) -> mouse panel symbol
sets = {}
for line in Path(GMT).read_text().splitlines():
    p = line.split("\t"); sets[p[0]] = p[2:]

# curated inflammation selection: (Reactome name, short display label)
SELECT = [
    ("REACTOME_CHEMOKINE_RECEPTORS_BIND_CHEMOKINES",      "Chemokine signaling"),
    ("REACTOME_NEUTROPHIL_DEGRANULATION",                 "Neutrophil degranulation"),
    ("REACTOME_TOLL_LIKE_RECEPTOR_CASCADES",              "TLR cascades"),
    ("REACTOME_TNFR2_NON_CANONICAL_NF_KB_PATHWAY",        "TNF / NF-kB"),
    ("REACTOME_INTERLEUKIN_1_FAMILY_SIGNALING",           "IL-1 family"),
    ("REACTOME_INTERLEUKIN_17_SIGNALING",                 "IL-17 signaling"),
    ("REACTOME_INTERLEUKIN_6_FAMILY_SIGNALING",           "IL-6 family"),
    ("REACTOME_INTERLEUKIN_10_SIGNALING",                 "IL-10 signaling"),
    ("REACTOME_INTERLEUKIN_12_FAMILY_SIGNALING",          "IL-12 family"),
    ("REACTOME_INTERLEUKIN_4_AND_INTERLEUKIN_13_SIGNALING", "IL-4 / IL-13"),
    ("REACTOME_INTERFERON_ALPHA_BETA_SIGNALING",          "IFN-alpha/beta"),
    ("REACTOME_INTERFERON_GAMMA_SIGNALING",               "IFN-gamma"),
    ("REACTOME_INTERFERON_SIGNALING",                     "Interferon (all)"),
]

out = {}
print(f"{'label':26s} {'panel':>5s}  genes")
for name, label in SELECT:
    if name not in sets:
        print(f"[MISS] {name}"); continue
    genes = sorted({panel_upper[g.upper()] for g in sets[name] if g.upper() in panel_upper})
    if len(genes) < 5:
        print(f"[skip <5] {label} ({len(genes)})"); continue
    out[label] = genes
    print(f"{label:26s} {len(genes):>5d}  {', '.join(genes[:14])}{' ...' if len(genes)>14 else ''}")

OUT.write_text(json.dumps(out, indent=2))
print(f"\n[write] {OUT}  ({len(out)} inflammation pathways)")
