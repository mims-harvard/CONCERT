#!/usr/bin/env python
"""Stroke: a lesion imposed on a healthy brain, predicted in a 2D and a 3D arm.

The script form of `notebooks/04_stroke_2d_and_3d_prediction.ipynb`.

A photothrombotic lesion cannot be placed at a chosen size and position in a measured animal.
So the lesion geometry is taken from the animal that has one and imposed on the sham animal,
whose tissue was never injured, and the model is asked what the sham tissue would look like if
it were. Two arms differ only in how the perturbation field is allowed to extend through the
stack: the 2D arm treats each section independently, the 3D arm couples four z levels.

Three decode passes per arm make one prediction:
  oncore  the core compartment set to the ICA label
  onrim   the rim compartment set to the PIA label
  off     every sham spot left unperturbed -- the baseline the other two are differenced against

The scored claim is the core-versus-rim direction: injury markers should separate the two
compartments in the same direction as the measurement.

Steps
  weights      what the two checkpoints hold and where their shapes differ
  decode       re-run the three passes for the chosen arms       (GPU, about 20 min per arm)
  assemble     build the predicted induction field per compartment, and read the measured side
  spatial      the section-by-section figure, measured beside predicted
  profiles     the rim-to-core depth profile and the compartment bar panels
  check        recomputed compartment statistics against the shipped ones

`decode` is off unless asked for: the shipped decodes are read instead.

Examples
  python scripts/04_stroke_2d_3d_prediction.py
  python scripts/04_stroke_2d_3d_prediction.py --gene Spp1 --sections 0 1
  python scripts/04_stroke_2d_3d_prediction.py --run-decode --arms 2d
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import concert_paths as cp

MODEL_SRC = cp.MODELS / "concert_spac_niche"      # the copy that produced these results
EXP = cp.EXPERIMENTS / "07_stroke_case_study"

SLIDES = cp.CASE_STUDY_ROOT / "stroke3d" / "slides_cr"     # compartment-labelled slides
ARM_DIR = {"2d": cp.OUTPUT_ROOT / "stroke_2d_cr", "3d": cp.OUTPUT_ROOT / "stroke_3d_cr"}

# region labels of the measured stroke animal (ICA / PIA), from the registration delivery
REGION_H5 = Path(cp._p(
    "CONCERT_STROKE_REGION_H5",
    "/n/holylfs06/LABS/mzitnik_lab/Lab/bof695/CONCERT/delivery/stroke-registration-benchmark/"
    "benchmark_artifacts/data/Mouse_brain_stroke_all_data.h5"))

STACKS = ["PT", "Sham"]        # PT = photothrombosis (real lesion), Sham = healthy control
PASSES = [("oncore", "ICA", "shape_core.txt"),
          ("onrim", "PIA", "shape_rim.txt"),
          ("off", "__basal__", None)]           # None -> every sham spot
GENES = ["Gm42418", "Scgb1a1", "Spp1", "Vim", "Lgals1", "Timp1"]
NB = 6                                          # depth bins, rim -> core


def decode_cmd(arm, shadow, pass_name, target, cells_file):
    """The eval call for one decode pass. Every flag has to match the training flags."""
    nz = {"2d": "1", "3d": "4"}[arm]
    slide_files = ",".join(str(SLIDES / f"stroke_paste2_{s}{i}.h5ad")
                           for s in STACKS for i in (1, 2, 3, 4))
    return [
        sys.executable, "-u", "run_concert_spac.py", "--stage", "eval",
        "--config", "config.yaml", "--sample", f"stroke_{arm}_cr_{pass_name}",
        "--multi_sample_files", slide_files,
        "--outdir", str(shadow), "--model_file", "model.pt", "--project_index", "stroke",
        "--hvg_max_rank", "2000", "--seed", "42",
        "--unassigned_label", "unperturbed", "--control_label", "__NONE__",
        "--min_cells_per_pert", "25",
        "--use_action_correction", "False", "--use_pad", "False",
        "--inducing_point_steps", "6", "--grid_inducing_points", "True",
        "--loc_range", "20.0", "--kernel_scale", "10.0", "--multi_kernel_mode", "True",
        "--use_expr_encoder", "True", "--expr_hidden_dims", "512", "256",
        "--use_basal_encoder_only", "True",
        "--dynamicVAE", "False", "--free_bits", "0.5", "--gp_free_bits", "1.0",
        "--use_nngp", "False", "--use_paracrine", "False",
        "--use_niche_interaction", "False", "--use_niche_film", "True",
        "--niche_film_dim", "8", "--niche_film_k", "25",
        "--niche_film_gcap", "3.0", "--niche_film_l2", "1e-3",
        "--use_pert_gate", "True", "--gate_hidden", "64",
        "--use_f_pert", "True", "--f_pert_clamp", "5.0",
        "--spatial_basal_only", "True", "--lfc_ref_tissue", "CTX",
        "--basal_field", "rff", "--basal_bayes", "True",
        "--moran_lambda", "1.0", "--disentangle_radius", "4.0",
        "--fpert_field", "gp_pp", "--fpert_coord_scale", "20.0",
        "--fpert_rff_features", "128", "--fpert_rff_bandwidth", "0.15",
        "--fpert_ls_min", "0.05", "--fpert_ls_max", "0.60",
        "--tile_by_stack", "True", "--z_stack_obs", "mouse",
        "--z_index_obsm", "section_index",
        "--fpert_n_z_levels", nz, "--fpert_z_gain", "0.25",
        "--ls_prior_lambda", "1.0", "--ls_prior_min", "3", "--ls_prior_max", "18",
        "--fpert_svgp_inducing", "16", "--fpert_svgp_ls_init", "5.0",
        "--pert_cells", str(cells_file),
        "--target_cell_tissue", "CTX", "--target_cell_perturbation", target,
        "--verbosity", "1",
    ]


def log1p_cpm(x):
    import numpy as np
    s = x.sum(1, keepdims=True)
    s[s == 0] = 1.0
    return np.log1p(x / s * 1e4)


def norm_xy(xy):
    lo, hi = xy.min(0), xy.max(0)
    import numpy as np
    return (xy - lo) / np.maximum(hi - lo, 1e-8)


def load_stack(stack):
    import anndata as ad
    import numpy as np
    ps = sorted(SLIDES.glob(f"stroke_paste2_{stack}*.h5ad"))
    a = ad.concat([ad.read_h5ad(p) for p in ps], label="sf", keys=[p.stem for p in ps],
                  index_unique=None)
    section = np.unique(np.asarray(a.obsm["section_index"]).reshape(-1), return_inverse=True)[1]
    return a, np.asarray(a.obsm["array_coords"], np.float32), section


def build_delta(arm, source_dir=None):
    """Predicted induction per sham spot, assembled per compartment."""
    import anndata as ad
    import numpy as np

    d = Path(source_dir or ARM_DIR[arm])
    side = json.loads((d / "sham_cells.json").read_text())
    offsets = np.concatenate([[0], np.cumsum(side["counts"])])[:-1]
    rows = np.concatenate([np.arange(o, o + c)
                           for o, c, t in zip(offsets, side["counts"], side["is_target"]) if t])

    def read(pass_name):
        # The `_stroke_` segment is load-bearing: the arm directories also hold decodes from
        # the rim-thickness and observed-shape variants, named `onrim_obs_*`, `onrim_t1_*` and
        # so on. A looser glob picks one of those instead, in an order that is not defined.
        hits = sorted(d.glob(f"stroke_{arm}_cr_{pass_name}_stroke_*_perturbed_counts.h5ad"))
        if len(hits) != 1:
            raise FileNotFoundError(f"expected one {pass_name} decode in {d}, found {hits}")
        return log1p_cpm(np.asarray(ad.read_h5ad(hits[0]).X[rows], np.float32))

    l_core, l_rim, l_off = read("oncore"), read("onrim"), read("off")
    genes = np.loadtxt(d / "hvg_gene_names.txt", dtype=str)
    n_sham = l_off.shape[0]

    def inside(f):
        return np.isin(np.arange(n_sham), np.loadtxt(d / "shape_sets" / f, dtype=np.int64))

    p_in = inside("shape_lesion_local.txt")
    p_core = inside("shape_core_local.txt") & p_in
    p_rim = inside("shape_rim_local.txt") & p_in

    delta = np.zeros_like(l_off)
    delta[p_core] = l_core[p_core] - l_off[p_core]
    delta[p_rim] = l_rim[p_rim] - l_off[p_rim]
    return dict(delta=delta, genes=genes, in_lesion=p_in, core=p_core, rim=p_rim)


# ----------------------------------------------------------------------------- steps

def step_weights(env, out):
    """What the two checkpoints hold, and where the arms differ."""
    import torch

    sd = {}
    for arm, d in ARM_DIR.items():
        s = torch.load(str(d / "model.pt"), map_location="cpu", weights_only=False)
        sd[arm] = s.get("model_state_dict", s) if isinstance(s, dict) else s
        n = sum(v.numel() for v in sd[arm].values() if hasattr(v, "numel"))
        print(f"{arm}: {len(sd[arm])} tensors, {n:,} parameters")
    print("\ntensors whose shape differs between the arms:")
    for k in sorted(set(sd["2d"]) & set(sd["3d"])):
        a, b = sd["2d"][k], sd["3d"][k]
        if hasattr(a, "shape") and tuple(a.shape) != tuple(b.shape):
            print(f"  {k:<40} 2d {tuple(a.shape)}   3d {tuple(b.shape)}")
    only2, only3 = set(sd["2d"]) - set(sd["3d"]), set(sd["3d"]) - set(sd["2d"])
    print(f"\ntensors only in 2d: {sorted(only2) or 'none'}")
    print(f"tensors only in 3d: {sorted(only3) or 'none'}")
    print("\nperturbation vocabulary:",
          json.loads((ARM_DIR["2d"] / "pert_map.json").read_text())["pert_map"])


def step_decode(env, out):
    """Re-run the three decode passes for the chosen arms.

    `--stage eval` loads `<outdir>/model.pt`, so a symlink shadow of the checkpoint directory
    is built under the work directory rather than writing beside the checkpoint.
    """
    for arm in env["arms"]:
        shadow = out / f"{arm}_shadow"
        shadow.mkdir(parents=True, exist_ok=True)
        for f in ARM_DIR[arm].iterdir():           # checkpoint + geometry + sidecar files
            if not (shadow / f.name).exists():
                (shadow / f.name).symlink_to(f)
        for pass_name, target, shape_file in PASSES:
            cells = (shadow / "shape_sets" / shape_file) if shape_file else (shadow / "sham_cells.txt")
            print(f"\n=== {arm} / {pass_name}: {target} on {cells.name} ===")
            r = subprocess.run(decode_cmd(arm, shadow, pass_name, target, cells),
                               cwd=str(MODEL_SRC))
            print("exit code", r.returncode)
        env["source_dir"][arm] = shadow


def step_assemble(env, out):
    """Assemble the predicted induction field, and read the measured side."""
    import h5py
    import numpy as np
    import pandas as pd

    env["PRED"] = {arm: build_delta(arm, env["source_dir"].get(arm)) for arm in ARM_DIR}
    for arm, p in env["PRED"].items():
        print(f"{arm}: {p['delta'].shape[0]} sham spots x {p['delta'].shape[1]} genes; "
              f"imposed lesion {int(p['in_lesion'].sum())} spots "
              f"({int(p['core'].sum())} core, {int(p['rim'].sum())} rim)")

    a_pt, xy_pt, sec_pt = load_stack("PT")
    a_sh, xy_sh, sec_sh = load_stack("Sham")
    env.update(L_pt=log1p_cpm(np.asarray(a_pt.X.todense(), np.float32)),
               L_sh=log1p_cpm(np.asarray(a_sh.X.todense(), np.float32)),   # healthy baseline
               genes_obs=np.asarray(a_pt.var_names),
               xy_pt=xy_pt, xy_sh=xy_sh, sec_pt=sec_pt, sec_sh=sec_sh)

    with h5py.File(REGION_H5, "r") as f:
        bc = np.array([b.decode() for b in f["Barcode"][:]])
        reg = np.array([b.decode() for b in f["Celltype_coarse"][:]])
    region = pd.Series(reg, index=bc).loc[np.asarray(a_pt.obs_names)].to_numpy()
    env["o_core"], env["o_rim"] = region == "ICA", region == "PIA"
    env["o_in"] = env["o_core"] | env["o_rim"]

    print(f"\nstroke animal: {a_pt.n_obs} spots over {sec_pt.max() + 1} sections; "
          f"measured lesion {int(env['o_in'].sum())} spots "
          f"({int(env['o_core'].sum())} ICA, {int(env['o_rim'].sum())} PIA)")
    print(f"sham animal  : {a_sh.n_obs} spots over {sec_sh.max() + 1} sections")


def _style():
    sys.path.insert(0, str(cp.EXPERIMENTS / "03_dss_gut_case_study" / "figures"))
    import paper_style as ps
    ps.use()
    return ps


def step_spatial(env, out):
    """Section-by-section figure: measured beside predicted, with both region maps."""
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    ps = _style()
    core_c, rim_c, other_c = "#b2182b", "#ef8a62", "#dcdcdc"
    npt, nsh = norm_xy(env["xy_pt"]), norm_xy(env["xy_sh"])
    gene = env["gene"]

    for arm in env["arms_plot"]:
        p = env["PRED"][arm]
        if gene not in set(env["genes_obs"].tolist()) or gene not in set(p["genes"].tolist()):
            raise KeyError(f"{gene} is not on both gene axes")
        ov = env["L_pt"][:, int(np.where(env["genes_obs"] == gene)[0][0])]
        pv = p["delta"][:, int(np.where(p["genes"] == gene)[0][0])]
        keep = env["sections"] if env["sections"] is not None \
            else list(range(int(env["sec_pt"].max()) + 1))

        reg_cmap = ListedColormap([other_c, rim_c, core_c])
        o_reg = np.where(env["o_core"], 2, np.where(env["o_rim"], 1, 0))
        p_reg = np.where(p["core"], 2, np.where(p["rim"], 1, 0))
        vo, vp = np.quantile(ov, [.03, .97]), np.quantile(pv, [.03, .97])

        fig, ax = plt.subplots(4, len(keep), figsize=(2.5 * len(keep) + 2.0, 10.0), squeeze=False)
        for ci, s in enumerate(keep):
            mp, ms = env["sec_pt"] == s, env["sec_sh"] == s
            h0 = ax[0, ci].scatter(npt[mp, 0], npt[mp, 1], c=ov[mp], s=2.6, lw=0,
                                   cmap=ps.FIELD_CMAP, vmin=vo[0], vmax=vo[1], rasterized=True)
            h1 = ax[1, ci].scatter(nsh[ms, 0], nsh[ms, 1], c=pv[ms], s=2.6, lw=0,
                                   cmap=ps.FIELD_CMAP, vmin=vp[0], vmax=vp[1], rasterized=True)
            ax[2, ci].scatter(npt[mp, 0], npt[mp, 1], c=o_reg[mp], s=2.6, lw=0,
                              cmap=reg_cmap, vmin=0, vmax=2, rasterized=True)
            ax[3, ci].scatter(nsh[ms, 0], nsh[ms, 1], c=p_reg[ms], s=2.6, lw=0,
                              cmap=reg_cmap, vmin=0, vmax=2, rasterized=True)
            for r in range(4):
                ax[r, ci].set_xticks([]); ax[r, ci].set_yticks([])
                ax[r, ci].set_aspect("equal"); ax[r, ci].grid(False)
                for sp in ax[r, ci].spines.values():
                    sp.set_visible(False)
            ax[3, ci].set_xlabel(f"Section {s + 1}", fontsize=13)
        for r, lab in enumerate(["Measured\n(stroke animal)", "Predicted\n(sham animal)",
                                 "Measured regions", "Imposed regions"]):
            ax[r, 0].set_ylabel(lab, fontsize=12, labelpad=6)
        fig.subplots_adjust(wspace=0.02, hspace=0.02, left=0.12, right=0.87, top=0.99, bottom=0.06)
        for h, arow, lb in ((h0, ax[0, -1], f"measured {gene}"),
                            (h1, ax[1, -1], "predicted induction")):
            po = arow.get_position(); pad = 0.12 * po.height
            cax = fig.add_axes([0.885, po.y0 + pad, 0.016, po.height - 2 * pad])
            cb = fig.colorbar(h, cax=cax); cb.set_label(lb, fontsize=11, labelpad=2)
            cb.ax.tick_params(labelsize=9); cb.set_ticks([cb.vmin, cb.vmax])
        p_out = out / f"spatial_{arm}_{gene}.pdf"
        fig.savefig(p_out, dpi=300); plt.close(fig)
        print("written", p_out)


def _depth_of(xy, inside):
    import numpy as np
    from scipy.spatial import cKDTree
    d = np.zeros(len(xy))
    if inside.sum() and (~inside).sum():
        d[inside] = cKDTree(xy[~inside]).query(xy[inside])[0]
    return d


def _profile(depth, v, inside, nb=NB):
    import numpy as np
    d, x = depth[inside], v[inside]
    q = np.quantile(d, np.linspace(0, 1, nb + 1)); q[-1] += 1e-6
    b = [x[(d >= q[i]) & (d < q[i + 1])] for i in range(nb)]
    return (np.array([bb.mean() for bb in b]),
            np.array([bb.std() / max(np.sqrt(len(bb)), 1) for bb in b]))


def _compartment_stats(env, arm, gene):
    """Mean induction over healthy tissue, per compartment, measured and predicted."""
    import numpy as np
    p = env["PRED"][arm]
    ov = env["L_pt"][:, int(np.where(env["genes_obs"] == gene)[0][0])]
    pv = p["delta"][:, int(np.where(p["genes"] == gene)[0][0])]
    base = env["L_sh"][:, int(np.where(env["genes_obs"] == gene)[0][0])]  # healthy, same places
    return {"gene": gene, "arm": arm,
            "observed_core": float(ov[env["o_core"]].mean() - base[p["core"]].mean()),
            "observed_rim": float(ov[env["o_rim"]].mean() - base[p["rim"]].mean()),
            "predicted_core": float(pv[p["core"]].mean()),
            "predicted_rim": float(pv[p["rim"]].mean())}


def step_profiles(env, out):
    """Rim-to-core depth profiles and the compartment bars, plus the sign-match table."""
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    ps = _style()
    genes = [g for g in GENES
             if g in set(env["genes_obs"].tolist()) and g in set(env["PRED"]["2d"]["genes"].tolist())]
    stats = pd.DataFrame([_compartment_stats(env, a, g) for g in genes for a in ("2d", "3d")])
    stats["observed_core_minus_rim"] = stats.observed_core - stats.observed_rim
    stats["predicted_core_minus_rim"] = stats.predicted_core - stats.predicted_rim
    stats["sign_matches"] = (np.sign(stats.observed_core_minus_rim)
                             == np.sign(stats.predicted_core_minus_rim))
    print(stats.round(3).to_string(index=False))
    stats.to_csv(out / "compartment_stats.csv", index=False)
    for arm in ("2d", "3d"):
        s = stats[stats.arm == arm]
        print(f"{arm}: core-vs-rim direction matched on {int(s.sign_matches.sum())}/{len(s)} genes")
    env["stats"], env["genes"] = stats, genes

    dep_pred = _depth_of(env["xy_sh"], env["PRED"]["2d"]["in_lesion"])  # geometry is arm-independent
    dep_obs = _depth_of(env["xy_pt"], env["o_in"])
    core_genes = [g for g in ["Gm42418", "Scgb1a1"] if g in genes]
    if not core_genes:
        print("none of the core marker genes are on both axes; skipping the profile figure")
        return

    fig, axes = plt.subplots(2, len(core_genes) * 2, figsize=(3.2 * len(core_genes) * 2, 6.4),
                             squeeze=False)
    xs = np.arange(NB)
    col = 0
    for arm in ("2d", "3d"):
        p = env["PRED"][arm]
        for g in core_genes:
            gi_o = int(np.where(env["genes_obs"] == g)[0][0])
            gi_p = int(np.where(p["genes"] == g)[0][0])
            od, oe = _profile(dep_obs, env["L_pt"][:, gi_o], env["o_in"])
            od = od - float(env["L_sh"][:, gi_o][p["in_lesion"]].mean())   # induction over healthy
            pdp, pe = _profile(dep_pred, p["delta"][:, gi_p], p["in_lesion"])

            a0 = axes[0, col]
            a0.errorbar(xs, od, yerr=oe, color=ps.ACCENT, lw=2.0, marker="o", ms=5, label="measured")
            a0.errorbar(xs, pdp, yerr=pe, color="#2166ac", lw=2.0, marker="s", ms=5, ls="--",
                        label="CONCERT")
            a0.set_xticks([0, NB - 1]); a0.set_xticklabels(["Rim", "Core"], fontsize=12)
            a0.set_xlabel(f"{g}  ({arm})", fontsize=12)
            if col == 0:
                a0.set_ylabel("induction over healthy", fontsize=11); a0.legend(fontsize=9)

            st = _compartment_stats(env, arm, g)
            a1 = axes[1, col]
            w, x2 = 0.36, np.arange(2)
            a1.bar(x2 - w / 2, [st["observed_core"], st["observed_rim"]], w, color=ps.ACCENT,
                   label="measured")
            a1.bar(x2 + w / 2, [st["predicted_core"], st["predicted_rim"]], w, color="#2166ac",
                   label="CONCERT")
            a1.set_xticks(x2); a1.set_xticklabels(["Core", "Rim"], fontsize=12)
            a1.set_xlabel(f"{g}  ({arm})", fontsize=12)
            if col == 0:
                a1.set_ylabel("mean induction", fontsize=11); a1.legend(fontsize=9)
            col += 1
    fig.subplots_adjust(wspace=0.45, hspace=0.32)
    fig.savefig(out / "depth_profiles.pdf", dpi=300); plt.close(fig)
    print("written", out / "depth_profiles.pdf")


def step_check(env, out):
    """Recomputed compartment statistics against the ones shipped with the figures."""
    import pandas as pd

    core_genes = [g for g in ["Gm42418", "Scgb1a1"] if g in env["genes"]]
    rows = []
    for arm in ("2d", "3d"):
        shipped = json.loads(
            (ARM_DIR[arm] / f"figure_stats_patch_{arm}_cr.json").read_text())["genes"]
        for g in core_genes:
            if g not in shipped:
                continue
            here = _compartment_stats(env, arm, g)
            for k in ("observed_core", "observed_rim", "predicted_core", "predicted_rim"):
                rows.append({"arm": arm, "gene": g, "quantity": k, "shipped": shipped[g][k],
                             "recomputed": here[k], "abs_diff": abs(shipped[g][k] - here[k])})
    chk = pd.DataFrame(rows)
    if chk.empty:
        print("no shipped statistics to compare against")
        return
    print(chk.round(6).to_string(index=False))
    print(f"\nlargest absolute difference: {chk.abs_diff.max():.2e}")
    chk.to_csv(out / "shipped_vs_recomputed.csv", index=False)


STEPS = {"weights": step_weights, "decode": step_decode, "assemble": step_assemble,
         "spatial": step_spatial, "profiles": step_profiles, "check": step_check}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", nargs="+", choices=list(STEPS),
                    default=["weights", "assemble", "spatial", "profiles", "check"])
    ap.add_argument("--run-decode", action="store_true",
                    help="re-run the three decode passes instead of reading the shipped ones "
                         "(about 20 min per arm, GPU)")
    ap.add_argument("--arms", nargs="+", choices=["2d", "3d"], default=["2d"],
                    help="arms to re-decode with --run-decode")
    ap.add_argument("--gene", default="Gm42418", help="gene for the spatial figure")
    ap.add_argument("--sections", nargs="+", type=int, default=None,
                    help="sections for the spatial figure (default: all four)")
    ap.add_argument("--outdir", default=None, help="default: $CONCERT_WORKDIR/04_stroke")
    a = ap.parse_args()

    steps = list(a.steps)
    if a.run_decode and "decode" not in steps:
        steps.insert(0, "decode")
    if not a.run_decode and "decode" in steps:
        steps.remove("decode")
    for later, need in (("spatial", "assemble"), ("profiles", "assemble"), ("check", "profiles")):
        if later in steps and need not in steps:
            steps.insert(steps.index(later), need)

    out = Path(a.outdir) if a.outdir else cp.workdir("04_stroke")
    out.mkdir(parents=True, exist_ok=True)

    cp.section("0. Inputs")
    checks = [("model source", MODEL_SRC / "run_concert_spac.py"),
              ("figure code", EXP / "figures" / "fig_cr_patch.py"),
              ("slides (sham 1)", SLIDES / "stroke_paste2_Sham1.h5ad"),
              ("slides (stroke 1)", SLIDES / "stroke_paste2_PT1.h5ad"),
              ("region labels", REGION_H5)]
    for arm, d in ARM_DIR.items():
        checks += [(f"{arm} checkpoint", d / "model.pt"),
                   (f"{arm} gene axis", d / "hvg_gene_names.txt"),
                   (f"{arm} lesion shape", d / "shape_sets" / "shape_core_local.txt"),
                   (f"{arm} sham index", d / "sham_cells.json"),
                   (f"{arm} shipped stats", d / f"figure_stats_patch_{arm}_cr.json")]
    cp.check(checks)
    print(f"\noutput dir: {out}")
    if "decode" in steps:
        cp.describe_device()

    env = {"arms": a.arms, "arms_plot": list(ARM_DIR), "gene": a.gene, "sections": a.sections,
           "source_dir": {}}
    for name in STEPS:
        if name in steps:
            cp.section(f"Step: {name}")
            STEPS[name](env, out)

    print(f"\nDone. Everything written under {out}")


if __name__ == "__main__":
    main()
