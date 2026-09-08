#!/usr/bin/env python
"""DSS colitis time course: a longitudinal marker trajectory for animals measured once.

The script form of `notebooks/03_dss_colitis_per_mouse_marker_trend.ipynb`.

Every animal in this MERFISH cohort is sacrificed at one timepoint, so no animal has a
measured trajectory. The temporal model carries time as a conditioning variable, so the cells
of one animal can be re-decoded at a virtual day they were never measured at. Doing that for
each animal at days 0, 3, 9 and 21 turns 15 single observations into 15 curves, and the
cell-weighted mean of those curves is the population course of the injury -- which rises to a
day-9 peak and resolves by day 21, a shape interpolation between the observed timepoints
cannot produce.

The readout is a 20-gene inflammation marker score. The observed anchor uses 17 of them: the
three neutrophil genes (S100a8, S100a9, Ly6a) are dropped there because they are dominated by
infiltrating cells rather than by the resident tissue the model decodes.

Steps
  cohort       the animals, sections and cells behind the figure
  weights      what the temporal checkpoint holds (time kernel, field, vocabulary)
  decode       re-decode animals at all four virtual days                (GPU, 30-45 min, ~200 GB)
  trajectory   per-animal curves and the population course
  observed     the measured anchor per day
  figures      the trajectory figure, the predicted-vs-measured panels, the velocity table

`decode` is off unless asked for: the shipped decode is read instead, and the numbers and
figures still come out.

Examples
  python scripts/03_dss_colitis_marker_trend.py
  python scripts/03_dss_colitis_marker_trend.py --run-decode --mice 082421_D0_m6 092421_D3_m1
  python scripts/03_dss_colitis_marker_trend.py --run-decode --mice all --recompute-observed
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import concert_paths as cp

MODEL_SRC = cp.MODELS / "concert_temporal"
EXP = cp.EXPERIMENTS / "03_dss_gut_case_study"
ARCH_DIR = Path(str(cp.DSS_DIR) + "_arch")            # tissue-architecture descriptors
MANIFEST = cp.DSS_DIR / "manifest.csv"

RUN = cp.OUTPUT_ROOT / "dss_timekernel_markersign_timefield_fixedt"
CKPT = RUN / "model.pt"
SHIPPED = RUN / "case_study"
NUMBERS = cp.PAPER_NUMBERS / "case_study_colitis"

MARKERS_20 = ["Isg15", "Ifit1", "Tnf", "Cxcl1", "Nos2", "Il11", "Il1b", "Mmp3", "Mmp10",
              "Mmp13", "Plau", "Igfbp5", "Col18a1", "Mmp2", "S100a8", "S100a9", "Osm",
              "Cxcl10", "Cxcl5", "Ly6a"]
MARKERS_17 = [g for g in MARKERS_20 if g not in {"S100a8", "S100a9", "Ly6a"}]
DAYS = [0, 3, 9, 21]
DEFAULT_MICE = ["082421_D0_m6", "092421_D3_m1", "100221_D9_m5", "082421_D21_m1"]  # one per day


def decode_command(files, shadow):
    """The composition_eval call that re-decodes a set of sections at every virtual day.

    Every architecture flag has to match the ones training used or the weights will not load;
    they are the flags of experiments/03_dss_gut_case_study/train/.
    """
    return [
        sys.executable, "-u", "run_concert_spac.py", "--stage", "composition_eval",
        "--config", "config.yaml", "--sample", "dss_script",
        "--multi_sample_files", ",".join(files),
        "--outdir", str(shadow), "--model_file", "model.pt", "--project_index", "dss",
        "--case_study_deliverables", "True",
        "--use_timekernel", "True", "--use_shared_basal", "True", "--use_time_intensity", "False",
        "--arch_desc_dir", str(ARCH_DIR),
        "--use_inflammation_field", "True", "--field_mse_lambda", "1.0",
        "--marker_sign_positive", "True", "--time_in_field_mlp", "True",
        "--amp_anchor_offset", "0.0", "--amp_anchor_lambda", "0.0", "--amp_anchor_t", "0.0",
        "--use_time_conditioning", "True", "--time_col", "day_norm",
        "--time_fourier_bands", "6", "--time_emb_dim", "16",
        "--hvg_max_rank", "2000", "--seed", "42",
        "--inducing_point_steps", "6", "--grid_inducing_points", "True",
        "--loc_range", "20.0", "--kernel_scale", "10.0", "--multi_kernel_mode", "True",
        "--batch_size", "1024", "--train_size", "0.95",
        "--use_action_correction", "False", "--use_pad", "False", "--min_cells_per_pert", "25",
        "--use_expr_encoder", "True", "--expr_hidden_dims", "512", "256",
        "--use_basal_encoder_only", "True",
        "--dynamicVAE", "False", "--free_bits", "0.5", "--gp_free_bits", "1.0",
        "--use_nngp", "False", "--use_paracrine", "False", "--use_pert_gate", "False",
        "--use_f_pert", "True", "--f_pert_clamp", "5.0",
        "--spatial_basal_only", "True", "--lfc_lambda", "0.0",
        "--basal_field", "rff", "--basal_bayes", "True",
        "--moran_lambda", "1.0", "--disentangle_radius", "4.0",
        "--fpert_field", "gp_pp", "--fpert_coord_scale", "20.0",
        "--fpert_rff_features", "128", "--fpert_rff_bandwidth", "0.15",
        "--ls_prior_lambda", "1.0", "--ls_prior_min", "3", "--ls_prior_max", "18",
        "--fpert_svgp_inducing", "16", "--fpert_svgp_ls_init", "5.0",
        "--verbosity", "1",
    ]


# ----------------------------------------------------------------------------- steps

def step_cohort(env, out):
    """The animals, sections and cells behind the figure."""
    man, labels = env["man"], env["labels"]
    cohort = (man.groupby(["Mouse_ID", "Sample_type", "day"])
                .agg(n_sections=("file", "size"), n_cells=("n_cells", "sum"))
                .reset_index().sort_values(["day", "Mouse_ID"]))
    cohort = cohort.merge(labels[["figure_label", "mouse_id"]], left_on="Mouse_ID",
                          right_on="mouse_id", how="left").drop(columns="mouse_id")
    print(f"{len(man)} sections, {cohort.Mouse_ID.nunique()} animals, {cohort.n_cells.sum():,} cells")
    print(cohort.groupby("day").agg(n_mice=("Mouse_ID", "size"), n_sections=("n_sections", "sum"),
                                    n_cells=("n_cells", "sum")).to_string())
    cols = ["figure_label", "Mouse_ID", "Sample_type", "day", "n_sections", "n_cells"]
    print("\n" + cohort[cols].to_string(index=False))
    cohort[cols].to_csv(out / "cohort.csv", index=False)


def step_weights(env, out):
    """What the temporal checkpoint holds, by tensor group."""
    import torch

    sd = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    n = sum(v.numel() for v in sd.values() if hasattr(v, "numel"))
    print(f"{len(sd)} tensors, {n:,} parameters\n")
    for pat in ("time", "field", "amp", "niche", "fpert", "arch", "tier1"):
        hits = [k for k in sd if pat in k.lower()]
        if hits:
            print(f"{pat:<8} {len(hits):>3} tensors, e.g. {hits[0]}  {tuple(sd[hits[0]].shape)}")
    print("\nperturbation vocabulary:", json.loads((RUN / "pert_map.json").read_text())["pert_map"])


def step_decode(env, out):
    """Re-decode the chosen animals at every virtual day.

    composition_eval loads `<outdir>/model.pt`, so the script builds a symlink shadow of the
    checkpoint directory under the work directory rather than writing next to the checkpoint
    (which lives on a filesystem close to its group quota, where a failed write can truncate
    the target file to zero).
    """
    man = env["man"]
    shadow = out / "ckpt_shadow"
    shadow.mkdir(parents=True, exist_ok=True)
    for f in RUN.iterdir():
        if f.is_file() and not (shadow / f.name).exists():
            (shadow / f.name).symlink_to(f)

    mice = env["mice"] or sorted(man.Mouse_ID.unique())
    files = [str(cp.DSS_DIR / f) for f in man[man.Mouse_ID.isin(mice)].file]
    print(f"{len(mice)} animals, {len(files)} sections")
    cmd = decode_command(files, shadow)
    print("command:\n ", " ".join(cmd[:8]), "...")
    r = subprocess.run(cmd, cwd=str(MODEL_SRC))
    print("exit code", r.returncode)
    fresh = shadow / "case_study" / "per_mouse_trajectories.csv"
    if r.returncode == 0 and fresh.exists():
        env["fresh_traj"] = fresh
        print("fresh trajectories at", fresh)
    else:
        print("decode did not produce per_mouse_trajectories.csv; the shipped result is used below")


def step_trajectory(env, out):
    """Per-animal curves and the cell-weighted population course."""
    import pandas as pd

    src = env.get("fresh_traj") or (SHIPPED / "per_mouse_trajectories.csv")
    traj = pd.read_csv(src)
    traj = traj.merge(env["labels"][["figure_label", "mouse_id"]], left_on="mouse",
                      right_on="mouse_id", how="left").drop(columns="mouse_id")
    traj["mouse_number"] = traj.figure_label.str.extract(r"(\d+)").astype(int)
    traj = traj.sort_values("mouse_number")
    print("read", src, "\n")

    pop = {d: traj[f"pred_t{d}"].mul(traj.n_cells).sum() / traj.n_cells.sum() for d in DAYS}
    print("population predicted trajectory (cell-weighted):")
    for d in DAYS:
        print(f"  day {d:>2}: {pop[d]:.4f}")
    print(f"\npeak at day {max(pop, key=pop.get)}")
    cols = ["figure_label", "mouse", "observed_day", "n_cells"] + [f"pred_t{d}" for d in DAYS]
    print("\n" + traj[cols].round(4).to_string(index=False))
    traj[cols].to_csv(out / "per_mouse_trajectories.csv", index=False)
    env["traj"], env["pop"] = traj, pop


def step_observed(env, out):
    """The measured anchor per day.

    NOTE `per_mouse_trajectories.csv` also carries an `observed_marker_score` column, but that
    one is standardised (and negative). The count-space anchor is `per_mouse_observed.csv`, or
    the recomputation below.
    """
    import numpy as np
    import pandas as pd

    if env["recompute_observed"]:
        import anndata as ad
        import scipy.sparse as sp
        rows = []
        for mid, grp in env["man"].groupby("Mouse_ID"):
            ssum, ncell = 0.0, 0
            for f in grp.file:
                a = ad.read_h5ad(cp.DSS_DIR / f)
                idx = [list(a.var_names).index(g) for g in MARKERS_17]
                x = a.X[:, idx]
                x = x.toarray() if sp.issparse(x) else np.asarray(x)
                s = np.log1p(x).mean(1)
                ssum += float(s.sum()); ncell += s.shape[0]
            rows.append({"mouse": mid, "day": float(grp.day.iloc[0]), "n_cells": ncell,
                         "observed_marker_log1p": ssum / ncell})
        obs = pd.DataFrame(rows)
    else:
        obs = pd.read_csv(SHIPPED / "per_mouse_observed.csv")
        print("read", SHIPPED / "per_mouse_observed.csv")

    by_day = {}
    for d in DAYS:
        g = obs[np.isclose(obs.day, d)]
        by_day[d] = float(np.average(g.observed_marker_log1p, weights=g.n_cells))
    print("\nobserved marker score, cell-weighted pooled mean per day:")
    for d in DAYS:
        print(f"  day {d:>2}: {by_day[d]:.4f}   ({int(np.isclose(obs.day, d).sum())} animals)")
    print("\n" + obs.sort_values(["day", "mouse"]).round(4).to_string(index=False))
    obs.to_csv(out / "per_mouse_observed.csv", index=False)
    env["obs"], env["obs_by_day"] = obs, by_day


def step_figures(env, out):
    """The trajectory figure, the predicted-vs-measured panels and the velocity table."""
    import numpy as np
    import pandas as pd

    sys.path.insert(0, str(EXP / "figures"))
    import paper_style as ps          # shared day colours and animal labels; selects Agg
    ps.use()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    traj, pop, obs, obs_by_day = env["traj"], env["pop"], env["obs"], env["obs_by_day"]
    obs_by_mouse = dict(zip(obs.mouse, obs.observed_marker_log1p))

    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    for _, r in traj.iterrows():
        c = ps.day_color(r.observed_day)
        ax.plot(DAYS, [r[f"pred_t{d}"] for d in DAYS], "-", color=c, lw=1.6, alpha=0.85, zorder=2)
        ax.plot([r.observed_day], [r[f"pred_t{int(r.observed_day)}"]], "o", color=c, ms=6,
                mfc="white", mew=1.6, zorder=3)
        if r.mouse in obs_by_mouse:
            ax.plot([r.observed_day], [obs_by_mouse[r.mouse]], "D", color=c, ms=5, zorder=4)
    ax.plot(DAYS, [pop[d] for d in DAYS], "-", color="0.15", lw=3.0, zorder=5)
    ax.set_xticks(DAYS)
    ax.set_xlabel("virtual day the cells were re-decoded at", fontsize=12)
    ax.set_ylabel("predicted DSS-marker score", fontsize=12)
    ax.legend(handles=[Line2D([], [], color="0.15", lw=3, label="population mean"),
                       Line2D([], [], color="0.4", marker="o", mfc="white", ls="none",
                              label="prediction at the sampled day"),
                       Line2D([], [], color="0.4", marker="D", ls="none", label="measured value")]
                      + [Line2D([], [], color=ps.DAY_COLORS[float(d)], lw=2, label=f"sampled day {d}")
                         for d in DAYS], fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "per_mouse_trajectories.pdf", dpi=300); plt.close(fig)

    ratio = pd.DataFrame({"day": DAYS, "predicted": [pop[d] for d in DAYS],
                          "observed": [obs_by_day[d] for d in DAYS]})
    ratio["ratio_pred_over_obs"] = ratio.predicted / ratio.observed
    ratio["fold_over_day0_pred"] = ratio.predicted / ratio.predicted.iloc[0]
    ratio["fold_over_day0_obs"] = ratio.observed / ratio.observed.iloc[0]
    print(ratio.round(4).to_string(index=False))
    ratio.to_csv(out / "predicted_vs_observed_by_day.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    axes[0].plot(DAYS, ratio.predicted, "-o", color="#2166ac", lw=2, label="predicted")
    axes[0].plot(DAYS, ratio.observed, "-s", color=ps.ACCENT, lw=2, label="measured")
    axes[0].set_ylabel("DSS-marker score", fontsize=12)
    axes[0].legend(fontsize=10)
    axes[1].plot(DAYS, ratio.fold_over_day0_pred, "-o", color="#2166ac", lw=2, label="predicted")
    axes[1].plot(DAYS, ratio.fold_over_day0_obs, "-s", color=ps.ACCENT, lw=2, label="measured")
    axes[1].axhline(1.0, lw=0.8, color="0.6")
    axes[1].set_ylabel("fold change over day 0", fontsize=12)
    axes[1].legend(fontsize=10)
    for ax in axes:
        ax.set_xticks(DAYS)
        ax.set_xlabel("day", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "predicted_vs_observed.pdf", dpi=300); plt.close(fig)

    v = pd.DataFrame({"interval": ["0 to 3", "3 to 9", "9 to 21"],
                      "velocity": [(pop[3] - pop[0]) / 3, (pop[9] - pop[3]) / 6,
                                   (pop[21] - pop[9]) / 12]})
    print("\n" + v.round(5).to_string(index=False))
    print("sign flip between the day-3-to-9 and day-9-to-21 intervals:",
          bool(np.sign(v.velocity.iloc[1]) != np.sign(v.velocity.iloc[2])))
    v.to_csv(out / "population_velocity.csv", index=False)

    spread = traj["pred_t9"]
    print(f"\ncross-animal spread of the predicted score at day 9: s.d. {spread.std(ddof=0):.4f}, "
          f"range {spread.min():.4f} to {spread.max():.4f}")
    print("the same quantity under interpolation between population means: 0.0000 by construction")

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    d = traj.sort_values("pred_t9")
    ax.barh(d.figure_label, d.pred_t9, color=[ps.day_color(x) for x in d.observed_day])
    ax.set_xlabel("predicted DSS-marker score at day 9", fontsize=12)
    ax.tick_params(labelsize=9)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(out / "day9_per_mouse.pdf", dpi=300); plt.close(fig)
    print(f"\nfigures written under {out}")


STEPS = {"cohort": step_cohort, "weights": step_weights, "decode": step_decode,
         "trajectory": step_trajectory, "observed": step_observed, "figures": step_figures}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", nargs="+", choices=list(STEPS),
                    default=["cohort", "weights", "trajectory", "observed", "figures"])
    ap.add_argument("--run-decode", action="store_true",
                    help="re-decode on a GPU instead of reading the shipped decode "
                         "(adds the `decode` step)")
    ap.add_argument("--mice", nargs="+", default=DEFAULT_MICE,
                    help="animals to re-decode, or `all` for the full cohort "
                         "(~30-45 min and ~200 GB)")
    ap.add_argument("--recompute-observed", action="store_true",
                    help="read the raw sections for the measured anchor instead of the "
                         "shipped table (a few minutes of I/O, no GPU)")
    ap.add_argument("--outdir", default=None, help="default: $CONCERT_WORKDIR/03_dss_colitis")
    a = ap.parse_args()

    steps = list(a.steps)
    if a.run_decode and "decode" not in steps:
        steps.insert(steps.index("trajectory") if "trajectory" in steps else 0, "decode")
    if not a.run_decode and "decode" in steps:
        steps.remove("decode")
    # `figures` needs what `trajectory` and `observed` put in the environment
    if "figures" in steps:
        for need in ("trajectory", "observed"):
            if need not in steps:
                steps.insert(steps.index("figures"), need)

    out = Path(a.outdir) if a.outdir else cp.workdir("03_dss_colitis")
    out.mkdir(parents=True, exist_ok=True)

    cp.section("0. Inputs")
    cp.check([
        ("temporal model source", MODEL_SRC / "concert_spac.py"),
        ("case-study evaluation", EXP / "evaluation" / "composition_eval_dss.py"),
        ("figure style", EXP / "figures" / "paper_style.py"),
        ("section manifest", MANIFEST),
        ("architecture codes", ARCH_DIR),
        ("checkpoint", CKPT),
        ("vocabulary", RUN / "pert_map.json"),
        ("architecture scaler", RUN / "arch_scaler.npz"),
        ("shipped trajectories", SHIPPED / "per_mouse_trajectories.csv"),
        ("observed anchors", SHIPPED / "per_mouse_observed.csv"),
        ("mouse label map", NUMBERS / "mouse_id_map.csv"),
        ("readout calibration", NUMBERS / "calibration.json"),
    ])
    print(f"\ncheckpoint size: {CKPT.stat().st_size / 1e9:.2f} GB")
    print(f"output dir: {out}")
    print(f"\nre-decode panel (20): {', '.join(MARKERS_20)}")
    print(f"observed-anchor panel (17): {', '.join(MARKERS_17)}")

    import pandas as pd
    env = {"man": pd.read_csv(MANIFEST),
           "labels": pd.read_csv(NUMBERS / "mouse_id_map.csv"),
           "mice": None if a.mice == ["all"] else a.mice,
           "recompute_observed": a.recompute_observed}

    for name in STEPS:
        if name in steps:
            cp.section(f"Step: {name}")
            STEPS[name](env, out)

    print(f"\nDone. Everything written under {out}")


if __name__ == "__main__":
    main()
