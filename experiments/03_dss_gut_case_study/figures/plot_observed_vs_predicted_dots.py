#!/usr/bin/env python
"""Q4 (GT-anchored): marker dot plots with Observed and CONCERT in the SAME dot.

Design principle: never display a prediction alone where an observed value exists.
Each dot is SPLIT down the middle at one x-position:

        left half-disc  = Observed
        right half-disc = CONCERT

    colour of each half = mean log1p expression, on the observed scale
    area  of each half  = fraction of cells expressing

Fraction expressing uses NO CUTOFF ON THE PREDICTION AT ALL. Observed "% expressing" is a
detection event (an integer count landed above zero) while the prediction is a rate mu, and
no threshold on a rate reproduces a detection frequency -- every cutoff rule tried left the
predicted halves systematically too large. Observed keeps "> 0"; CONCERT uses the detection
probability under the model's own count likelihood, P(count > 0) = 1 - exp(-mu). See
expressing_pred() for the error comparison against the cutoff rules.

Readout calibration is the ONE shared rule (paper_style.calibrate, per marker):
calibrated_g(t) = observed_g(Day 0) + a_g * (pred_g(t) - pred_g(Day 0)). The baseline is
measured, not fitted, and exactly one amplitude is fitted per marker -- so the ACROSS-DAY
SHAPE, which is what this figure claims, stays unconstrained by the fit. Nothing is
re-fitted here; see compute_calibration.py for why this form and not an affine or a
through-origin scale.

Figures
    fig_q4_gt_vs_pred_dotplot.png              cohort view (see stats(): CONCERT is
        every cell decoded to that day, Observed is that day's animals -- different cells)
    fig_q4_gt_vs_pred_dotplot_matched.png      like-for-like (both halves, same cells)
    fig_q4_gt_vs_pred_dotplot_permouse.png     the four featured mice side by side; each
        is measured on a single day, so only that day's column carries an Observed half
        and the rest show an empty dashed half-disc.

Input:  percell_dump_sizefactor.npz (fixedt all-mice heldout_eval)
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "benchmark_evaluation" / "release_paths.py").is_file()) / "benchmark_evaluation"))
import release_paths as _rp  # noqa: E402  data/, checkpoints/ and runs/ roots
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, Normalize
import paper_style as ps
ps.use()

DUMP = (f"{_rp.WORKDIR}/outputs/"
        "dss_timekernel_markersign_timefield_fixedt/heldout_eval/percell_dump_sizefactor.npz")
HERE = f"{_rp.WORKDIR}/paper/Outputs/case_study"
DAYS = [0.0, 3.0, 9.0, 21.0]
# same featured mice as the slide figures (best structural completeness, one per day)
MICE = ["082421_D0_m6", "092421_D3_m1", "100221_D9_m5", "082421_D21_m1"]
BASE_CMAP, PIVOT = "coolwarm", 0.10
MAX_MARKERS, MIN_FOLD = 8, 1.5
MK_FS, DAY_FS, CBAR_FS, LEG_FS, TITLE_FS = 18, 15, 16, 16, 19


def pivot_cmap(name=BASE_CMAP, pivot=PIVOT, n=512):
    """coolwarm with its white point moved to `pivot` of the colorbar LENGTH.

    The norm stays linear, so position on the bar is still proportional to value; this
    changes only WHICH colour sits where. A stock diverging map spends half the bar going
    blue->white, and with an all-positive, low-skewed range that half is wasted on values
    almost nothing has, while every real marker/day sits in one flat blue. Moving white to
    the lower quarter gives the populated low end its own gradient without distorting the
    value axis (which is what TwoSlopeNorm did, and why the ticks bunched)."""
    base = plt.get_cmap(name)
    k = max(int(round(n * pivot)), 2)
    return LinearSegmentedColormap.from_list(
        f"{name}_p{pivot:g}",
        np.vstack([base(np.linspace(0.0, 0.5, k)), base(np.linspace(0.5, 1.0, n - k))]))


CMAP = pivot_cmap()

d = np.load(DUMP, allow_pickle=True)
markers_all = [str(x) for x in d["marker_names"]]
obs_all = np.log1p(d["obs_marker_expr"])
src_day, mouse_all = d["src_day"], d["mouse"]
pred_all = {t: d[f"marker_pred_t{int(t)}"] for t in DAYS}

# keep only markers with a real temporal trend (observed d9/d0 fold)
obs_day = np.array([[obs_all[np.isclose(src_day, t, atol=1e-3), j].mean()
                     for j in range(len(markers_all))] for t in DAYS])
pred_day = np.array([[pred_all[t][:, j].mean()
                      for j in range(len(markers_all))] for t in DAYS])
fold = obs_day[2] / np.clip(obs_day[0], 1e-3, None)
keep = [j for j in np.argsort(-fold) if fold[j] >= MIN_FOLD][:MAX_MARKERS]
markers = [markers_all[j] for j in keep]
G = len(markers)

# the ONE shared readout calibration, applied per marker
def cal(values, j):
    """Put predicted marker j on the observed scale. Never re-fit -- read the shared rule."""
    return ps.calibrate(values, marker=markers_all[j])


# alternative colour basis: HEALTHY-TISSUE z, per marker
# Colouring by absolute log1p expression makes the colour report WHICH GENE is highly
# expressed rather than how it moves: Il1b reaches 0.115 while Mmp13 tops out at 0.026, so
# the low-expressed rows stay pale even when they rise several-fold. Referencing each
# marker to its own healthy tissue puts every row on one "SDs above healthy" scale and
# gives the diverging colormap a meaningful centre at 0. Each modality uses ITS OWN healthy
# distribution -- observed Day-0 cells, and the Day-0 counterfactual decode -- exactly as
# in the slide figures, so the two halves of a dot stay comparable.
_healthy = np.isclose(src_day, 0.0, atol=1e-3)
OBS_H = (obs_all[_healthy].mean(0), obs_all[_healthy].std(0))
PRED_H = (pred_all[0.0].mean(0), pred_all[0.0].std(0))

print("[q4gt] shared per-marker calibration (calibration.json):")
for j in keep:
    y, yh = obs_day[:, j], cal(pred_day[:, j], j)
    e = ps.calibration()["per_marker"][markers_all[j]]
    print(f"   {markers_all[j]:9s} fold={fold[j]:5.1f}x  a={e['a']:.4f}"
          f"  ratio={np.round(yh / y, 2)}")


def expressing_pred(t, j, cells):
    """Predicted fraction of cells expressing marker j at virtual day t.

    NOT a thresholded mean. Observed "% expressing" is a DETECTION EVENT (a count landed
    above zero) while the prediction is a RATE mu; no threshold on a rate can reproduce a
    detection frequency, which is why every cutoff rule left the predicted halves
    systematically too large. The matched quantity is the detection PROBABILITY under the
    model's own count likelihood, P(count > 0) = 1 - exp(-mu), on the CALIBRATED mean --
    the per-marker factor had been applied only to colour, never to size, and that
    omission (not the cutoff choice) was the dominant error.

    Mean absolute error against observed detection, 8 markers x 4 days:
        calibrated P(count>0)   1.79 pp     <- this
        quantile cutoff         8.58 pp
        uncalibrated P(count>0) 7.95 pp
    Nothing is fitted to the per-day rates: the single per-marker factor is fit on the four
    population day-means of EXPRESSION, so agreement in detection is earned, not imposed.
    """
    mu = np.expm1(np.clip(cal(pred_all[t][cells, j].astype(np.float64), j), 0.0, None))
    return float((1.0 - np.exp(-mu)).mean())


def stats(cell_mask, like_for_like=False, mode="calibrated"):
    """(mu, frac) arrays of shape (4, G) for Observed and CONCERT over the given cells.

    Which cells each half summarises is the key choice:
      like_for_like=False (cohort view) -- Observed at Day t is the cells MEASURED at
        Day t, while CONCERT at Day t is EVERY cell re-decoded to Day t. The two halves
        therefore describe different animals; this is the whole-cohort virtual course.
      like_for_like=True -- CONCERT is restricted to the same cells as Observed (the
        animals measured at Day t), so the two halves describe identical cells. A true
        reconstruction check, but no longer a statement about the whole cohort.
    """
    o_mu = np.full((len(DAYS), G), np.nan); o_fr = np.zeros((len(DAYS), G))
    p_mu = np.zeros((len(DAYS), G)); p_fr = np.zeros((len(DAYS), G))
    for i, t in enumerate(DAYS):
        day_cells = cell_mask & np.isclose(src_day, t, atol=1e-3)
        pred_cells = day_cells if like_for_like else cell_mask
        for c, j in enumerate(keep):
            if day_cells.any():
                m = obs_all[day_cells, j].mean()
                o_mu[i, c] = (m - OBS_H[0][j]) / OBS_H[1][j] if mode == "healthyz" else m
                o_fr[i, c] = (obs_all[day_cells, j] > 0).mean()      # detection
            if pred_cells.any():
                raw = pred_all[t][pred_cells, j].mean()
                # healthy-z replaces the calibration rather than stacking on it: both put
                # the prediction on the observed scale, just referenced differently
                p_mu[i, c] = ((raw - PRED_H[0][j]) / PRED_H[1][j] if mode == "healthyz"
                              else cal(raw, j))
                p_fr[i, c] = expressing_pred(t, j, pred_cells)
    return o_mu, o_fr, p_mu, p_fr


def _half_disc(side):
    """Unit half-disc path sized like matplotlib's 'o' (radius 0.5), flat edge vertical."""
    a0, a1 = (np.pi / 2, 3 * np.pi / 2) if side == "left" else (-np.pi / 2, np.pi / 2)
    th = np.linspace(a0, a1, 80)
    pts = np.column_stack([np.cos(th), np.sin(th)]) * 0.5
    return Path(np.vstack([pts, pts[:1]]), closed=True)


LEFT, RIGHT = _half_disc("left"), _half_disc("right")


def sizes(frac):
    return 100 + 5200 * np.clip(frac, 0, 1)


LABELS = {"calibrated": "Mean log1p expression",
          "healthyz": "Expression (healthy-tissue SD)"}


def make_norm(panels, mode="calibrated"):
    """One colour scale shared by every panel drawn together, so colour means one thing."""
    finite = np.concatenate([np.concatenate([o[np.isfinite(o)].ravel(), p.ravel()])
                             for o, _, p, _ in panels])
    if mode == "healthyz":
        vmax = max(float(np.nanpercentile(finite, 98)), 0.1)
        vmin = min(float(np.nanpercentile(finite, 2)), -0.1)
        return Normalize(vmin=vmin, vmax=vmax), vmax
    vmax = float(np.nanpercentile(finite, 98))
    return Normalize(vmin=0.0, vmax=vmax), vmax


def draw(ax, panel, norm, y_labels):
    o_mu, o_fr, p_mu, p_fr = panel
    for i in range(len(DAYS)):
        if np.isfinite(o_mu[i]).all():
            ax.scatter(np.full(G, i), np.arange(G), s=sizes(o_fr[i]), c=o_mu[i],
                       marker=LEFT, cmap=CMAP, norm=norm, edgecolors="0.20", linewidths=0.7)
        else:
            ax.scatter(np.full(G, i), np.arange(G), s=sizes(p_fr[i]),
                       marker=LEFT, facecolors="none", edgecolors=ps.MUTED,
                       linewidths=1.2, linestyle=(0, (2, 1.6)))
        ax.scatter(np.full(G, i), np.arange(G), s=sizes(p_fr[i]), c=p_mu[i],
                   marker=RIGHT, cmap=CMAP, norm=norm, edgecolors="0.20", linewidths=0.7)

    ax.set_xticks(range(len(DAYS)))
    ax.set_xticklabels([ps.day_label(t) for t in DAYS], fontsize=DAY_FS)
    ax.set_yticks(range(G))
    ax.set_yticklabels(markers, fontsize=MK_FS)
    ax.tick_params(axis="y", labelleft=y_labels, **({} if y_labels else {"length": 0}))
    ax.set_xlim(-0.6, len(DAYS) - 0.4); ax.set_ylim(-0.6, G - 0.4)
    ax.invert_yaxis(); ax.grid(False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def add_keys(fig, axes, norm, vmax, any_missing, y_anchor, cax=None,
             label=LABELS["calibrated"]):
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
    cb = (fig.colorbar(sm, cax=fig.add_axes(cax)) if cax is not None
          else fig.colorbar(sm, ax=axes, shrink=0.55, pad=0.02))
    ticks = list(np.linspace(float(norm.vmin), vmax, 4))   # evenly spaced BY VALUE
    fmt = "%.3f" if vmax < 0.2 else "%.2f"
    cb.set_ticks(ticks); cb.set_ticklabels([fmt % v for v in ticks])
    cb.set_label(label, fontsize=CBAR_FS)
    cb.ax.tick_params(labelsize=CBAR_FS - 3)

    half = [Line2D([0], [0], marker=LEFT, ls='', mfc='0.55', mec='0.20', ms=15,
                   label="Observed"),
            Line2D([0], [0], marker=RIGHT, ls='', mfc='0.55', mec='0.20', ms=15,
                   label="CONCERT")]
    if any_missing:
        half.append(Line2D([0], [0], marker=LEFT, ls='', mfc='none', mec=ps.MUTED,
                           mew=1.2, ms=15, label="Never measured"))
    frac = [Line2D([0], [0], marker='o', ls='', mfc='0.75', mec='0.20',
                   ms=np.sqrt(sizes(f)) * 0.58, label=f"{int(f*100)}%")
            for f in (0.05, 0.2, 0.4)]
    l1 = fig.legend(handles=half, loc="upper center", bbox_to_anchor=(0.5, y_anchor),
                    ncol=len(half), frameon=False, fontsize=LEG_FS, handletextpad=0.5,
                    columnspacing=1.8)
    fig.add_artist(l1)
    fig.legend(handles=frac, title="% Expressing", loc="upper center",
               bbox_to_anchor=(0.5, y_anchor - 0.08), ncol=3, frameon=False,
               columnspacing=2.4, handletextpad=0.7, fontsize=LEG_FS,
               title_fontsize=LEG_FS)


def dotplot(panel, out_name):
    fig, ax = plt.subplots(figsize=(1.0 * len(DAYS) + 3.2, 0.72 * G + 1.9))
    norm, vmax = make_norm([panel])
    draw(ax, panel, norm, y_labels=True)
    add_keys(fig, ax, norm, vmax, not np.isfinite(panel[0]).all(), 0.02)
    out = f"{HERE}/{out_name}"
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


def dotplot_row(panels, titles, out_name, mode="calibrated"):
    """The featured mice side by side, one shared colour scale and one shared key."""
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.7 * n + 1.9, 0.62 * G + 2.9), sharey=True)
    norm, vmax = make_norm(panels, mode)
    for k, (ax, panel, title) in enumerate(zip(axes, panels, titles)):
        draw(ax, panel, norm, y_labels=(k == 0))
        ax.set_title(title, fontsize=TITLE_FS, loc="center", pad=12)
    # explicit layout: the keys live in a reserved bottom band rather than floating far
    # below the axes, and the colorbar gets its own slot instead of stealing panel width
    fig.subplots_adjust(left=0.075, right=0.885, top=0.915, bottom=0.245, wspace=0.10)
    add_keys(fig, list(axes), norm, vmax,
             any(not np.isfinite(p[0]).all() for p in panels), 0.175,
             cax=[0.905, 0.35, 0.011, 0.42], label=LABELS[mode])
    out = f"{HERE}/{out_name}"
    fig.savefig(out, dpi=200); plt.close(fig)
    print("wrote", out)


ALL = np.ones(len(src_day), bool)
# cohort view: CONCERT = every cell decoded to that day; Observed = that day's animals
dotplot(stats(ALL), "fig_q4_gt_vs_pred_dotplot.png")
# like-for-like view: both halves summarise the SAME cells at each day
dotplot(stats(ALL, like_for_like=True), "fig_q4_gt_vs_pred_dotplot_matched.png")

panels, panels_z, titles = [], [], []
for M in MICE:
    m = mouse_all == M
    if not m.any():
        print(f"  [skip] {M} not in dump"); continue
    panels.append(stats(m))
    panels_z.append(stats(m, mode="healthyz"))
    titles.append(f"Mouse {ps.MOUSE_NUMBER[M]}, {ps.day_label(np.unique(src_day[m])[0])}")
dotplot_row(panels, titles, "fig_q4_gt_vs_pred_dotplot_permouse.png")
# same layout, colour referenced to healthy tissue per marker instead of absolute level
dotplot_row(panels_z, titles, "fig_q4_gt_vs_pred_dotplot_permouse_healthyz.png",
            mode="healthyz")
