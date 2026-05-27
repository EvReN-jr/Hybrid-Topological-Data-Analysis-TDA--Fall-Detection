"""
Results & Discussion Figures — from V15 DB
Generates four publication-quality figures:
  fig_f2_bars.pdf          – grouped bar chart: General F₂ by dataset × classifier
  fig_sweep_boxplot.pdf    – per-subject ΔF₂ distributions (LR and SVM panels)
  fig_general_vs_personal.pdf – scatter: General F₂ vs Personal F₂ per subject
  fig_bayesian_convergence.pdf – Optuna best-so-far F₂ vs trial number

DB: tda_proje/Results_V15.db  (tda_label = 'server')
"""

import sqlite3
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH    = os.path.join(SCRIPT_DIR, "..", "..", "tda_proje", "Results_V15.db")
OUT_DIR    = SCRIPT_DIR

plt.rcParams.update({
    "font.family": "serif",
    "font.size":   9,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.dpi":  150,
    "text.usetex": False,
})

DATASETS  = ["MobiFall", "SisFall", "FAD_40Hz"]
DS_LABELS = ["MobiFall", "SisFall", "FAD 40 Hz"]
MODELS    = ["LogReg", "SVM"]
CLF_LABELS = {"LogReg": "LR", "SVM": "SVM"}

# Colors
CLF_COLOR = {"LogReg": "#2B5EA7", "SVM": "#C0392B"}
DS_COLOR  = {"MobiFall": "#27AE60", "SisFall": "#2B5EA7", "FAD_40Hz": "#E67E22"}


# ── DB helpers ────────────────────────────────────────────────────────────────

def load_metrics(db_path):
    """Return dict {dataset: {model: {subject: row}}} for tda_label='server'."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    data = {}
    for ds in DATASETS:
        data[ds] = {}
        for model in MODELS:
            rows = conn.execute(
                f"SELECT * FROM metrics_{ds}_server WHERE model=?", (model,)
            ).fetchall()
            data[ds][model] = {r["subject"]: dict(r) for r in rows}
    conn.close()
    return data


def load_naive_repeats(db_path):
    """Return dict {dataset: {model: [f2, ...]}} for tda_label='server'."""
    conn = sqlite3.connect(db_path)
    data = {}
    for ds in DATASETS:
        data[ds] = {}
        for model in MODELS:
            rows = conn.execute(
                f"SELECT f2 FROM naive_repeats_{ds}_server WHERE model=?", (model,)
            ).fetchall()
            data[ds][model] = [r[0] for r in rows]
    conn.close()
    return data


def load_optuna_trials(db_path):
    """Return dict {dataset: [f2_per_trial]} sorted by trial index."""
    conn = sqlite3.connect(db_path)
    data = {}
    for ds in DATASETS:
        rows = conn.execute(
            f"SELECT trial, f2 FROM optuna_trials_{ds}_server "
            f"WHERE state='TrialState.COMPLETE' ORDER BY trial"
        ).fetchall()
        data[ds] = [(r[0], r[1]) for r in rows]
    conn.close()
    return data


def compute_sweep_gains(metrics):
    """
    For each subject × dataset × model compute
      ΔF₂(s) = max_τ F₂(τ, s) − F₂(τ̂₀(s), s)
    where τ̂₀(s) = g_thr (PR-curve trained threshold).

    Returns dict {dataset: {model: [delta_f2, ...]}}
    """
    gains = {}
    for ds, ds_data in metrics.items():
        gains[ds] = {}
        for model, subj_data in ds_data.items():
            deltas = []
            for subj, row in subj_data.items():
                g_thr = row["g_thr"]
                sweep = json.loads(row["thr_sweep_json"])
                # F₂ at the trained threshold (closest in sweep grid)
                f2_baseline = min(sweep, key=lambda e: abs(e["threshold"] - g_thr))["f2"]
                f2_best     = max(e["f2"] for e in sweep)
                deltas.append(f2_best - f2_baseline)
            gains[ds][model] = deltas
    return gains


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Grouped bar chart: General F₂ by dataset × classifier × protocol
# ════════════════════════════════════════════════════════════════════════════

def fig_f2_bars(metrics, naive_repeats, out_dir):
    protocols = ["Naive", "General", "Personal"]
    prot_cols  = {
        "Naive":    {"mean": None,         "ci": None},
        "General":  {"mean": "g_f2_mean",  "ci": "g_f2_ci"},
        "Personal": {"mean": "p_f2_mean",  "ci": "p_f2_ci"},
    }
    bar_colors = {
        "Naive":    {"LogReg": "#8FB8DE", "SVM": "#F0A8A8"},
        "General":  {"LogReg": "#2B5EA7", "SVM": "#C0392B"},
        "Personal": {"LogReg": "#1A3A6B", "SVM": "#7B241C"},
    }
    hatches = {"Naive": "//", "General": "", "Personal": "xx"}

    n_ds   = len(DATASETS)
    n_clf  = len(MODELS)
    n_prot = len(protocols)
    group_w = 0.88
    bar_w   = group_w / (n_clf * n_prot + 0.8)

    fig, ax = plt.subplots(figsize=(8.0, 3.4))

    for di, ds in enumerate(DATASETS):
        base_x = di
        for ci, model in enumerate(MODELS):
            for pi, prot in enumerate(protocols):
                offset = (ci * n_prot + pi - (n_clf * n_prot - 1) / 2) * bar_w
                x = base_x + offset

                if prot == "Naive":
                    vals = naive_repeats[ds][model]
                    mean_f2 = np.mean(vals)
                    ci_f2   = 1.96 * np.std(vals) / np.sqrt(len(vals))
                else:
                    subj_data = metrics[ds][model]
                    mean_f2 = np.mean([r[prot_cols[prot]["mean"]]
                                       for r in subj_data.values()])
                    ci_f2   = np.mean([r[prot_cols[prot]["ci"]]
                                       for r in subj_data.values()])

                bar = ax.bar(x, mean_f2, width=bar_w * 0.92,
                             color=bar_colors[prot][model],
                             hatch=hatches[prot],
                             edgecolor="white", linewidth=0.5, zorder=2)
                ax.errorbar(x, mean_f2, yerr=ci_f2, fmt="none",
                            ecolor="black", elinewidth=0.7, capsize=2, zorder=3)

    ax.set_xticks(range(n_ds))
    ax.set_xticklabels(DS_LABELS, fontsize=8)
    ax.set_ylabel(r"Mean $F_2$ score", fontsize=8)
    ax.set_ylim(0.55, 1.02)
    ax.axhline(1.0, color="gray", lw=0.4, ls="--", alpha=0.4)
    ax.set_title(
        r"$F_2$ by dataset, classifier, and evaluation protocol "
        "(error bars: mean CI across subjects)", pad=5
    )

    # Legend
    legend_handles = []
    for prot in protocols:
        for model in MODELS:
            legend_handles.append(
                mpatches.Patch(
                    facecolor=bar_colors[prot][model],
                    hatch=hatches[prot],
                    edgecolor="gray", linewidth=0.5,
                    label=f"{CLF_LABELS[model]} – {prot}",
                )
            )
    ax.legend(handles=legend_handles, ncol=3, fontsize=6.5,
              frameon=True, framealpha=0.9, loc="lower right",
              bbox_to_anchor=(1.0, 0.0))

    fig.subplots_adjust()
    for ext in [".pdf", ".png"]:
        fig.savefig(os.path.join(out_dir, f"fig_f2_bars{ext}"),
                    bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("Saved: fig_f2_bars.pdf")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Per-subject ΔF₂ distributions (boxplots + jittered points)
# ════════════════════════════════════════════════════════════════════════════

def fig_sweep_boxplot(gains, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2),
                              sharey=False, gridspec_kw={"wspace": 0.32})

    for ax, model in zip(axes, MODELS):
        positions = np.arange(len(DATASETS)) * 1.2
        data_all = [gains[ds][model] for ds in DATASETS]

        bp = ax.boxplot(
            data_all,
            positions=positions,
            widths=0.45,
            patch_artist=True,
            notch=False,
            medianprops=dict(color="black", linewidth=1.4),
            whiskerprops=dict(linewidth=0.8),
            capprops=dict(linewidth=0.8),
            flierprops=dict(marker="o", markersize=3,
                            markerfacecolor="gray", alpha=0.5),
        )
        for patch, ds in zip(bp["boxes"], DATASETS):
            patch.set_facecolor(DS_COLOR[ds])
            patch.set_alpha(0.75)

        # Jittered individual points
        rng = np.random.default_rng(99)
        for pos, ds in zip(positions, DATASETS):
            vals = gains[ds][model]
            jit  = rng.uniform(-0.12, 0.12, len(vals))
            ax.scatter(pos + jit, vals,
                       s=14, color=DS_COLOR[ds],
                       edgecolors="white", linewidths=0.4,
                       alpha=0.9, zorder=3)

        ax.axhline(0, color="gray", lw=0.5, ls="--", alpha=0.5)
        ax.set_xticks(positions)
        ax.set_xticklabels(DS_LABELS, fontsize=7.5)
        ax.set_ylabel(r"$\Delta F_2(s) = F_2(\tau^*(s)) - F_2(\hat\tau_0(s))$", fontsize=7.5)
        ax.set_title(f"{CLF_LABELS[model]} classifier", pad=4)

        # Annotate max gain subject
        for pos, ds in zip(positions, DATASETS):
            vals = gains[ds][model]
            vmax = max(vals)
            ax.text(pos, vmax + 0.004, f"+{vmax:.3f}",
                    ha="center", va="bottom", fontsize=5.8, color=DS_COLOR[ds])

    fig.suptitle(
        r"Per-subject threshold sweep gain $\Delta F_2(s)$ — "
        r"baseline $\hat\tau_0(s)$ is the PR-curve trained threshold",
        fontsize=9, y=1.02,
    )
    fig.subplots_adjust()
    for ext in [".pdf", ".png"]:
        fig.savefig(os.path.join(out_dir, f"fig_sweep_boxplot{ext}"),
                    bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("Saved: fig_sweep_boxplot.pdf")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Scatter: General F₂ vs Personal F₂ per subject
# ════════════════════════════════════════════════════════════════════════════

def fig_general_vs_personal(metrics, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4),
                              sharex=False, sharey=False,
                              layout="constrained",
                              gridspec_kw={"wspace": 0.35})

    for ax, model in zip(axes, MODELS):
        all_gen, all_per = [], []

        for ds in DATASETS:
            subj_data = metrics[ds][model]
            gen = [r["g_f2_mean"] for r in subj_data.values()]
            per = [r["p_f2_mean"] for r in subj_data.values()]
            ax.scatter(gen, per,
                       color=DS_COLOR[ds], s=22,
                       edgecolors="white", linewidths=0.4,
                       alpha=0.85, zorder=3, label=ds.replace("_", " "))
            all_gen.extend(gen)
            all_per.extend(per)

        lim_lo = min(min(all_gen), min(all_per)) - 0.02
        lim_hi = max(max(all_gen), max(all_per)) + 0.02
        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi],
                "k--", lw=0.8, alpha=0.4, label="General = Personal")

        ax.set_xlim(lim_lo, lim_hi)
        ax.set_ylim(lim_lo, lim_hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"General $F_2$ (LOSO)", fontsize=8)
        ax.set_ylabel(r"Personal $F_2$ (per-subject train)", fontsize=8)
        ax.set_title(f"{CLF_LABELS[model]} classifier", pad=4)
        ax.legend(fontsize=6.5, frameon=True, framealpha=0.88, loc="upper left")

        # Shade region where Personal < General
        ax.fill_between([lim_lo, lim_hi], [lim_lo, lim_hi], [lim_lo, lim_lo],
                        alpha=0.06, color="#C0392B")
        ax.text((lim_lo + lim_hi) * 0.5, lim_lo + 0.01,
                "Personal < General", fontsize=6, color="#C0392B",
                ha="center", alpha=0.7)

    fig.suptitle(
        "General (LOSO) vs Personal (per-subject) $F_2$ — "
        "points below diagonal: personalisation degrades performance",
        fontsize=9, y=1.02,
    )
    for ext in [".pdf", ".png"]:
        fig.savefig(os.path.join(out_dir, f"fig_general_vs_personal{ext}"),
                    bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("Saved: fig_general_vs_personal.pdf")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Bayesian optimisation convergence (best-so-far F₂ vs trial)
# ════════════════════════════════════════════════════════════════════════════

def fig_bayesian_convergence(optuna_data, out_dir):
    fig, ax = plt.subplots(figsize=(6.5, 3.2))

    for ds in DATASETS:
        trials = optuna_data[ds]
        f2_vals = np.array([f2 for _, f2 in trials])
        best_so_far = np.maximum.accumulate(f2_vals)
        trial_nums  = np.array([t for t, _ in trials])

        ax.plot(trial_nums, best_so_far,
                color=DS_COLOR[ds], lw=1.4,
                label=ds.replace("_", " "))
        ax.scatter(trial_nums[::20], best_so_far[::20],
                   color=DS_COLOR[ds], s=14, zorder=3,
                   edgecolors="white", linewidths=0.4)

        # Annotate final best value
        ax.annotate(f"{best_so_far[-1]:.4f}",
                    xy=(trial_nums[-1], best_so_far[-1]),
                    xytext=(trial_nums[-1] - 22, best_so_far[-1] - 0.004),
                    fontsize=6.5, color=DS_COLOR[ds],
                    ha="right")

    ax.set_xlabel("Optuna trial number", fontsize=8)
    ax.set_ylabel(r"Best $F_2$ so far (averaged LR + SVM proxy)", fontsize=8)
    ax.set_title(
        "Bayesian optimisation convergence — "
        r"200 trials per dataset, TPE sampler", pad=5
    )
    ax.legend(fontsize=7.5, frameon=True, framealpha=0.9, loc="lower right")
    ax.set_xlim(-2, 205)
    ax.set_ylim(
        min(optuna_data[ds][0][1] for ds in DATASETS) * 0.985,
        max(max(f2 for _, f2 in optuna_data[ds]) for ds in DATASETS) * 1.005,
    )

    # Add a vertical dashed line at trial 50 (early convergence reference)
    ax.axvline(50, color="gray", ls=":", lw=0.7, alpha=0.5)
    ax.text(51, ax.get_ylim()[0] + 0.003, "trial 50", fontsize=6, color="gray")

    fig.subplots_adjust()
    for ext in [".pdf", ".png"]:
        fig.savefig(os.path.join(out_dir, f"fig_bayesian_convergence{ext}"),
                    bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("Saved: fig_bayesian_convergence.pdf")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"Loading data from {DB_PATH}")
    metrics       = load_metrics(DB_PATH)
    naive_repeats = load_naive_repeats(DB_PATH)
    optuna_data   = load_optuna_trials(DB_PATH)
    gains         = compute_sweep_gains(metrics)

    print("\n--- Sweep gain summary (mean ± std) ---")
    for ds in DATASETS:
        for model in MODELS:
            g = gains[ds][model]
            print(f"  {ds:12s} {CLF_LABELS[model]:3s}: "
                  f"mean={np.mean(g):.4f}  std={np.std(g):.4f}  "
                  f"max={np.max(g):.4f}  n={len(g)}")

    os.chdir(OUT_DIR)
    fig_f2_bars(metrics, naive_repeats, OUT_DIR)
    fig_sweep_boxplot(gains, OUT_DIR)
    fig_general_vs_personal(metrics, OUT_DIR)
    fig_bayesian_convergence(optuna_data, OUT_DIR)

    print("\nDone. Generated files:")
    figs = ["fig_f2_bars", "fig_sweep_boxplot",
            "fig_general_vs_personal", "fig_bayesian_convergence"]
    for f in figs:
        for ext in [".pdf", ".png"]:
            print(f"  {f}{ext}")
