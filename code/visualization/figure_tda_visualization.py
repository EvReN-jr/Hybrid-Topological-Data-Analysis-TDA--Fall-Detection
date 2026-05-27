"""
TDA Visualization Figures — Methodology Section
Generates three publication-quality figures:
  fig_rips_filtration.pdf  – Rips(X, 2ε) at four ε scales (like Figure 1 in 2102.01956)
  fig_delay_embedding.pdf  – delay-embedded fall vs ADL windows (2D projection)
  fig_persistence_diagram.pdf – H₀ and H₁ persistence diagrams for fall vs ADL

Dependencies: numpy, scipy, matplotlib (no TDA library required)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from scipy.spatial.distance import squareform, pdist
from itertools import combinations

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.dpi": 150,
    "text.usetex": False,
})


# ── Persistent Homology (H₀ + H₁) via boundary-matrix reduction ─────────────

def _boundary_matrix_reduction(filtration):
    """
    Standard persistence algorithm (Edelsbrunner-Letscher-Zomorodian) over Z/2.

    filtration : list of (birth_value, simplex_tuple) sorted by
                 (birth_value, dim) with dim = len(simplex)-1.

    Returns dict {dim: [(birth, death), ...]}  where death=inf means essential.
    """
    N = len(filtration)
    simplex_to_idx = {s: i for i, (_, s) in enumerate(filtration)}

    # Boundary columns (sparse, as sets of row indices)
    columns = []
    for _, sigma in filtration:
        dim = len(sigma) - 1
        if dim == 0:
            columns.append(set())
        elif dim == 1:
            a, b = sigma
            columns.append({simplex_to_idx[(a,)], simplex_to_idx[(b,)]})
        elif dim == 2:
            a, b, c = sigma
            columns.append({
                simplex_to_idx[(a, b)],
                simplex_to_idx[(a, c)],
                simplex_to_idx[(b, c)],
            })
        else:
            columns.append(set())

    # Left-to-right column reduction
    pivots = {}  # row_index → col_index that claimed it
    for j in range(N):
        while columns[j]:
            low = max(columns[j])
            if low not in pivots:
                pivots[low] = j
                break
            columns[j] ^= columns[pivots[low]]   # symmetric_difference in-place

    # Collect pairs
    pairs = {0: [], 1: []}
    for i, (birth_i, sigma_i) in enumerate(filtration):
        d = len(sigma_i) - 1
        if d > 1:
            continue
        if i in pivots:
            j = pivots[i]
            death = filtration[j][0]
            if death > birth_i:
                pairs[d].append((birth_i, death))
        elif not columns[i]:
            pairs[d].append((birth_i, np.inf))

    return pairs


def compute_persistence_rips(pts, max_dim=1):
    """Compute H₀ and H₁ for the Rips complex of a 2-D point set."""
    n = len(pts)
    D = squareform(pdist(pts))

    filt = []
    for i in range(n):
        filt.append((0.0, (i,)))
    for i in range(n):
        for j in range(i + 1, n):
            filt.append((D[i, j] / 2, (i, j)))
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                b = max(D[i, j], D[i, k], D[j, k]) / 2
                filt.append((b, (i, j, k)))

    filt.sort(key=lambda x: (x[0], len(x[1]) - 1))
    return _boundary_matrix_reduction(filt)


# ── Rips complex drawing helper ───────────────────────────────────────────────

def draw_rips(ax, pts, eps, title,
              pt_color="#2B5EA7", edge_color="#1A1A1A",
              tri_color="#AEC6E8", highlight_new=None):
    """Draw Rips(pts, 2ε) on ax."""
    n = len(pts)
    D = squareform(pdist(pts))

    # Triangles (drawn first so edges/points sit on top)
    tri_patches = []
    for i, j, k in combinations(range(n), 3):
        if D[i, j] <= 2 * eps and D[i, k] <= 2 * eps and D[j, k] <= 2 * eps:
            tri_patches.append(Polygon(pts[[i, j, k]], closed=True))
    if tri_patches:
        pc = PatchCollection(tri_patches, facecolor=tri_color, edgecolor="none", alpha=0.55)
        ax.add_collection(pc)

    # Edges
    for i, j in combinations(range(n), 2):
        if D[i, j] <= 2 * eps:
            lw, col = (1.0, edge_color)
            ax.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]],
                    lw=lw, color=col, zorder=2, solid_capstyle="round")

    # Points
    ax.scatter(pts[:, 0], pts[:, 1], s=28, zorder=3,
               color=pt_color, edgecolors="white", linewidths=0.6)

    ax.set_title(title, pad=4)
    ax.set_aspect("equal")
    ax.axis("off")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Rips filtration at four scales
# ════════════════════════════════════════════════════════════════════════════

def fig_rips_filtration():
    # 6 points on a unit circle (regular hexagon) — produces a clean H₁ feature
    theta = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    pts_circle = np.column_stack([np.cos(theta), np.sin(theta)])

    # Add a few noise points for visual interest
    rng = np.random.default_rng(7)
    jitter = rng.normal(0, 0.07, pts_circle.shape)
    pts = pts_circle + jitter

    # Pairwise distances (for label annotations)
    D = squareform(pdist(pts))
    consec_dists = [D[i, (i + 1) % 6] for i in range(6)]
    d_consec = np.mean(consec_dists)                 # ≈ 1.0
    d_skip2  = np.mean([D[i, (i + 2) % 6] for i in range(6)])  # ≈ √3

    eps_vals = [0.0, d_consec / 2 * 1.04, d_skip2 / 2 * 0.97, d_skip2 / 2 * 1.08]
    labels = [
        r"$\varepsilon = 0$",
        rf"$\varepsilon = {eps_vals[1]:.2f}$" + "\n(H₀ merges, H₁ born)",
        rf"$\varepsilon = {eps_vals[2]:.2f}$" + "\n(H₁ persists)",
        rf"$\varepsilon = {eps_vals[3]:.2f}$" + "\n(H₁ dies)",
    ]

    fig, axes = plt.subplots(1, 4, figsize=(7.5, 2.1),
                              gridspec_kw={"wspace": 0.08})
    for ax, eps, lbl in zip(axes, eps_vals, labels):
        draw_rips(ax, pts, eps, lbl)

    # Legend
    leg_elements = [
        mpatches.Patch(facecolor="#AEC6E8", edgecolor="none", label="2-simplex (triangle)"),
        plt.Line2D([0], [0], color="#1A1A1A", lw=1.0, label="1-simplex (edge)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#2B5EA7",
                   markersize=5, label="0-simplex (point)"),
    ]
    fig.legend(handles=leg_elements, loc="lower center", ncol=3,
               frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.08))

    fig.suptitle(
        r"Rips$(X,\, 2\varepsilon)$ filtration — simplicial complex grows as $\varepsilon$ increases",
        fontsize=9, y=1.02,
    )
    fig.savefig("fig_rips_filtration.pdf", bbox_inches="tight")
    fig.savefig("fig_rips_filtration.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("Saved: fig_rips_filtration.pdf")
    return pts  # return for reuse


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Delay-embedded fall vs ADL windows
# ════════════════════════════════════════════════════════════════════════════

def _delay_embed(signal, m, tau):
    """Takens delay embedding: returns point cloud in R^m."""
    N = len(signal)
    L = N - (m - 1) * tau
    pts = np.array([signal[i:i + (m - 1) * tau + 1:tau] for i in range(L)])
    return pts


def fig_delay_embedding():
    rng = np.random.default_rng(42)
    fs, T = 50, 2.0
    t = np.linspace(0, T, int(fs * T), endpoint=False)

    # ADL: slow sinusoidal (walking cadence ~2 Hz), low amplitude
    adl = 0.5 * np.sin(2 * np.pi * 1.8 * t) + 0.15 * np.sin(2 * np.pi * 5.3 * t) \
        + rng.normal(0, 0.04, len(t))

    # Fall: quiet pre-impact → sharp impact spike → post-fall stillness
    fall = np.zeros(len(t))
    impact = int(0.55 * len(t))
    pre  = slice(0, impact)
    imp  = slice(impact, impact + int(0.06 * len(t)))
    post = slice(impact + int(0.06 * len(t)), len(t))
    fall[pre]  = rng.normal(0, 0.08, impact)
    fall[imp]  = np.linspace(0, 4.0, imp.stop - imp.start) * rng.normal(1, 0.05, imp.stop - imp.start)
    fall[post] = np.linspace(fall[imp.stop - 1], 0, post.stop - post.start) \
                + rng.normal(0, 0.05, post.stop - post.start)

    tau, m = 3, 2  # m=2 for 2-D visualization; thesis uses m=3–5 in R³⁺

    adl_pts  = _delay_embed(adl,  m, tau)
    fall_pts = _delay_embed(fall, m, tau)

    fig, axes = plt.subplots(1, 4, figsize=(8.5, 2.2),
                              gridspec_kw={"wspace": 0.35, "hspace": 0.0})

    # ── Signal panels ────────────────────────────────────────────────────────
    axes[0].plot(t, adl,  color="#2B5EA7", lw=0.7, alpha=0.85)
    axes[0].set_title("ADL signal (walking)", pad=3)
    axes[0].set_xlabel("Time (s)"); axes[0].set_ylabel("Acceleration")
    axes[0].set_xlim(0, T); axes[0].tick_params(labelsize=7)

    axes[1].plot(t, fall, color="#C0392B", lw=0.7, alpha=0.85)
    axes[1].axvspan(0.55 * T, min(0.61 * T, T), color="#F5CBA7", alpha=0.5,
                    label="impact")
    axes[1].set_title("Fall signal (impact at 1.1 s)", pad=3)
    axes[1].set_xlabel("Time (s)"); axes[1].set_ylabel("Acceleration")
    axes[1].set_xlim(0, T); axes[1].tick_params(labelsize=7)
    axes[1].legend(fontsize=6.5, frameon=False)

    # ── Delay-embedding cloud panels ─────────────────────────────────────────
    axes[2].scatter(adl_pts[:, 0], adl_pts[:, 1],
                    s=3, color="#2B5EA7", alpha=0.6, linewidths=0)
    axes[2].set_title(rf"Delay embedding $\tau={tau}$ (ADL)", pad=3)
    axes[2].set_xlabel(r"$x(t)$"); axes[2].set_ylabel(rf"$x(t+{tau})$")
    axes[2].tick_params(labelsize=7)
    axes[2].set_aspect("equal", adjustable="box")

    axes[3].scatter(fall_pts[:, 0], fall_pts[:, 1],
                    s=3, color="#C0392B", alpha=0.6, linewidths=0)
    axes[3].set_title(rf"Delay embedding $\tau={tau}$ (Fall)", pad=3)
    axes[3].set_xlabel(r"$x(t)$"); axes[3].set_ylabel(rf"$x(t+{tau})$")
    axes[3].tick_params(labelsize=7)
    axes[3].set_aspect("equal", adjustable="box")

    fig.suptitle(
        "Delay embedding converts a 1-D accelerometer window into a point cloud in $\\mathbb{R}^2$",
        fontsize=9, y=1.02,
    )
    fig.savefig("fig_delay_embedding.pdf", bbox_inches="tight")
    fig.savefig("fig_delay_embedding.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("Saved: fig_delay_embedding.pdf")
    return adl_pts, fall_pts


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Persistence diagrams (H₀ + H₁) for fall vs ADL
# ════════════════════════════════════════════════════════════════════════════

def _persistence_diagram_ax(ax, pairs, title, max_val=None):
    """Plot a persistence diagram on ax."""
    h0 = [(b, d) for b, d in pairs.get(0, []) if d < np.inf]
    h0_ess = [(b, d) for b, d in pairs.get(0, []) if d == np.inf]
    h1 = [(b, d) for b, d in pairs.get(1, []) if d < np.inf]
    h1_ess = [(b, d) for b, d in pairs.get(1, []) if d == np.inf]

    all_finite = [d for _, d in h0] + [b for b, _ in h0] + \
                 [d for _, d in h1] + [b for b, _ in h1]
    if max_val is None:
        max_val = max(all_finite, default=1.0) * 1.15

    ax.plot([0, max_val], [0, max_val], "k--", lw=0.7, alpha=0.4)

    if h0:
        b0, d0 = zip(*h0)
        ax.scatter(b0, d0, c="#2B5EA7", s=22, marker="o",
                   zorder=3, label=r"$H_0$ (finite)")
    if h0_ess:
        b0e = [b for b, _ in h0_ess]
        ax.scatter(b0e, [max_val * 0.97] * len(b0e),
                   c="#2B5EA7", s=22, marker="^",
                   zorder=3, label=r"$H_0$ (essential)")
    if h1:
        b1, d1 = zip(*h1)
        ax.scatter(b1, d1, c="#C0392B", s=28, marker="s",
                   zorder=3, label=r"$H_1$ (finite)")
    if h1_ess:
        b1e = [b for b, _ in h1_ess]
        ax.scatter(b1e, [max_val * 0.97] * len(b1e),
                   c="#C0392B", s=28, marker="D",
                   zorder=3, label=r"$H_1$ (essential)")

    ax.set_xlim(-0.02 * max_val, max_val)
    ax.set_ylim(-0.02 * max_val, max_val * 1.05)
    ax.set_xlabel(r"Birth $\varepsilon$", fontsize=8)
    ax.set_ylabel(r"Death $\varepsilon$", fontsize=8)
    ax.set_title(title, pad=4)
    ax.legend(fontsize=6.5, frameon=True, framealpha=0.85, loc="lower right")
    return max_val


def fig_persistence_diagram(adl_pts, fall_pts):
    # Subsample to ≤70 pts for speed (full cloud has ~95 pts)
    rng = np.random.default_rng(0)

    def subsample(pts, n=55):
        idx = rng.choice(len(pts), min(n, len(pts)), replace=False)
        return pts[idx]

    adl_sub  = subsample(adl_pts,  50)
    fall_sub = subsample(fall_pts, 50)

    print("Computing persistence for ADL cloud ...")
    adl_pairs  = compute_persistence_rips(adl_sub)
    print("Computing persistence for fall cloud ...")
    fall_pairs = compute_persistence_rips(fall_sub)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1),
                              gridspec_kw={"wspace": 0.38})

    mv_adl  = _persistence_diagram_ax(axes[0], adl_pairs,
                                       "Persistence diagram — ADL")
    mv_fall = _persistence_diagram_ax(axes[1], fall_pairs,
                                       "Persistence diagram — Fall",
                                       max_val=max(mv_adl,
                                                   _persistence_diagram_ax.__defaults__[0]
                                                   if False else 0))

    # Add text annotations highlighting the long-living H₁
    for ax, pairs, label in [(axes[0], adl_pairs, "ADL"),
                              (axes[1], fall_pairs, "Fall")]:
        h1_long = [(b, d) for b, d in pairs.get(1, [])
                   if d < np.inf and (d - b) > 0.05]
        if h1_long:
            b, d = max(h1_long, key=lambda x: x[1] - x[0])
            ax.annotate(f"loop\n$\\Delta\\varepsilon={d-b:.2f}$",
                        xy=(b, d), xytext=(b + 0.02, d - 0.04),
                        fontsize=6.5, color="#C0392B",
                        arrowprops=dict(arrowstyle="->", color="#C0392B",
                                        lw=0.7))

    fig.suptitle(
        "Persistent homology diagrams — points above the diagonal live longer",
        fontsize=9, y=1.02,
    )
    fig.savefig("fig_persistence_diagram.pdf", bbox_inches="tight")
    fig.savefig("fig_persistence_diagram.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("Saved: fig_persistence_diagram.pdf")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    fig_rips_filtration()
    adl_pts, fall_pts = fig_delay_embedding()
    fig_persistence_diagram(adl_pts, fall_pts)

    print("\nDone. Generated files:")
    for f in ["fig_rips_filtration", "fig_delay_embedding", "fig_persistence_diagram"]:
        for ext in [".pdf", ".png"]:
            print(f"  {f}{ext}")
