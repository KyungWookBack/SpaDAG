"""Plot the truth and the SpaDAG posterior for one reversal pair of the example dataset.

Usage::

    python docs/plot_example_recovery.py --summary runs/chain1/trace/summary.npz \
        --data artifacts/data/example_reversal_p50_n500 --out docs/example_recovery.png

The pair is the lexicographically first unordered pair whose two directions
are both present in the true triangle graphs, chosen from the truth alone.
The top row shows the direction k -> j, the bottom row j -> k; the left column
is the truth (present or absent in each triangle) and the right column the
posterior inclusion probability of the fitted chain.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

BLUE, ORANGE, GRID, BG = "#1f4e79", "#c55a11", "#9aa4ab", "#f4f5f6"


def first_reversal_pair(truth: np.ndarray) -> tuple[int, int]:
    p = truth.shape[1]
    for k in range(p):
        for j in range(k + 1, p):
            if truth[:, j, k].any() and truth[:, k, j].any():
                return k, j
    raise ValueError("no pair is present in both directions")


def panel(ax, points, tri, values, obs, color, binary):
    cmap = LinearSegmentedColormap.from_list("pip", [BG, color])
    face = cmap(np.where(values > 0.5, 1.0, 0.0)) if binary else cmap(np.clip(values, 0, 1))
    ax.add_collection(PolyCollection(points[tri], facecolors=face, edgecolors=GRID, linewidths=0.7))
    ax.scatter(obs[:, 0], obs[:, 1], s=2.5, color="#4b5563", alpha=0.45, lw=0, zorder=3)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    return cmap


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--summary", type=Path, required=True, help="summary.npz of a fitted chain")
    parser.add_argument("--data", type=Path, required=True, help="example bundle with edge_triangle.npy")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=170)
    args = parser.parse_args(argv)

    points = np.load(args.data / "mesh_points.npy")
    tri = np.load(args.data / "mesh_triangles.npy")
    obs = np.load(args.data / "obs_points.npy")
    truth = np.load(args.data / "edge_triangle.npy")
    with np.load(args.summary, allow_pickle=False) as z:
        prob = np.asarray(z["edge_prob"], dtype=float)
    if prob.shape != truth.shape:
        raise ValueError(f"edge_prob {prob.shape} and truth {truth.shape} differ")

    k, j = first_reversal_pair(truth)
    directions = [((j, k), rf"$X_{{{k + 1}}} \rightarrow X_{{{j + 1}}}$", BLUE),
                  ((k, j), rf"$X_{{{j + 1}}} \rightarrow X_{{{k + 1}}}$", ORANGE)]

    fig, axes = plt.subplots(2, 2, figsize=(7.6, 7.2))
    for row, ((child, parent), label, color) in enumerate(directions):
        panel(axes[row, 0], points, tri, truth[:, child, parent].astype(float), obs, color, binary=True)
        cmap = panel(axes[row, 1], points, tri, prob[:, child, parent], obs, color, binary=False)
        axes[row, 0].set_ylabel(label, fontsize=15, labelpad=10)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
        cbar = fig.colorbar(sm, ax=axes[row, 1], fraction=0.046, pad=0.03)
        cbar.set_label("PIP", fontsize=11)
        cbar.ax.tick_params(labelsize=9)
    axes[0, 0].set_title("Truth", fontsize=15, pad=8)
    axes[0, 1].set_title("SpaDAG posterior", fontsize=15, pad=8)
    fig.text(0.5, 0.015, "Dots: observation locations", ha="center", fontsize=10, color="#687477")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, facecolor="white")
    print(f"pair X{k + 1} <-> X{j + 1}; saved {args.out}")


if __name__ == "__main__":
    main()
