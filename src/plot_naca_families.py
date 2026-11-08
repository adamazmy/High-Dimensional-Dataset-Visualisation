"""Reproduce the headline NACA result: KMeans (k=5) at Re = 200,000 recovers the
NACA 4-series families from performance data alone.

Saves docs/images/naca_kmeans_families.png and prints cluster membership.

Run with:  python src/plot_naca_families.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from airfoil_features import FEATURE_COLUMNS, load_airfoil_data, summarise_at_reynolds

REYNOLDS = 200_000
K = 5
OUT_PATH = Path(__file__).resolve().parent.parent / "docs" / "images" / "naca_kmeans_families.png"


def describe(cluster_rows) -> str:
    """Short human-readable label for a cluster from its mean geometry."""
    camber = cluster_rows["maxCamber"].mean()
    thickness = cluster_rows["maxThickness"].mean()
    if camber >= 0.04:
        return "High camber"
    kind = "Symmetric" if camber == 0 else "Cambered"
    return f"{kind}, {'thick' if thickness >= 0.15 else 'thin'}"


def main() -> None:
    summary = summarise_at_reynolds(load_airfoil_data(), REYNOLDS)
    X = StandardScaler().fit_transform(summary[FEATURE_COLUMNS])
    summary["cluster"] = KMeans(n_clusters=K, random_state=42, n_init=10).fit_predict(X)
    sil = silhouette_score(X, summary["cluster"])

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    var = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(10, 6.5))
    colors = plt.get_cmap("tab10").colors
    for c, rows in summary.groupby("cluster"):
        idx = rows.index
        names = ", ".join(n.replace("naca", "") for n in rows["airfoilName"])
        ax.scatter(X_pca[idx, 0], X_pca[idx, 1], s=110, color=colors[c], edgecolor="white",
                   linewidth=1, label=f"{describe(rows)}  ({names})", zorder=3)
        print(f"Cluster {c} - {describe(rows)}: {list(rows['airfoilName'])}")
        for i in idx:
            ax.annotate(summary.at[i, "airfoilName"].upper(), (X_pca[i, 0], X_pca[i, 1]),
                        xytext=(6, 5), textcoords="offset points", fontsize=8, color="#333")

    ax.set_xlabel(f"PC1 ({var[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({var[1]:.1f}% variance)")
    ax.set_title(f"KMeans (k={K}) on NACA 4-series airfoils at Re = {REYNOLDS:,}  |  "
                 f"silhouette = {sil:.3f}", fontweight="bold")
    ax.grid(alpha=0.3, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(title="Cluster (NACA designations)", loc="upper center",
              bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8.5, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Silhouette: {sil:.3f}\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()
