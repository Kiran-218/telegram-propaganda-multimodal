"""
M5 channel-level k-means partitioning (interpretive complement to HDBSCAN).

HDBSCAN found no density-based clusters -- the channels form a continuous
gradient (confirmed visually in the PCA plot). k-means will nonetheless
PARTITION that gradient into k pieces. This is NOT a claim of discrete natural
groups; k-means always returns clusters even on continuous/random data. The
value is interpretive: do the imposed partitions map onto something meaningful
(e.g. a size/activity gradient), and how well-separated are they?

Validation:
  - silhouette score per k (higher = better separation; low/modest values here
    are EXPECTED and are themselves evidence of continuity, not failure).
  - per-cluster feature medians so partitions can be interpreted.
  - Kruskal-Wallis on mismatch_mean across the chosen partition (does the
    proxy differ? expected: negligibly, within the mismatch noise band).

Honest framing for the writeup: k-means is reported as an exploratory partition
of a continuous population, explicitly NOT as discovery of discrete channel
types. Silhouette contextualises how weak/strong the imposed separation is.

Outputs (under --outdir):
  kmeans_silhouette.csv        k vs silhouette (+ inertia)
  kmeans_channel_partition.csv channels + chosen-k cluster label + features
  kmeans_summary.txt           silhouette sweep + profiles + interpretation
  kmeans_2d.png                PCA scatter coloured by chosen-k partition

Usage:
  python kmeans_analysis.py \
      --table channel_features_with_reuse.csv \
      --outdir m5_outputs \
      --k 3
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import kruskal

CLUSTER_FEATURES = [
    "mismatch_mean", "mismatch_std", "participants_count",
    "channel_age_days", "forward_rate", "posts_per_day",
    "mean_views", "reuse_rate_cross",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="channel_features_with_reuse.csv")
    ap.add_argument("--outdir", default="m5_outputs")
    ap.add_argument("--k", type=int, default=3,
                    help="chosen k for the reported partition")
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=6)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.decomposition import PCA

    df = pd.read_csv(args.table)
    feats = [f for f in CLUSTER_FEATURES if f in df.columns]
    X = df[feats].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median())
    Xz = StandardScaler().fit_transform(X)

    # silhouette sweep
    sweep = []
    for k in range(args.k_min, args.k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        lab = km.fit_predict(Xz)
        sil = silhouette_score(Xz, lab) if k >= 2 else np.nan
        sweep.append({"k": k, "silhouette": sil, "inertia": km.inertia_})
    sweep_df = pd.DataFrame(sweep)
    sweep_df.to_csv(os.path.join(args.outdir, "kmeans_silhouette.csv"),
                    index=False)

    # chosen-k partition
    km = KMeans(n_clusters=args.k, n_init=10, random_state=42)
    labels = km.fit_predict(Xz)
    df_out = df.copy()
    df_out["kmeans_cluster"] = labels
    df_out.to_csv(os.path.join(args.outdir, "kmeans_channel_partition.csv"),
                  index=False, encoding="utf-8-sig")
    chosen_sil = silhouette_score(Xz, labels)

    # ---- summary ----
    L = []
    def line(s=""): L.append(s)
    line("M5 K-MEANS PARTITIONING (interpretive complement)")
    line("=" * 50)
    line(f"Channels (n)   : {len(df)}")
    line(f"Features       : {feats}")
    line("")
    line("Silhouette sweep (higher = better-separated; low is EXPECTED here")
    line("and is evidence of a continuous population, not discrete groups):")
    for _, r in sweep_df.iterrows():
        mark = "  <- chosen" if int(r["k"]) == args.k else ""
        line(f"  k={int(r['k'])}: silhouette={r['silhouette']:.4f}{mark}")
    line("")
    line("Silhouette interpretation:")
    line("  > 0.5  strong separation; ~0.25-0.5 weak/moderate; < 0.25 essentially")
    line("  no real separation (partitions are arbitrary cuts of a continuum).")
    line("")
    line(f"Chosen k = {args.k}, silhouette = {chosen_sil:.4f}")
    line("")
    line("Per-cluster feature medians (raw scale):")
    prof = df_out.groupby("kmeans_cluster")[feats].median()
    line(prof.to_string())
    line("")
    line("Cluster sizes:")
    for c in sorted(set(labels)):
        line(f"  cluster {c}: {(labels == c).sum()}")
    line("")

    # mismatch across partition
    groups = [g["mismatch_mean"].dropna().values
              for _, g in df_out.groupby("kmeans_cluster")]
    if len(groups) >= 2:
        h, p = kruskal(*groups)
        mm_meds = df_out.groupby("kmeans_cluster")["mismatch_mean"].median()
        spread = mm_meds.max() - mm_meds.min()
        line(f"Kruskal-Wallis, mismatch_mean across partition: "
             f"H={h:.3f}, p={p:.4f}")
        line(f"  mismatch median spread across clusters: {spread:.4f} "
             f"(corpus std ~0.037)")
        if spread < 0.037:
            line("  -> any difference is WITHIN the mismatch noise band; not")
            line("     substantively meaningful even if statistically flagged.")
    line("")
    line("Framing: k-means imposes k partitions on a continuous gradient; it")
    line("does NOT establish discrete channel types. Report alongside the")
    line("HDBSCAN null and the PCA gradient as a consistent picture: channels")
    line("vary by degree (mainly a size/activity axis), not by kind.")

    summary = "\n".join(L)
    with open(os.path.join(args.outdir, "kmeans_summary.txt"), "w",
              encoding="utf-8") as f:
        f.write(summary + "\n")
    print(summary)

    # ---- 2D visualisation (PCA) at k=2 (best silhouette) ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        k_plot = 2                          # was 3; k=2 is the best silhouette
        km = KMeans(n_clusters=k_plot, random_state=42, n_init=10).fit(Xz)
        labels = km.labels_

        pca = PCA(n_components=2)
        xy = pca.fit_transform(Xz)
        var = pca.explained_variance_ratio_

        plt.rcParams.update({"font.family": "serif"})
        plt.figure(figsize=(7, 5.5))
        colours = ["#0072B2", "#D55E00"]    # blue, orange (colour-blind-safe)
        for c in range(k_plot):
            m = labels == c
            plt.scatter(xy[m, 0], xy[m, 1], c=colours[c], label=f"cluster {c}",
                        alpha=0.85, s=45, edgecolor="white", linewidth=0.4)
        plt.xlabel(f"PC1 ({var[0]*100:.1f}% variance)")
        plt.ylabel(f"PC2 ({var[1]*100:.1f}% variance)")
        for s in ("top", "right"):
            plt.gca().spines[s].set_visible(False)
        plt.legend(fontsize=8, frameon=False)
        plt.tight_layout()

        fig_path = os.path.join(args.outdir, "kmeans_2d.png")
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        print(f"\n2D visualisation saved: {fig_path}  (k={k_plot})")
    except ImportError:
        print("\nmatplotlib not available; skipped 2D plot.")

    print(f"\nArtifacts written to {args.outdir}/:")
    print("  kmeans_silhouette.csv, kmeans_channel_partition.csv,")
    print("  kmeans_summary.txt, kmeans_2d.png")


if __name__ == "__main__":
    main()
