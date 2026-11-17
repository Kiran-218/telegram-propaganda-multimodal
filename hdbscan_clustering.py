"""
M5 channel-level HDBSCAN clustering (supplementary, exploratory).

Clusters the ~92 channels on standardised behavioural + mismatch features to
see whether natural groupings emerge. Per the IPP this is a SUPPLEMENTARY
unsupervised step, not a core claim.

Feature set (standardised / z-scored -- features span very different scales):
    mismatch_mean, mismatch_std, participants_count, channel_age_days,
    forward_rate, posts_per_day, mean_views, reuse_rate_cross
Deliberately EXCLUDED:
    n_reuse_partners  (rho 0.91 with reuse_rate_cross -- redundant, would
                       double-weight the reuse axis)
    mismatch_var      (redundant with mismatch_std)
    verified/scam/fake/restricted (no variance across the sample)

Honest caveats (stated in the summary):
    - n=92 is small for density-based clustering; HDBSCAN may find few/no
      clusters and label many channels as noise. That is a valid result.
    - DBCV is noisy at small n; reported but treated as exploratory.
    - min_cluster_size is NOT tuned down to manufacture clusters.

After clustering, checks whether clusters differ on mismatch_mean (the proxy)
via a Kruskal-Wallis test -- i.e. does channel structure track mismatch?
(Expected: no, consistent with the post-level null.)

Outputs (under --outdir):
    channel_clusters.csv         channel + cluster label + all features
    cluster_summary.txt          DBCV, cluster sizes, per-cluster profiles
    channel_clusters_2d.png      2D PCA scatter coloured by cluster

Usage:
  python hdbscan_clustering.py \
      --table channel_features_with_reuse.csv \
      --outdir m5_outputs \
      --min-cluster-size 5
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
    ap.add_argument("--min-cluster-size", type=int, default=5)
    ap.add_argument("--min-samples", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    try:
        import hdbscan
    except ImportError:
        raise SystemExit("Need hdbscan: pip install hdbscan")
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    df = pd.read_csv(args.table)
    feats = [f for f in CLUSTER_FEATURES if f in df.columns]
    X = df[feats].apply(pd.to_numeric, errors="coerce").fillna(
        df[feats].median())

    Xz = StandardScaler().fit_transform(X)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        metric="euclidean",
        gen_min_span_tree=True,
    )
    labels = clusterer.fit_predict(Xz)
    df_out = df.copy()
    df_out["cluster"] = labels

    # DBCV (HDBSCAN exposes it as relative_validity_)
    try:
        dbcv = clusterer.relative_validity_
    except Exception:
        dbcv = float("nan")

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())

    df_out.to_csv(os.path.join(args.outdir, "channel_clusters.csv"),
                  index=False, encoding="utf-8-sig")

    # ---- summary ----
    L = []
    def line(s=""): L.append(s)
    line("M5 HDBSCAN CLUSTERING (supplementary, exploratory)")
    line("=" * 50)
    line(f"Channels (n)        : {len(df)}")
    line(f"Features (standardised): {feats}")
    line(f"min_cluster_size    : {args.min_cluster_size}")
    line(f"Clusters found      : {n_clusters}")
    line(f"Noise channels      : {n_noise} / {len(df)}")
    line(f"DBCV (relative_validity_): {dbcv:.4f}")
    line("")
    line("NOTE: n=92 is small for density clustering. Few/no clusters and")
    line("substantial noise are valid outcomes, not failures. DBCV is noisy")
    line("at this n; treat clustering as exploratory/supplementary.")
    line("")

    # cluster sizes
    line("Cluster sizes:")
    for lab in sorted(set(labels)):
        name = "noise" if lab == -1 else f"cluster {lab}"
        line(f"  {name:>10}: {(labels == lab).sum()}")
    line("")

    # per-cluster feature profiles (median, on raw scale)
    if n_clusters > 0:
        line("Per-cluster feature medians (raw scale):")
        prof = df_out.groupby("cluster")[feats].median()
        line(prof.to_string())
        line("")

        # does channel structure track the mismatch proxy?
        groups = [g["mismatch_mean"].dropna().values
                  for _, g in df_out[df_out["cluster"] != -1].groupby("cluster")]
        if len(groups) >= 2 and all(len(g) > 0 for g in groups):
            try:
                h, p = kruskal(*groups)
                line(f"Kruskal-Wallis, mismatch_mean across clusters: "
                     f"H={h:.3f}, p={p:.4f}")
                if p >= 0.05:
                    line("  -> clusters do NOT differ significantly on mismatch;")
                    line("     channel structure is not driven by the proxy,")
                    line("     consistent with the post-level null.")
                else:
                    line("  -> clusters differ on mismatch; report cautiously")
                    line("     given weak/tight mismatch distribution.")
            except Exception as e:
                line(f"Kruskal-Wallis could not be computed: {e}")
    else:
        line("No clusters found -- channels do not form dense natural groups")
        line("on these features. This is a legitimate finding: the channel")
        line("population is fairly continuous rather than discretely grouped.")

    summary = "\n".join(L)
    with open(os.path.join(args.outdir, "cluster_summary.txt"), "w",
              encoding="utf-8") as f:
        f.write(summary + "\n")
    print(summary)

    # ---- 2D visualisation (PCA) ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        pca = PCA(n_components=2)
        xy = pca.fit_transform(Xz)
        var = pca.explained_variance_ratio_

        plt.rcParams.update({"font.family": "serif"})
        plt.figure(figsize=(7, 5.5))

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters == 0:
            # all channels are noise: plot as one set, no legend
            plt.scatter(xy[:, 0], xy[:, 1], c="#0072B2",
                        alpha=0.85, s=45, edgecolor="white", linewidth=0.4)
        else:
            # (kept for completeness if run at other settings)
            for lab in sorted(set(labels)):
                mask = labels == lab
                if lab == -1:
                    plt.scatter(xy[mask, 0], xy[mask, 1], c="#BBBBBB",
                                marker="o", label="unclustered",
                                alpha=0.75, s=40, edgecolor="white", linewidth=0.4)
                else:
                    plt.scatter(xy[mask, 0], xy[mask, 1], label=f"cluster {lab}",
                                alpha=0.85, s=55, edgecolor="white", linewidth=0.4)
            plt.legend(fontsize=8, frameon=False)

        plt.xlabel(f"PC1 ({var[0]*100:.1f}% variance)")
        plt.ylabel(f"PC2 ({var[1]*100:.1f}% variance)")
        for s in ("top", "right"):
            plt.gca().spines[s].set_visible(False)
        plt.tight_layout()

        fig_path = os.path.join(args.outdir, "channel_clusters_2d.png")
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        print(f"\n2D visualisation saved: {fig_path}")
        print(f"(PCA captures {sum(var)*100:.1f}% of variance in 2D)")
    except ImportError:
        print("\nmatplotlib not available; skipped 2D plot "
              "(pip install matplotlib to enable).")

    print(f"\nArtifacts written to {args.outdir}/:")
    print("  channel_clusters.csv, cluster_summary.txt, channel_clusters_2d.png")


if __name__ == "__main__":
    main()
