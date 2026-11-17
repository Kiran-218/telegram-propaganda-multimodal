"""
M5 channel-level Spearman correlation analysis.

Computes Spearman rank correlations between all continuous channel features on
the ~92-channel table, with p-values, and saves everything as draft-ready
artifacts. Spearman (not Pearson) because: features are skewed (participant
counts, forward rates span orders of magnitude), the relationships need not be
linear, and rank correlation is robust to the outliers we know are present
(e.g. reuse_rate_cross max 0.51).

The central question this addresses (IPP secondary contribution): does the
mismatch proxy co-occur with behavioural indicators of coordination? i.e. do
channels with higher mismatch_mean also show higher image reuse, forward
rates, etc.? Given the post-level null (AUC 0.542) and the tight mismatch
spread (std 0.037), weak mismatch correlations are the expected, honest result
-- reported as such, not tuned.

Multiple-comparison note: with many feature pairs, some p<0.05 will occur by
chance. A Benjamini-Hochberg FDR correction is applied and both raw and
corrected significance are reported, so the writeup can be honest about which
correlations survive correction.

Outputs (all under --outdir):
  spearman_matrix_rho.csv     full rho matrix
  spearman_matrix_p.csv       full p-value matrix
  spearman_pairs.csv          every pair: rho, p_raw, p_fdr, significant flags
  spearman_summary.txt        human-readable summary for the draft

Usage:
  python spearman_analysis.py \
      --table channel_features_with_reuse.csv \
      --outdir m5_outputs
"""

import argparse
import itertools
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# features to correlate (continuous only; flags excluded -- no variance).
# reuse counts n_reused_cross / n_images_hashed are components of the rate, so
# only the rate + partners breadth are used to avoid trivial self-correlation.
FEATURES = [
    "mismatch_mean",
    "mismatch_std",
    "participants_count",
    "channel_age_days",
    "forward_rate",
    "posts_per_day",
    "mean_views",
    "reuse_rate_cross",
    "n_reuse_partners",
]


def benjamini_hochberg(pvals):
    """Return FDR-corrected p-values (BH)."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    corrected = ranked * n / (np.arange(n) + 1)
    # enforce monotonicity
    corrected = np.minimum.accumulate(corrected[::-1])[::-1]
    corrected = np.clip(corrected, 0, 1)
    out = np.empty(n)
    out[order] = corrected
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="channel_features_with_reuse.csv")
    ap.add_argument("--outdir", default="m5_outputs")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.table)

    feats = [f for f in FEATURES if f in df.columns]
    missing = [f for f in FEATURES if f not in df.columns]
    sub = df[feats].apply(pd.to_numeric, errors="coerce")

    n_channels = len(sub)

    # full matrices
    rho_mat = pd.DataFrame(index=feats, columns=feats, dtype=float)
    p_mat = pd.DataFrame(index=feats, columns=feats, dtype=float)
    for a, b in itertools.product(feats, feats):
        rho, p = spearmanr(sub[a], sub[b], nan_policy="omit")
        rho_mat.loc[a, b] = rho
        p_mat.loc[a, b] = p

    rho_mat.to_csv(os.path.join(args.outdir, "spearman_matrix_rho.csv"))
    p_mat.to_csv(os.path.join(args.outdir, "spearman_matrix_p.csv"))

    # unique pairs
    pair_rows = []
    for a, b in itertools.combinations(feats, 2):
        rho, p = spearmanr(sub[a], sub[b], nan_policy="omit")
        pair_rows.append({"feature_a": a, "feature_b": b, "rho": rho, "p_raw": p})
    pairs = pd.DataFrame(pair_rows)
    pairs["p_fdr"] = benjamini_hochberg(pairs["p_raw"].values)
    pairs["sig_raw"] = pairs["p_raw"] < 0.05
    pairs["sig_fdr"] = pairs["p_fdr"] < 0.05
    pairs = pairs.sort_values("p_raw").reset_index(drop=True)
    pairs.to_csv(os.path.join(args.outdir, "spearman_pairs.csv"), index=False)

    # text summary
    lines = []
    lines.append("M5 SPEARMAN CORRELATION ANALYSIS")
    lines.append("=" * 40)
    lines.append(f"Channels (n)        : {n_channels}")
    lines.append(f"Features correlated : {len(feats)}")
    if missing:
        lines.append(f"Features missing    : {missing}")
    lines.append(f"Unique pairs        : {len(pairs)}")
    lines.append("")
    lines.append("Method: Spearman rank correlation (robust to skew/outliers,")
    lines.append("no linearity assumption). BH-FDR applied for multiple tests.")
    lines.append("")
    lines.append("Correlations significant AFTER FDR correction (p_fdr < 0.05):")
    sig = pairs[pairs["sig_fdr"]]
    if len(sig) == 0:
        lines.append("  NONE survive FDR correction.")
    else:
        for _, r in sig.iterrows():
            lines.append(f"  {r['feature_a']:>18} ~ {r['feature_b']:<18} "
                         f"rho={r['rho']:+.3f}  p_fdr={r['p_fdr']:.4f}")
    lines.append("")
    lines.append("Raw-significant but NOT surviving FDR (p_raw<0.05, p_fdr>=0.05):")
    marg = pairs[(pairs["sig_raw"]) & (~pairs["sig_fdr"])]
    if len(marg) == 0:
        lines.append("  none")
    else:
        for _, r in marg.iterrows():
            lines.append(f"  {r['feature_a']:>18} ~ {r['feature_b']:<18} "
                         f"rho={r['rho']:+.3f}  p_raw={r['p_raw']:.4f}")
    lines.append("")
    lines.append("Mismatch-proxy correlations specifically (the IPP question):")
    mm = pairs[(pairs["feature_a"] == "mismatch_mean") |
               (pairs["feature_b"] == "mismatch_mean")]
    for _, r in mm.iterrows():
        other = r["feature_b"] if r["feature_a"] == "mismatch_mean" else r["feature_a"]
        flag = "FDR-sig" if r["sig_fdr"] else ("raw-sig" if r["sig_raw"] else "ns")
        lines.append(f"  mismatch_mean ~ {other:<18} rho={r['rho']:+.3f}  "
                     f"p_raw={r['p_raw']:.4f}  [{flag}]")
    lines.append("")
    lines.append("Interpretation guidance:")
    lines.append("  Given the post-level null and tight mismatch spread, weak or")
    lines.append("  null mismatch correlations are expected and legitimate to")
    lines.append("  report. Non-mismatch correlations (e.g. among behavioural")
    lines.append("  features) may still be informative about channel structure.")

    summary = "\n".join(lines)
    with open(os.path.join(args.outdir, "spearman_summary.txt"), "w",
              encoding="utf-8") as f:
        f.write(summary + "\n")

    print(summary)
    print(f"\nArtifacts written to {args.outdir}/:")
    print("  spearman_matrix_rho.csv, spearman_matrix_p.csv,")
    print("  spearman_pairs.csv, spearman_summary.txt")


if __name__ == "__main__":
    main()
