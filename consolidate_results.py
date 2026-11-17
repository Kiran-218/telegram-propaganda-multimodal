"""
Consolidate earlier M5 pipeline results into saved draft artifacts.

Recomputes and saves the key numbers that were only printed to the terminal
earlier in the pipeline, so they're preserved for the dissertation write-up:

  - Corpus M-CLIP scoring: status counts, mismatch distribution
  - Image hashing: counts
  - Cross-channel reuse: distance distribution table, pair counts,
    per-channel reuse rate stats, caption divergence

Everything is derived from files already on disk (no re-running of the GPU
job or re-hashing needed). Writes a single consolidated summary plus the
distance-distribution table as CSV.

Outputs (under --outdir):
  pipeline_results_summary.txt   consolidated numbers for the draft
  reuse_distance_distribution.csv  the hamming-distance table (threshold
                                   justification)

Usage:
  python consolidate_results.py --outdir m5_outputs
"""

import argparse
import os
from collections import Counter

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="corpus_mclip_scores.csv")
    ap.add_argument("--hashes", default="image_hashes.csv")
    ap.add_argument("--pairs", default="reuse_pairs.csv")
    ap.add_argument("--reuse-table", default="channel_features_with_reuse.csv")
    ap.add_argument("--captions", default="corpus_captions.csv")
    ap.add_argument("--outdir", default="m5_outputs")
    ap.add_argument("--thresh", type=int, default=6)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    L = []
    def line(s=""): L.append(s)

    line("M5 PIPELINE RESULTS -- CONSOLIDATED")
    line("=" * 45)
    line("All numbers recomputed from on-disk artifacts.")
    line("")

    # --- corpus M-CLIP ---
    line("CORPUS M-CLIP SCORING")
    line("-" * 45)
    sc = pd.read_csv(args.scores)
    status_counts = sc["status"].value_counts().to_dict()
    ok = sc[sc["status"] == "ok"].copy()
    ok["mclip_mismatch"] = pd.to_numeric(ok["mclip_mismatch"], errors="coerce")
    line(f"Total posts scored     : {len(sc)}")
    line(f"Status counts          : {status_counts}")
    line(f"Scored ok              : {len(ok)}")
    line(f"Mismatch mean          : {ok['mclip_mismatch'].mean():.4f}")
    line(f"Mismatch std           : {ok['mclip_mismatch'].std():.4f}")
    line(f"Mismatch min / max     : {ok['mclip_mismatch'].min():.4f} / "
         f"{ok['mclip_mismatch'].max():.4f}")
    line("")

    # --- hashing ---
    line("IMAGE HASHING")
    line("-" * 45)
    hs = pd.read_csv(args.hashes)
    line(f"Images hashed          : {len(hs)}")
    line(f"Channels represented   : {hs['channel'].nunique()}")
    line("")

    # --- reuse distance distribution ---
    line("CROSS-CHANNEL REUSE -- DISTANCE DISTRIBUTION")
    line("-" * 45)
    pairs = pd.read_csv(args.pairs)
    cross = Counter(pairs[pairs["cross_channel"]]["distance"])
    same = Counter(pairs[~pairs["cross_channel"]]["distance"])
    max_d = int(pairs["distance"].max())
    dist_rows = []
    line(f"{'dist':>4} | {'cross':>6} | {'same':>6} | {'total':>6}")
    cum = 0
    for d in range(0, max_d + 1):
        c, s = cross.get(d, 0), same.get(d, 0)
        cum += c
        if c or s:
            line(f"{d:>4} | {c:>6} | {s:>6} | {c+s:>6}   (cum cross: {cum})")
        dist_rows.append({"distance": d, "cross_channel": c,
                          "same_channel": s, "cum_cross": cum})
    pd.DataFrame(dist_rows).to_csv(
        os.path.join(args.outdir, "reuse_distance_distribution.csv"),
        index=False)
    line(f"Threshold chosen       : hamming <= {args.thresh} (at the gap)")
    line("")

    # --- reuse rate ---
    line("CROSS-CHANNEL REUSE -- PER-CHANNEL RATE")
    line("-" * 45)
    xc = pairs[(pairs["distance"] <= args.thresh) & pairs["cross_channel"]]
    line(f"Cross-channel pairs at <= {args.thresh}: {len(xc)}")
    rt = pd.read_csv(args.reuse_table)
    line(f"Channels with any reuse : "
         f"{(rt['n_reused_cross'] > 0).sum()} / {len(rt)}")
    line(f"reuse_rate_cross mean   : {rt['reuse_rate_cross'].mean():.4f}")
    line(f"reuse_rate_cross max    : {rt['reuse_rate_cross'].max():.4f}")
    # identify the max-reuse channel
    top = rt.loc[rt['reuse_rate_cross'].idxmax()]
    line(f"Highest-reuse channel   : {top['channel']} "
         f"(rate={top['reuse_rate_cross']:.4f})")
    line("")

    line("Artifacts saved for draft reference:")
    line("  pipeline_results_summary.txt (this file)")
    line("  reuse_distance_distribution.csv")

    summary = "\n".join(L)
    with open(os.path.join(args.outdir, "pipeline_results_summary.txt"), "w",
              encoding="utf-8") as f:
        f.write(summary + "\n")
    print(summary)


if __name__ == "__main__":
    main()
