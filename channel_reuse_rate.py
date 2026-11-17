"""
Per-channel cross-channel image-reuse rate (step 2 of image reuse).

Threshold LOCKED at hamming <= 6, chosen at the empirical gap in the
distance distribution (cross-channel pairs: 407@0, 90@2, 12@4, 3@6, then a
trough before a coincidental rebound at 10).

Defines, per channel:
  - n_images_hashed        images successfully hashed for the channel
  - n_reused_cross         images in this channel that have >=1 cross-channel
                           near-duplicate (hamming <= THRESH) in ANOTHER channel
  - reuse_rate_cross       n_reused_cross / n_images_hashed   (the FEATURE)
  - n_reuse_partners       number of distinct OTHER channels this channel
                           shares an image with (a coordination-breadth proxy)

Also produces a descriptive caption-divergence report on the cross-channel
reuse groups: of the reused image pairs, how many carry meaningfully different
captions (consistent with repurposing rather than plain resharing). NOTE: the
scrape captured forward COUNTS, not forward SOURCES, so the IPP's
"no forward relationship" filter cannot be fully applied; this is reported
honestly as a limitation, and caption divergence is the available proxy.

The reuse_rate_cross column joins onto channel_features.csv as the final M5
feature, ready for Spearman correlations and HDBSCAN clustering.

Usage:
  python channel_reuse_rate.py \
      --pairs reuse_pairs.csv \
      --hashes image_hashes.csv \
      --captions corpus_captions.csv \
      --channel-table channel_features.csv \
      --thresh 6 \
      --out channel_features_with_reuse.csv
"""

import argparse
import re
from collections import defaultdict

import pandas as pd

_URL_RE = re.compile(r"https?://\S+|t\.me/\S+")
_WS_RE = re.compile(r"\s+")


def light_clean(text):
    if not isinstance(text, str):
        return ""
    return _WS_RE.sub(" ", _URL_RE.sub(" ", text)).strip().lower()


def jaccard(a, b):
    sa, sb = set(a.split()), set(b.split())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="reuse_pairs.csv")
    ap.add_argument("--hashes", default="image_hashes.csv")
    ap.add_argument("--captions", default="corpus_captions.csv")
    ap.add_argument("--channel-table", default="channel_features.csv")
    ap.add_argument("--thresh", type=int, default=6)
    ap.add_argument("--out", default="channel_features_with_reuse.csv")
    ap.add_argument("--caption-divergence", type=float, default=0.5,
                    help="jaccard BELOW this = 'divergent captions' (descriptive)")
    args = ap.parse_args()

    pairs = pd.read_csv(args.pairs)
    hashes = pd.read_csv(args.hashes)

    # apply the locked threshold + cross-channel filter
    xc = pairs[(pairs["distance"] <= args.thresh) & (pairs["cross_channel"])].copy()
    print(f"Cross-channel pairs at hamming <= {args.thresh}: {len(xc)}")

    # per-channel denominator: images hashed per channel
    imgs_per_channel = hashes.groupby("channel")["ann_id"].nunique().to_dict()

    # which images (ann_ids) are involved in a cross-channel reuse, per channel;
    # and which OTHER channels each channel partners with
    reused_ann = defaultdict(set)      # channel -> set of its ann_ids reused
    partners = defaultdict(set)        # channel -> set of other channels
    for _, r in xc.iterrows():
        ca, cb = r["channel_a"], r["channel_b"]
        reused_ann[ca].add(r["ann_id_a"])
        reused_ann[cb].add(r["ann_id_b"])
        partners[ca].add(cb)
        partners[cb].add(ca)

    rows = []
    for ch, n_img in imgs_per_channel.items():
        n_reused = len(reused_ann.get(ch, set()))
        rows.append({
            "channel": ch,
            "n_images_hashed": n_img,
            "n_reused_cross": n_reused,
            "reuse_rate_cross": (n_reused / n_img) if n_img else 0.0,
            "n_reuse_partners": len(partners.get(ch, set())),
        })
    reuse_df = pd.DataFrame(rows)

    # join onto the channel table (left join: keep all analysis channels)
    tbl = pd.read_csv(args.channel_table)
    merged = tbl.merge(reuse_df, on="channel", how="left")
    # channels with no reuse get 0, not NaN
    for c in ["n_images_hashed", "n_reused_cross", "reuse_rate_cross",
              "n_reuse_partners"]:
        merged[c] = merged[c].fillna(0)
    merged.to_csv(args.out, index=False, encoding="utf-8-sig")

    # ---- descriptive caption-divergence report on reuse pairs ----
    caps = pd.read_csv(args.captions)[["ann_id", "caption"]]
    cap_map = dict(zip(caps["ann_id"].astype(str),
                       caps["caption"].map(light_clean)))
    div, tot = 0, 0
    for _, r in xc.iterrows():
        a = cap_map.get(str(r["ann_id_a"]), "")
        b = cap_map.get(str(r["ann_id_b"]), "")
        tot += 1
        if jaccard(a, b) < args.caption_divergence:
            div += 1

    print(f"\nChannels with any cross-channel reuse: "
          f"{(reuse_df['n_reused_cross'] > 0).sum()} / {len(reuse_df)}")
    print(f"reuse_rate_cross: mean={merged['reuse_rate_cross'].mean():.4f} "
          f"max={merged['reuse_rate_cross'].max():.4f}")
    print(f"\nCaption divergence on cross-channel reuse pairs (descriptive):")
    print(f"  {div}/{tot} pairs have divergent captions "
          f"(jaccard < {args.caption_divergence})")
    print("  -> consistent with repurposing rather than plain resharing.")
    print("  LIMITATION: forward sources not captured in scrape, so the")
    print("  'no forward relationship' filter could not be fully applied.")
    print(f"\nWritten: {args.out}")
    print("This table is the M5 feature set. Next: Spearman + HDBSCAN.")


if __name__ == "__main__":
    main()
