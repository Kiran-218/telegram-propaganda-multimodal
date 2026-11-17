"""
Perceptual image-reuse: hashing + near-duplicate discovery + distance report.

Step 1 of the image-reuse feature. This script:
  1. Perceptual-hashes every image referenced by corpus_captions.csv (phash,
     16x16 -> 256-bit hash), caching hashes so re-runs are instant.
  2. Finds all near-duplicate pairs up to a generous hamming distance
     (--max-dist, default 12) using a BK-tree -- NOT a brute-force 9k x 9k
     comparison.
  3. Tags each pair as cross-channel (different channels) or same-channel.
  4. Prints the hamming-distance distribution, split by cross/same channel,
     so the reuse threshold can be chosen on evidence rather than guessed.

It does NOT yet compute the per-channel reuse rate -- that's step 2, after
you've seen the distribution and picked a threshold.

Reuse definition (IPP): the coordination signal of interest is CROSS-CHANNEL
reuse -- the same image in two or more different channels. Caption divergence
and forward relationships are reported descriptively later; note that forward
SOURCES were not captured in the scrape (only forward counts), so the
"no forward relationship" filter can only be applied partially.

Outputs:
  image_hashes.csv        ann_id, channel, post_id, image_path, phash (hex)
  reuse_pairs.csv         all near-dup pairs <= max_dist, with distance and
                          cross_channel flag
  (distance histogram printed to stdout)

Usage:
  python image_reuse_hash.py \
      --csv corpus_captions.csv \
      --root "C:\\Users\\kiran\\OneDrive - University of Edinburgh\\Dissertation" \
      --max-dist 12

Requires: pip install imagehash pillow
"""

import argparse
import os
from collections import Counter, defaultdict

import pandas as pd
from PIL import Image

try:
    import imagehash
except ImportError:
    raise SystemExit("Need imagehash: pip install imagehash pillow")

Image.MAX_IMAGE_PIXELS = None  # silence decompression-bomb warnings on big imgs

HASH_SIZE = 16   # 16x16 -> 256-bit phash; more discriminative than default 8x8


def norm(p):
    return str(p).replace("\\", "/")


# ---------------------------------------------------------------------------
# BK-tree for hamming-distance near-neighbour search
# ---------------------------------------------------------------------------
class BKTree:
    def __init__(self):
        self.tree = None  # (hash_int, {dist: subtree})

    @staticmethod
    def _ham(a, b):
        return bin(a ^ b).count("1")

    def add(self, h):
        if self.tree is None:
            self.tree = (h, {})
            return
        node_h, children = self.tree
        while True:
            d = self._ham(h, node_h)
            if d in children:
                node_h, children = children[d]
            else:
                children[d] = (h, {})
                return

    def query(self, h, max_dist):
        if self.tree is None:
            return []
        out = []
        stack = [self.tree]
        while stack:
            node_h, children = stack.pop()
            d = self._ham(h, node_h)
            if d <= max_dist:
                out.append((node_h, d))
            for edge_d, child in children.items():
                if abs(edge_d - d) <= max_dist:
                    stack.append(child)
        return out


def compute_hashes(df, root, cache_path):
    """Hash every referenced image, caching to cache_path."""
    cache = {}
    if os.path.exists(cache_path):
        prev = pd.read_csv(cache_path)
        cache = dict(zip(prev["ann_id"].astype(str), prev["phash"]))
        print(f"Loaded {len(cache)} cached hashes.")

    rows = []
    n_new, n_missing, n_err = 0, 0, 0
    for _, r in df.iterrows():
        aid = str(r["ann_id"])
        rel = norm(r["image_path"])
        if aid in cache and isinstance(cache[aid], str):
            rows.append({"ann_id": aid, "channel": r["channel"],
                         "post_id": r["post_id"], "image_path": rel,
                         "phash": cache[aid]})
            continue
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            n_missing += 1
            continue
        try:
            with Image.open(path) as im:
                h = imagehash.phash(im.convert("RGB"), hash_size=HASH_SIZE)
            rows.append({"ann_id": aid, "channel": r["channel"],
                         "post_id": r["post_id"], "image_path": rel,
                         "phash": str(h)})
            n_new += 1
        except Exception:
            n_err += 1
        if (n_new % 500) == 0 and n_new:
            print(f"  hashed {n_new} new ...", end="\r")

    out = pd.DataFrame(rows)
    out.to_csv(cache_path, index=False, encoding="utf-8-sig")
    print(f"\nHashes: {len(out)} total ({n_new} new, {n_missing} missing, "
          f"{n_err} errored). Cached -> {cache_path}")
    return out


def hex_to_int(hexstr):
    return int(hexstr, 16)


def find_pairs(hashes_df, max_dist):
    """BK-tree near-dup search. Returns a DataFrame of pairs <= max_dist."""
    ann_ids = hashes_df["ann_id"].tolist()
    channels = hashes_df["channel"].tolist()
    hints = [hex_to_int(h) for h in hashes_df["phash"].tolist()]

    tree = BKTree()
    # map hash-int -> list of indices sharing it (dups collapse in the tree,
    # so keep our own index by value)
    by_hash = defaultdict(list)
    for i, hv in enumerate(hints):
        by_hash[hv].append(i)

    pairs = []
    added = set()
    for hv in by_hash:
        tree.add(hv)

    seen_pair = set()
    for i, hv in enumerate(hints):
        neighbours = tree.query(hv, max_dist)
        for nh, d in neighbours:
            for j in by_hash[nh]:
                if j <= i:
                    continue
                key = (i, j)
                if key in seen_pair:
                    continue
                seen_pair.add(key)
                pairs.append({
                    "ann_id_a": ann_ids[i], "channel_a": channels[i],
                    "ann_id_b": ann_ids[j], "channel_b": channels[j],
                    "distance": d,
                    "cross_channel": channels[i] != channels[j],
                })
    return pd.DataFrame(pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="corpus_captions.csv")
    ap.add_argument("--root", default=".",
                    help="folder that CONTAINS the media/ tree")
    ap.add_argument("--max-dist", type=int, default=12)
    ap.add_argument("--hash-cache", default="image_hashes.csv")
    ap.add_argument("--pairs-out", default="reuse_pairs.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    hashes = compute_hashes(df, args.root, args.hash_cache)

    print(f"\nSearching near-duplicate pairs (hamming <= {args.max_dist}) ...")
    pairs = find_pairs(hashes, args.max_dist)
    pairs.to_csv(args.pairs_out, index=False, encoding="utf-8-sig")
    print(f"Found {len(pairs)} pairs. Written -> {args.pairs_out}")

    # ---- distance distribution report ----
    print("\nHamming-distance distribution (choose threshold from this):")
    print(f"{'dist':>4} | {'cross-chan':>10} | {'same-chan':>9} | {'total':>6}")
    print("-" * 40)
    cross = Counter(pairs[pairs["cross_channel"]]["distance"])
    same = Counter(pairs[~pairs["cross_channel"]]["distance"])
    cum_cross = 0
    for d in range(0, args.max_dist + 1):
        c, s = cross.get(d, 0), same.get(d, 0)
        cum_cross += c
        print(f"{d:>4} | {c:>10} | {s:>9} | {c+s:>6}   (cum cross-chan: {cum_cross})")

    n_cross_channels = pairs[pairs["cross_channel"]]["channel_a"].nunique()
    print(f"\nCross-channel pairs involve images from multiple channels.")
    print("Look for a GAP in the cross-channel column: true duplicates cluster")
    print("at low distances (0-6), coincidental matches rise again later. Pick")
    print("the threshold at the gap. Then run the per-channel reuse-rate step.")


if __name__ == "__main__":
    main()
