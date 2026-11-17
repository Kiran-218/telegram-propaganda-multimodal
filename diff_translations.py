"""
Compare the 600M and 1.3B NLLB translations of the same captions.

Joins the two translated CSVs on ann_id and reports, for each row, how much
the English differs -- so you can eyeball where the larger model actually
changed something before deciding which version feeds annotation.

Usage:
    python diff_translations.py \
        --a annotation_sample_translated.csv \
        --b annotation_sample_translated_1.3B.csv \
        --top 30

Difference metric: token-level Jaccard distance (1 - overlap) on lowercased
word sets, plus a raw character-length delta. Simple, dependency-free, and
good enough to surface the captions worth reading side by side.
"""

import argparse
import pandas as pd


def tokens(s):
    return set(str(s).lower().split())


def jaccard_distance(a, b):
    ta, tb = tokens(a), tokens(b)
    if not ta and not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return 1.0 - (inter / union if union else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="annotation_sample_translated.csv",
                    help="600M translation CSV")
    ap.add_argument("--b", default="annotation_sample_translated_1.3B.csv",
                    help="1.3B translation CSV")
    ap.add_argument("--top", type=int, default=30,
                    help="how many biggest-difference rows to print")
    ap.add_argument("--out", default="translation_diff.csv",
                    help="full per-row diff written here")
    args = ap.parse_args()

    a = pd.read_csv(args.a)[["ann_id", "caption_en"]].rename(
        columns={"caption_en": "en_600M"})
    b = pd.read_csv(args.b)[["ann_id", "caption_en"]].rename(
        columns={"caption_en": "en_1.3B"})

    # normalise the join key: strip whitespace/BOM, force to string, so the
    # merge can't silently misalign between files written by different scripts
    a["ann_id"] = a["ann_id"].astype(str).str.strip()
    b["ann_id"] = b["ann_id"].astype(str).str.strip()

    # alignment sanity check: how many ann_ids actually match?
    ids_a, ids_b = set(a["ann_id"]), set(b["ann_id"])
    print(f"ann_ids in 600M file : {len(ids_a)}")
    print(f"ann_ids in 1.3B file : {len(ids_b)}")
    print(f"ann_ids in common    : {len(ids_a & ids_b)}")
    if len(ids_a & ids_b) < 0.5 * max(len(ids_a), len(ids_b)):
        print("WARNING: few shared ann_ids - the files may not align on ann_id.")
    print()

    m = a.merge(b, on="ann_id", how="inner")
    m["en_600M"] = m["en_600M"].fillna("")
    m["en_1.3B"] = m["en_1.3B"].fillna("")

    # post-merge diagnostic: confirm both sides actually carry text
    print(f"after merge: 600M non-empty = "
          f"{m['en_600M'].str.strip().ne('').sum()}, "
          f"1.3B non-empty = {m['en_1.3B'].str.strip().ne('').sum()}\n")

    m["jaccard_dist"] = [jaccard_distance(x, y)
                         for x, y in zip(m["en_600M"], m["en_1.3B"])]
    m["char_delta"] = m["en_1.3B"].str.len() - m["en_600M"].str.len()
    m["identical"] = m["en_600M"].str.strip() == m["en_1.3B"].str.strip()

    n = len(m)
    n_same = int(m["identical"].sum())
    print(f"Rows compared        : {n}")
    print(f"Identical            : {n_same}  ({100*n_same/n:.1f}%)")
    print(f"Differ at all        : {n - n_same}  ({100*(n-n_same)/n:.1f}%)")
    print(f"Mean Jaccard distance: {m['jaccard_dist'].mean():.3f}")
    print()

    top = m.sort_values("jaccard_dist", ascending=False).head(args.top)
    print(f"--- Top {args.top} biggest differences ---\n")
    for _, r in top.iterrows():
        if r["identical"]:
            continue
        print(f"[{r['ann_id']}]  jaccard={r['jaccard_dist']:.2f}  "
              f"char_delta={int(r['char_delta']):+d}")
        print(f"  600M: {str(r['en_600M'])[:160]}")
        print(f"  1.3B: {str(r['en_1.3B'])[:160]}")
        print()

    m.sort_values("jaccard_dist", ascending=False).to_csv(args.out, index=False,
                                                           encoding="utf-8-sig")
    print(f"Full diff written: {args.out}")


if __name__ == "__main__":
    main()
