"""
Compute inter-annotator agreement for the pilot from two label files.

Each label file is an annotation_labels.csv with columns:
    ann_id, label, notes, timestamp
where label is one of: propagandistic / neutral / unclear.

Reports:
  - overlap count (posts both people labelled)
  - three-class Cohen's Kappa  (the honest headline: includes 'unclear')
  - binary Cohen's Kappa       (propaganda vs neutral, 'unclear' dropped)
  - raw agreement percentages
  - a list of every disagreement, so you know which posts to discuss and which
    boundary rules to sharpen before the full pass.

Usage:
    python compute_kappa.py --me annotation_labels.csv --them annotator2_labels.csv
"""

import argparse
import pandas as pd


def load(path):
    df = pd.read_csv(path)
    df = df[df["label"].astype(str).str.strip() != ""]
    return dict(zip(df["ann_id"].astype(str), df["label"].astype(str).str.strip()))


def cohen_kappa(pairs):
    """pairs: list of (label_a, label_b). Returns kappa, po, categories."""
    cats = sorted({c for pair in pairs for c in pair})
    idx = {c: i for i, c in enumerate(cats)}
    k = len(cats)
    n = len(pairs)
    if n == 0:
        return float("nan"), float("nan"), cats

    conf = [[0] * k for _ in range(k)]
    for a, b in pairs:
        conf[idx[a]][idx[b]] += 1

    po = sum(conf[i][i] for i in range(k)) / n
    row = [sum(conf[i]) for i in range(k)]
    col = [sum(conf[i][j] for i in range(k)) for j in range(k)]
    pe = sum((row[i] / n) * (col[i] / n) for i in range(k))
    kappa = (po - pe) / (1 - pe) if (1 - pe) else float("nan")
    return kappa, po, cats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--me", default="annotation_labels.csv")
    ap.add_argument("--them", default="annotator2_labels.csv")
    args = ap.parse_args()

    a = load(args.me)
    b = load(args.them)

    shared = sorted(set(a) & set(b))
    if not shared:
        raise SystemExit("No overlapping ann_ids between the two files.")

    print(f"You labelled     : {len(a)}")
    print(f"They labelled    : {len(b)}")
    print(f"Overlap (scored) : {len(shared)}\n")

    # three-class
    pairs3 = [(a[i], b[i]) for i in shared]
    k3, po3, cats3 = cohen_kappa(pairs3)
    print(f"Three-class kappa: {k3:.3f}   (raw agreement {po3*100:.1f}%)")
    print(f"  categories: {', '.join(cats3)}")

    # binary: drop any pair where either side said 'unclear'
    pairs2 = [(a[i], b[i]) for i in shared
              if a[i] != "unclear" and b[i] != "unclear"]
    if pairs2:
        k2, po2, _ = cohen_kappa(pairs2)
        print(f"Binary kappa     : {k2:.3f}   (raw agreement {po2*100:.1f}%)"
              f"   [n={len(pairs2)}, 'unclear' dropped]")
    else:
        print("Binary kappa     : n/a (no non-unclear overlap)")

    # disagreements
    disagree = [(i, a[i], b[i]) for i in shared if a[i] != b[i]]
    print(f"\nDisagreements: {len(disagree)} of {len(shared)}")
    if disagree:
        print("  ann_id      you            them")
        for i, la, lb in disagree:
            print(f"  {i:<10}  {la:<14} {lb}")
        print("\nThese are the posts to discuss before the full pass.")
        # flag the highest-risk boundary explicitly
        mob = [d for d in disagree if "propagandistic" in (d[1], d[2])
               and "neutral" in (d[1], d[2])]
        if mob:
            print(f"({len(mob)} are propagandistic-vs-neutral splits - the "
                  f"manner-not-topic boundary, your highest-risk disagreement point.)")

    if k3 == k3 and k3 < 0.6:  # not nan and below target
        print("\nNote: three-class kappa is below 0.6. That's common on a first "
              "pilot - review the disagreements above, sharpen those specific "
              "boundary rules, then re-run the pilot or proceed with the refined "
              "criteria as your plan allows.")


if __name__ == "__main__":
    main()
