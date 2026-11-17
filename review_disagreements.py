"""
Build a side-by-side review of the posts where the two annotators disagreed.

Joins the pilot captions with both label files, keeps only the disagreements,
and prints (and writes to CSV) each post's caption, image path, both labels,
and both annotators' notes -- so the calibration discussion works from the
concrete posts rather than in the abstract.

Usage:
    python review_disagreements.py \
        --pilot pilot_30.csv \
        --me annotation_labels.csv \
        --them annotator2_labels.csv

Writes:
    disagreements_review.csv
"""

import argparse
import pandas as pd


def load_labels(path, who):
    df = pd.read_csv(path)
    df["ann_id"] = df["ann_id"].astype(str).str.strip()
    keep = df[["ann_id", "label"]].rename(columns={"label": f"label_{who}"})
    if "notes" in df.columns:
        keep[f"notes_{who}"] = df["notes"].fillna("")
    else:
        keep[f"notes_{who}"] = ""
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", default="pilot_30.csv")
    ap.add_argument("--me", default="annotation_labels.csv")
    ap.add_argument("--them", default="annotator2_labels.csv")
    ap.add_argument("--out", default="disagreements_review.csv")
    args = ap.parse_args()

    pilot = pd.read_csv(args.pilot)
    pilot["ann_id"] = pilot["ann_id"].astype(str).str.strip()
    # caption_en if present, else caption
    cap_col = "caption_en" if "caption_en" in pilot.columns else "caption"
    pilot_keep = pilot[["ann_id", "channel", "image_path", cap_col]].rename(
        columns={cap_col: "caption"})

    me = load_labels(args.me, "you")
    them = load_labels(args.them, "them")

    m = pilot_keep.merge(me, on="ann_id").merge(them, on="ann_id")
    dis = m[m["label_you"] != m["label_them"]].copy()
    dis = dis.sort_values("ann_id").reset_index(drop=True)

    print(f"{len(dis)} disagreements of {len(m)} posts\n")
    print("=" * 70)
    for r in dis.itertuples():
        print(f"\n[{r.ann_id}]  channel: {r.channel}")
        print(f"  YOU : {r.label_you:<14}  THEM: {r.label_them}")
        print(f"  image: {r.image_path}")
        cap = str(r.caption).replace("\n", " ").strip()
        print(f"  caption: {cap[:400]}")
        if str(r.notes_you).strip():
            print(f"  your note : {str(r.notes_you).strip()[:200]}")
        if str(r.notes_them).strip():
            print(f"  their note: {str(r.notes_them).strip()[:200]}")
        print("-" * 70)

    # categorise for the discussion
    def kind(a, b):
        s = {a, b}
        if s == {"neutral", "propagandistic"}:
            return "manner-not-topic (prop vs neutral)"
        if "unclear" in s:
            return "unclear-threshold"
        return "other"

    dis["disagreement_type"] = [kind(a, b) for a, b in
                                zip(dis["label_you"], dis["label_them"])]

    print("\nDisagreement types:")
    for t, n in dis["disagreement_type"].value_counts().items():
        print(f"  {t}: {n}")

    dis.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"\nWritten: {args.out}")


if __name__ == "__main__":
    main()
