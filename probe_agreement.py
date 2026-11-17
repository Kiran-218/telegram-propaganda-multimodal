#!/usr/bin/env python3
"""
probe_agreement.py -- Where do the text and image probes agree and disagree?

Reads the per-post out-of-fold probabilities (from probe_predictions.py) and
answers two questions:

1. AGREEMENT: do the two modalities carry the same signal or different signal?
   - correlation between p_text and p_image across all posts
   - a 2x2 correct/incorrect cross-tab at the 0.5 threshold:
       both right | both wrong | text-right/image-wrong | image-right/text-wrong
   If "both right" + "both wrong" dominate -> modalities are redundant (they
   see the same posts), which explains why fusion does not beat text alone.
   If the off-diagonal cells are large -> the modalities are complementary.

2. DISAGREEMENT DUMP: the posts where the two probes most disagree, with the
   ENGLISH caption and image path attached, so you can look at what each
   modality is keying on. Sorted by |p_text - p_image| (biggest disagreement
   first). Posts whose NLLB translation was flagged as low-quality are marked
   so you know which English captions to trust.

Inputs:
  --pred     output/probe_per_post.csv                 (ann_id,y_true,p_text,p_image)
  --src      data/annotation_sample_translated_1.3B.csv (ann_id,caption_en,image_path)
  --tqflags  output/gold_tq_flags.csv                  (id/ann_id, tq_exclude)

Outputs:
  output/agreement_summary.txt        the stats, for the write-up
  output/probe_disagreement.csv       disagreement posts w/ English captions
"""
import argparse, os, sys
import numpy as np
import pandas as pd


def norm_cols(df):
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="output/probe_per_post.csv")
    ap.add_argument("--src", default="data/annotation_sample_translated_1.3B.csv")
    ap.add_argument("--tqflags", default="output/gold_tq_flags.csv")
    ap.add_argument("--outdir", default="output")
    ap.add_argument("--top", type=int, default=40,
                    help="how many top-disagreement posts to dump")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    pred = norm_cols(pd.read_csv(args.pred))
    pred["ann_id"] = pred["ann_id"].astype(str).str.strip()

    src = norm_cols(pd.read_csv(args.src))
    src["ann_id"] = src["ann_id"].astype(str).str.strip()
    keep = ["ann_id"]
    for c in ("caption_en", "image_path", "channel"):
        if c in src.columns:
            keep.append(c)
    src = src[keep]

    tq = norm_cols(pd.read_csv(args.tqflags))
    if "ann_id" not in tq.columns and "id" in tq.columns:
        tq = tq.rename(columns={"id": "ann_id"})
    tq["ann_id"] = tq["ann_id"].astype(str).str.strip()
    tq = tq[["ann_id", "tq_exclude"]]

    df = pred.merge(src, on="ann_id", how="left").merge(tq, on="ann_id", how="left")
    df["tq_exclude"] = df["tq_exclude"].fillna(0).astype(int)

    # binary predictions at 0.5
    df["pred_text"] = (df["p_text"] >= 0.5).astype(int)
    df["pred_image"] = (df["p_image"] >= 0.5).astype(int)
    df["text_correct"] = (df["pred_text"] == df["y_true"])
    df["image_correct"] = (df["pred_image"] == df["y_true"])

    # ---- agreement stats ----
    r_pearson = df["p_text"].corr(df["p_image"], method="pearson")
    r_spearman = df["p_text"].corr(df["p_image"], method="spearman")

    both_right = int((df["text_correct"] & df["image_correct"]).sum())
    both_wrong = int((~df["text_correct"] & ~df["image_correct"]).sum())
    text_only = int((df["text_correct"] & ~df["image_correct"]).sum())
    image_only = int((~df["text_correct"] & df["image_correct"]).sum())
    n = len(df)

    L = []
    L.append("TEXT vs IMAGE PROBE -- agreement analysis")
    L.append("=" * 50)
    L.append(f"n posts = {n}")
    L.append("")
    L.append("Probability correlation (do the probes rank posts alike?)")
    L.append(f"  Pearson  r = {r_pearson:.3f}")
    L.append(f"  Spearman r = {r_spearman:.3f}")
    L.append("")
    L.append("Correct / incorrect cross-tab @ 0.5 threshold")
    L.append(f"  both correct              : {both_right:3d}  ({100*both_right/n:.1f}%)")
    L.append(f"  both wrong                : {both_wrong:3d}  ({100*both_wrong/n:.1f}%)")
    L.append(f"  text correct, image wrong : {text_only:3d}  ({100*text_only/n:.1f}%)")
    L.append(f"  image correct, text wrong : {image_only:3d}  ({100*image_only/n:.1f}%)")
    L.append("")
    agree_frac = (both_right + both_wrong) / n
    L.append(f"  agreement (same outcome)  : {100*agree_frac:.1f}%")
    L.append("")
    L.append("how to read this")
    L.append("-" * 50)
    if agree_frac >= 0.75 and r_spearman >= 0.5:
        L.append("  The probes largely agree per-post and their probabilities are")
        L.append("  correlated -> the two modalities are picking up much the SAME")
        L.append("  signal (likely topic/content that surfaces in both caption and")
        L.append("  image). This explains why fusion does not beat text alone: the")
        L.append("  image adds little INDEPENDENT information over the text.")
    else:
        L.append("  The probes disagree on a substantial share of posts and/or their")
        L.append("  probabilities are only weakly correlated -> the modalities carry")
        L.append("  partly INDEPENDENT signal. That the (stronger) text probe still")
        L.append("  wins overall, despite this independence, suggests the linear")
        L.append("  fusion could not exploit the image's separate contribution at")
        L.append("  this sample size.")

    summary = "\n".join(L)
    with open(f"{args.outdir}/agreement_summary.txt", "w") as f:
        f.write(summary + "\n")
    print(summary)

    # ---- disagreement dump ----
    df["disagreement"] = (df["p_text"] - df["p_image"]).abs()
    cols = ["ann_id", "y_true", "p_text", "p_image", "disagreement",
            "text_correct", "image_correct", "tq_exclude"]
    for c in ("channel", "image_path", "caption_en"):
        if c in df.columns:
            cols.append(c)
    dump = df.sort_values("disagreement", ascending=False)[cols].head(args.top)
    dump.to_csv(f"{args.outdir}/probe_disagreement.csv", index=False,
                encoding="utf-8-sig")
    print(f"\nWrote {args.outdir}/agreement_summary.txt")
    print(f"Wrote {args.outdir}/probe_disagreement.csv  "
          f"(top {args.top} disagreement posts; tq_exclude=1 => trust the "
          f"English caption less)")


if __name__ == "__main__":
    sys.exit(main())
