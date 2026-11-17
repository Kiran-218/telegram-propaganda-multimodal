#!/usr/bin/env python3
"""
evaluate_mismatch.py  --  M4 core evaluation.

Tests the research question: does the M-CLIP cross-modal mismatch score
(a PROXY SIGNAL under investigation) track human propaganda judgements on
the 400-post gold set?

Blind-evaluation discipline: exclusions are declared by data-quality rules
(unreadable image, translation-collapse), NOT by whether they improve the
result. Unclear labels are dropped for the binary evaluation as pre-registered.

Inputs (all already on the cluster):
  --scores   mclip_scores.csv          (ann_id, mclip_similarity, mclip_mismatch, status)
  --labels   annotation_labels.csv     (ann_id, label, ...)
  --tqflags  gold_tq_flags.csv         (id, tq_exclude, ...)   [optional but recommended]

Outputs:
  - prints AUC-ROC, point-biserial r, Spearman rho, group means, exclusion audit
  - writes eval_merged.csv (joined table with an 'evaluable' flag) for the write-up
"""

import argparse
import sys
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="output/mclip_scores.csv")
    ap.add_argument("--labels", required=True,
                    help="path to annotation_labels.csv (the 400 gold labels)")
    ap.add_argument("--tqflags", default=None,
                    help="path to gold_tq_flags.csv (translation-quality exclusions)")
    ap.add_argument("--out", default="output/eval_merged.csv")
    args = ap.parse_args()

    scores = pd.read_csv(args.scores)
    labels = pd.read_csv(args.labels)
    for d in (scores, labels):
        d.columns = [c.strip().lstrip("\ufeff") for c in d.columns]
        if "ann_id" in d:
            d["ann_id"] = d["ann_id"].astype(str).str.strip()
    labels["label"] = labels["label"].astype(str).str.strip().str.lower()

    m = scores.merge(labels[["ann_id", "label"]], on="ann_id", how="inner")
    print(f"Joined scores+labels: {len(m)} posts")

    # --- declare exclusions (data-quality, not result-driven) ---------------
    m["excl_reason"] = ""

    bad_img = m["status"].astype(str) != "ok"
    m.loc[bad_img, "excl_reason"] = "unreadable_image"

    tq_ids = set()
    if args.tqflags:
        tq = pd.read_csv(args.tqflags)
        tq.columns = [c.strip().lstrip("\ufeff") for c in tq.columns]
        idcol = "id" if "id" in tq.columns else "ann_id"
        tq[idcol] = tq[idcol].astype(str).str.strip()
        tq_ids = set(tq.loc[tq["tq_exclude"] == 1, idcol])
        hit = m["ann_id"].isin(tq_ids) & (m["excl_reason"] == "")
        m.loc[hit, "excl_reason"] = "translation_collapse"

    # unclear -> excluded from BINARY eval (pre-registered), tracked separately
    is_unclear = m["label"] == "unclear"

    # evaluable = readable image AND not tq-excluded AND binary label
    m["evaluable"] = (m["excl_reason"] == "") & (~is_unclear)

    n_badimg = int(bad_img.sum())
    n_tq = int((m["excl_reason"] == "translation_collapse").sum())
    n_unclear = int(is_unclear.sum())
    n_eval = int(m["evaluable"].sum())

    print("\n--- exclusion audit ---")
    print(f"  unreadable image:      {n_badimg}")
    print(f"  translation_collapse:  {n_tq}")
    print(f"  unclear label:         {n_unclear}")
    print(f"  => evaluable (binary): {n_eval}")

    ev = m[m["evaluable"]].copy()
    ev["y"] = (ev["label"] == "propagandistic").astype(int)
    x = ev["mclip_mismatch"].astype(float).values
    y = ev["y"].values

    print(f"\n  class balance: propagandistic={int(y.sum())}, "
          f"neutral={int((1 - y).sum())}")

    # --- core metrics -------------------------------------------------------
    # Hypothesis: higher mismatch -> more likely propagandistic.
    auc = roc_auc_score(y, x)
    r_pb, p_pb = stats.pointbiserialr(y, x)
    rho, p_rho = stats.spearmanr(x, y)

    prop_mean = x[y == 1].mean()
    neut_mean = x[y == 0].mean()
    u, p_mw = stats.mannwhitneyu(x[y == 1], x[y == 0], alternative="two-sided")

    print("\n=== CORE RESULT: mismatch vs human label ===")
    print(f"  AUC-ROC:                 {auc:.3f}")
    print(f"  point-biserial r:        {r_pb:+.3f}  (p={p_pb:.3g})")
    print(f"  Spearman rho:            {rho:+.3f}  (p={p_rho:.3g})")
    print(f"  mean mismatch (prop):    {prop_mean:.4f}")
    print(f"  mean mismatch (neutral): {neut_mean:.4f}")
    print(f"  difference:              {prop_mean - neut_mean:+.4f}")
    print(f"  Mann-Whitney U:          U={u:.0f}, p={p_mw:.3g}")

    # Youden-J operating point, for reporting only (not for tuning)
    fpr, tpr, thr = roc_curve(y, x)
    j = tpr - fpr
    best = int(np.argmax(j))
    print(f"\n  (Youden-J threshold={thr[best]:.4f}: "
          f"TPR={tpr[best]:.2f}, FPR={fpr[best]:.2f}) [reporting only]")

    m.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"\nWrote {args.out}")

    # --- honest reading -----------------------------------------------------
    print("\n--- how to read this ---")
    if auc >= 0.70:
        print("  AUC >= 0.70: mismatch carries clear, useful signal for the label.")
    elif auc >= 0.60:
        print("  AUC 0.60-0.70: modest but real signal; consistent with a weak")
        print("  proxy that is informative but not sufficient alone.")
    elif auc >= 0.55:
        print("  AUC 0.55-0.60: weak signal, barely above chance. Report honestly;")
        print("  the tight mismatch distribution (std~0.037) limits discrimination.")
    else:
        print("  AUC ~0.50: mismatch does not separate the classes on its own.")
        print("  This is a legitimate finding: cross-modal mismatch alone is not")
        print("  a viable proxy here. Frame as such; it does not sink the project.")
    print("  Whatever the value: this was computed once, blind, on the frozen")
    print("  gold set with pre-declared exclusions. Report it as-is.")


if __name__ == "__main__":
    sys.exit(main())
