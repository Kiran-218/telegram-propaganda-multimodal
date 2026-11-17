#!/usr/bin/env python3
"""
embedding_probe.py -- Does the M-CLIP embedding SPACE carry a propaganda signal
that the single cosine-similarity number missed?

Background
----------
The core mismatch result collapsed each post's two 512-dim M-CLIP embeddings
(text, image) into ONE number -- cosine similarity -- and found it did not
separate propagandistic from neutral posts (AUC 0.542, a null). That test asks
only whether the two modalities are *aligned*. It throws away everything else in
the embeddings.

This script asks the sharper question: is there ANY discriminative signal in the
full cross-modal embedding representation that a supervised model can extract?
We concatenate the text and image embeddings (1024 features) and train a
regularised linear classifier to predict the human label.

Why this is honest and not an overfitting trap
-----------------------------------------------
n is small (345) and the feature space is large (1024). A classifier can easily
"find" signal in noise at this ratio. Three safeguards make the result trustworthy:

1. NESTED cross-validation. The OUTER loop measures performance on data the model
   never saw; the INNER loop picks the regularisation strength C. Because C is
   chosen inside the inner loop only, we never tune on the data we score on --
   no optimistic leakage.

2. A LABEL-PERMUTATION test. We shuffle the labels many times (destroying any
   real signal) and re-run the entire nested-CV pipeline each time, building a
   distribution of AUCs that arise from noise alone. The real AUC gets a p-value
   against that null distribution. This is the decisive check: if the real AUC
   does not clearly beat the shuffled distribution, any apparent signal is noise.

3. A SIMPLE linear model with strong L2 regularisation and balanced class
   weights (propaganda is the minority: 115 vs 230). No MLP -- a flexible model
   on this little data is exactly how you fool yourself.

Interpretation either way
-------------------------
 - If the real AUC beats the permutation null (low p): the cross-modal signal
   EXISTS in the embedding space but not in the cosine dimension -- a genuine,
   interesting deepening of the null. Report as a weak, high-variance signal at
   this n, NOT a deployable detector.
 - If it does not: cross-modal absence is now ROBUST -- not an artefact of one
   crude metric. This strengthens the null.

Inputs (paths match the mismatch evaluation)
--------------------------------------------
  --emb      mclip_embeddings.npz      (ann_ids, text_emb[N,512], image_emb[N,512])
  --labels   annotation_labels.csv     (ann_id, label)   label in {propagandistic,neutral,unclear}
  --tqflags  output/gold_tq_flags.csv  (ann_id, tq_exclude)  1 = drop

Evaluable set (identical to evaluate_mismatch.py):
  drop unreadable images (already absent from the .npz),
  drop tq_exclude == 1, drop label == unclear  ->  expect n=345 (115 prop / 230 neut)

Outputs (draft artifacts)
-------------------------
  output/probe_outer_folds.csv     per-outer-fold AUC / F1 / chosen C
  output/probe_permutation.csv     the permuted-label AUCs (null distribution)
  output/probe_summary.txt         plain-language summary for the write-up
"""

import argparse
import sys
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.metrics import roc_auc_score, f1_score


def norm_cols(df):
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
    if "ann_id" in df.columns:
        df["ann_id"] = df["ann_id"].astype(str).str.strip()
    return df


def build_pipeline():
    # Standardise features (embeddings are already L2-normed per-vector, but
    # per-dimension standardisation helps the linear model and the scaler is
    # fit INSIDE each training fold via the pipeline, so no leakage).
    return Pipeline([
        ("scaler", StandardScaler()),
        # liblinear uses L2 by default; regularisation strength comes from C
        # (tuned in the inner CV loop). Not passing penalty explicitly keeps
        # this compatible across sklearn versions.
        ("clf", LogisticRegression(
            solver="liblinear",
            class_weight="balanced", max_iter=2000)),
    ])


C_GRID = {"clf__C": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]}


def nested_cv_auc(X, y, seed, n_outer=5, n_inner=5, return_folds=False):
    """Run nested stratified CV. Returns the mean outer-fold AUC.

    Outer loop: honest evaluation on held-out folds.
    Inner loop (GridSearchCV): picks C using ONLY the outer-training data.
    """
    outer = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=seed)
    fold_rows = []
    aucs = []
    for k, (tr, te) in enumerate(outer.split(X, y)):
        inner = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed)
        gs = GridSearchCV(build_pipeline(), C_GRID, scoring="roc_auc",
                          cv=inner, n_jobs=-1)
        gs.fit(X[tr], y[tr])
        proba = gs.predict_proba(X[te])[:, 1]
        auc = roc_auc_score(y[te], proba)
        pred = (proba >= 0.5).astype(int)
        f1 = f1_score(y[te], pred, pos_label=1)
        aucs.append(auc)
        fold_rows.append({"outer_fold": k, "auc": auc, "f1_prop": f1,
                          "chosen_C": gs.best_params_["clf__C"],
                          "n_test": len(te)})
    mean_auc = float(np.mean(aucs))
    if return_folds:
        return mean_auc, fold_rows
    return mean_auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default="mclip_embeddings.npz")
    ap.add_argument("--labels", default="annotation_labels.csv")
    ap.add_argument("--tqflags", default="output/gold_tq_flags.csv")
    ap.add_argument("--outdir", default="output")
    ap.add_argument("--n-perm", type=int, default=1000,
                    help="number of label permutations for the null distribution")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--modality", choices=["both", "text", "image"],
                    default="both",
                    help="which embeddings to use as features: 'text' (512), "
                         "'image' (512), or 'both' concatenated (1024). Run all "
                         "three to test whether the image modality adds signal "
                         "over text alone -- the core multimodal question.")
    args = ap.parse_args()

    import os
    os.makedirs(args.outdir, exist_ok=True)

    # ---- load embeddings -----------------------------------------------------
    data = np.load(args.emb, allow_pickle=True)
    emb_ids = np.array([str(a).strip() for a in data["ann_ids"]])
    text_emb = data["text_emb"].astype(np.float32)     # [N,512]
    image_emb = data["image_emb"].astype(np.float32)   # [N,512]
    emb_df = pd.DataFrame({"ann_id": emb_ids,
                           "row": np.arange(len(emb_ids))})

    # ---- load labels + tq flags ---------------------------------------------
    gold = norm_cols(pd.read_csv(args.labels))
    gold["label"] = gold["label"].astype(str).str.strip().str.lower()
    tq = norm_cols(pd.read_csv(args.tqflags))
    # The tq-flags file keys its ID column 'id' (it was produced from the
    # translated captions CSV), whereas everything else uses 'ann_id'. Rename
    # so the join key is consistent.
    if "ann_id" not in tq.columns and "id" in tq.columns:
        tq = tq.rename(columns={"id": "ann_id"})
    tq["ann_id"] = tq["ann_id"].astype(str).str.strip()

    # ---- reproduce the evaluable set exactly --------------------------------
    m = emb_df.merge(gold[["ann_id", "label"]], on="ann_id", how="inner")
    m = m.merge(tq[["ann_id", "tq_exclude"]], on="ann_id", how="left")
    m["tq_exclude"] = m["tq_exclude"].fillna(0).astype(int)

    n_joined = len(m)
    m = m[m["tq_exclude"] == 0]
    n_after_tq = len(m)
    m = m[m["label"] != "unclear"]
    n_after_unclear = len(m)

    # binary target: propagandistic = 1, neutral = 0
    m = m[m["label"].isin(["propagandistic", "neutral"])].copy()
    m["y"] = (m["label"] == "propagandistic").astype(int)

    rows = m["row"].to_numpy()
    # Feature set depends on the chosen modality. Running text-only, image-only,
    # and both lets us see whether the IMAGE half adds discriminative signal
    # over the text half alone -- i.e. whether the result is genuinely
    # multimodal or is really the text embedding doing the work.
    if args.modality == "text":
        X = text_emb[rows]                                          # [n,512]
    elif args.modality == "image":
        X = image_emb[rows]                                         # [n,512]
    else:
        X = np.concatenate([text_emb[rows], image_emb[rows]], axis=1)   # [n,1024]
    y = m["y"].to_numpy()

    n = len(y)
    n_prop = int(y.sum())
    n_neut = n - n_prop

    print("--- evaluable-set reconstruction ---")
    print(f"  embeddings joined to labels: {n_joined}")
    print(f"  after tq_exclude==0:         {n_after_tq}")
    print(f"  after dropping unclear:      {n_after_unclear}")
    print(f"  => evaluable (binary):       {n}  "
          f"(propagandistic={n_prop}, neutral={n_neut})")
    print(f"  feature matrix: {X.shape}")
    if not (n == 345 and n_prop == 115 and n_neut == 230):
        print("  NOTE: this differs from the expected 345/115/230. Not fatal, but")
        print("  check the join keys if you expected an exact match.")

    # ---- real nested-CV AUC --------------------------------------------------
    print("\n--- nested cross-validation (real labels) ---")
    real_auc, fold_rows = nested_cv_auc(X, y, seed=args.seed, return_folds=True)
    fold_df = pd.DataFrame(fold_rows)
    tag = args.modality
    fold_df.to_csv(f"{args.outdir}/probe_outer_folds_{tag}.csv", index=False)
    print(fold_df.to_string(index=False))
    print(f"  mean outer-fold AUC: {real_auc:.4f}  "
          f"(std {fold_df['auc'].std():.4f})")

    # ---- permutation null ----------------------------------------------------
    # For each permutation, shuffle y and re-run the SAME nested-CV pipeline.
    # This is the honest test: what AUC does this whole procedure produce when
    # there is definitely no real signal? nested-CV is expensive, so this loop
    # is the slow part -- n-perm * (outer*inner*grid) fits. 1000 perms is the
    # standard; drop to 200-500 if wall-time is tight (report the number used).
    print(f"\n--- permutation test ({args.n_perm} shuffles) ---")
    rng = np.random.default_rng(args.seed)
    perm_aucs = np.empty(args.n_perm, dtype=np.float64)
    for i in range(args.n_perm):
        y_perm = rng.permutation(y)
        # use a per-permutation seed so folds vary; keeps the null honest
        perm_aucs[i] = nested_cv_auc(X, y_perm, seed=int(rng.integers(1e9)))
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{args.n_perm} done", end="\r")
    print()

    pd.DataFrame({"perm_auc": perm_aucs}).to_csv(
        f"{args.outdir}/probe_permutation_{tag}.csv", index=False)

    # p-value: fraction of permuted AUCs >= real AUC (one-sided, +1 smoothing)
    p_val = (np.sum(perm_aucs >= real_auc) + 1) / (args.n_perm + 1)
    null_mean = float(perm_aucs.mean())
    null_p95 = float(np.percentile(perm_aucs, 95))

    # ---- summary -------------------------------------------------------------
    lines = []
    lines.append("M-CLIP EMBEDDING PROBE -- supervised signal beyond cosine")
    lines.append("=" * 58)
    lines.append(f"modality    = {args.modality}")
    lines.append(f"evaluable n = {n}  (propagandistic={n_prop}, neutral={n_neut})")
    feat_desc = {"both": "512 text + 512 image, concatenated",
                 "text": "512 text only",
                 "image": "512 image only"}[args.modality]
    lines.append(f"features    = {X.shape[1]}  ({feat_desc})")
    lines.append("")
    lines.append(f"real mean outer-fold AUC : {real_auc:.4f} "
                 f"(std {fold_df['auc'].std():.4f})")
    lines.append(f"permutation null mean AUC: {null_mean:.4f}")
    lines.append(f"permutation null 95th pct: {null_p95:.4f}")
    lines.append(f"permutations             : {args.n_perm}")
    lines.append(f"one-sided p-value        : {p_val:.4f}")
    lines.append("")
    lines.append("comparison line-up (for the write-up)")
    lines.append("-" * 58)
    lines.append("  cosine mismatch (1-dim)      : AUC 0.542  (null, post-level)")
    lines.append(f"  embedding probe (1024-dim)   : AUC {real_auc:.3f}  "
                 f"(perm p={p_val:.3f})")
    lines.append("")
    lines.append("how to read this")
    lines.append("-" * 58)
    if p_val < 0.05 and real_auc > null_p95:
        lines.append("  The supervised probe beats the permutation null. There IS")
        lines.append("  discriminative signal in the cross-modal embedding space that")
        lines.append("  the single cosine dimension missed. Report as a weak, high-")
        lines.append("  variance signal at this n -- a deepening of the null, NOT a")
        lines.append("  working detector. The cosine metric, not the representation,")
        lines.append("  was the limiting factor.")
    else:
        lines.append("  The supervised probe does NOT beat the permutation null. Even")
        lines.append("  with the full 1024-dim representation and a trained classifier,")
        lines.append("  no cross-modal signal is recoverable. This STRENGTHENS the")
        lines.append("  null: the absence is a property of the cross-modal signal for")
        lines.append("  this corpus, not an artefact of the crude cosine metric.")
    lines.append("")
    lines.append("  Either way: computed with nested CV (no tuning leakage) and a")
    lines.append("  label-permutation null (honest significance at small n).")

    summary = "\n".join(lines)
    with open(f"{args.outdir}/probe_summary_{tag}.txt", "w") as f:
        f.write(summary + "\n")
    print("\n" + summary)
    print(f"\nWrote {args.outdir}/probe_outer_folds_{tag}.csv")
    print(f"Wrote {args.outdir}/probe_permutation_{tag}.csv")
    print(f"Wrote {args.outdir}/probe_summary_{tag}.txt")


if __name__ == "__main__":
    sys.exit(main())
