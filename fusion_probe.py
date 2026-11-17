#!/usr/bin/env python3
"""
fusion_probe.py -- Does LATE fusion of the two modalities beat text alone?

Context
-------
The concatenation ("both", 1024-dim) probe scored AUC 0.742, BELOW text-only
(0.782). That could mean either (a) the image genuinely adds nothing useful, or
(b) naive concatenation is just a bad way to fuse -- the extra 512 image
dimensions dilute the stronger text signal at this small n. This script
distinguishes those by testing LATE fusion instead of feature concatenation.

Late fusion
-----------
Train the text probe and the image probe SEPARATELY on the training fold, get
each one's predicted probability on the held-out fold, then AVERAGE the two
probabilities and score that. Each modality uses its own model; only the
decisions are combined -- a standard, gentler fusion that does not inflate
dimensionality.

Interpretation
--------------
 - late-fusion AUC > text-only (0.782): the image DOES carry complementary
   signal; concatenation simply fused it badly. A genuine multimodal result.
 - late-fusion AUC <= text-only: fusion does not help however it is done. The
   "text dominates; the visual channel adds no usable signal on top of text"
   conclusion is then robust against the "you fused badly" objection.

Everything else -- the n=345 evaluable set, nested CV, permutation null, fold
seeding -- is identical to embedding_probe.py, so the late-fusion AUC sits on
the same axis as the text / image / both numbers.

Outputs (draft artifacts):
  output/probe_outer_folds_latefusion.csv
  output/probe_permutation_latefusion.csv
  output/probe_summary_latefusion.txt
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, f1_score


def norm_cols(df):
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
    if "ann_id" in df.columns:
        df["ann_id"] = df["ann_id"].astype(str).str.strip()
    return df


def build_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver="liblinear", class_weight="balanced", max_iter=2000)),
    ])


C_GRID = {"clf__C": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]}


def fit_probe_proba(Xtr, ytr, Xte, seed, n_inner=5):
    """Fit one modality's probe (inner-CV C selection) on the training fold and
    return held-out probabilities plus the chosen C."""
    inner = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed)
    gs = GridSearchCV(build_pipeline(), C_GRID, scoring="roc_auc",
                      cv=inner, n_jobs=-1)
    gs.fit(Xtr, ytr)
    return gs.predict_proba(Xte)[:, 1], gs.best_params_["clf__C"]


def latefusion_cv_auc(Xt, Xi, y, seed, n_outer=5, n_inner=5, return_folds=False):
    """Nested CV where, per outer fold, a text probe and an image probe are
    trained separately and their held-out probabilities are AVERAGED."""
    outer = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=seed)
    fold_rows, aucs = [], []
    for k, (tr, te) in enumerate(outer.split(Xt, y)):
        pt, ct = fit_probe_proba(Xt[tr], y[tr], Xt[te], seed, n_inner)
        pi, ci = fit_probe_proba(Xi[tr], y[tr], Xi[te], seed, n_inner)
        proba = (pt + pi) / 2.0                      # late fusion: mean of probs
        auc = roc_auc_score(y[te], proba)
        f1 = f1_score(y[te], (proba >= 0.5).astype(int), pos_label=1)
        aucs.append(auc)
        fold_rows.append({"outer_fold": k, "auc": auc, "f1_prop": f1,
                          "chosen_C_text": ct, "chosen_C_image": ci,
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
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--text-auc", type=float, default=0.782,
                    help="text-only AUC to compare against in the summary")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    data = np.load(args.emb, allow_pickle=True)
    emb_ids = np.array([str(a).strip() for a in data["ann_ids"]])
    text_emb = data["text_emb"].astype(np.float32)
    image_emb = data["image_emb"].astype(np.float32)
    emb_df = pd.DataFrame({"ann_id": emb_ids, "row": np.arange(len(emb_ids))})

    gold = norm_cols(pd.read_csv(args.labels))
    gold["label"] = gold["label"].astype(str).str.strip().str.lower()
    tq = norm_cols(pd.read_csv(args.tqflags))
    if "ann_id" not in tq.columns and "id" in tq.columns:
        tq = tq.rename(columns={"id": "ann_id"})
    tq["ann_id"] = tq["ann_id"].astype(str).str.strip()

    m = emb_df.merge(gold[["ann_id", "label"]], on="ann_id", how="inner")
    m = m.merge(tq[["ann_id", "tq_exclude"]], on="ann_id", how="left")
    m["tq_exclude"] = m["tq_exclude"].fillna(0).astype(int)
    m = m[m["tq_exclude"] == 0]
    m = m[m["label"] != "unclear"]
    m = m[m["label"].isin(["propagandistic", "neutral"])].copy()
    m["y"] = (m["label"] == "propagandistic").astype(int)

    rows = m["row"].to_numpy()
    Xt = text_emb[rows]
    Xi = image_emb[rows]
    y = m["y"].to_numpy()
    n, n_prop = len(y), int(y.sum())

    print("--- evaluable-set reconstruction ---")
    print(f"  => evaluable (binary): {n}  "
          f"(propagandistic={n_prop}, neutral={n-n_prop})")
    print(f"  text features: {Xt.shape}, image features: {Xi.shape}")
    if not (n == 345 and n_prop == 115):
        print("  NOTE: differs from expected 345/115 -- check join keys.")

    print("\n--- nested CV (late fusion: mean of text & image probs) ---")
    real_auc, fold_rows = latefusion_cv_auc(Xt, Xi, y, seed=args.seed,
                                            return_folds=True)
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(f"{args.outdir}/probe_outer_folds_latefusion.csv", index=False)
    print(fold_df.to_string(index=False))
    print(f"  mean outer-fold AUC: {real_auc:.4f}  (std {fold_df['auc'].std():.4f})")

    print(f"\n--- permutation test ({args.n_perm} shuffles) ---")
    rng = np.random.default_rng(args.seed)
    perm_aucs = np.empty(args.n_perm)
    for i in range(args.n_perm):
        y_perm = rng.permutation(y)
        perm_aucs[i] = latefusion_cv_auc(Xt, Xi, y_perm,
                                         seed=int(rng.integers(1e9)))
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{args.n_perm} done", end="\r")
    print()
    pd.DataFrame({"perm_auc": perm_aucs}).to_csv(
        f"{args.outdir}/probe_permutation_latefusion.csv", index=False)
    p_val = (np.sum(perm_aucs >= real_auc) + 1) / (args.n_perm + 1)

    L = []
    L.append("LATE-FUSION PROBE -- averaged text & image decision probabilities")
    L.append("=" * 62)
    L.append(f"evaluable n = {n}  (propagandistic={n_prop}, neutral={n-n_prop})")
    L.append(f"real mean outer-fold AUC : {real_auc:.4f} "
             f"(std {fold_df['auc'].std():.4f})")
    L.append(f"permutation null mean AUC: {perm_aucs.mean():.4f}")
    L.append(f"permutation p-value      : {p_val:.4f}  ({args.n_perm} perms)")
    L.append("")
    L.append("comparison line-up (for the write-up)")
    L.append("-" * 62)
    L.append("  text only      : AUC {:.3f}".format(args.text_auc))
    L.append("  image only     : AUC 0.640")
    L.append("  concat (both)  : AUC 0.742")
    L.append(f"  late fusion    : AUC {real_auc:.3f}  (perm p={p_val:.3f})")
    L.append("")
    L.append("how to read this")
    L.append("-" * 62)
    if real_auc > args.text_auc:
        L.append("  Late fusion BEATS text-only. The image carries complementary")
        L.append("  signal that concatenation fused badly; averaging the two")
        L.append("  decisions recovers it. This is a genuine multimodal gain.")
    else:
        L.append("  Late fusion does NOT beat text-only. Fusion does not help even")
        L.append("  when done gently (decision averaging), not just under naive")
        L.append("  concatenation. Conclusion: the text modality dominates and the")
        L.append("  visual channel adds no usable signal ON TOP OF text -- robust")
        L.append("  against the 'you fused badly' objection. (The image is still")
        L.append("  independently informative at AUC 0.640; it just does not")
        L.append("  complement the stronger text signal.)")

    summary = "\n".join(L)
    with open(f"{args.outdir}/probe_summary_latefusion.txt", "w") as f:
        f.write(summary + "\n")
    print("\n" + summary)
    print(f"\nWrote {args.outdir}/probe_outer_folds_latefusion.csv")
    print(f"Wrote {args.outdir}/probe_permutation_latefusion.csv")
    print(f"Wrote {args.outdir}/probe_summary_latefusion.txt")


if __name__ == "__main__":
    sys.exit(main())
