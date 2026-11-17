#!/usr/bin/env python3
"""
probe_predictions.py -- Save per-post out-of-fold predictions for the text and
image probes, so we can compare WHERE each modality succeeds and fails.

This does NOT recompute AUCs or run permutations. It reruns the same nested-CV
folds (same seed as embedding_probe.py) but records, for every post, the
held-out predicted probability from the text probe and from the image probe.
Because every post is in exactly one held-out outer fold, each post gets one
honest out-of-fold probability per modality.

Output:
  output/probe_per_post.csv
    ann_id, y_true, p_text, p_image
  (y_true: 1=propagandistic, 0=neutral)

Same evaluable set (n=345) and same fold seeding as the other probes, so these
predictions correspond exactly to the reported AUCs.
"""
import argparse, os, sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV


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


def oof_probabilities(X, y, seed, n_outer=5, n_inner=5):
    """Return an array of out-of-fold predicted probabilities, one per post,
    each produced by a model that did NOT see that post in training."""
    oof = np.full(len(y), np.nan)
    outer = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=seed)
    for tr, te in outer.split(X, y):
        inner = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed)
        gs = GridSearchCV(build_pipeline(), C_GRID, scoring="roc_auc",
                          cv=inner, n_jobs=-1)
        gs.fit(X[tr], y[tr])
        oof[te] = gs.predict_proba(X[te])[:, 1]
    return oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default="mclip_embeddings.npz")
    ap.add_argument("--labels", default="annotation_labels.csv")
    ap.add_argument("--tqflags", default="output/gold_tq_flags.csv")
    ap.add_argument("--outdir", default="output")
    ap.add_argument("--seed", type=int, default=42)
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
    print(f"evaluable n = {len(y)} (prop={int(y.sum())}, neut={len(y)-int(y.sum())})")

    print("computing out-of-fold text-probe probabilities ...")
    p_text = oof_probabilities(Xt, y, args.seed)
    print("computing out-of-fold image-probe probabilities ...")
    p_image = oof_probabilities(Xi, y, args.seed)

    out = pd.DataFrame({"ann_id": m["ann_id"].to_numpy(),
                        "y_true": y, "p_text": p_text, "p_image": p_image})
    out.to_csv(f"{args.outdir}/probe_per_post.csv", index=False)
    print(f"Wrote {args.outdir}/probe_per_post.csv  ({len(out)} rows)")


if __name__ == "__main__":
    sys.exit(main())
