"""
Compute per-post M-CLIP mismatch scores for the annotation sample.

For each post with an image and a Russian caption:
  - encode the ORIGINAL Russian caption with the M-CLIP text encoder
    (M-CLIP/XLM-Roberta-Large-Vit-B-32)
  - encode the image with the matching CLIP image encoder
    (ViT-B-32-quickgelu, OpenAI weights)
  - L2-normalise both embeddings and take their cosine similarity
  - mismatch = 1 - cosine similarity  (high mismatch = low alignment)

The mismatch score is a PROXY SIGNAL UNDER INVESTIGATION; it is not a
propaganda label. Scores are produced blind to the human annotations.

Input : annotation_sample.csv   (needs ann_id, caption, image_path)
Output: mclip_scores.csv        (ann_id, mclip_similarity, mclip_mismatch, status)

Resumable: writes after every row; on restart, rows already scored are skipped.
Robust: a missing/corrupt image is recorded with status != 'ok' rather than
crashing the run.
"""

import argparse
import os

import pandas as pd
import torch
from PIL import Image

from multilingual_clip import pt_multilingual_clip
import transformers
import open_clip

TXT_MODEL = "M-CLIP/XLM-Roberta-Large-Vit-B-32"
IMG_MODEL = "ViT-B-32-quickgelu"
IMG_PRETRAINED = "openai"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def norm_path(p):
    return str(p).replace("\\", "/")


def load_models():
    print(f"Device: {DEVICE}")
    print("Loading M-CLIP text encoder ...")
    tmodel = pt_multilingual_clip.MultilingualCLIP.from_pretrained(TXT_MODEL)
    tmodel = tmodel.to(DEVICE).eval()
    tok = transformers.AutoTokenizer.from_pretrained(TXT_MODEL)
    print("Loading CLIP image encoder ...")
    imodel, _, preprocess = open_clip.create_model_and_transforms(
        IMG_MODEL, pretrained=IMG_PRETRAINED)
    imodel = imodel.to(DEVICE).eval()
    return tmodel, tok, imodel, preprocess


def embed_text(tmodel, tok, text):
    # Replicates MultilingualCLIP.forward but keeps tokens on the model's
    # device (the wrapper tokenises on CPU, causing a device mismatch on GPU).
    # Pooling is identical to the wrapper: attention-masked mean, then the
    # linear projection. truncation added for XLM-R's 512-token limit.
    txt_tok = tok([text], padding=True, truncation=True, max_length=512,
                  return_tensors="pt")
    txt_tok = {k: v.to(DEVICE) for k, v in txt_tok.items()}
    with torch.no_grad():
        embs = tmodel.transformer(**txt_tok)[0]
        att = txt_tok["attention_mask"]
        embs = (embs * att.unsqueeze(2)).sum(dim=1) / att.sum(dim=1)[:, None]
        embs = tmodel.LinearTransformation(embs)
        embs = embs / embs.norm(dim=-1, keepdim=True)
    return embs


def embed_image(imodel, preprocess, path):
    img = Image.open(path).convert("RGB")
    ix = preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = imodel.encode_image(ix)              # [1, 512]
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="annotation_sample.csv")
    ap.add_argument("--out", dest="outp", default="mclip_scores.csv")
    ap.add_argument("--media-root", default=".",
                    help="folder that contains the media/ tree; image_path in "
                         "the CSV is resolved against this")
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    for col in ("ann_id", "caption", "image_path"):
        if col not in df.columns:
            raise SystemExit(f"Input CSV missing '{col}' column.")

    # resume: load any existing scores and skip those ann_ids
    done = set()
    if os.path.exists(args.outp):
        prev = pd.read_csv(args.outp)
        done = set(prev["ann_id"].astype(str))
        print(f"Resuming: {len(done)} rows already scored.")
        out_rows = prev.to_dict("records")
    else:
        out_rows = []

    tmodel, tok, imodel, preprocess = load_models()

    total = len(df)
    for idx, row in df.iterrows():
        aid = str(row["ann_id"])
        if aid in done:
            continue

        status = "ok"
        similarity = None
        try:
            cap = "" if pd.isna(row["caption"]) else str(row["caption"]).strip()
            rel = norm_path(row["image_path"])
            img_path = os.path.join(args.media_root, rel)

            if not cap:
                status = "empty_caption"
            elif not os.path.exists(img_path):
                status = "missing_image"
            else:
                temb = embed_text(tmodel, tok, cap)
                iemb = embed_image(imodel, preprocess, img_path)
                similarity = float((temb @ iemb.T).squeeze())
        except Exception as e:                      # corrupt image, decode error, etc.
            status = f"error:{type(e).__name__}"

        mismatch = (1.0 - similarity) if similarity is not None else None
        out_rows.append({"ann_id": aid,
                         "mclip_similarity": similarity,
                         "mclip_mismatch": mismatch,
                         "status": status})

        # write after every row so a crash never loses progress
        pd.DataFrame(out_rows).to_csv(args.outp, index=False, encoding="utf-8-sig")
        print(f"  {idx+1}/{total}  {aid}  status={status}"
              + (f"  sim={similarity:.4f}" if similarity is not None else ""),
              end="\r")

    print()
    scored = pd.DataFrame(out_rows)
    ok = (scored["status"] == "ok").sum()
    print(f"Done. {ok}/{len(scored)} scored ok; "
          f"{len(scored)-ok} had missing/empty/errored inputs.")
    print(f"Written: {args.outp}")


if __name__ == "__main__":
    main()
