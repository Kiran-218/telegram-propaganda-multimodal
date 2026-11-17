"""
Compute per-post M-CLIP mismatch scores AND SAVE THE FULL EMBEDDINGS.

This is mclip_score.py with one addition: as well as writing the cosine
similarity / mismatch to the CSV (exactly as before, identical numbers), it
also saves the full 512-dim text embedding and 512-dim image embedding for
every successfully scored post. Those saved embeddings are what the later
supervised-classifier experiment needs -- the original script computed them
and then threw them away, keeping only the single cosine number.

NOTHING about the similarity computation has changed, so re-running this
reproduces the same mclip_scores.csv you already have. The only new behaviour
is the extra embeddings file.

For each post with an image and a Russian caption:
  - encode the ORIGINAL Russian caption with the M-CLIP text encoder
    (M-CLIP/XLM-Roberta-Large-Vit-B-32)
  - encode the image with the matching CLIP image encoder
    (ViT-B-32-quickgelu, OpenAI weights)
  - L2-normalise both embeddings and take their cosine similarity
  - mismatch = 1 - cosine similarity  (high mismatch = low alignment)
  - >>> NEW: also keep both normalised embeddings for this post <<<

The mismatch score is a PROXY SIGNAL UNDER INVESTIGATION; it is not a
propaganda label. Scores are produced blind to the human annotations.

Input : annotation_sample.csv   (needs ann_id, caption, image_path)
Output: mclip_scores.csv        (ann_id, mclip_similarity, mclip_mismatch, status)  [unchanged]
        mclip_embeddings.npz     (NEW: ann_ids, text_emb, image_emb arrays)

Resumable: writes after every row; on restart, rows already scored are skipped.
The embeddings file is rebuilt from an in-memory store at the end of the run,
and also flushed periodically, so a crash mid-run does not lose scored rows.
Robust: a missing/corrupt image is recorded with status != 'ok' rather than
crashing the run.
"""

import argparse
import os

import numpy as np                      # NEW: for saving embeddings
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


# NEW: helper that writes the accumulated embeddings to a single .npz file.
# We store three aligned arrays: the ann_ids (str), the text embeddings, and
# the image embeddings. Row i of text_emb / image_emb belongs to ann_ids[i].
# Only 'ok' posts (both embeddings present) are stored.
def save_embeddings(path, emb_store):
    if not emb_store:
        return
    ann_ids = np.array([a for (a, _, _) in emb_store])
    text_emb = np.stack([t for (_, t, _) in emb_store]).astype(np.float32)
    image_emb = np.stack([i for (_, _, i) in emb_store]).astype(np.float32)
    np.savez(path, ann_ids=ann_ids, text_emb=text_emb, image_emb=image_emb)


# NEW: on resume, load any embeddings already saved so we don't recompute them.
# Returns a list of (ann_id, text_vec, image_vec) tuples.
def load_embeddings(path):
    store = []
    if os.path.exists(path):
        data = np.load(path, allow_pickle=True)
        ann_ids = data["ann_ids"]
        text_emb = data["text_emb"]
        image_emb = data["image_emb"]
        for i in range(len(ann_ids)):
            store.append((str(ann_ids[i]), text_emb[i], image_emb[i]))
        print(f"Resuming: {len(store)} embeddings already saved.")
    return store


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="annotation_sample.csv")
    ap.add_argument("--out", dest="outp", default="mclip_scores.csv")
    # NEW: where to write the embeddings archive
    ap.add_argument("--emb-out", dest="emb_outp", default="mclip_embeddings.npz",
                    help="output .npz file for the full text/image embeddings")
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

    # NEW: resume the embeddings store in lockstep with the CSV resume above
    emb_store = load_embeddings(args.emb_outp)

    tmodel, tok, imodel, preprocess = load_models()

    total = len(df)
    for idx, row in df.iterrows():
        aid = str(row["ann_id"])
        if aid in done:
            continue

        status = "ok"
        similarity = None
        temb_vec = None                 # NEW
        iemb_vec = None                 # NEW
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
                # NEW: pull the vectors off the GPU and keep them (512-dim each)
                temb_vec = temb.squeeze(0).cpu().numpy()
                iemb_vec = iemb.squeeze(0).cpu().numpy()
        except Exception as e:                      # corrupt image, decode error, etc.
            status = f"error:{type(e).__name__}"

        mismatch = (1.0 - similarity) if similarity is not None else None
        out_rows.append({"ann_id": aid,
                         "mclip_similarity": similarity,
                         "mclip_mismatch": mismatch,
                         "status": status})

        # NEW: only store embeddings for successfully scored posts
        if status == "ok" and temb_vec is not None and iemb_vec is not None:
            emb_store.append((aid, temb_vec, iemb_vec))

        # write after every row so a crash never loses progress
        pd.DataFrame(out_rows).to_csv(args.outp, index=False, encoding="utf-8-sig")
        # NEW: flush embeddings periodically (every 25 rows) to bound loss on crash
        if (idx + 1) % 25 == 0:
            save_embeddings(args.emb_outp, emb_store)
        print(f"  {idx+1}/{total}  {aid}  status={status}"
              + (f"  sim={similarity:.4f}" if similarity is not None else ""),
              end="\r")

    print()
    # NEW: final flush of all embeddings
    save_embeddings(args.emb_outp, emb_store)

    scored = pd.DataFrame(out_rows)
    ok = (scored["status"] == "ok").sum()
    print(f"Done. {ok}/{len(scored)} scored ok; "
          f"{len(scored)-ok} had missing/empty/errored inputs.")
    print(f"Written: {args.outp}")
    print(f"Written: {args.emb_outp}  ({len(emb_store)} embeddings)")   # NEW


if __name__ == "__main__":
    main()
