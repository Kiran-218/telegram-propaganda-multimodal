"""
Translate the Russian captions in the annotation sample to English using
NLLB-200 (1.3B distilled) on GPU (fp16), for the MLP cluster via Slurm.
For ANNOTATION AID only: the Russian 'caption' column is preserved unchanged
so the M-CLIP pipeline still consumes the original Russian.

Input : annotation_sample.csv              (must have 'caption' and 'ann_id')
Output: annotation_sample_translated_1.3B.csv  (adds 'caption_en')

Differences vs the 600M CPU version:
  - Model: facebook/nllb-200-distilled-1.3B
  - Runs on CUDA in fp16 (falls back to CPU/fp32 if no GPU is visible)
  - Beam search (num_beams=4) instead of greedy decoding
Everything else (chunking, resume/skip, per-row incremental writes,
utf-8-sig output, Russian-column preservation) is unchanged.

Resumable: writes after every row, skips rows already translated on restart.
"""

import argparse
import os
import re

import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_NAME = "facebook/nllb-200-distilled-1.3B"
SRC_LANG   = "rus_Cyrl"
TGT_LANG   = "eng_Latn"

MAX_TOKENS = 400      # NLLB source limit; long captions are chunked
NUM_BEAMS  = 4        # beam search: affordable on GPU, better quality

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16 if DEVICE == "cuda" else torch.float32


def split_long(text, tokenizer):
    """Split a caption into <=MAX_TOKENS chunks on sentence boundaries."""
    if not text or not text.strip():
        return [""]
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, cur = [], ""
    for s in sentences:
        trial = (cur + " " + s).strip()
        if len(tokenizer(trial)["input_ids"]) <= MAX_TOKENS:
            cur = trial
        else:
            if cur:
                chunks.append(cur)
            if len(tokenizer(s)["input_ids"]) > MAX_TOKENS:
                words, w = s.split(), ""
                for word in words:
                    t2 = (w + " " + word).strip()
                    if len(tokenizer(t2)["input_ids"]) <= MAX_TOKENS:
                        w = t2
                    else:
                        if w:
                            chunks.append(w)
                        w = word
                if w:
                    chunks.append(w)
                cur = ""
            else:
                cur = s
    if cur:
        chunks.append(cur)
    return chunks or [""]


def translate_text(text, tokenizer, model, bos_id):
    """Translate one caption (chunked, beam search on GPU)."""
    chunks = split_long(text, tokenizer)
    outs = []
    for ch in chunks:
        if not ch.strip():
            outs.append("")
            continue
        enc = tokenizer(ch, return_tensors="pt", truncation=True,
                        max_length=MAX_TOKENS).to(DEVICE)
        with torch.no_grad():
            gen = model.generate(**enc, forced_bos_token_id=bos_id,
                                 max_length=MAX_TOKENS, num_beams=NUM_BEAMS)
        outs.append(tokenizer.batch_decode(gen, skip_special_tokens=True)[0])
    return " ".join(o for o in outs if o).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="inp",  default="annotation_sample.csv")
    ap.add_argument("--out", dest="outp",
                    default="annotation_sample_translated_1.3B.csv")
    args = ap.parse_args()

    print(f"Device: {DEVICE} ({DTYPE})")
    print(f"Loading {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang=SRC_LANG)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME, torch_dtype=DTYPE).to(DEVICE).eval()
    bos_id = tokenizer.convert_tokens_to_ids(TGT_LANG)

    df = pd.read_csv(args.inp)
    for col in ("caption", "ann_id"):
        if col not in df.columns:
            raise SystemExit(f"Input CSV missing '{col}' column.")

    # resume: load any partial output and keep existing translations
    done = {}
    if os.path.exists(args.outp):
        prev = pd.read_csv(args.outp)
        if "caption_en" in prev.columns:
            done = dict(zip(prev["ann_id"].astype(str),
                            prev["caption_en"].fillna("")))
        print(f"Resuming: {sum(1 for v in done.values() if v)} rows already done.")

    if "caption_en" not in df.columns:
        df["caption_en"] = ""

    total = len(df)
    for idx, row in df.iterrows():
        aid = str(row["ann_id"])
        if done.get(aid):                 # already translated -> skip
            df.at[idx, "caption_en"] = done[aid]
            continue
        text = "" if pd.isna(row["caption"]) else str(row["caption"])
        df.at[idx, "caption_en"] = translate_text(text, tokenizer, model, bos_id)
        # write after every row so a crash never loses progress
        df.to_csv(args.outp, index=False, encoding="utf-8-sig")
        print(f"  {idx+1}/{total}  ({aid})", end="\r")

    print()
    df.to_csv(args.outp, index=False, encoding="utf-8-sig")
    print(f"Written: {args.outp}  ('caption_en' added, Russian preserved)")


if __name__ == "__main__":
    main()
