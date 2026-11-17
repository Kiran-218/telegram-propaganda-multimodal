#!/usr/bin/env python3
"""
llm_baseline_infer.py -- Zero-shot text-only LLM baseline (M4 comparator).

Mistral-7B-Instruct-v0.3 in 4-bit classifies each post's TRANSLATED caption as
propagandistic / neutral, using the project's own annotation criteria in the
prompt. NOT fine-tuned. NOT an annotator. A baseline evaluated against the gold
set (see llm_baseline_eval.py).

Resumable and ann_id-keyed (mirrors the translation/mclip pattern): re-running
skips ann_ids already present in the output CSV.

Input : annotation_sample_translated_1.3B.csv  (ann_id, caption_en, ...)
Output: llm_baseline_labels.csv  (ann_id, llm_label, llm_raw)

Design notes:
- English prompt on translated text (matches IPP LLM-baseline design).
- Binary output forced (propagandistic/neutral); the baseline is evaluated on
  the binary set. Unparseable output is recorded as 'abstain', NOT silently
  bucketed, so it can't distort the metrics.
- Criteria are lifted from the frozen annotation scheme: manner-not-topic;
  propaganda = manipulative persuasion (emotional/rhetorical) not rational
  argument; a post is neutral if it merely informs/depicts without that
  persuasive work.
"""
import argparse
import csv
import os
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

SYSTEM = """You are annotating Russian-language Telegram posts from the Russia-Ukraine war for PROPAGANDA, judging only the text provided (translated to English).

Definition (manner, not topic): a post is PROPAGANDISTIC when it works to persuade the audience through emotional or rhetorical manipulation rather than rational, factual argument -- e.g. loaded emotional language, glorification framed as persuasion, demonization of an out-group, appeals designed to provoke rather than inform. A war topic alone does NOT make a post propagandistic.

A post is NEUTRAL when it primarily informs, reports, or depicts without that persuasive manipulative work, even if it concerns the war, the military, or casualties.

You must choose exactly one label. Answer with a single word: PROPAGANDISTIC or NEUTRAL."""

USER_TMPL = """Post text:
\"\"\"{caption}\"\"\"

Label (one word, PROPAGANDISTIC or NEUTRAL):"""


def build_prompt(tok, caption):
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER_TMPL.format(caption=caption.strip()[:4000])},
    ]
    return tok.apply_chat_template(messages, tokenize=False,
                                   add_generation_prompt=True)


def parse_label(text):
    t = text.upper()
    # take the first explicit label token that appears
    m = re.search(r"\b(PROPAGANDISTIC|NEUTRAL)\b", t)
    if m:
        return "propagandistic" if m.group(1) == "PROPAGANDISTIC" else "neutral"
    return "abstain"


def load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if row.get("ann_id"):
                    done.add(row["ann_id"])
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp",
                    default="data/annotation_sample_translated_1.3B.csv")
    ap.add_argument("--out", default="output/llm_baseline_labels.csv")
    ap.add_argument("--caption-col", default="caption_en")
    args = ap.parse_args()

    done = load_done(args.out)
    print(f"Already done: {len(done)} posts (will skip these)")

    rows = []
    with open(args.inp, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["ann_id"] not in done:
                rows.append(row)
    print(f"To classify: {len(rows)} posts")
    if not rows:
        print("Nothing to do.")
        return

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    print(f"Loading {MODEL} in 4-bit...")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, quantization_config=bnb, device_map="auto",
        attn_implementation="eager")  # bypass triton/SDPA conflict on this env
    model.eval()

    write_header = not os.path.exists(args.out)
    out_f = open(args.out, "a", newline="", encoding="utf-8-sig")
    writer = csv.DictWriter(out_f, fieldnames=["ann_id", "llm_label", "llm_raw"])
    if write_header:
        writer.writeheader()

    n_abstain = 0
    for i, row in enumerate(rows, 1):
        ann = row["ann_id"]
        cap = row.get(args.caption_col, "") or ""
        if not cap.strip():
            writer.writerow({"ann_id": ann, "llm_label": "abstain",
                             "llm_raw": "(empty caption)"})
            n_abstain += 1
            continue

        prompt = build_prompt(tok, cap)
        inputs = tok(prompt, return_tensors="pt",
                     truncation=True, max_length=4096).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=8, do_sample=False,
                pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0][inputs["input_ids"].shape[1]:],
                         skip_special_tokens=True).strip()
        label = parse_label(gen)
        if label == "abstain":
            n_abstain += 1
        writer.writerow({"ann_id": ann, "llm_label": label,
                         "llm_raw": gen.replace("\n", " ")[:120]})
        out_f.flush()
        if i % 25 == 0:
            print(f"  {i}/{len(rows)}  (abstain so far: {n_abstain})")

    out_f.close()
    print(f"\nDone. Wrote {args.out}. Abstentions: {n_abstain}")
    if n_abstain > 0:
        print("Abstentions are recorded explicitly; the eval script reports")
        print("them separately and does not count them as either class.")


if __name__ == "__main__":
    sys.exit(main())
