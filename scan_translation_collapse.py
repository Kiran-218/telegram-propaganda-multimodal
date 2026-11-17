# scan_translation_collapse.py
# Scans caption_en for repetition-loop / collapsed translations.
import csv
import re
from collections import Counter

CSV_IN = "annotation_sample_translated_1.3B.csv"
CSV_OUT = "translation_collapse_flags.csv"
CAPTION_COL = "caption_en"
ID_COL = "ann_id"  # change if your key column differs

# --- tunable thresholds ---
MAX_NGRAM = 4              # check repeated phrases up to 4 tokens long
MIN_CONSEC_REPEATS = 3     # e.g. "X X X" -> flag
UNIQUE_RATIO_FLOOR = 0.45  # unique_tokens/total_tokens below this = suspicious
MIN_TOKENS_FOR_RATIO = 8   # don't apply ratio test to very short captions
TOP_TOKEN_SHARE = 0.30     # single non-stopword token >30% of caption = suspicious

STOP = {"the","a","an","of","to","and","in","on","for","is","are","was","were",
        "it","that","this","with","as","at","by","from","be","has","have","had",
        "s","not","or","but","its","their","his","her"}


def tokenize(text):
    return re.findall(r"\b\w+\b", str(text).lower())


def max_consecutive_ngram_repeat(tokens):
    """Return (n, phrase, count) for the worst consecutive n-gram repetition."""
    worst = (0, "", 0)
    for n in range(1, MAX_NGRAM + 1):
        if len(tokens) < n * 2:
            continue
        i = 0
        while i + n <= len(tokens):
            gram = tuple(tokens[i:i + n])
            count = 1
            j = i + n
            while j + n <= len(tokens) and tuple(tokens[j:j + n]) == gram:
                count += 1
                j += n
            if count >= MIN_CONSEC_REPEATS and count * n > worst[2] * worst[0]:
                worst = (n, " ".join(gram), count)
            i = j if count > 1 else i + 1
    return worst


def top_token_share(tokens):
    content = [t for t in tokens if t not in STOP]
    if not content:
        return 0.0, ""
    c = Counter(content)
    tok, cnt = c.most_common(1)[0]
    return cnt / len(content), tok


def analyze(text):
    tokens = tokenize(text)
    n_tok = len(tokens)
    flags = []
    details = {}

    # 1. consecutive n-gram repetition
    n, phrase, count = max_consecutive_ngram_repeat(tokens)
    if count >= MIN_CONSEC_REPEATS:
        flags.append("consec_repeat")
        details["repeat"] = f'"{phrase}" x{count} ({n}-gram)'

    # 2. low unique-token ratio
    if n_tok >= MIN_TOKENS_FOR_RATIO:
        ratio = len(set(tokens)) / n_tok
        details["unique_ratio"] = round(ratio, 2)
        if ratio < UNIQUE_RATIO_FLOOR:
            flags.append("low_unique_ratio")

    # 3. single token dominance
    share, tok = top_token_share(tokens)
    if n_tok >= MIN_TOKENS_FOR_RATIO and share > TOP_TOKEN_SHARE:
        flags.append("token_dominance")
        details["dominant"] = f'"{tok}" {round(share*100)}%'

    return flags, details, n_tok


def main():
    rows_out = []
    total = flagged = 0
    with open(CSV_IN, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            caption = row.get(CAPTION_COL, "")
            flags, details, n_tok = analyze(caption)
            if flags:
                flagged += 1
                rows_out.append({
                    "id": row.get(ID_COL, ""),
                    "n_tokens": n_tok,
                    "flags": ";".join(flags),
                    "detail": " | ".join(f"{k}={v}" for k, v in details.items()),
                    "caption_en": caption,
                })

    rows_out.sort(key=lambda r: r["flags"])
    with open(CSV_OUT, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f, fieldnames=["id", "n_tokens", "flags", "detail", "caption_en"])
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Scanned {total} posts, flagged {flagged}.")
    by_flag = Counter(fl for r in rows_out for fl in r["flags"].split(";"))
    for fl, c in by_flag.most_common():
        print(f"  {fl}: {c}")
    print(f"Wrote {CSV_OUT}")


if __name__ == "__main__":
    main()