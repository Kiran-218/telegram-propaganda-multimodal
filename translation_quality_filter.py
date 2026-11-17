# translation_quality_filter.py
# One deterministic rule, applied identically to the 400-post gold set AND the
# full ~9500-post corpus. Flags translation-degenerate captions for exclusion.
# No per-post human judgement — this is a pipeline stage, not feature engineering.
#
# Usage:
#   python translation_quality_filter.py <input.csv> <output.csv>
# e.g.
#   python translation_quality_filter.py annotation_sample_translated_1.3B.csv gold_tq_flags.csv
#   python translation_quality_filter.py all_9500_translated.csv corpus_tq_flags.csv
#
# Output columns: ann_id (or row index), tq_exclude (0/1), tq_reason, metrics
# Apply IDENTICALLY downstream: keep rows where tq_exclude == 0.

import csv
import re
import sys
from collections import Counter

CAPTION_COL = "caption_en"
ID_COL = "ann_id"  # falls back to row index if absent

# --- single documented threshold rule ---------------------------------------
# A caption is excluded if it shows automated-detectable translation
# degeneration by ANY of these reproducible signals:
MIN_CONSEC_WORD_REPEAT = 4     # "oregon oregon oregon oregon" -> exclude
MAX_NGRAM = 3                  # check 1..3-gram consecutive loops
UNIQUE_RATIO_FLOOR = 0.40      # unique/total tokens below this = degenerate
MIN_TOKENS = 8                # skip ratio test on very short captions
TOP_TOKEN_SHARE = 0.35        # one content token > this share = degenerate

STOP = {"the", "a", "an", "of", "to", "and", "in", "on", "for", "is", "are",
        "was", "were", "it", "that", "this", "with", "as", "at", "by", "from",
        "be", "s", "not", "or", "but", "its", "we", "our", "you"}


def tokenize(text):
    return re.findall(r"\b\w+\b", str(text).lower())


def worst_consec_repeat(tokens):
    worst = (0, "", 0)  # (n, phrase, count)
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
            if count > worst[2]:
                worst = (n, " ".join(gram), count)
            i = j if count > 1 else i + 1
    return worst


def assess(text):
    tokens = tokenize(text)
    n = len(tokens)
    reasons = []
    metrics = {}

    ncnt = worst_consec_repeat(tokens)
    metrics["max_consec"] = ncnt[2]
    if ncnt[2] >= MIN_CONSEC_WORD_REPEAT:
        reasons.append("consec_repeat")

    if n >= MIN_TOKENS:
        uniq = len(set(tokens)) / n
        metrics["unique_ratio"] = round(uniq, 3)
        if uniq < UNIQUE_RATIO_FLOOR:
            reasons.append("low_unique_ratio")

        content = [t for t in tokens if t not in STOP]
        if content:
            top = Counter(content).most_common(1)[0][1]
            share = top / len(content)
            metrics["top_share"] = round(share, 3)
            if share > TOP_TOKEN_SHARE:
                reasons.append("token_dominance")

    exclude = 1 if reasons else 0
    return exclude, ";".join(reasons), metrics


def main():
    if len(sys.argv) != 3:
        print("usage: python translation_quality_filter.py <in.csv> <out.csv>")
        sys.exit(1)
    src, dst = sys.argv[1], sys.argv[2]

    rows = []
    total = excluded = 0
    reason_counts = Counter()
    with open(src, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        has_id = ID_COL in (reader.fieldnames or [])
        for idx, row in enumerate(reader):
            total += 1
            cap = row.get(CAPTION_COL, "") or ""
            ex, reason, metrics = assess(cap)
            if ex:
                excluded += 1
                for r in reason.split(";"):
                    reason_counts[r] += 1
            rows.append({
                "id": row.get(ID_COL, "") if has_id else str(idx),
                "tq_exclude": ex,
                "tq_reason": reason,
                "max_consec": metrics.get("max_consec", ""),
                "unique_ratio": metrics.get("unique_ratio", ""),
                "top_share": metrics.get("top_share", ""),
            })

    with open(dst, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "tq_exclude", "tq_reason",
            "max_consec", "unique_ratio", "top_share"])
        writer.writeheader()
        writer.writerows(rows)

    pct = 100 * excluded / total if total else 0
    print(f"Input:    {src}")
    print(f"Total captions:   {total}")
    print(f"Excluded (tq):    {excluded}  ({pct:.2f}%)")
    for r, c in reason_counts.most_common():
        print(f"    {r}: {c}")
    print(f"Wrote {dst}")
    print("\nApply downstream: keep rows where tq_exclude == 0, on BOTH gold and corpus.")


if __name__ == "__main__":
    main()
