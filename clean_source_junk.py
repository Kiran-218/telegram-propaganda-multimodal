# clean_source_junk.py
# Strips repeated-URL / markup runs from caption_en and writes caption_en_clean.
# Flags anything still degenerate after cleaning (still_bad = 1).
#
# Input:  annotation_sample_translated_1.3B.csv  (has ann_id, caption_en)
# Output: captions_cleaned.csv  (ann_id, still_bad, removed_chars,
#                                caption_en_clean, caption_en)
#
# Safe to run on all 400 — posts with no junk pass through unchanged.

import csv
import re

CSV_IN = "annotation_sample_translated_1.3B.csv"
CSV_OUT = "captions_cleaned.csv"
ID_COL = "ann_id"
CAPTION_COL = "caption_en"

# ------------------------------------------------------------------ cleaning
# 1. Collapsed URL runs: the same host/path fragment chained with / repeatedly.
#    e.g. twitter.com/twitter.com/twitter.com...   t.me/x/x/x...
#    Strategy: find URLs, then within them collapse immediately-repeated
#    slash-separated segments down to a single occurrence.
URL_RE = re.compile(r'https?://[^\s\]\)]+', re.IGNORECASE)


def collapse_repeated_segments(url):
    """Collapse consecutive duplicate '/'-separated segments in a URL."""
    # split off scheme
    m = re.match(r'(https?://)(.*)', url, re.IGNORECASE)
    if not m:
        return url
    scheme, rest = m.group(1), m.group(2)
    segs = rest.split('/')
    out = []
    for s in segs:
        if out and s == out[-1] and s != "":
            continue  # drop immediate duplicate segment
        out.append(s)
    return scheme + "/".join(out)


def clean_urls(text):
    return URL_RE.sub(lambda m: collapse_repeated_segments(m.group(0)), text)


# 2. Bare repeated markup tokens outside URLs: nobj.nobj.nobj, obj.obj,
#    tml.tml, and similar dotted single-token loops (>=3 repeats).
DOTTED_LOOP_RE = re.compile(
    r'\b(\w{1,12})(?:\.\1\b){2,}\.?', re.IGNORECASE)


def clean_dotted_loops(text):
    return DOTTED_LOOP_RE.sub(lambda m: m.group(1), text)


# 3. Generic consecutive whole-token repetition (>=4 times) left over after
#    the above — e.g. "oregon oregon oregon...". This is the genuine-collapse
#    signature; if cleaning it leaves almost nothing, still_bad will catch it.
WORD_LOOP_RE = re.compile(r'\b(\w+)(?:\s+\1\b){3,}', re.IGNORECASE)


def clean_word_loops(text):
    return WORD_LOOP_RE.sub(lambda m: m.group(1), text)


def clean_caption(text):
    t = str(text)
    t = clean_urls(t)
    t = clean_dotted_loops(t)
    t = clean_word_loops(t)
    # tidy whitespace
    t = re.sub(r'\s+', ' ', t).strip()
    return t


# ------------------------------------------------------------- still-bad test
# After cleaning, a caption is "still_bad" if it's degenerate on its own terms:
# very low unique-token ratio, or dominated by one token, or too short to use.
def is_still_bad(text):
    toks = re.findall(r'\b\w+\b', text.lower())
    if len(toks) < 4:
        return True
    uniq_ratio = len(set(toks)) / len(toks)
    if uniq_ratio < 0.40:
        return True
    stop = {"the", "a", "an", "of", "to", "and", "in", "on", "for", "is",
            "s", "it", "that", "this", "with", "as", "at", "by", "from"}
    content = [t for t in toks if t not in stop]
    if content:
        top = max(set(content), key=content.count)
        if content.count(top) / len(content) > 0.35:
            return True
    return False


def main():
    rows_out = []
    n_changed = n_bad = 0
    with open(CSV_IN, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ann = row.get(ID_COL, "")
            orig = row.get(CAPTION_COL, "") or ""
            cleaned = clean_caption(orig)
            removed = len(orig) - len(cleaned)
            still_bad = is_still_bad(cleaned)
            if cleaned != orig.strip():
                n_changed += 1
            if still_bad:
                n_bad += 1
            rows_out.append({
                ID_COL: ann,
                "still_bad": 1 if still_bad else 0,
                "removed_chars": removed,
                "caption_en_clean": cleaned,
                "caption_en": orig,
            })

    with open(CSV_OUT, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f, fieldnames=[ID_COL, "still_bad", "removed_chars",
                           "caption_en_clean", "caption_en"])
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Processed {len(rows_out)} captions.")
    print(f"  captions altered by cleaning: {n_changed}")
    print(f"  still_bad after cleaning:     {n_bad}")
    print(f"Wrote {CSV_OUT}")

    # show the posts where cleaning removed the most, so you can eyeball
    big = sorted(rows_out, key=lambda r: r["removed_chars"], reverse=True)[:15]
    print("\nMost-cleaned captions (eyeball these):")
    for r in big:
        if r["removed_chars"] == 0:
            continue
        tag = "STILL BAD" if r["still_bad"] else "recovered"
        print(f"  {r[ID_COL]}  -{r['removed_chars']:>5} chars  [{tag}]")


if __name__ == "__main__":
    main()
