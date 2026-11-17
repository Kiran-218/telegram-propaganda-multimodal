"""
Draw a proportional, stratified annotation sample from the annotatable pairs.

- Proportional to each channel's pair count, with a floor (every channel with
  pairs gets at least MIN_PER_CHANNEL) and a ceiling (no channel exceeds
  MAX_PER_CHANNEL), remainder distributed proportionally.
- Reproducible via fixed SEED.
- Output is annotation-ready and BLIND: annotators see channel, post id,
  caption, and image path only. No mismatch scores, no metadata flags,
  no labels -- so judgements aren't biased.

Outputs:
  annotation_sample.csv   (for annotators: includes an empty 'label' column)
  annotation_sample.json  (full record, for your own provenance)
"""

import csv
import json
import random
import re
from pathlib import Path

# ==============================
# CONFIG
# ==============================
INPUT_JSON   = "telegram_channels_full_data.json"
MEDIA_DIR    = "media"
OUT_CSV      = "annotation_sample.csv"
OUT_JSON     = "annotation_sample.json"

TARGET_N         = 400
MIN_PER_CHANNEL  = 1     # every channel with pairs is represented
MAX_PER_CHANNEL  = 12    # no single channel dominates
MIN_CAPTION_CHARS = 10   # must match the audit threshold
SEED             = 42

# ==============================
# HELPERS (mirror audit_dataset.py)
# ==============================
def normalize_username(raw: str) -> str:
    u = (raw or "").strip()
    for p in ("https://", "http://"):
        if u.startswith(p):
            u = u[len(p):]
    if u.startswith("t.me/"):
        u = u[len("t.me/"):]
    return u.lstrip("@").strip("/")

_URL_RE      = re.compile(r"https?://\S+|t\.me/\S+")
_MD_LINK_RE  = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_MD_MARKS_RE = re.compile(r"[*_`~]+")
_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF"
    "\U00002190-\U000021FF\U00002B00-\U00002BFF\uFE0F\u200d\u20e3]+"
)

def clean_caption(text: str) -> str:
    if not text:
        return ""
    t = _MD_LINK_RE.sub(r"\1", text)
    t = _URL_RE.sub(" ", t)
    t = _MD_MARKS_RE.sub("", t)
    t = _EMOJI_RE.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()

def has_meaningful_caption(text: str) -> bool:
    return len(clean_caption(text)) >= MIN_CAPTION_CHARS


# ==============================
# BUILD POOL
# ==============================
def build_pool():
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    media_root = Path(MEDIA_DIR)

    by_channel = {}   # username -> list of pair records
    for entry in data:
        meta = entry.get("metadata") or {}
        uname = normalize_username(meta.get("username")
                                   or normalize_username(entry.get("username", "")))
        channel_dir = media_root / uname
        if not channel_dir.is_dir():
            continue
        img_ids = {p.stem for p in channel_dir.glob("*.jpg")}
        if not img_ids:
            continue

        pairs = []
        for post in (entry.get("recent_posts") or []):
            pid = str(post.get("id"))
            if pid in img_ids and has_meaningful_caption(post.get("text", "")):
                pairs.append({
                    "channel": uname,
                    "post_id": pid,
                    "image_path": str(channel_dir / f"{pid}.jpg"),
                    "caption": post.get("text", ""),
                    "date": post.get("date"),
                })
        if pairs:
            by_channel[uname] = pairs
    return by_channel


# ==============================
# ALLOCATE (proportional w/ floor & ceiling)
# ==============================
def allocate(by_channel, target):
    counts = {c: len(p) for c, p in by_channel.items()}
    total = sum(counts.values())
    channels = list(counts)

    # start everyone at the floor
    alloc = {c: min(MIN_PER_CHANNEL, counts[c]) for c in channels}
    remaining = target - sum(alloc.values())

    # distribute the rest proportionally, respecting ceiling & availability
    if remaining > 0:
        # proportional weights on the *remaining* capacity
        while remaining > 0:
            # capacity left per channel
            cap = {c: min(MAX_PER_CHANNEL, counts[c]) - alloc[c] for c in channels}
            avail = [c for c in channels if cap[c] > 0]
            if not avail:
                break
            weights = [counts[c] for c in avail]
            wsum = sum(weights)
            progressed = False
            for c, w in zip(avail, weights):
                if remaining <= 0:
                    break
                give = max(1, round(remaining * w / wsum))
                give = min(give, cap[c], remaining)
                if give > 0:
                    alloc[c] += give
                    remaining -= give
                    progressed = True
            if not progressed:
                break
    return alloc


def main():
    random.seed(SEED)
    by_channel = build_pool()
    total_pairs = sum(len(p) for p in by_channel.values())
    print(f"Channels with pairs : {len(by_channel)}")
    print(f"Total annotatable   : {total_pairs}")

    alloc = allocate(by_channel, TARGET_N)
    sample = []
    for c, n in alloc.items():
        picks = random.sample(by_channel[c], min(n, len(by_channel[c])))
        sample.extend(picks)

    random.shuffle(sample)   # mix channels so annotators don't see runs
    # stable annotation id
    for i, rec in enumerate(sample, 1):
        rec["ann_id"] = f"A{i:04d}"

    print(f"Sample drawn        : {len(sample)} posts "
          f"across {len(alloc)} channels")
    print(f"Per-channel range   : "
          f"{min(alloc.values())}..{max(alloc.values())}")

    # full provenance JSON
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"seed": SEED, "target": TARGET_N,
                   "min_per_channel": MIN_PER_CHANNEL,
                   "max_per_channel": MAX_PER_CHANNEL,
                   "sample": sample}, f, ensure_ascii=False, indent=2)

    # blind annotation CSV
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ann_id", "channel", "post_id", "image_path",
                    "caption", "label", "notes"])
        for r in sample:
            w.writerow([r["ann_id"], r["channel"], r["post_id"],
                        r["image_path"], r["caption"], "", ""])

    print(f"\nWritten: {OUT_CSV}  (label/notes columns blank, for annotators)")
    print(f"Written: {OUT_JSON} (provenance, seed={SEED})")
    print("\nLabel scheme reminder: propagandistic / neutral / unclear")


if __name__ == "__main__":
    main()