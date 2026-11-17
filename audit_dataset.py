"""
Dataset audit for the cross-modal mismatch project.

For each channel it reports:
  - total posts in JSON
  - posts with a downloaded image on disk
  - posts with a "meaningful" Russian caption (markdown/URLs/emoji stripped)
  - posts that have BOTH  -> the annotatable / scoreable set
  - image files on disk and channel disk footprint

Writes a per-channel table to audit_summary.json and prints totals.
No GPU, runs locally.
"""

import json
import re
from pathlib import Path

# ==============================
# CONFIG
# ==============================
INPUT_JSON   = "telegram_channels_full_data.json"
MEDIA_DIR    = "media"
OUTPUT_JSON  = "audit_summary.json"

# A caption counts as "meaningful" if, after stripping markdown / URLs /
# emoji / whitespace, this many characters of real text remain.
# Start permissive; inspect the distribution, then tune.
MIN_CAPTION_CHARS = 10

# ==============================
# HELPERS
# ==============================
def normalize_username(raw: str) -> str:
    """https://t.me/x , t.me/x , @x , x  ->  x"""
    u = (raw or "").strip()
    for p in ("https://", "http://"):
        if u.startswith(p):
            u = u[len(p):]
    if u.startswith("t.me/"):
        u = u[len("t.me/"):]
    return u.lstrip("@").strip("/")


# Telegram markdown bold/italic, inline links [txt](url), bare URLs, emoji.
_URL_RE       = re.compile(r"https?://\S+|t\.me/\S+")
_MD_LINK_RE   = re.compile(r"\[([^\]]*)\]\([^)]*\)")   # [text](url) -> text
_MD_MARKS_RE  = re.compile(r"[*_`~]+")
# Rough emoji / symbol / pictograph ranges + variation selectors.
_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF"
    "\U00002190-\U000021FF\U00002B00-\U00002BFF\uFE0F\u200d\u20e3]+"
)

def clean_caption(text: str) -> str:
    """Strip markdown, links and emoji; return remaining real text."""
    if not text:
        return ""
    t = _MD_LINK_RE.sub(r"\1", text)   # keep link label, drop the url
    t = _URL_RE.sub(" ", t)
    t = _MD_MARKS_RE.sub("", t)
    t = _EMOJI_RE.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def has_meaningful_caption(text: str) -> bool:
    return len(clean_caption(text)) >= MIN_CAPTION_CHARS


def dir_size_mb(path: Path) -> float:
    total = sum(f.stat().st_size for f in path.glob("*") if f.is_file())
    return round(total / (1024 * 1024), 1)


# ==============================
# MAIN
# ==============================
def main():
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    media_root = Path(MEDIA_DIR)
    per_channel = []
    caption_len_samples = []  # for distribution reporting

    for entry in data:
        meta = entry.get("metadata") or {}
        # prefer the already-bare metadata username, fall back to the URL
        uname = meta.get("username") or normalize_username(entry.get("username", ""))
        uname = normalize_username(uname)

        posts = entry.get("recent_posts") or []
        channel_dir = media_root / uname

        # image ids present on disk (filename stem -> post id)
        if channel_dir.is_dir():
            img_ids = {p.stem for p in channel_dir.glob("*.jpg")}
            img_files = len(img_ids)
            footprint = dir_size_mb(channel_dir)
        else:
            img_ids = set()
            img_files = 0
            footprint = 0.0

        total_posts = len(posts)
        with_image = 0
        with_caption = 0
        pairs = 0  # both image AND meaningful caption  -> annotatable

        for post in posts:
            pid = str(post.get("id"))
            cap_ok = has_meaningful_caption(post.get("text", ""))
            img_ok = pid in img_ids

            if cap_ok:
                with_caption += 1
                caption_len_samples.append(len(clean_caption(post.get("text", ""))))
            if img_ok:
                with_image += 1
            if cap_ok and img_ok:
                pairs += 1

        per_channel.append({
            "username": uname,
            "verified": meta.get("verified"),
            "scam": meta.get("scam"),
            "fake": meta.get("fake"),
            "participants_count": meta.get("participants_count"),
            "total_posts": total_posts,
            "posts_with_image": with_image,
            "posts_with_caption": with_caption,
            "annotatable_pairs": pairs,
            "image_files_on_disk": img_files,
            "footprint_mb": footprint,
        })

    # ---- totals ----
    def s(k): return sum(c[k] for c in per_channel)
    totals = {
        "channels": len(per_channel),
        "total_posts": s("total_posts"),
        "posts_with_image": s("posts_with_image"),
        "posts_with_caption": s("posts_with_caption"),
        "annotatable_pairs": s("annotatable_pairs"),
        "image_files_on_disk": s("image_files_on_disk"),
        "footprint_mb": round(sum(c["footprint_mb"] for c in per_channel), 1),
    }

    out = {"config": {"MIN_CAPTION_CHARS": MIN_CAPTION_CHARS},
           "totals": totals,
           "per_channel": per_channel}
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # ---- console report ----
    print("=" * 60)
    print("DATASET AUDIT")
    print("=" * 60)
    print(f"Channels                : {totals['channels']}")
    print(f"Total posts             : {totals['total_posts']}")
    print(f"Posts with image        : {totals['posts_with_image']}")
    print(f"Posts with caption      : {totals['posts_with_caption']}")
    print(f"ANNOTATABLE PAIRS       : {totals['annotatable_pairs']}   <-- effective dataset")
    print(f"Image files on disk     : {totals['image_files_on_disk']}")
    print(f"Total footprint (MB)    : {totals['footprint_mb']}")

    if caption_len_samples:
        caption_len_samples.sort()
        n = len(caption_len_samples)
        def pct(p): return caption_len_samples[min(int(p * n), n - 1)]
        print("\nCleaned caption length (chars) distribution:")
        print(f"  min {caption_len_samples[0]}  "
              f"p25 {pct(.25)}  median {pct(.5)}  "
              f"p75 {pct(.75)}  max {caption_len_samples[-1]}")
        print(f"  (threshold currently {MIN_CAPTION_CHARS}; "
              f"tune MIN_CAPTION_CHARS and re-run if needed)")

    # flag channels where image filenames don't match any post id
    print("\nChannels with images on disk but 0 matched pairs "
          "(possible id-mismatch, inspect):")
    flagged = [c["username"] for c in per_channel
               if c["image_files_on_disk"] > 0 and c["annotatable_pairs"] == 0]
    print("  " + (", ".join(flagged) if flagged else "none"))

    print(f"\nWritten: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()