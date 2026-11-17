"""
Build the FULL-CORPUS caption/image table for M-CLIP scoring.

This is sampling.py's build_pool() with the sampling step removed: it emits
EVERY valid image-caption pair in the dataset, not a 400-post sample. The pool
definition is byte-for-byte identical to sampling.py -- same username
normalisation, same caption-cleaning threshold (MIN_CAPTION_CHARS), same image
resolution (media/<channel>/<post_id>.jpg matched by .jpg stem) -- so the
corpus scores are directly comparable to the gold-set scores.

Key: ann_id = "<channel>__<post_id>" (stable, reproducible, joinable back to
posts and up to channels). mclip_score.py requires an 'ann_id' column; this
supplies it. The 400 gold posts also appear here under their channel__post_id
key, so corpus-vs-gold scores can be cross-checked on that overlap.

Output columns (exactly what mclip_score.py needs, plus provenance):
  ann_id, channel, post_id, image_path, caption, date

Output image_path is written RELATIVE (media/<channel>/<post_id>.jpg) so it
resolves on the cluster against --media-root, regardless of OS path style.

Usage:
  python build_corpus_csv.py \
      --in telegram_channels_full_data.json \
      --media-root . \
      --out corpus_captions.csv
"""

import argparse
import csv
import json
import re
from pathlib import Path

# --- must match sampling.py / audit_dataset.py exactly ---
MIN_CAPTION_CHARS = 10


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


def build_all_pairs(input_json: str, media_dir: str):
    """Identical pool logic to sampling.py.build_pool(), but returns a flat
    list of ALL valid pairs (no per-channel dict, no sampling).

    media_dir is the 'media' folder itself (matching sampling.py's
    MEDIA_DIR='media'), NOT the folder that contains it."""
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    media_root = Path(media_dir)

    pairs = []
    channels_with_pairs = set()
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

        for post in (entry.get("recent_posts") or []):
            pid = str(post.get("id"))
            if pid in img_ids and has_meaningful_caption(post.get("text", "")):
                pairs.append({
                    "ann_id": f"{uname}__{pid}",
                    "channel": uname,
                    "post_id": pid,
                    # RELATIVE path, forward slashes: resolves on the cluster
                    "image_path": f"media/{uname}/{pid}.jpg",
                    "caption": post.get("text", ""),
                    "date": post.get("date"),
                })
                channels_with_pairs.add(uname)
    return pairs, channels_with_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="telegram_channels_full_data.json")
    ap.add_argument("--media-dir", default="media",
                    help="the media/ folder itself (matches sampling.py's "
                         "MEDIA_DIR='media'); used to check which images exist. "
                         "Output image_path is always written relative as "
                         "media/<channel>/<post_id>.jpg.")
    ap.add_argument("--out", dest="outp", default="corpus_captions.csv")
    args = ap.parse_args()

    pairs, channels = build_all_pairs(args.inp, args.media_dir)

    # stable ordering: by channel then numeric post id
    pairs.sort(key=lambda r: (r["channel"], int(r["post_id"])))

    with open(args.outp, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ann_id", "channel", "post_id", "image_path", "caption", "date"])
        for r in pairs:
            w.writerow([r["ann_id"], r["channel"], r["post_id"],
                        r["image_path"], r["caption"], r["date"]])

    # check for duplicate keys (should be none; a channel shouldn't have two
    # posts with the same id)
    keys = [r["ann_id"] for r in pairs]
    dupes = len(keys) - len(set(keys))

    print(f"Channels with pairs : {len(channels)}")
    print(f"Total valid pairs   : {len(pairs)}")
    if dupes:
        print(f"WARNING: {dupes} duplicate ann_id keys -- investigate before scoring.")
    print(f"Written: {args.outp}")
    print("\nSanity check vs sampling.py: 'Total valid pairs' here should equal "
          "the 'Total annotatable' number sampling.py printed (~9,054).")


if __name__ == "__main__":
    main()
