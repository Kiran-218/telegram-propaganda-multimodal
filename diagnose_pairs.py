"""
Diagnose channels that have images on disk but 0 matched annotatable pairs.

For each flagged channel it prints (no caption text, privacy-safe):
  - number of posts in JSON
  - number of .jpg files on disk
  - how many post-ids overlap with image-filename stems
  - a few sample JSON post-ids vs a few image-filename stems

This tells us whether the JSON posts and the images are simply different
sets of posts (the usual cause) or a username/path mismatch.
"""

import json
import re
from pathlib import Path

INPUT_JSON = "telegram_channels_full_data.json"
MEDIA_DIR  = "media"


def normalize_username(raw: str) -> str:
    u = (raw or "").strip()
    for p in ("https://", "http://"):
        if u.startswith(p):
            u = u[len(p):]
    if u.startswith("t.me/"):
        u = u[len("t.me/"):]
    return u.lstrip("@").strip("/")


def main():
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    media_root = Path(MEDIA_DIR)

    for entry in data:
        meta = entry.get("metadata") or {}
        uname = meta.get("username") or normalize_username(entry.get("username", ""))
        uname = normalize_username(uname)

        posts = entry.get("recent_posts") or []
        channel_dir = media_root / uname

        if not channel_dir.is_dir():
            continue
        img_stems = {p.stem for p in channel_dir.glob("*.jpg")}
        if not img_stems:
            continue

        post_ids = {str(p.get("id")) for p in posts}
        overlap = post_ids & img_stems

        # only report the problem channels
        if overlap:
            continue

        print("-" * 50)
        print(f"channel folder      : {uname}")
        print(f"posts in JSON       : {len(posts)}")
        print(f"image files on disk : {len(img_stems)}")
        print(f"id overlap          : {len(overlap)}")
        # privacy-safe: ids are just integers, no content
        sample_posts = sorted(list(post_ids))[:5]
        sample_imgs  = sorted(list(img_stems))[:5]
        print(f"sample JSON post-ids: {sample_posts}")
        print(f"sample image stems  : {sample_imgs}")
        # also check: does the folder name appear as a DIFFERENT key anywhere?
        # (renamed-channel detection handled separately if needed)


if __name__ == "__main__":
    main()