"""
Re-scrape post metadata for channels whose JSON posts don't overlap their
downloaded images (the 34 flagged by diagnose_pairs.py).

Strategy
--------
For each channel, the downloaded images define the target window. We read the
min/max image id on disk and use Telethon's offset_id to page BACKWARD from
just above the newest image, so the scraped posts cover the same id range as
the images we already have. This maximises recovered image+caption pairs.

Safety
------
Writes to a SEPARATE file (rescrape_34.json). It does NOT touch
telegram_channels_full_data.json. A separate merge step does that.
"""

import asyncio
import json
import random
from pathlib import Path

from telethon import TelegramClient
from telethon.errors import FloodWaitError
from telethon.tl.functions.channels import GetFullChannelRequest
from tqdm import tqdm

# ==============================
# CONFIG
# ==============================
import os
api_id = int(os.environ["TG_API_ID"])
api_hash = os.environ["TG_API_HASH"]

channels = [
    
    # Add your 150 channels here (without @)
]

MAIN_JSON   = "telegram_channels_full_data.json"
MEDIA_DIR   = "media"
OUTPUT_JSON = "rescrape_34.json"

# Pull enough posts to comfortably span each channel's image id range.
# Image ranges spanned ~1000-4000 ids, so 800 posts back from the anchor
# is generous. Bump if any channel still under-covers (we report this).
POST_LIMIT  = 800
DELAY_RANGE = (2, 5)

# ==============================
# HELPERS
# ==============================
def normalize_username(raw: str) -> str:
    u = (raw or "").strip()
    for p in ("https://", "http://"):
        if u.startswith(p):
            u = u[len(p):]
    if u.startswith("t.me/"):
        u = u[len("t.me/"):]
    return u.lstrip("@").strip("/")


def find_flagged_channels():
    """Return [(username, min_img_id, max_img_id), ...] for channels with
    images on disk but zero overlap with current JSON post ids."""
    with open(MAIN_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    media_root = Path(MEDIA_DIR)
    flagged = []

    for entry in data:
        meta = entry.get("metadata") or {}
        uname = normalize_username(meta.get("username")
                                   or normalize_username(entry.get("username", "")))
        channel_dir = media_root / uname
        if not channel_dir.is_dir():
            continue

        img_ids = []
        for p in channel_dir.glob("*.jpg"):
            try:
                img_ids.append(int(p.stem))
            except ValueError:
                pass
        if not img_ids:
            continue

        post_ids = {p.get("id") for p in (entry.get("recent_posts") or [])}
        if post_ids & set(img_ids):
            continue  # already overlaps -> not flagged

        flagged.append((uname, min(img_ids), max(img_ids)))

    return flagged


# ==============================
# SCRAPE
# ==============================
async def fetch_posts(client, username, min_img_id, max_img_id):
    """Scrape posts paging backward from just above the newest image id."""
    result = {"username": username,
              "min_img_id": min_img_id,
              "max_img_id": max_img_id,
              "recent_posts": []}

    try:
        entity = await client.get_entity(username)
        # also refresh metadata so the merge can update participant counts etc.
        full = await client(GetFullChannelRequest(entity))
        result["metadata"] = {
            "id": entity.id,
            "title": entity.title,
            "username": entity.username,
            "creation_date": entity.date.isoformat() if entity.date else None,
            "verified": entity.verified,
            "restricted": entity.restricted,
            "scam": entity.scam,
            "fake": entity.fake,
            "gigagroup": entity.gigagroup,
            "megagroup": entity.megagroup,
            "participants_count": full.full_chat.participants_count,
            "description": full.full_chat.about,
        }

        # offset_id starts iteration just ABOVE the newest image, walking back.
        async for message in client.iter_messages(
                entity, limit=POST_LIMIT, offset_id=max_img_id + 1):
            result["recent_posts"].append({
                "id": message.id,
                "date": message.date.isoformat() if message.date else None,
                "text": message.text,
                "views": message.views,
                "forwards": message.forwards,
                "replies": message.replies.replies if message.replies else None,
            })
    except FloodWaitError as e:
        print(f"  Flood wait {e.seconds}s ...")
        await asyncio.sleep(e.seconds)
        return await fetch_posts(client, username, min_img_id, max_img_id)
    except Exception as e:
        result["error"] = str(e)

    return result


async def main():
    flagged = find_flagged_channels()
    print(f"Flagged channels to re-scrape: {len(flagged)}\n")

    out = []
    async with TelegramClient("research_session", api_id, api_hash) as client:
        for uname, lo, hi in tqdm(flagged, desc="Re-scrape"):
            print(f"\nProcessing: {uname}  (images {lo}..{hi})")
            res = await fetch_posts(client, uname, lo, hi)

            # quick coverage report: how many scraped ids fall in image range
            scraped_ids = {p["id"] for p in res["recent_posts"]}
            img_lo, img_hi = lo, hi
            covered = sum(1 for i in scraped_ids if img_lo <= i <= img_hi)
            lowest = min(scraped_ids) if scraped_ids else None
            print(f"  scraped={len(scraped_ids)} posts, "
                  f"lowest id={lowest}, "
                  f"ids within image range={covered}")
            if lowest is not None and lowest > img_lo:
                print(f"  ! lowest scraped id ({lowest}) > min image id "
                      f"({img_lo}) -- raise POST_LIMIT to cover the tail")

            out.append(res)
            await asyncio.sleep(random.uniform(*DELAY_RANGE))

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nWritten: {OUTPUT_JSON}  ({len(out)} channels)")
    print("Next: run merge_rescrape.py to fold these into the main JSON.")


if __name__ == "__main__":
    asyncio.run(main())