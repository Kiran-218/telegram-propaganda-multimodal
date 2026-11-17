import asyncio
import json
import random
from pathlib import Path

from telethon import TelegramClient
from telethon.errors import FloodWaitError
from telethon.tl.types import MessageMediaPhoto
from tqdm import tqdm

# ==============================
# CONFIG
# ==============================
import os
api_id = int(os.environ["TG_API_ID"])
api_hash = os.environ["TG_API_HASH"]

INPUT_JSON  = "telegram_channels_full_data.json"
OUTPUT_DIR  = "media"
POST_LIMIT  = 500
DELAY_RANGE = (2, 5)
MAX_RETRIES = 3

# ==============================
# HELPERS
# ==============================
def normalize_username(raw: str) -> str:
    """Reduce any of https://t.me/x, t.me/x, @x, x  ->  x"""
    u = raw.strip()
    for prefix in ("https://", "http://"):
        if u.startswith(prefix):
            u = u[len(prefix):]
    if u.startswith("t.me/"):
        u = u[len("t.me/"):]
    u = u.lstrip("@").strip("/")
    return u


# ==============================
# MAIN
# ==============================
async def download_channel_media(client, username: str, channel_dir: Path):
    """Download images only for one channel."""
    stats = {"images": 0, "skipped": 0, "errors": 0}

    try:
        entity = await client.get_entity(username)
    except Exception as e:
        print(f"  x Could not get entity for {username}: {e}")
        stats["errors"] += 1
        return stats

    async for message in client.iter_messages(entity, limit=POST_LIMIT):
        if not message.media:
            continue

        # Images only -- video/audio are out of scope
        if not isinstance(message.media, MessageMediaPhoto):
            stats["skipped"] += 1
            continue

        filename = channel_dir / f"{message.id}.jpg"
        if filename.exists():
            stats["images"] += 1
            continue

        for attempt in range(MAX_RETRIES):
            try:
                await client.download_media(message, file=str(filename))
                stats["images"] += 1
                break
            except FloodWaitError as e:
                print(f"  Flood wait {e.seconds}s ...")
                await asyncio.sleep(e.seconds)
            except Exception:
                if attempt == MAX_RETRIES - 1:
                    stats["errors"] += 1
                else:
                    await asyncio.sleep(2 ** attempt)

    return stats


async def main():
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    usernames = [normalize_username(entry["username"])
                 for entry in all_data if entry.get("username")]
    print(f"Found {len(usernames)} channels in JSON.\n")

    output_root = Path(OUTPUT_DIR)
    output_root.mkdir(exist_ok=True)

    summary = {}

    async with TelegramClient("research_session", api_id, api_hash) as client:
        for username in tqdm(usernames, desc="Channels"):
            channel_dir = output_root / username
            done_marker = channel_dir / ".done"

            # Resume: skip channels already finished
            if done_marker.exists():
                print(f"\nSkipping (already done): {username}")
                continue

            print(f"\nProcessing: {username}")
            channel_dir.mkdir(exist_ok=True)

            stats = await download_channel_media(client, username, channel_dir)
            summary[username] = stats

            # Mark complete only if no errors
            if stats["errors"] == 0:
                done_marker.touch()

            print(f"  images={stats['images']} "
                  f"skipped={stats['skipped']} errors={stats['errors']}")

            await asyncio.sleep(random.uniform(*DELAY_RANGE))

    with open("download_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    total_imgs = sum(s["images"] for s in summary.values())
    print(f"\nDone. Total images this run: {total_imgs}")
    print("Summary written to download_summary.json")


if __name__ == "__main__":
    asyncio.run(main())