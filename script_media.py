import asyncio
import json
import os
import random
import time
from pathlib import Path

from telethon import TelegramClient
from telethon.errors import FloodWaitError
from telethon.tl.types import MessageMediaPhoto, MessageMediaDocument
from tqdm import tqdm

# ==============================
# CONFIG
# ==============================
import os
api_id = int(os.environ["TG_API_ID"])
api_hash = os.environ["TG_API_HASH"]      # your api_hash here

INPUT_JSON     = "telegram_channels_full_data.json"
OUTPUT_DIR     = "media"
MAX_VIDEO_SIZE = 50 * 1024 * 1024   # 50 MB cap for videos
POST_LIMIT     = 500
DELAY_RANGE    = (2, 5)

# ==============================
# HELPERS
# ==============================
PHOTO_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
AUDIO_EXTS = {".mp3", ".ogg", ".wav", ".m4a", ".flac", ".opus"}

def get_extension(mime_type: str) -> str:
    mapping = {
        "image/jpeg":       ".jpg",
        "image/png":        ".png",
        "image/webp":       ".webp",
        "video/mp4":        ".mp4",
        "video/x-matroska": ".mkv",
        "video/webm":       ".webm",
        "audio/mpeg":       ".mp3",
        "audio/ogg":        ".ogg",
        "audio/opus":       ".opus",
        "audio/mp4":        ".m4a",
    }
    return mapping.get(mime_type, "")

# ==============================
# MAIN
# ==============================
async def download_channel_media(client, username: str, channel_dir: Path):
    """Download images, videos (<=50MB), and audio for one channel."""
    stats = {"images": 0, "videos": 0, "audio": 0, "skipped": 0, "errors": 0}

    try:
        entity = await client.get_entity(username)
    except Exception as e:
        print(f"  ✗ Could not get entity for {username}: {e}")
        return stats

    async for message in client.iter_messages(entity, limit=POST_LIMIT):
        if not message.media:
            continue

        for attempt in range(3):  # retry up to 3 times
            try:
                # ── PHOTOS ──────────────────────────────────────
                if isinstance(message.media, MessageMediaPhoto):
                    filename = channel_dir / f"{message.id}.jpg"
                    if not filename.exists():
                        await client.download_media(message, file=str(filename))
                    stats["images"] += 1
                    break

                # ── DOCUMENTS (video / audio / gif) ─────────────
                elif isinstance(message.media, MessageMediaDocument):
                    doc       = message.media.document
                    mime      = doc.mime_type or ""
                    ext       = get_extension(mime)
                    file_size = doc.size

                    if not ext:
                        stats["skipped"] += 1
                        break

                    if ext in VIDEO_EXTS:
                        if file_size > MAX_VIDEO_SIZE:
                            stats["skipped"] += 1
                            break
                        filename = channel_dir / f"{message.id}{ext}"
                        if not filename.exists():
                            await client.download_media(message, file=str(filename))
                        stats["videos"] += 1
                        break

                    elif ext in AUDIO_EXTS:
                        filename = channel_dir / f"{message.id}{ext}"
                        if not filename.exists():
                            await client.download_media(message, file=str(filename))
                        stats["audio"] += 1
                        break

                    else:
                        stats["skipped"] += 1
                        break

            except FloodWaitError as e:
                print(f"  Flood wait {e.seconds}s …")
                await asyncio.sleep(e.seconds)
            except ConnectionError as e:
                print(f"  Connection error on message {message.id}, attempt {attempt+1}/3: {e}")
                await asyncio.sleep(5)
                if attempt == 2:
                    stats["errors"] += 1
            except Exception as e:
                stats["errors"] += 1
                break

    return stats


async def main():
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    usernames = [
        entry["username"].replace("https://t.me/", "").replace("@", "").strip()
        for entry in all_data if entry.get("username")
    ]
    print(f"Found {len(usernames)} channels in JSON.\n")

    # ── TEST MODE: change to usernames[:10] for 10 channels, remove line for full run ──
    usernames = usernames[:10]
    print("TEST MODE: running on first 10 channels only.\n")

    output_root = Path(OUTPUT_DIR)
    output_root.mkdir(exist_ok=True)

    summary = {}

    async with TelegramClient("research_session", api_id, api_hash) as client:
        for username in tqdm(usernames, desc="Channels"):
            print(f"\nProcessing: {username}")

            channel_dir = output_root / username
            channel_dir.mkdir(exist_ok=True)

            stats = await download_channel_media(client, username, channel_dir)
            summary[username] = stats

            print(f"  ✓ images={stats['images']}  videos={stats['videos']}  "
                  f"audio={stats['audio']}  skipped={stats['skipped']}  errors={stats['errors']}")

            await asyncio.sleep(random.uniform(*DELAY_RANGE))

    # ── SUMMARY ──────────────────────────────────────────────────────
    total_images = sum(v["images"] for v in summary.values())
    total_videos = sum(v["videos"] for v in summary.values())
    total_audio  = sum(v["audio"]  for v in summary.values())
    total_skip   = sum(v["skipped"] for v in summary.values())

    print("\n" + "="*50)
    print("DOWNLOAD COMPLETE")
    print(f"  Total images : {total_images}")
    print(f"  Total videos : {total_videos}")
    print(f"  Total audio  : {total_audio}")
    print(f"  Skipped      : {total_skip}  (docs, oversized videos, unknown types)")
    print(f"  Media saved to: {output_root.resolve()}")

    with open("media_download_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("  Summary saved to: media_download_summary.json")


if __name__ == "__main__":
    asyncio.run(main())