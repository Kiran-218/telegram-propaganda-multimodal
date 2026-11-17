import asyncio
import json
import random
import time
from datetime import datetime

import pandas as pd
from telethon import TelegramClient
from telethon.errors import FloodWaitError
from telethon.tl.functions.channels import GetFullChannelRequest
from telethon.tl.types import ChannelParticipantsAdmins
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

OUTPUT_JSON = "telegram_channels_full_data.json"
RECENT_POST_LIMIT = 500   # Adjust if needed
DELAY_RANGE = (2, 5)     # Random delay between channels (seconds)

# ==============================
# MAIN SCRIPT
# ==============================

async def fetch_channel_data(client, username):
    data = {
        "username": username,
        "metadata": {},
        "admins": [],
        "recent_posts": [],
        "statistics": {},
    }

    try:
        entity = await client.get_entity(username)
        full = await client(GetFullChannelRequest(entity))

        # ==============================
        # BASIC METADATA
        # ==============================

        data["metadata"] = {
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

        # ==============================
        # ADMINS
        # ==============================

        # admins = await client.get_participants(
        #     entity,
        #     filter=ChannelParticipantsAdmins
        # )

        # for admin in admins:
        #     data["admins"].append({
        #         "id": admin.id,
        #         "username": admin.username,
        #         "first_name": admin.first_name,
        #         "last_name": admin.last_name,
        #         "bot": admin.bot,
        #     })

        # ==============================
        # RECENT POSTS
        # ==============================

        # async for message in client.iter_messages(entity, limit=RECENT_POST_LIMIT):
        #     data["recent_posts"].append({
        #         "id": message.id,
        #         "date": message.date.isoformat() if message.date else None,
        #         "text": message.text,
        #         "views": message.views,
        #         "forwards": message.forwards,
        #         "replies": message.replies.replies if message.replies else None,
        #     })
        
       
        try:
            async for message in client.iter_messages(entity, limit=RECENT_POST_LIMIT):
                data["recent_posts"].append({
                    "id": message.id,
                    "date": message.date.isoformat() if message.date else None,
                    "text": message.text,
                    "views": message.views,
                    "forwards": message.forwards,
                    "replies": message.replies.replies if message.replies else None,
                })
        except Exception as e:
            data["recent_posts_error"] = str(e)

        # ==============================
        # STATISTICS (if accessible)
        # ==============================

        try:
            stats = await client.get_stats(entity)
            data["statistics"] = str(stats)  # Raw dump (can parse if needed)
        except Exception:
            data["statistics"] = "Not accessible (requires admin privileges)"

    except FloodWaitError as e:
        print(f"Flood wait for {e.seconds} seconds...")
        time.sleep(e.seconds)
        return await fetch_channel_data(client, username)

    except Exception as e:
        data["error"] = str(e)

    return data


async def main():
    all_data = []

    async with TelegramClient("research_session", api_id, api_hash) as client:
        for username in tqdm(channels):
            print(f"\nProcessing: {username}")

            result = await fetch_channel_data(client, username)
            all_data.append(result)

            # Random delay to avoid rate limits
            await asyncio.sleep(random.uniform(*DELAY_RANGE))

    # Save JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    print("\n✅ Data collection complete.")
    print(f"Saved to: {OUTPUT_JSON}")


if __name__ == "__main__":
    asyncio.run(main())