"""
Merge rescrape_34.json into telegram_channels_full_data.json.

For each re-scraped channel, replace that channel's recent_posts (and refresh
its metadata) in the main JSON. All other channels are left untouched.

Safety: writes a timestamped backup of the main JSON before modifying.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path

MAIN_JSON     = "telegram_channels_full_data.json"
RESCRAPE_JSON = "rescrape_34.json"


def normalize_username(raw: str) -> str:
    u = (raw or "").strip()
    for p in ("https://", "http://"):
        if u.startswith(p):
            u = u[len(p):]
    if u.startswith("t.me/"):
        u = u[len("t.me/"):]
    return u.lstrip("@").strip("/")


def key_of(entry):
    meta = entry.get("metadata") or {}
    return normalize_username(meta.get("username")
                              or normalize_username(entry.get("username", "")))


def main():
    with open(MAIN_JSON, "r", encoding="utf-8") as f:
        main_data = json.load(f)
    with open(RESCRAPE_JSON, "r", encoding="utf-8") as f:
        rescrape = json.load(f)

    # backup first
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = f"telegram_channels_full_data.backup_{stamp}.json"
    shutil.copy(MAIN_JSON, backup)
    print(f"Backup written: {backup}")

    # index re-scraped channels by normalized username
    rescrape_by_key = {}
    for r in rescrape:
        if r.get("recent_posts"):           # only use successful scrapes
            rescrape_by_key[normalize_username(r["username"])] = r

    updated, skipped = [], []
    for entry in main_data:
        k = key_of(entry)
        if k in rescrape_by_key:
            r = rescrape_by_key[k]
            old_n = len(entry.get("recent_posts") or [])
            entry["recent_posts"] = r["recent_posts"]
            if r.get("metadata"):
                entry["metadata"] = r["metadata"]
            entry["rescraped"] = True
            updated.append((k, old_n, len(r["recent_posts"])))

    matched_keys = {u for u, _, _ in updated}
    for k in rescrape_by_key:
        if k not in matched_keys:
            skipped.append(k)

    with open(MAIN_JSON, "w", encoding="utf-8") as f:
        json.dump(main_data, f, ensure_ascii=False, indent=2)

    print(f"\nUpdated {len(updated)} channels:")
    for k, old_n, new_n in updated:
        print(f"  {k:25s}  posts {old_n} -> {new_n}")

    if skipped:
        print(f"\n! Re-scraped but NOT matched in main JSON "
              f"(check username): {skipped}")

    print(f"\nMain JSON updated. Re-run audit_dataset.py to see new pair count.")


if __name__ == "__main__":
    main()