"""
draw_heldout.py  --  Draw the HELD-OUT VALIDATION sample (30 unseen posts).

This is the reliability gate: a fresh set of posts NEITHER annotator has seen
or discussed, drawn with the SAME stratification logic as the main sample, so
kappa on it is a clean, memory-free estimate of scheme reliability.

It mirrors draw_sample.py (same filtering, same floor/ceiling allocation, same
SEED-based reproducibility) but:
  - EXCLUDES posts already in the existing annotation_sample.csv, so the
    held-out set does not overlap what you've already drawn.
  - Uses a DIFFERENT SEED, so it doesn't just re-pick the same posts.
  - Numbers posts H0001..H0030 (separate namespace; cannot collide with Axxxx).

Outputs:
  heldout_sample.csv   (blind: ann_id, channel, post_id, image_path,
                        caption, label, notes  -- label/notes blank)
  heldout_sample.json  (provenance: seed, params, excluded count, sample)
"""

import csv
import json
import os
import random
import re
from pathlib import Path

# ==============================
# CONFIG  (mirror draw_sample.py; change only where noted)
# ==============================
INPUT_JSON        = "telegram_channels_full_data.json"
MEDIA_DIR         = "media"
EXISTING_CSV      = "annotation_sample.csv"   # the already-drawn 400
OUT_CSV           = "heldout_sample.csv"
OUT_JSON          = "heldout_sample.json"

TARGET_N          = 30      # validation round size
MIN_PER_CHANNEL   = 1
MAX_PER_CHANNEL   = 12
MIN_CAPTION_CHARS = 10      # must match the audit/main-sample threshold
SEED              = 4242    # DIFFERENT from the main sample's 42, so we don't
                            # re-draw identical picks; still fixed & reproducible
ID_PREFIX         = "H"     # H0001.. -> separate namespace from A0001..

# What to hold out from the existing sample.
#   "full_sample" (default, recommended): exclude ALL posts already in
#       annotation_sample.csv -> held-out is disjoint from your gold set.
#   "pilot_only": exclude only the 30 pilot posts (A0001-A0030) -> held-out
#       may include not-yet-labelled posts from the existing 400 (they'd then
#       be double-represented once you run the full pass).
EXCLUDE           = "full_sample"
PILOT_IDS         = {f"A{i:04d}" for i in range(1, 31)}

# ==============================
# HELPERS (identical to draw_sample.py)
# ==============================
def normalize_username(raw: str) -> str:
    u = (raw or "").strip()
    for p in ("https://", "http://"):
        if u.startswith(p):
            u = u[len(p):]
    if u.startswith("t.me/"):
        u = u[len("t.me/"):]
    return u.lstrip("@").strip("/")


_URL_RE   = re.compile(r"https?://\S+")
_WS_RE    = re.compile(r"\s+")


def clean_caption(text: str) -> str:
    t = _URL_RE.sub("", text or "")
    t = _WS_RE.sub(" ", t).strip()
    return t


def load_existing_keys(media_root="."):
    """Return the set of (channel, post_id) already drawn, per EXCLUDE mode."""
    keys = set()
    path = EXISTING_CSV
    if not os.path.exists(path):
        alt = os.path.join(media_root, EXISTING_CSV)
        path = alt if os.path.exists(alt) else None
    if path is None:
        print(f"WARNING: {EXISTING_CSV} not found (here or under --root) -- "
              f"nothing excluded. Held-out set may overlap your existing sample.")
        return keys
    with open(path, "r", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if EXCLUDE == "pilot_only" and r.get("ann_id") not in PILOT_IDS:
                continue
            keys.add((r["channel"], str(r["post_id"])))
    return keys


def build_pairs(media_root=".", json_path=None):
    """Reconstruct the annotatable image-caption pairs from the source JSON,
    mirroring how draw_sample.py filters them. The stored image_path stays
    relative (media/<channel>/<id>.jpg) so downstream bundling is unchanged;
    only the existence CHECK is resolved against media_root."""
    with open(json_path or INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    # data may be a list of channels or a dict keyed by username
    channels = data.values() if isinstance(data, dict) else data

    by_channel = {}
    for ch in channels:
        uname = normalize_username(
            ch.get("username") or ch.get("metadata", {}).get("username", ""))
        if not uname:
            continue
        posts = ch.get("recent_posts", []) or []
        pairs = []
        for p in posts:
            pid = str(p.get("id") or p.get("post_id") or "")
            if not pid:
                continue
            cap = clean_caption(p.get("text") or p.get("caption") or "")
            if len(cap) < MIN_CAPTION_CHARS:
                continue
            img_rel = os.path.join(MEDIA_DIR, uname, f"{pid}.jpg")
            img_abs = os.path.join(media_root, img_rel)
            if not os.path.exists(img_abs):
                continue
            pairs.append({"channel": uname, "post_id": pid,
                          "image_path": img_rel, "caption": cap})
        if pairs:
            by_channel[uname] = pairs
    return by_channel


def allocate(by_channel, target):
    """Allocate `target` posts across channels.

    Two regimes:
      (a) target >= number of channels -> floor MIN_PER_CHANNEL, ceiling
          MAX_PER_CHANNEL, remainder proportional (same policy as
          draw_sample.py, used for the large main sample).
      (b) target <  number of channels -> a floor of 1 per channel is
          arithmetically impossible, so instead pick `target` channels at
          random and take exactly 1 post from each. This maximises channel
          diversity, which is what a reliability round needs: 30 independent
          contexts rather than several near-identical posts from one channel.
    """
    chans = list(by_channel)
    sizes = {c: len(by_channel[c]) for c in chans}
    total = sum(sizes.values())

    if total < target:
        print(f"WARNING: only {total} eligible unseen pairs available "
              f"(< target {target}). Drawing all of them.")
        return {c: min(sizes[c], MAX_PER_CHANNEL) for c in chans}

    # --- regime (b): fewer posts wanted than channels available ---
    if target < len(chans):
        picked = random.sample(chans, target)
        print(f"NOTE: target ({target}) < channels ({len(chans)}); "
              f"drawing 1 post from each of {target} randomly chosen channels "
              f"to maximise channel diversity.")
        return {c: 1 for c in picked}

    # --- regime (a): original proportional policy ---
    alloc = {c: MIN_PER_CHANNEL for c in chans}
    remaining = target - sum(alloc.values())
    if remaining > 0:
        weight_total = sum(max(0, sizes[c] - MIN_PER_CHANNEL) for c in chans)
        if weight_total > 0:
            for c in chans:
                extra = int(round(
                    remaining * max(0, sizes[c] - MIN_PER_CHANNEL) / weight_total))
                alloc[c] = min(MAX_PER_CHANNEL, min(sizes[c], alloc[c] + extra))

    def total_alloc():
        return sum(alloc.values())
    order = sorted(chans, key=lambda c: sizes[c], reverse=True)
    i = 0
    guard = 0
    while total_alloc() != target and guard < 100000:
        c = order[i % len(order)]
        if total_alloc() < target and alloc[c] < min(MAX_PER_CHANNEL, sizes[c]):
            alloc[c] += 1
        elif total_alloc() > target and alloc[c] > MIN_PER_CHANNEL:
            alloc[c] -= 1
        i += 1
        guard += 1
    return alloc


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Draw the held-out validation sample.")
    ap.add_argument("--root", default=".",
                    help=r'Folder containing media\<channel>\<id>.jpg '
                         r'(e.g. "C:\Users\kiran\OneDrive - University of '
                         r'Edinburgh\Dissertation"). Defaults to current folder.')
    args = ap.parse_args()
    media_root = args.root

    random.seed(SEED)

    json_path = os.path.join(media_root, INPUT_JSON)
    if not os.path.exists(json_path):
        # fall back to current folder for backward compatibility
        if os.path.exists(INPUT_JSON):
            json_path = INPUT_JSON
        else:
            raise SystemExit(
                f"Missing {INPUT_JSON} (looked under --root and here).")
    if not os.path.isdir(os.path.join(media_root, MEDIA_DIR)):
        raise SystemExit(
            f"No '{MEDIA_DIR}' folder under --root: {media_root!r}\n"
            f"       Pass the folder that CONTAINS media/, e.g.\n"
            f'       python draw_heldout.py --root "C:\\Users\\kiran\\'
            f'OneDrive - University of Edinburgh\\Dissertation"')

    excluded = load_existing_keys(media_root)
    print(f"Exclude mode        : {EXCLUDE}  "
          f"({len(excluded)} posts held out from the draw)")

    by_channel_all = build_pairs(media_root, json_path)

    # remove already-sampled posts
    by_channel = {}
    dropped = 0
    for c, pairs in by_channel_all.items():
        keep = [p for p in pairs if (p["channel"], p["post_id"]) not in excluded]
        dropped += len(pairs) - len(keep)
        if keep:
            by_channel[c] = keep
    print(f"Eligible unseen pool: "
          f"{sum(len(v) for v in by_channel.values())} pairs "
          f"across {len(by_channel)} channels  "
          f"({dropped} excluded as already-sampled)")

    alloc = allocate(by_channel, TARGET_N)

    sample = []
    for c, n in alloc.items():
        if n <= 0:
            continue
        picks = random.sample(by_channel[c], min(n, len(by_channel[c])))
        sample.extend(picks)

    random.shuffle(sample)   # mix channels so annotators don't see runs
    for i, rec in enumerate(sample, 1):
        rec["ann_id"] = f"{ID_PREFIX}{i:04d}"

    # safety check: no ID or post collision with the existing sample
    assert all(r["ann_id"].startswith(ID_PREFIX) for r in sample)
    overlap = [(r["channel"], r["post_id"]) for r in sample
               if (r["channel"], r["post_id"]) in excluded]
    assert not overlap, f"Held-out overlaps excluded posts: {overlap[:5]}"
    assert len(sample) == TARGET_N, (
        f"Drew {len(sample)} posts but TARGET_N is {TARGET_N}. "
        f"Allocation is wrong -- do not use this sample.")

    print(f"Held-out drawn      : {len(sample)} posts "
          f"across {len(set(r['channel'] for r in sample))} channels")
    if alloc:
        used = {c: n for c, n in alloc.items() if n > 0}
        print(f"Per-channel range   : {min(used.values())}..{max(used.values())}")

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"seed": SEED, "target": TARGET_N,
                   "exclude_mode": EXCLUDE,
                   "excluded_count": len(excluded),
                   "min_per_channel": MIN_PER_CHANNEL,
                   "max_per_channel": MAX_PER_CHANNEL,
                   "id_prefix": ID_PREFIX,
                   "sample": sample}, f, ensure_ascii=False, indent=2)

    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ann_id", "channel", "post_id", "image_path",
                    "caption", "label", "notes"])
        for r in sample:
            w.writerow([r["ann_id"], r["channel"], r["post_id"],
                        r["image_path"], r["caption"], "", ""])

    print(f"\nWritten: {OUT_CSV}   (label/notes blank, blind for annotators)")
    print(f"Written: {OUT_JSON}  (provenance, seed={SEED})")
    print("\nNOTE: captions here are the RAW Russian text. If your annotators")
    print("label from English, run your translation step on heldout_sample.csv")
    print("to add caption_en before bundling (same as the main sample).")
    print("\nLabel scheme reminder: propagandistic / neutral / unclear")


if __name__ == "__main__":
    main()
