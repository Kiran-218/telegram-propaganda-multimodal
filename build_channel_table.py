"""
Build the M5 channel-level feature table (spine for the channel analysis).

Joins THREE feature groups into one row-per-channel dataframe (~92 channels):

  1. Mismatch aggregates  (from corpus_mclip_scores.csv)
       - n_scored              posts with a valid M-CLIP score in this channel
       - mismatch_mean
       - mismatch_std
       - mismatch_var

  2. Channel metadata     (from telegram_channels_full_data.json -> metadata)
       - participants_count
       - channel_age_days      days between creation_date and REF_DATE
       - verified / scam / fake / restricted   (0/1)

  3. Activity aggregates  (from the same JSON -> recent_posts)
       - n_posts               total posts scraped for the channel
       - forward_rate          mean forwards per post
       - posts_per_day         posting frequency over the observed span
       - mean_views

NOTE on data sources:
  * The channel key is recovered from the M-CLIP 'ann_id' (== "<channel>__<post_id>").
  * Activity features are computed by AGGREGATING recent_posts, NOT from the
    JSON 'statistics' field -- that field is a stringified raw dump and is not
    reliably parseable.
  * Mismatch is a PROXY SIGNAL under investigation; it is one feature group of
    several here, not a propaganda label. Its per-channel spread is expected to
    be small (corpus std ~0.037).

Usage:
  python build_channel_table.py \
      --scores corpus_mclip_scores.csv \
      --json   telegram_channels_full_data.json \
      --out    channel_features.csv \
      --ref-date 2026-06-09

Channel mismatch is summarised threshold-free (mean, std, var). Mismatch is a
PROXY SIGNAL under investigation -- one feature group of several here, not a
propaganda label. Its per-channel spread is expected to be small (corpus std
~0.037). The verified/scam/fake flags are retained but were near-empty across
the sampled war channels, so no flag-based grouping / Mann-Whitney test is run.
"""

import argparse
import json
import re
from datetime import datetime, timezone

import pandas as pd


def normalize_username(raw: str) -> str:
    u = (raw or "").strip()
    for p in ("https://", "http://"):
        if u.startswith(p):
            u = u[len(p):]
    if u.startswith("t.me/"):
        u = u[len("t.me/"):]
    return u.lstrip("@").strip("/")


def parse_iso(s):
    """Parse an ISO timestamp; return timezone-aware UTC datetime or None."""
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def channel_from_ann_id(ann_id: str) -> str:
    # ann_id == "<channel>__<post_id>"; channel may itself contain no "__"
    return str(ann_id).rsplit("__", 1)[0]


# ---------------------------------------------------------------------------
# 1. mismatch aggregates
# ---------------------------------------------------------------------------
def mismatch_aggregates(scores_csv):
    df = pd.read_csv(scores_csv)
    # keep only successfully scored rows
    df = df[df["status"] == "ok"].copy()
    df["channel"] = df["ann_id"].map(channel_from_ann_id)
    df["mclip_mismatch"] = pd.to_numeric(df["mclip_mismatch"], errors="coerce")
    df = df.dropna(subset=["mclip_mismatch"])

    g = df.groupby("channel")["mclip_mismatch"]
    agg = pd.DataFrame({
        "n_scored": g.size(),
        "mismatch_mean": g.mean(),
        "mismatch_std": g.std(),
        "mismatch_var": g.var(),
    })
    return agg.reset_index()


# ---------------------------------------------------------------------------
# 2 + 3. metadata + activity, from the JSON
# ---------------------------------------------------------------------------
def metadata_and_activity(json_path, ref_date):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for entry in data:
        meta = entry.get("metadata") or {}
        uname = normalize_username(meta.get("username")
                                   or normalize_username(entry.get("username", "")))
        if not uname:
            continue

        # --- metadata features ---
        created = parse_iso(meta.get("creation_date"))
        age_days = (ref_date - created).days if created else None

        def as01(v):
            return 1 if v is True else (0 if v is False else None)

        # --- activity features (aggregate recent_posts) ---
        posts = entry.get("recent_posts") or []
        n_posts = len(posts)
        fwds, views, dates = [], [], []
        for p in posts:
            fw = p.get("forwards")
            if isinstance(fw, (int, float)):
                fwds.append(fw)
            vw = p.get("views")
            if isinstance(vw, (int, float)):
                views.append(vw)
            d = parse_iso(p.get("date"))
            if d:
                dates.append(d)

        forward_rate = (sum(fwds) / len(fwds)) if fwds else None
        mean_views = (sum(views) / len(views)) if views else None

        posts_per_day = None
        if len(dates) >= 2:
            span_days = (max(dates) - min(dates)).days
            posts_per_day = n_posts / span_days if span_days > 0 else None

        rows.append({
            "channel": uname,
            "participants_count": meta.get("participants_count"),
            "channel_age_days": age_days,
            "verified": as01(meta.get("verified")),
            "scam": as01(meta.get("scam")),
            "fake": as01(meta.get("fake")),
            "restricted": as01(meta.get("restricted")),
            "n_posts": n_posts,
            "forward_rate": forward_rate,
            "posts_per_day": posts_per_day,
            "mean_views": mean_views,
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="corpus_mclip_scores.csv")
    ap.add_argument("--json", default="telegram_channels_full_data.json")
    ap.add_argument("--out", default="channel_features.csv")
    ap.add_argument("--ref-date", default=None,
                    help="reference date for channel age (ISO, default: today UTC)")
    args = ap.parse_args()

    ref_date = parse_iso(args.ref_date) if args.ref_date else datetime.now(timezone.utc)

    mm = mismatch_aggregates(args.scores)
    ma = metadata_and_activity(args.json, ref_date)

    # LEFT-join metadata onto the channels that actually have scored posts:
    # the analysis population is the ~92 channels with valid image-caption pairs.
    tbl = mm.merge(ma, on="channel", how="left")

    # order columns sensibly
    front = ["channel", "n_scored", "n_posts"]
    mismatch_cols = ["mismatch_mean", "mismatch_std", "mismatch_var"]
    behav = ["participants_count", "channel_age_days", "forward_rate",
             "posts_per_day", "mean_views"]
    flags = ["verified", "scam", "fake", "restricted"]
    cols = front + mismatch_cols + behav + flags
    tbl = tbl[[c for c in cols if c in tbl.columns]]

    tbl = tbl.sort_values("channel").reset_index(drop=True)
    tbl.to_csv(args.out, index=False, encoding="utf-8-sig")

    print(f"Channels in table     : {len(tbl)}")
    print(f"Reference date (age)  : {ref_date.date().isoformat()}")
    miss_meta = tbl["participants_count"].isna().sum()
    if miss_meta:
        print(f"NOTE: {miss_meta} channels had no matched metadata "
              f"(check username normalisation).")
    # quick flag tallies
    for fcol in flags:
        if fcol in tbl.columns:
            print(f"  {fcol}=1 : {(tbl[fcol] == 1).sum()} channels")
    print(f"Written: {args.out}")
    print("\nNote: verified/scam/fake flags are retained but near-empty across "
          "the sample; documented-but-unused (no flag-based grouping possible).")
    print("Next: perceptual image-reuse rate joins onto this table, then "
          "Spearman correlations and HDBSCAN clustering.")


if __name__ == "__main__":
    main()
