"""
draw_overlap.py  --  Draw the 100-post OVERLAP SUBSET for the second annotator.

The full 400-post gold set is annotated by you. A random subset is also
annotated independently by the second annotator, and Cohen's kappa on that
subset is the reported inter-annotator reliability for the gold set.

Design decisions baked in (documented in the output JSON):
  - The subset is drawn RANDOMLY from the 400, not taken as the first N,
    so it is representative of the whole gold set and kappa on it credibly
    generalises to the posts he did not label.
  - The 30 pilot posts (A0001-A0030) are EXCLUDED from the draw by default.
    They stay in the 400-post gold set, but both annotators have already
    seen and discussed them across two pilot rounds, so kappa computed on
    them would be inflated by that prior exposure. Drawing the overlap from
    the 370 never-discussed posts keeps the reported reliability clean.

Usage:
    python draw_overlap.py --root "C:\\...\\Dissertation"

Reads (looked for here, then under --root):
    annotation_sample_translated_1.3B.csv   (the 400, with caption_en)
Writes:
    overlap_100.csv    (blind: label/notes blank -- bundle this for him)
    overlap_100.json   (provenance: seed, exclusions, the drawn ann_ids)
"""

import argparse
import csv
import json
import os
import random

# ==============================
# CONFIG
# ==============================
SOURCE_CSV   = "annotation_sample_translated_1.3B.csv"   # the 400 w/ caption_en
OUT_CSV      = "overlap_100.csv"
OUT_JSON     = "overlap_100.json"

TARGET_N     = 100
SEED         = 100100        # fixed & documented, distinct from other draws

# Exclude the pilot posts from the overlap draw (see module docstring).
EXCLUDE_PILOT = True
PILOT_IDS     = {f"A{i:04d}" for i in range(1, 31)}

# Columns the annotation server needs. Everything else is dropped so the
# bundle stays blind (no existing labels, no model output, no channel flags).
KEEP_COLS = ["ann_id", "channel", "post_id", "image_path",
             "caption", "caption_en"]
BLANK_COLS = ["label", "notes"]


def resolve(path, root):
    """Find a file here, else under --root."""
    if os.path.exists(path):
        return path
    alt = os.path.join(root, path)
    return alt if os.path.exists(alt) else None


def main():
    ap = argparse.ArgumentParser(
        description="Draw the 100-post overlap subset for the second annotator.")
    ap.add_argument("--root", default=".",
                    help="Folder holding the dataset files, if not the current one.")
    ap.add_argument("--n", type=int, default=TARGET_N,
                    help=f"Overlap size (default {TARGET_N}).")
    ap.add_argument("--include-pilot", action="store_true",
                    help="Include the 30 pilot posts in the draw "
                         "(default: exclude them).")
    args = ap.parse_args()

    src = resolve(SOURCE_CSV, args.root)
    if src is None:
        raise SystemExit(
            f"Missing {SOURCE_CSV} (looked here and under --root).\n"
            f"       This should be your 400-post sample WITH the caption_en column.")

    with open(src, "r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    print(f"Source: {src}")
    print(f"Loaded : {len(rows)} posts")

    missing_cols = [c for c in KEEP_COLS if c not in rows[0]]
    if missing_cols:
        raise SystemExit(
            f"{SOURCE_CSV} is missing expected column(s): {missing_cols}\n"
            f"       Columns present: {list(rows[0].keys())}")

    exclude_pilot = EXCLUDE_PILOT and not args.include_pilot
    if exclude_pilot:
        pool = [r for r in rows if r["ann_id"] not in PILOT_IDS]
        print(f"Excluded: {len(rows) - len(pool)} pilot posts "
              f"(A0001-A0030; already seen and discussed in both pilots)")
    else:
        pool = list(rows)
        print("NOTE: pilot posts INCLUDED in the draw (--include-pilot). "
              "Kappa on those posts is inflated by prior exposure.")

    if len(pool) < args.n:
        raise SystemExit(
            f"Pool has only {len(pool)} posts but --n is {args.n}.")

    random.seed(SEED)
    picked = random.sample(pool, args.n)
    picked.sort(key=lambda r: r["ann_id"])   # stable, readable order

    # sanity checks
    ids = [r["ann_id"] for r in picked]
    assert len(set(ids)) == len(ids), "duplicate ann_id in draw"
    if exclude_pilot:
        assert not (set(ids) & PILOT_IDS), "pilot post leaked into overlap draw"
    assert len(picked) == args.n

    print(f"Pool   : {len(pool)} eligible posts")
    print(f"Drawn  : {len(picked)} posts for the overlap subset")
    chans = {r["channel"] for r in picked}
    print(f"Channels represented: {len(chans)}")

    # blind CSV for the annotator
    header = KEEP_COLS + BLANK_COLS
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in picked:
            w.writerow([r.get(c, "") for c in KEEP_COLS] + ["", ""])

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "seed": SEED,
            "n": args.n,
            "source_csv": SOURCE_CSV,
            "source_total": len(rows),
            "pilot_excluded": exclude_pilot,
            "pool_size": len(pool),
            "ann_ids": ids,
        }, f, ensure_ascii=False, indent=2)

    print(f"\nWritten: {OUT_CSV}   (blind -- label/notes blank)")
    print(f"Written: {OUT_JSON}  (provenance, seed={SEED})")
    print(f"\nNext: build his bundle from {OUT_CSV}, and annotate all 400 yourself.")
    print("Kappa is computed on these 100 ann_ids only.")


if __name__ == "__main__":
    main()
