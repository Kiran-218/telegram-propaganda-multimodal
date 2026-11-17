# apply_translation_quality.py
# Adds a translation_quality column to the gold set and reports how many
# flagged (collapse / source_junk) posts received a substantive label.
#
# Output: annotation_labels_full_flagged.csv  (original + translation_quality)
# The original annotation_labels_full.csv is NOT modified.

import csv

LABELS_IN = "annotation_labels_gold_400.csv"
LABELS_OUT = "annotation_labels_full_flagged.csv"
ID_COL = "ann_id"
LABEL_COL = "label"

# --- Reviewed flag lists (each ann_id in exactly one bucket) -----------------
COLLAPSE = {
    "A0367", "A0228", "A0243", "A0192", "A0184", "A0177", "A0194", "A0398",
    "A0391", "A0318", "A0290", "A0256", "A0197", "A0353", "A0130", "A0079",
    "A0135", "A0017",
}
SOURCE_JUNK = {
    "A0305", "A0190", "A0134", "A0076", "A0156", "A0242", "A0324", "A0252",
    "A0090", "A0196", "A0232", "A0012",
}
# Everything else -> "ok". A0104, A0231, A0371, A0128, A0352 reviewed-and-kept.


def quality_for(ann_id):
    if ann_id in COLLAPSE:
        return "collapse"
    if ann_id in SOURCE_JUNK:
        return "source_junk"
    return "ok"


def main():
    # sanity: no id in two buckets
    overlap = COLLAPSE & SOURCE_JUNK
    if overlap:
        raise SystemExit(f"ERROR: ids in both buckets: {sorted(overlap)}")

    rows = []
    fieldnames = None
    seen = set()
    with open(LABELS_IN, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        for row in reader:
            ann = row.get(ID_COL, "")
            seen.add(ann)
            row["translation_quality"] = quality_for(ann)
            rows.append(row)

    if "translation_quality" not in fieldnames:
        fieldnames.append("translation_quality")

    with open(LABELS_OUT, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # --- report -------------------------------------------------------------
    flagged = COLLAPSE | SOURCE_JUNK
    missing = sorted(flagged - seen)

    def tally(bucket):
        counts = {}
        substantive = []
        for r in rows:
            if r.get(ID_COL) in bucket:
                lab = (r.get(LABEL_COL) or "").strip().lower()
                counts[lab] = counts.get(lab, 0) + 1
                if lab and lab != "unclear":
                    substantive.append((r.get(ID_COL), lab))
        return counts, substantive

    print(f"Gold set rows: {len(rows)}")
    print(f"Flagged total: {len(flagged)}  "
          f"(collapse={len(COLLAPSE)}, source_junk={len(SOURCE_JUNK)})")
    if missing:
        print(f"WARNING: {len(missing)} flagged ids not found in gold set: {missing}")
    print()

    for name, bucket in (("collapse", COLLAPSE), ("source_junk", SOURCE_JUNK)):
        counts, substantive = tally(bucket)
        found = sum(counts.values())
        print(f"[{name}]  found {found}/{len(bucket)} in gold set")
        for lab in ("propagandistic", "neutral", "unclear"):
            if lab in counts:
                print(f"    {lab:15s} {counts[lab]}")
        other = {k: v for k, v in counts.items()
                 if k not in ("propagandistic", "neutral", "unclear")}
        for lab, c in other.items():
            print(f"    {lab or '(blank)':15s} {c}")
        print(f"  -> {len(substantive)} got a SUBSTANTIVE (non-unclear) label"
              f"  {[a for a, _ in substantive]}")
        print()

    print(f"Wrote {LABELS_OUT}")


if __name__ == "__main__":
    main()
