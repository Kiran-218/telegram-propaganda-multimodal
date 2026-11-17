"""
build_bundle_p2.py  --  Build the PILOT 2 annotation bundle.

Run this in your Dissertation folder (where pilot_30.csv and the media/
images already live), the same place you ran the pilot-1 build.

    python build_bundle_p2.py

It produces:
    pilot2_bundle/            (the assembled folder)
    pilot2_bundle.zip         (zip it up ready to send)

What goes in the bundle:
    annotate_server.py   -- adjusted: reads pilot_30.csv, writes
                            annotator2_labels_p2.csv  (distinct from pilot 1)
    annotate.html        -- unchanged copy of your UI
    pilot_30.csv         -- the same 30 posts (A0001-A0030)
    media/<channel>/<post_id>.jpg  -- ONLY the 30 referenced images
    README_MAC.md        -- run instructions for your Mac annotator

Nothing else from your dataset is copied in. The bundle starts blank
(no labels), so he begins from a clean slate.
"""

import argparse
import csv
import os
import re
import shutil
import sys
import zipfile

# ---- config -------------------------------------------------------------
SERVER_SRC  = "annotate_server.py"    # your existing server
HTML_SRC    = "annotate.html"         # your existing UI
IMG_COL     = "image_path"            # column holding media\<ch>\<id>.jpg

# Per-round presets. Pick one with --round.
ROUNDS = {
    "pilot2": {
        "csv":    "pilot_30.csv",
        "outdir": "pilot2_bundle",
        "zip":    "pilot2_bundle.zip",
        "labels": "annotator2_labels_p2.csv",
        "desc":   "Pilot 2 (same 30 posts, revised scheme)",
        "same_posts_note": True,
    },
    "heldout": {
        "csv":    "heldout_sample_translated.csv",
        "outdir": "heldout_bundle",
        "zip":    "heldout_bundle.zip",
        "labels": "annotator2_labels_heldout.csv",
        "desc":   "Held-out validation round (30 unseen posts, H0001-H0030)",
        "same_posts_note": False,
    },
    "overlap": {
        "csv":    "overlap_100.csv",
        "outdir": "overlap_bundle",
        "zip":    "overlap_bundle.zip",
        "labels": "annotator2_labels_overlap.csv",
        "desc":   "Gold-set overlap round (100 posts for reliability)",
        "same_posts_note": False,
    },
}
# Set by main() from --round
PILOT_CSV = OUT_DIR = OUT_ZIP = OUT_LABELS = None
ROUND_DESC = ""
SAME_POSTS = False
# -------------------------------------------------------------------------


def die(msg):
    print("ERROR:", msg)
    sys.exit(1)


def norm(p):
    return (p or "").replace("\\", "/")


def main():
    # 0. parse --root (where the media/ folder actually lives).
    #    Defaults to the current folder, so old behaviour is unchanged if omitted.
    ap = argparse.ArgumentParser(
        description="Build an annotation bundle for a given round.")
    ap.add_argument("--round", default="heldout", choices=sorted(ROUNDS),
                    help="Which round to build (default: heldout).")
    ap.add_argument("--root", default=".",
                    help=r'Folder containing media\<channel>\<post_id>.jpg '
                         r'(e.g. "C:\Users\kiran\OneDrive - University of Edinburgh\Dissertation"). '
                         r'Defaults to the current folder.')
    args = ap.parse_args()
    media_root = args.root

    global PILOT_CSV, OUT_DIR, OUT_ZIP, OUT_LABELS, ROUND_DESC, SAME_POSTS
    cfg = ROUNDS[args.round]
    PILOT_CSV  = cfg["csv"]
    OUT_DIR    = cfg["outdir"]
    OUT_ZIP    = cfg["zip"]
    OUT_LABELS = cfg["labels"]
    ROUND_DESC = cfg["desc"]
    SAME_POSTS = cfg["same_posts_note"]
    print(f"Round: {args.round}  --  {ROUND_DESC}")

    # 1. sanity checks
    for f in (PILOT_CSV, SERVER_SRC, HTML_SRC):
        if not os.path.exists(f):
            die(f"Missing {f} in this folder. Run this from your Dissertation folder.")
    if not os.path.isdir(os.path.join(media_root, "media")):
        die(f"No 'media' folder found under --root: {media_root!r}\n"
            f"       Pass the folder that CONTAINS media/, e.g.\n"
            f'       python build_bundle_p2.py --root "C:\\Users\\kiran\\OneDrive - University of Edinburgh\\Dissertation"')

    # 2. fresh output dir. On Windows/OneDrive the old folder can be locked
    #    (sync in progress, or a running server). Try to remove it; if that
    #    fails, fall back to a fresh timestamped folder so the build still runs.
    if os.path.exists(OUT_DIR):
        try:
            shutil.rmtree(OUT_DIR)
        except PermissionError:
            import time
            stamp = time.strftime("%H%M%S")
            OUT_DIR = f"{OUT_DIR}_{stamp}"
            OUT_ZIP = OUT_ZIP.replace(".zip", f"_{stamp}.zip")
            print(f"NOTE: previous output folder was locked; "
                  f"building into {OUT_DIR}/ instead.")
    os.makedirs(OUT_DIR)

    # 3. read the pilot CSV, collect image paths
    with open(PILOT_CSV, "r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        die(f"{PILOT_CSV} has no rows.")
    if IMG_COL not in rows[0]:
        die(f"{PILOT_CSV} has no '{IMG_COL}' column. Columns are: {list(rows[0].keys())}")

    print(f"Pilot CSV: {len(rows)} posts ({rows[0]['ann_id']}..{rows[-1]['ann_id']}).")

    # 4. copy exactly the referenced images, preserving media/<channel>/ structure
    copied, missing = 0, []
    for r in rows:
        rel = norm(r[IMG_COL])
        src = os.path.join(media_root, rel)
        if not os.path.exists(src):
            missing.append(src)
            continue
        dst = os.path.join(OUT_DIR, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
    print(f"Copied {copied} images.", end="")
    if missing:
        print(f"  WARNING: {len(missing)} referenced images not found:")
        for m in missing[:10]:
            print("   -", m)
    else:
        print("  All present.")

    # 5. copy the pilot CSV in
    shutil.copy2(PILOT_CSV, os.path.join(OUT_DIR, PILOT_CSV))

    # 6. copy HTML unchanged
    shutil.copy2(HTML_SRC, os.path.join(OUT_DIR, "annotate.html"))

    # 7. write the ADJUSTED server: reads pilot_30.csv, writes annotator2_labels_p2.csv
    with open(SERVER_SRC, "r", encoding="utf-8") as f:
        server = f.read()

    # swap the two hard-coded filenames
    server_new, n_in = re.subn(
        r'CSV_IN\s*=\s*"[^"]*"',
        f'CSV_IN   = "{PILOT_CSV}"',
        server, count=1)
    server_new, n_out = re.subn(
        r'CSV_OUT\s*=\s*"[^"]*"',
        f'CSV_OUT  = "{OUT_LABELS}"',
        server_new, count=1)

    if n_in != 1 or n_out != 1:
        die("Could not locate CSV_IN / CSV_OUT lines in annotate_server.py "
            "to rewrite. Check the file hasn't changed shape.")

    # also fix the two user-facing print lines that name the output file
    server_new = server_new.replace(
        "Labels save automatically to annotation_labels.csv",
        f"Labels save automatically to {OUT_LABELS}")

    with open(os.path.join(OUT_DIR, "annotate_server.py"), "w", encoding="utf-8") as f:
        f.write(server_new)

    # 8. README for the Mac annotator
    if SAME_POSTS:
        intro = """This is the **second** pilot round. It is the **same 30 posts** as before,
but the labelling criteria have been revised. Please:

1. **Read the updated `Annotation_Scheme.docx` first** (sent alongside this
   bundle). A few boundary rules changed — read it fresh before you start.
2. **Do not look at your previous labels.** Label each post on its own, using
   the revised scheme. Independence matters for the agreement calculation.
3. **Add a short note** whenever a post is not obvious — which cue you saw,
   or why it was borderline. The notes are as useful as the labels this round."""
    else:
        intro = f"""This is the **{ROUND_DESC.lower()}**: {len(rows)} posts you have **not seen
before** ({rows[0]['ann_id']}-{rows[-1]['ann_id']}). Same labelling criteria as
the last round. Please:

1. **Use the same `Annotation_Scheme.docx`** you used last time (sent alongside
   this bundle). Nothing in the criteria has changed.
2. **Label independently.** Please don't discuss any of these posts with me
   until we've both finished — this round measures how consistently the
   criteria are applied, so independence is what makes the number meaningful.
3. **Add a short note** whenever a post is not obvious — which cue you saw,
   or why it was borderline."""

    readme = f"""# {ROUND_DESC} — annotation instructions (Mac)

{intro}

## Running the tool

You need Python 3 (most Macs have it). In Terminal:

```
cd <folder where you unzipped this>
python3 annotate_server.py
```

Then open the printed address (http://localhost:8000) in your browser.

- Keys: **1** propagandistic · **2** neutral · **3** unclear
- ←/→ move between posts · Enter jumps to next unlabelled
- Labels save automatically to **{OUT_LABELS}**

When you have labelled all 30, press **Ctrl+C** in Terminal to stop the
server, then send me back the file **{OUT_LABELS}**.

Nothing leaves your machine — the tool runs entirely locally.
"""
    with open(os.path.join(OUT_DIR, "README_MAC.md"), "w", encoding="utf-8") as f:
        f.write(readme)

    # 9. zip it
    if os.path.exists(OUT_ZIP):
        os.remove(OUT_ZIP)
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(OUT_DIR):
            for name in files:
                full = os.path.join(root, name)
                arc = os.path.relpath(full, OUT_DIR)
                z.write(full, arc)

    # 10. integrity report
    with zipfile.ZipFile(OUT_ZIP) as z:
        names = z.namelist()
    n_imgs = sum(1 for n in names if n.lower().endswith((".jpg", ".jpeg", ".png", ".webp")))
    print()
    print(f"Built {OUT_ZIP}: {len(names)} files total "
          f"({n_imgs} images + {len(names) - n_imgs} other).")
    print(f"Expected: {len(rows)} images + 4 files "
          f"(server, html, csv, readme) = {len(rows) + 4}.")
    if n_imgs == len(rows) and len(names) == len(rows) + 4:
        print("Integrity OK — bundle is complete and self-contained.")
    else:
        print("CHECK: file count differs from expected. Review the warnings above.")

    print(f"\nThe server in this bundle:")
    print(f"  reads   {PILOT_CSV}")
    print(f"  writes  {OUT_LABELS}   (so it won't collide with his pilot-1 file)")
    print(f"\nSend him: {OUT_ZIP}  +  the frozen Annotation_Scheme.docx")


if __name__ == "__main__":
    main()
