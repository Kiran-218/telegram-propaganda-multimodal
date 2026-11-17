"""
Assemble a self-contained pilot bundle to send to the second annotator.

Copies ONLY the 30 pilot images (nothing else from your image set), the
annotation server + page, the pilot CSV, and a Mac README into one folder,
then zips it.

Run this AFTER make_pilot.py has produced pilot_30.csv.

Usage:
    python build_pilot_bundle.py \
        --pilot pilot_30.csv \
        --server annotate_server.py \
        --page annotate.html \
        --media media \
        --out pilot_bundle

Produces:
    pilot_bundle/            (the folder)
    pilot_bundle.zip         (zipped, ready to send)

The annotator's server copy is adjusted to read pilot_30.csv and write
annotation_labels.csv; you rename his returned file when you receive it.
"""

import argparse
import os
import re
import shutil
import zipfile

import pandas as pd


README = """# Propaganda annotation - pilot (30 posts)

Thanks for helping with the pilot. This runs entirely on your own Mac;
nothing is uploaded anywhere.

## 1. One-time setup
You need Python 3 (macOS usually has it). Check by opening **Terminal** and running:

    python3 --version

If that prints a version (3.8 or higher), you're set. If not, install it from
https://www.python.org/downloads/ .

No other packages are needed - the tool uses only Python's standard library.

## 2. Run the tool
In Terminal, navigate into this folder (drag the folder onto the Terminal window
after typing `cd `, then press Enter), then run:

    python3 annotate_server.py

You'll see a line like:  Open  http://localhost:8000  in your browser
Open that address in Safari or Chrome.

## 3. Annotate
For each post you'll see the image and the English caption. Choose one label:

- **1 = Propagandistic** - the caption/image persuades by *manner* (loaded language,
  fear, manipulation), not merely by being about the war or taking a side.
- **2 = Neutral** - factual or plainly presented, even if partisan.
- **3 = Unclear** - genuinely can't tell / not enough to judge.

Use the number keys (1/2/3) or click. Arrow keys move between posts; Enter jumps
to the next unlabelled one. Add an optional note for borderline cases. Everything
saves automatically.

**Please label all 30.** Judge each post on its own; don't try to guess my labels.

## 4. Send results back
When done, close the tool (Ctrl+C in Terminal). Send me back the single file:

    annotation_labels.csv

That's it - thank you!
"""


def norm(p):
    return str(p).replace("\\", "/")


def patch_server(src_text):
    """Point the server's input CSV at pilot_30.csv (leave output as-is)."""
    return re.sub(r'CSV_IN\s*=\s*"[^"]*"', 'CSV_IN   = "pilot_30.csv"', src_text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", default="pilot_30.csv")
    ap.add_argument("--server", default="annotate_server.py")
    ap.add_argument("--page", default="annotate.html")
    ap.add_argument("--root", default=".",
                    help="folder that CONTAINS the media/ tree "
                         "(image paths in the CSV are resolved against this)")
    ap.add_argument("--out", default="pilot_bundle")
    args = ap.parse_args()

    for f in (args.pilot, args.server, args.page):
        if not os.path.exists(f):
            raise SystemExit(f"Missing required file: {f}")

    # fresh output dir
    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    os.makedirs(args.out)

    # 1. pilot CSV
    shutil.copy(args.pilot, os.path.join(args.out, "pilot_30.csv"))

    # 2. server (patched to read pilot_30.csv) + page
    with open(args.server, "r", encoding="utf-8") as f:
        server_text = f.read()
    with open(os.path.join(args.out, "annotate_server.py"), "w",
              encoding="utf-8") as f:
        f.write(patch_server(server_text))
    shutil.copy(args.page, os.path.join(args.out, "annotate.html"))

    # 3. README
    with open(os.path.join(args.out, "README.md"), "w", encoding="utf-8") as f:
        f.write(README)

    # 4. exactly the 30 pilot images, preserving media/<channel>/<id> structure
    df = pd.read_csv(args.pilot)
    copied, missing = 0, []
    for rel in df["image_path"].map(norm):
        # rel looks like "media/<channel>/<post_id>.jpg" (relative).
        # Resolve it against --root (the folder that CONTAINS media/).
        src = os.path.join(args.root, rel)
        if os.path.exists(src):
            dst = os.path.join(args.out, rel)   # keep media/<channel>/ layout
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(src, dst)
            copied += 1
        else:
            missing.append(rel)

    print(f"Copied {copied} images into {args.out}/")
    if missing:
        print(f"WARNING: {len(missing)} images not found - check paths:")
        for m in missing[:10]:
            print(f"  {m}")

    # 5. zip it
    zip_path = args.out + ".zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(args.out):
            for name in files:
                full = os.path.join(root, name)
                z.write(full, os.path.relpath(full, os.path.dirname(args.out)))

    print(f"\nBundle ready: {zip_path}")
    print("Send that zip to your annotator. He unzips, runs annotate_server.py,")
    print("and sends back annotation_labels.csv.")


if __name__ == "__main__":
    main()
