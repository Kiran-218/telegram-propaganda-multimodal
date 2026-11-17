"""
Collect exactly the images referenced in the annotation sample into a single
zip, preserving the media/<channel>/<id>.jpg structure, ready to upload to the
cluster for M-CLIP scoring.

Run on the laptop, from the Dissertation folder.

Usage (adjust --root to the folder that contains your media/ tree):
    python collect_sample_images.py --csv annotation_sample.csv --root "PATH_TO_FOLDER_WITH_media" --out sample_media

Produces:
    sample_media/                 (folder with media/<channel>/... inside)
    sample_media.zip              (upload this to the cluster)
"""

import argparse
import os
import shutil
import zipfile

import pandas as pd


def norm(p):
    return str(p).replace("\\", "/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="annotation_sample.csv")
    ap.add_argument("--root", default=".",
                    help="folder that CONTAINS the media/ tree")
    ap.add_argument("--out", default="sample_media")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if "image_path" not in df.columns:
        raise SystemExit("CSV missing 'image_path' column.")

    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    os.makedirs(args.out)

    copied, missing = 0, []
    for rel in df["image_path"].map(norm):
        src = os.path.join(args.root, rel)
        if os.path.exists(src):
            dst = os.path.join(args.out, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(src, dst)
            copied += 1
        else:
            missing.append(rel)

    print(f"Copied {copied} images into {args.out}/")
    if missing:
        print(f"WARNING: {len(missing)} images not found (first few):")
        for m in missing[:10]:
            print(f"  {m}")

    zip_path = args.out + ".zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for r, _, files in os.walk(args.out):
            for name in files:
                full = os.path.join(r, name)
                z.write(full, os.path.relpath(full, args.out))

    print(f"Zipped: {zip_path}")
    print("Upload this to the cluster and unzip it inside ~/dissertation/mclip/")


if __name__ == "__main__":
    main()
