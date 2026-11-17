"""
Collect ONLY the images referenced by corpus_captions.csv into a single zip,
for one efficient transfer to the cluster (avoids per-file sftp overhead and
skips all the .mp4 videos that M-CLIP never opens).

Mirrors the gold-set collect+zip method, but for the full ~9,054-post corpus.
The zip preserves the media/<channel>/<post_id>.jpg structure, so unzipping it
inside ~/dissertation/mclip/ places images exactly where mclip_score.py expects
(image_path in the CSV is media/<channel>/<post_id>.jpg, resolved vs
--media-root .).

Usage:
  python collect_corpus_images.py \
      --csv corpus_captions.csv \
      --root "C:\\Users\\kiran\\OneDrive - University of Edinburgh\\Dissertation" \
      --out corpus_media

Produces corpus_media.zip. Upload that one file, then on the cluster:
  cd ~/dissertation/mclip && unzip -q corpus_media.zip
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
    ap.add_argument("--csv", default="corpus_captions.csv")
    ap.add_argument("--root", default=".",
                    help="folder that CONTAINS the media/ tree")
    ap.add_argument("--out", default="corpus_media")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if "image_path" not in df.columns:
        raise SystemExit("CSV missing 'image_path' column.")

    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    os.makedirs(args.out)

    copied, missing = 0, []
    paths = df["image_path"].map(norm).unique()
    for rel in paths:
        src = os.path.join(args.root, rel)
        if os.path.exists(src):
            dst = os.path.join(args.out, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(src, dst)
            copied += 1
        else:
            missing.append(rel)

    print(f"Referenced images : {len(paths)}")
    print(f"Copied            : {copied}")
    if missing:
        print(f"WARNING: {len(missing)} images not found on disk (first few):")
        for m in missing[:10]:
            print(f"  {m}")
        print("  If this number is large, check --root before uploading.")

    zip_path = args.out + ".zip"
    print(f"Zipping -> {zip_path} (this can take a minute for ~22k files) ...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for r, _, files in os.walk(args.out):
            for name in files:
                full = os.path.join(r, name)
                z.write(full, os.path.relpath(full, args.out))

    size_gb = os.path.getsize(zip_path) / (1024 ** 3)
    print(f"Done. {zip_path} is {size_gb:.2f} GB.")
    print("Upload just this one file, then unzip inside ~/dissertation/mclip/")


if __name__ == "__main__":
    main()
