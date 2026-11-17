"""
Split the translated annotation sample into a 30-post pilot set.

Takes the first 30 rows by ann_id (A0001..A0030) so you and your second
annotator label the SAME 30 posts independently for the pilot Cohen's Kappa.

Usage:
    python make_pilot.py --in annotation_sample_translated_1.3B.csv

Writes:
    pilot_30.csv    (the 30 pilot rows, same columns as input)
    main_370.csv    (the remaining rows, for the full pass afterwards)
"""

import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp",
                    default="annotation_sample_translated_1.3B.csv")
    ap.add_argument("--n", type=int, default=30, help="pilot size")
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    if "ann_id" not in df.columns:
        raise SystemExit("Input CSV missing 'ann_id' column.")

    # sort by ann_id so the pilot is deterministic (A0001..A0030)
    df = df.sort_values("ann_id").reset_index(drop=True)

    pilot = df.iloc[:args.n].copy()
    main = df.iloc[args.n:].copy()

    pilot.to_csv("pilot_30.csv", index=False, encoding="utf-8-sig")
    main.to_csv("main_370.csv", index=False, encoding="utf-8-sig")

    print(f"Pilot : {len(pilot)} rows -> pilot_30.csv "
          f"({pilot['ann_id'].iloc[0]}..{pilot['ann_id'].iloc[-1]})")
    print(f"Main  : {len(main)} rows -> main_370.csv")


if __name__ == "__main__":
    main()
