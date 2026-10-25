"""Regenerate mean +/- std summaries from one or more hybrid five-seed CSVs.

Usage:
  python make_summary.py
  python make_summary.py five_seed_python_clean_calibrated_fullmodel.csv
"""

import argparse
import glob
import os

import pandas as pd

from run_five_seeds import summarize


def main():
    parser = argparse.ArgumentParser(description="Summarize hybrid five-seed CSV files")
    parser.add_argument("csv", nargs="*", help="CSV file(s); default: all five_seed_*.csv")
    args = parser.parse_args()

    paths = args.csv or sorted(glob.glob("five_seed_*.csv"))
    paths = [p for p in paths if not p.endswith("_summary.csv")]
    if not paths:
        raise SystemExit("No five_seed_*.csv files found.")

    for path in paths:
        df = pd.read_csv(path)
        out_path = os.path.splitext(path)[0] + "_summary.csv"
        summarize(df).to_csv(out_path, index=False)
        print(f"[ok] Wrote {out_path}")


if __name__ == "__main__":
    main()
