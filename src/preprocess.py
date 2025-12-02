# src/preprocess.py
import pandas as pd
from pathlib import Path
from dateutil import parser
import numpy as np

RAW = Path(__file__).resolve().parents[1] / "data" / "raw_reviews.csv"
OUT = Path(__file__).resolve().parents[1] / "data" / "clean_reviews.csv"

def parse_date(s):
    try:
        return pd.to_datetime(s)
    except Exception:
        try:
            return pd.to_datetime(parser.parse(s))
        except Exception:
            return pd.NaT

def main():
    df = pd.read_csv(RAW)
    # Basic stats
    print("Raw rows:", len(df))
    # Drop reviews with empty content
    df['review'] = df['review'].astype(str).str.strip()
    df = df[df['review'].notna() & (df['review'] != '')].copy()
    # Normalize date
    df['review_date'] = df['at'].apply(parse_date)
    # Keep ISO date
    df['review_date_iso'] = df['review_date'].dt.strftime('%Y-%m-%d')
    # Deduplicate by review text + package
    before = len(df)
    df.drop_duplicates(subset=['package','review'], inplace=True)
    after = len(df)
    print(f"Dropped {before-after} duplicates")
    # Missing rate
    missing_rate = df.isnull().mean()
    print("Missing rate per column:\n", missing_rate)
    # Keep or fill missing numeric rating
    df['score'] = pd.to_numeric(df['score'], errors='coerce').fillna(0).astype(int)
    # Keep only relevant columns
    out = df[['review_id','bank','app_name','package','review','score','review_date_iso','source']]
    out.to_csv(OUT, index=False)
    print("Saved clean data to", OUT, "rows:", len(out))

if __name__ == "__main__":
    main()
