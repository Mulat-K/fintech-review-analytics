# src/scraper.py
import json
import time
import pandas as pd
from google_play_scraper import reviews, Sort
from tqdm import tqdm
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[1] / "apps.json"
OUT = Path(__file__).resolve().parents[1] / "data" / "raw_reviews.csv"
TARGET_PER_APP = 500   # aim for 500 to have buffer (you requested 400 min)

def fetch_reviews_for_package(package_name, target=TARGET_PER_APP, lang='en', country='us'):
    all_reviews = []
    batch = 200  # google_play_scraper supports up to 200
    cursor = None
    while len(all_reviews) < target:
        cnt = min(batch, target - len(all_reviews))
        rv, cursor = reviews(
            package_name,
            lang=lang,
            country=country,
            sort=Sort.MOST_RELEVANT,
            count=cnt,
            continuation_token=cursor
        )
        if not rv:
            break
        all_reviews.extend(rv)
        if cursor is None:
            break
        time.sleep(1)  # polite pause
    return all_reviews

def main():
    apps = json.loads(open(CONFIG).read())
    rows = []
    for a in apps:
        pkg = a['package']
        print(f"Fetching reviews for {a['bank']} ({pkg})")
        rv = fetch_reviews_for_package(pkg, target=TARGET_PER_APP)
        print(f"Fetched {len(rv)} reviews for {a['bank']}")
        for r in rv:
            rows.append({
                "bank": a['bank'],
                "app_name": a['app_name'],
                "package": pkg,
                "review_id": r.get('reviewId'),
                "review": r.get('content'),
                "score": r.get('score'),
                "at": r.get('at').isoformat() if r.get('at') else None,
                "reply_text": r.get('replyContent'),
                "reply_date": r.get('repliedAt').isoformat() if r.get('repliedAt') else None,
                "source": "google_play"
            })
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print("Saved raw reviews to", OUT)

if __name__ == "__main__":
    main()
