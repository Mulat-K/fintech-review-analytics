import pandas as pd
from google_play_scraper import reviews, Sort

APP_PACKAGES = {
    'CBE': 'com.combanketh.mobilebanking', # Verify these IDs
    'BOA': 'com.boa.boaMobileBanking', 
    'Dashen': 'com.dashen.dashensuperapp'
}

def scrape_bank_reviews(bank_name, app_id, count=500):
    print(f"Scraping {bank_name}...")
    result, _ = reviews(
        app_id,
        lang='en', # Defaults to English
        country='et', # Storefront for Ethiopia
        sort=Sort.NEWEST,
        count=count
    )
    
    df = pd.DataFrame(result)
    df['bank_name'] = bank_name
    df['source'] = 'Google Play'
    
    # Keep only necessary columns
    df = df[['content', 'score', 'at', 'bank_name', 'source', 'reviewId']]
    df.rename(columns={'content': 'review_text', 'score': 'rating', 'at': 'review_date', 'reviewId': 'review_id'}, inplace=True)
    return df

all_reviews = []
for bank, app_id in APP_PACKAGES.items():
    try:
        df = scrape_bank_reviews(bank, app_id)
        all_reviews.append(df)
    except Exception as e:
        print(f"Error scraping {bank}: {e}")

final_df = pd.concat(all_reviews, ignore_index=True)
final_df.to_csv("data/raw/raw_reviews.csv", index=False)
print("Scraping Complete.")