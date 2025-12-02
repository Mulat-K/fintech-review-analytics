import pandas as pd
from google_play_scraper import reviews, Sort

# Define App IDs (Verify these on Google Play Store URLs)
APPS = {
    "CBE": "com.combanketh.mobilebanking",
    "BOA": "com.boa.boaMobileBanking",
    "Dashen": "com.dashen.dashensuperapp"
}

def scrape_app_reviews(app_name, app_id, target_count=400):
    print(f"Scraping {app_name}...")
    result, _ = reviews(
        app_id,
        lang='en', 
        country='ETB',
        sort=Sort.NEWEST, 
        count=target_count
    )
    
    # Convert to DataFrame
    df = pd.DataFrame(result)
    df['bank'] = app_name
    df['source'] = 'Google Play'
    df['review_date'] = pd.to_datetime(df['at'])
    
    # Select only necessary columns
    df = df[['content', 'score', 'review_date', 'bank', 'source']]
    df.rename(columns={'content': 'review_text', 'score': 'rating'}, inplace=True)
    
    return df

# Main execution
all_reviews = []
for bank, app_id in APPS.items():
    try:
        df = scrape_app_reviews(bank, app_id, target_count=600) # Scraping slightly more to account for duplicates
        all_reviews.append(df)
    except Exception as e:
        print(f"Error scraping {bank}: {e}")

# Combine all data
final_df = pd.concat(all_reviews, ignore_index=True)

# Preprocessing: Remove duplicates and missing text
final_df.dropna(subset=['review_text'], inplace=True)
final_df.drop_duplicates(subset=['review_text', 'bank'], inplace=True)

# Save to CSV
final_df.to_csv("bank_reviews_raw.csv", index=False)
print(f"Successfully saved {len(final_df)} reviews to bank_reviews_raw.csv")