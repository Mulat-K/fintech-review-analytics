import pandas as pd
import numpy as np
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')
stop_words = list(stopwords.words('english'))

# Custom stop words relevant to banking
custom_stops = ['app', 'bank', 'mobile', 'banking', 'cbe', 'boa', 'dashen', 'ethiopia', 'money']
stop_words.extend(custom_stops)

def analyze_data():
    # Load Data
    df = pd.read_csv("bank_reviews_raw.csv")
    print("Data loaded. Starting Sentiment Analysis...")

    # 1. SENTIMENT ANALYSIS (using DistilBERT)
    # This model gives POSITIVE/NEGATIVE labels
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    # Helper function to handle truncation (DistilBERT has a token limit)
    def get_sentiment(text):
        try:
            # Truncate text to 512 tokens just in case
            result = sentiment_pipeline(text[:512])[0]
            return result['label'], result['score']
        except:
            return "NEUTRAL", 0.0

    # Apply sentiment analysis (This might take a few minutes)
    df[['sentiment_label', 'sentiment_score']] = df['review_text'].apply(lambda x: pd.Series(get_sentiment(str(x))))

    # 2. KEYWORD EXTRACTION (Thematic)
    print("Starting Keyword Extraction...")

    def extract_keywords(text_series):
        vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=10, ngram_range=(1,2))
        try:
            tfidf_matrix = vectorizer.fit_transform(text_series)
            feature_names = vectorizer.get_feature_names_out()
            return ", ".join(feature_names)
        except ValueError:
            return "insufficient_data"

    # Group by Bank and Sentiment to find pain points (Negative) vs drivers (Positive)
    # We will create a summary dataframe for themes
    themes = []
    
    for bank in df['bank'].unique():
        for label in ['POSITIVE', 'NEGATIVE']:
            subset = df[(df['bank'] == bank) & (df['sentiment_label'] == label)]
            if not subset.empty:
                keywords = extract_keywords(subset['review_text'])
                themes.append({
                    'bank': bank,
                    'sentiment': label,
                    'top_keywords': keywords
                })

    theme_df = pd.DataFrame(themes)
    print("\n--- Identified Themes ---")
    print(theme_df)

    # Save processed data
    df.to_csv("bank_reviews_processed.csv", index=False)
    theme_df.to_csv("bank_themes.csv", index=False)
    print("Analysis complete. Files saved.")

if __name__ == "__main__":
    analyze_data()