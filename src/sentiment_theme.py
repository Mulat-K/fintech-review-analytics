# src/sentiment_theme.py
import pandas as pd
from pathlib import Path
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import spacy
from tqdm import tqdm
import json
import re

CLEAN = Path(__file__).resolve().parents[1] / "data" / "clean_reviews.csv"
OUT = Path(__file__).resolve().parents[1] / "data" / "reviews_with_sentiment_themes.csv"

# theme mapping: keywords -> theme label (extend as you find domain terms)
THEME_KEYWORDS = {
    "login": "Account Access",
    "password": "Account Access",
    "fingerprint": "Authentication & Security",
    "face id": "Authentication & Security",
    "slow": "Performance",
    "loading": "Performance",
    "transfer": "Transactions",
    "send money": "Transactions",
    "crash": "Stability",
    "bug": "Stability",
    "error": "Stability",
    "support": "Customer Support",
    "customer service": "Customer Support",
    "ui": "UI/UX",
    "interface": "UI/UX",
    "update": "Updates & Releases",
    "feature": "Feature Request",
    "payment": "Transactions",
    "otp": "Authentication & Security",
    "balance": "Account Info"
}

# Preload spaCy for tokenization if needed
nlp = spacy.blank("en")

def hf_sentiment_pipeline(texts, model_name="distilbert-base-uncased-finetuned-sst-2-english", batch_size=32):
    pipe = pipeline("sentiment-analysis", model=model_name, device=-1)
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        out = pipe(batch)
        results.extend(out)
    return results

def vader_sentiment(texts):
    analyzer = SentimentIntensityAnalyzer()
    res = [analyzer.polarity_scores(t) for t in texts]
    return res

def map_to_label_hf(result, neutral_threshold=0.60):
    # result example: {'label': 'POSITIVE', 'score': 0.998}
    label = result['label']
    score = result['score']
    if score < neutral_threshold:
        return "neutral", score if label=="POSITIVE" else -score
    return ("positive", score) if label=="POSITIVE" else ("negative", -score)

def extract_tfidf_keywords(corpus, top_n=10, ngram_range=(1,2)):
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, stop_words='english', max_df=0.85, min_df=2)
    X = vectorizer.fit_transform(corpus)
    feature_names = np.array(vectorizer.get_feature_names_out())
    # sum tfidf for each term across docs
    sums = X.sum(axis=0).A1
    top_idx = sums.argsort()[::-1][:top_n]
    return list(feature_names[top_idx]), feature_names, sums

def rule_based_themes(text):
    text_l = text.lower()
    themes = set()
    for kw, theme in THEME_KEYWORDS.items():
        if kw in text_l:
            themes.add(theme)
    return list(themes) if themes else ["Other"]

def clean_text_for_tfidf(text):
    text = re.sub(r'http\S+','',text)
    text = re.sub(r'[^A-Za-z0-9\s]',' ',text)
    return ' '.join(text.split()).lower()

def main():
    df = pd.read_csv(CLEAN)
    texts = df['review'].fillna('').astype(str).tolist()

    print("Running HuggingFace sentiment (may take a while)...")
    try:
        hf_res = hf_sentiment_pipeline(texts)
        mapped = [map_to_label_hf(r) for r in hf_res]
        df['sentiment_label'] = [m[0] for m in mapped]
        df['sentiment_score'] = [m[1] for m in mapped]
    except Exception as e:
        print("HF pipeline failed, falling back to VADER:", e)
        vader_res = vader_sentiment(texts)
        df['sentiment_score'] = [r['compound'] for r in vader_res]
        df['sentiment_label'] = df['sentiment_score'].apply(lambda x: 'positive' if x>=0.05 else ('negative' if x<=-0.05 else 'neutral'))

    # Keyword extraction per bank
    df['clean_review'] = df['review'].apply(clean_text_for_tfidf)
    keywords_by_bank = {}
    for bank, group in df.groupby('bank'):
        corpus = group['clean_review'].tolist()
        if len(corpus) < 5:
            keywords_by_bank[bank] = []
            continue
        top_features, feature_names, sums = extract_tfidf_keywords(corpus, top_n=20)
        keywords_by_bank[bank] = top_features

    # assign rule-based themes per row
    df['themes'] = df['review'].apply(rule_based_themes)

    # Save keywords mapping for manual review
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print("Saved sentiment + theme file to", OUT)
    # Also save bank keywords
    json.dump(keywords_by_bank, open(Path(OUT).parent / "keywords_by_bank.json", "w"), indent=2)
    print("Saved keywords_by_bank.json")

if __name__ == "__main__":
    main()
