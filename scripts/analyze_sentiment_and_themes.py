import pandas as pd
import numpy as np
import os
import sys

# Add src to path to import custom modules
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if module_path not in sys.path:
    sys.path.append(module_path)

from utils.analyzer import SentimentAnalyzer, ThematicAnalyzer
from utils.preprocessor import ReviewPreprocessor

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'google_play_reviews.csv')
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'reviews_with_sentiment_themes.csv')

# 1. Load Data
df_reviews = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df_reviews)} reviews from {DATA_PATH}")

# 2. Preprocess Data
preprocessor = ReviewPreprocessor()
df_reviews = preprocessor.preprocess_data(df_reviews.to_dict('records'), bank_name=None)

# Try spaCy-based preprocessing for review text
try:
    cleaned_texts = preprocessor.preprocess_texts_with_spacy(df_reviews['review'].astype(str).fillna('').tolist())
    if cleaned_texts and any(cleaned_texts):
        df_reviews['review_text_for_sentiment'] = cleaned_texts
        print("spaCy-based preprocessing (lemmatization, stopword removal) applied.")
    else:
        raise Exception('spaCy returned empty results')
except Exception as e:
    df_reviews['review_text_for_sentiment'] = df_reviews['review'].astype(str).fillna('')
    print(f"spaCy preprocessing not used: {e}. Using raw text.")

# 3. Sentiment Analysis
sentiment_analyzer = SentimentAnalyzer()
df_reviews['review_text_for_sentiment'] = df_reviews['review_text_for_sentiment'].astype(str).fillna('')
review_texts = df_reviews['review_text_for_sentiment'].tolist()

if sentiment_analyzer.sentiment_pipeline:
    print(f"Starting sentiment prediction for {len(review_texts)} reviews...")
    sentiments = sentiment_analyzer.predict_sentiment(review_texts)
    print(f"Finished sentiment prediction.")
    if sentiments and len(sentiments) == len(df_reviews):
        df_reviews['sentiment_label'] = [s['label'] for s in sentiments]
        df_reviews['sentiment_score'] = [s['score'] for s in sentiments]
    else:
        print("Could not add sentiment labels/scores. Mismatch in length or empty results.")
        df_reviews['sentiment_label'] = 'Error'
        df_reviews['sentiment_score'] = np.nan
else:
    print("Sentiment model not loaded. Skipping sentiment analysis.")
    df_reviews['sentiment_label'] = 'Not Processed'
    df_reviews['sentiment_score'] = np.nan

# 3b. VADER Sentiment Analysis (if available)
try:
    from nltk.sentiment import SentimentIntensityAnalyzer
    vader_analyzer = SentimentIntensityAnalyzer()
    vader_scores = df_reviews['review_text_for_sentiment'].apply(lambda x: vader_analyzer.polarity_scores(str(x)))
    df_reviews['vader_compound'] = vader_scores.apply(lambda x: x['compound'])
    df_reviews['vader_label'] = df_reviews['vader_compound'].apply(lambda x: 'POSITIVE' if x > 0.05 else ('NEGATIVE' if x < -0.05 else 'NEUTRAL'))
    print("VADER sentiment analysis complete.")
except Exception as e:
    print(f"VADER sentiment analysis not available: {e}")
    df_reviews['vader_compound'] = np.nan
    df_reviews['vader_label'] = 'Not Processed'

# 3c. TextBlob Sentiment Analysis (if available)
try:
    from textblob import TextBlob
    df_reviews['textblob_polarity'] = df_reviews['review_text_for_sentiment'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    df_reviews['textblob_label'] = df_reviews['textblob_polarity'].apply(lambda x: 'POSITIVE' if x > 0.05 else ('NEGATIVE' if x < -0.05 else 'NEUTRAL'))
    print("TextBlob sentiment analysis complete.")
except Exception as e:
    print(f"TextBlob sentiment analysis not available: {e}")
    df_reviews['textblob_polarity'] = np.nan
    df_reviews['textblob_label'] = 'Not Processed'

# Print summary statistics for all three methods
print("\nSentiment label distribution (DistilBERT):")
print(df_reviews['sentiment_label'].value_counts())
print("\nSentiment label distribution (VADER):")
print(df_reviews['vader_label'].value_counts())
print("\nSentiment label distribution (TextBlob):")
print(df_reviews['textblob_label'].value_counts())

# 4. Thematic Analysis
# Extract keywords for each review
print("Extracting keywords for thematic analysis...")
thematic_analyzer = ThematicAnalyzer()
keywords_per_review = thematic_analyzer.extract_keywords(df_reviews['review_text_for_sentiment'].tolist())
df_reviews['keywords'] = [', '.join(kws) for kws in keywords_per_review]

# Group themes per bank
# ---
# Theme grouping logic:
# We use a rule-based approach to group keywords into 3-5 overarching themes per bank. The rules are:
#   - 'Account Access Issues': keywords matching login, access, account, password, sign in, log in, reset
#   - 'Transaction Performance': transfer, transaction, send, receive, delay, slow, fail, crash, error, network
#   - 'User Interface & Experience': ui, interface, design, easy, difficult, navigation, user-friendly, screen, button
#   - 'Customer Support': support, help, service, customer, response, contact
#   - 'Feature Requests': feature, add, missing, request, option, update, function, improve
# Any keyword not matching these patterns is grouped as 'Other'.
# Only the 3-5 most populated themes are kept per bank.
# ---
theme_results = {}
for bank, group in df_reviews.groupby('bank'):
    bank_keywords = [kws for kws in thematic_analyzer.extract_keywords(group['review_text_for_sentiment'].tolist())]
    themes = thematic_analyzer.group_themes(bank_keywords, n_themes=4)
    theme_results[bank] = themes
    print(f"Themes for {bank}: {themes}")
    # Assign the most likely theme to each review (first matching theme)
    review_themes = []
    for kws in bank_keywords:
        assigned = None
        for theme, theme_kws in themes.items():
            if any(kw in theme_kws for kw in kws):
                assigned = theme
                break
        review_themes.append(assigned if assigned else 'Other')
    df_reviews.loc[group.index, 'identified_theme'] = review_themes

# 5. Aggregate Sentiment by Bank and Rating
sentiment_summary = df_reviews.groupby(['bank', 'rating', 'sentiment_label']).size().unstack(fill_value=0)
print("\nSentiment Distribution by Bank and Rating:")
print(sentiment_summary)

# 6. Save Results
df_reviews.reset_index(inplace=True)
df_reviews.rename(columns={'index': 'review_id', 'review': 'review_text'}, inplace=True)
columns_to_save = ['review_id', 'review_text', 'sentiment_label', 'sentiment_score', 'identified_theme', 'bank', 'rating', 'date', 'source', 'keywords']
df_reviews[columns_to_save].to_csv(OUTPUT_PATH, index=False)
print(f"Results saved to {OUTPUT_PATH}")

# 7. KPI Checks
# Sentiment scores for 90%+ reviews
processed_sentiment_count = df_reviews[df_reviews['sentiment_label'].isin(['POSITIVE', 'NEGATIVE'])].shape[0]
total_reviews = len(df_reviews)
sentiment_coverage = (processed_sentiment_count / total_reviews) * 100 if total_reviews > 0 else 0
print(f"Sentiment analysis coverage: {sentiment_coverage:.2f}%")

# 3+ themes per bank with examples
for bank, themes in theme_results.items():
    print(f"\nThemes for {bank} (count: {len(themes)}):")
    for theme, keywords in themes.items():
        print(f"  {theme}: {keywords}") 