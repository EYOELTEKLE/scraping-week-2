# Placeholder for SentimentAnalyzer and ThematicAnalyzer classes
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import re
from collections import Counter, defaultdict

class SentimentAnalyzer:
    def __init__(self, model_name='distilbert-base-uncased-finetuned-sst-2-english'):
        """Initializes the Sentiment Analyzer with a specified model."""
        try:
            self.sentiment_pipeline = pipeline(
                'sentiment-analysis', 
                model=model_name,
                # Explicitly specify truncation if reviews can be long
                # tokenizer_kwargs={'truncation': True, 'max_length': 512} 
            )
            print(f"Sentiment pipeline loaded successfully with model: {model_name}")
        except Exception as e:
            print(f"Error loading sentiment pipeline model {model_name}: {e}")
            self.sentiment_pipeline = None

    def predict_sentiment(self, text_list):
        """
        Predicts sentiment for a list of texts.

        Args:
            text_list (list of str): A list of texts to analyze.

        Returns:
            list: A list of dictionaries, where each dictionary contains 'label' and 'score'.
                  Returns an empty list if the pipeline is not loaded or an error occurs.
        """
        if not self.sentiment_pipeline:
            print("Sentiment pipeline not loaded. Cannot predict.")
            return []
        if not isinstance(text_list, list):
            text_list = [text_list] # Ensure it's a list for pipeline
        
        try:
            # The pipeline handles tokenization and batching internally for lists.
            # Ensure texts are strings
            processed_text_list = [str(text) if text is not None else "" for text in text_list]
            results = self.sentiment_pipeline(processed_text_list, truncation=True)
            return results
        except Exception as e:
            print(f"Error during sentiment prediction: {e}")
            return []

class ThematicAnalyzer:
    def __init__(self, max_keywords=10, ngram_range=(1,2), stop_words='english'):
        self.max_keywords = max_keywords
        self.ngram_range = ngram_range
        self.stop_words = stop_words

    def extract_keywords(self, texts):
        """
        Extracts significant keywords and n-grams from a list of texts using TF-IDF.
        Args:
            texts (list of str): List of review texts.
        Returns:
            list of list: Each sublist contains top keywords/ngrams for the corresponding text.
        """
        if not texts:
            return []
        vectorizer = TfidfVectorizer(ngram_range=self.ngram_range, stop_words=self.stop_words, max_features=1000)
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = np.array(vectorizer.get_feature_names_out())
        keywords_per_doc = []
        for row in tfidf_matrix:
            row_data = row.toarray().flatten()
            top_indices = row_data.argsort()[-self.max_keywords:][::-1]
            keywords = feature_names[top_indices][row_data[top_indices] > 0]
            keywords_per_doc.append(list(keywords))
        return keywords_per_doc

    def group_themes(self, keywords_list, n_themes=4):
        """
        Groups related keywords into overarching themes using rule-based clustering.
        Args:
            keywords_list (list of list): List of keyword lists per review.
            n_themes (int): Number of themes to group into.
        Returns:
            dict: Mapping from theme name to list of example keywords.
        """
        # Flatten all keywords
        all_keywords = [kw for kws in keywords_list for kw in kws]
        if not all_keywords:
            return {}
        # Count most common keywords
        keyword_counts = Counter(all_keywords)
        most_common = [kw for kw, _ in keyword_counts.most_common(30)]
        # Simple rule-based grouping by keyword patterns
        theme_rules = {
            'Account Access Issues': re.compile(r'(login|access|account|password|sign in|log in|reset)', re.I),
            'Transaction Performance': re.compile(r'(transfer|transaction|send|receive|delay|slow|fail|crash|error|network)', re.I),
            'User Interface & Experience': re.compile(r'(ui|interface|design|easy|difficult|navigation|user[- ]?friendly|screen|button)', re.I),
            'Customer Support': re.compile(r'(support|help|service|customer|response|contact)', re.I),
            'Feature Requests': re.compile(r'(feature|add|missing|request|option|update|function|improve)', re.I),
        }
        theme_keywords = defaultdict(list)
        for kw in most_common:
            matched = False
            for theme, pattern in theme_rules.items():
                if pattern.search(kw):
                    theme_keywords[theme].append(kw)
                    matched = True
                    break
            if not matched:
                theme_keywords['Other'].append(kw)
        # Limit to n_themes most populated themes
        sorted_themes = sorted(theme_keywords.items(), key=lambda x: len(x[1]), reverse=True)
        grouped = {theme: kws[:6] for theme, kws in sorted_themes[:n_themes]}
        return grouped
