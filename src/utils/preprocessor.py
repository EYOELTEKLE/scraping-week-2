import pandas as pd

class ReviewPreprocessor:
    def __init__(self):
        """Initializes the ReviewPreprocessor."""
        pass

    def preprocess_data(self, reviews_list, bank_name):
        """
        Preprocesses a list of raw review data.

        Args:
            reviews_list (list): A list of dictionaries, where each dictionary is a raw review.
            bank_name (str): The name of the bank/app for tagging.

        Returns:
            pandas.DataFrame: A preprocessed DataFrame with columns: 
                              ['review', 'rating', 'date', 'bank', 'source'].
        """
        if not reviews_list:
            print(f"No reviews provided for preprocessing for {bank_name}.")
            # Return an empty DataFrame with expected columns if input is empty
            return pd.DataFrame(columns=['review', 'rating', 'date', 'bank', 'source'])

        df = pd.DataFrame(reviews_list)
        print(f"Preprocessing {len(df)} reviews for {bank_name}.")

        # Ensure essential columns are present (already handled by scraper, but good for robustness)
        # Expected columns from scraper: 'review', 'rating', 'date', 'userName'
        # We will drop 'userName' if it exists and isn't needed for final output.
        if 'userName' in df.columns:
            df = df.drop(columns=['userName'])

        # 1. Date Normalization: Convert 'date' column to YYYY-MM-DD
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        else:
            print("Warning: 'date' column not found for date normalization.")
            df['date'] = None # Or some default date / error handling

        # 2. Handle Missing Data (report, then simple drop for reviews)
        initial_rows = len(df)
        # For 'review' and 'rating', rows with missing values are less useful
        df.dropna(subset=['review', 'rating'], inplace=True)
        if len(df) < initial_rows:
            print(f"Dropped {initial_rows - len(df)} rows with missing 'review' or 'rating'.")

        # 3. Remove Duplicates: Based on 'review' text and 'date'
        # (A user might post the same review text on different dates, which could be valid)
        # (Or, a user might edit a review, keeping the same date - scraper might pick up both)
        # For simplicity, we'll consider review text + date as unique enough.
        # More robust would be review_id if available from scraper.
        initial_rows = len(df)
        df.drop_duplicates(subset=['review', 'date'], keep='first', inplace=True)
        if len(df) < initial_rows:
            print(f"Dropped {initial_rows - len(df)} duplicate reviews.")

        # 4. Add 'bank' and 'source' columns
        df['bank'] = bank_name
        df['source'] = 'Google Play'

        # 5. Select and Reorder Columns
        final_columns = ['review', 'rating', 'date', 'bank', 'source']
        # Ensure all final columns exist, add if missing (e.g. if date was missing initially)
        for col in final_columns:
            if col not in df.columns:
                df[col] = None 
        df = df[final_columns]

        print(f"Preprocessing complete for {bank_name}. {len(df)} reviews remaining.")
        print("Missing data summary after preprocessing:")
        print(df.isnull().sum())

        return df

    def preprocess_texts_with_spacy(self, texts):
        """
        Preprocesses texts using spaCy: tokenization, stop-word removal, lemmatization.
        Args:
            texts (list of str): List of review texts.
        Returns:
            list of str: Cleaned, lemmatized texts.
        """
        try:
            import spacy
            nlp = spacy.load('en_core_web_sm')
        except Exception as e:
            print(f"spaCy not available or model not loaded: {e}")
            return [str(t) for t in texts]
        cleaned = []
        for doc in nlp.pipe(texts, disable=["ner", "parser"]):
            tokens = [token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha]
            cleaned.append(' '.join(tokens))
        return cleaned
