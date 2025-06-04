from google_play_scraper import Sort, reviews_all
import pandas as pd

class PlayStoreScraper:
    def __init__(self):
        """Initializes the PlayStoreScraper."""
        pass

    def get_reviews(self, app_id, app_name, lang='en', country='us', count=400):
        """
        Fetches a specified number of reviews for a given app ID.

        Args:
            app_id (str): The Google Play Store ID of the app.
            app_name (str): The name of the bank/app for tagging.
            lang (str, optional): Language code for reviews. Defaults to 'en'.
            country (str, optional): Country code for reviews. Defaults to 'us'.
            count (int, optional): The target number of reviews to fetch. Defaults to 400.

        Returns:
            list: A list of dictionaries, where each dictionary represents a review.
                  Returns an empty list if an error occurs or no reviews are found.
        """
        print(f"Starting to scrape reviews for {app_name} (ID: {app_id}). Target: {count} reviews.")
        try:
            result = reviews_all(
                app_id,
                sleep_milliseconds=0, # Optional: delay between requests
                lang=lang,            # defaults to 'en'
                country=country,      # defaults to 'us'
                sort=Sort.NEWEST,     # start with newest
            )
            
            reviews_df = pd.DataFrame(result)
            
            if reviews_df.empty:
                print(f"No reviews found for {app_name} ({app_id}).")
                return []

            # Select and rename columns
            # userName, content, score, at
            reviews_df = reviews_df[['userName', 'content', 'score', 'at']]
            reviews_df.rename(columns={
                'content': 'review',
                'score': 'rating',
                'at': 'date'
            }, inplace=True)
            
            # Ensure we don't exceed the available reviews
            actual_fetched_count = min(len(reviews_df), count)
            print(f"Successfully fetched {len(reviews_df)} reviews in total for {app_name}. Taking the newest {actual_fetched_count}.")
            
            return reviews_df.head(actual_fetched_count).to_dict('records')
            
        except Exception as e:
            print(f"An error occurred while scraping reviews for {app_name} ({app_id}): {e}")
            return []
