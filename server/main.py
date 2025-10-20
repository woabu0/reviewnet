from google_play_scraper import Sort, reviews_all
import pandas as pd

def collect_reviews(app_id, output_file="reviews.csv"):
    print(f"Collecting reviews for {app_id} ...")
    
    # Scrape all reviews
    result = reviews_all(
        app_id,
        sleep_milliseconds=0,
        lang='en',  # language
        country='us',  # country
        sort=Sort.NEWEST
    )
    
    # Convert to DataFrame
    df = pd.DataFrame(result)
    print(f"Collected {len(df)} reviews")
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")

# Example
collect_reviews("com.foodiBd", "foodi_reviews.csv")
