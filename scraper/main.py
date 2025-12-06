from google_play_scraper import Sort, reviews_all
import pandas as pd

def collect_all_reviews(app_id, output_file="reviews.csv", lang="en", country="us"):
    print(f"Collecting ALL reviews for {app_id} ...")

    # Scrape ALL reviews
    result = reviews_all(
        app_id,
        lang=lang,      # language
        country=country,   # country
        sort=Sort.NEWEST
    )

    # Convert to DataFrame and keep only the 'content' column
    df = pd.DataFrame(result)

    if 'content' in df.columns:
        df = df[['content']]
    else:
        print("Error: 'content' column not found in scraped data.")
        return

    print(f"Collected {len(df)} reviews")

    # Save to CSV file
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"Saved to {output_file}")

# Example usage
collect_all_reviews("com.now.munchies.app", "munchies.csv")
