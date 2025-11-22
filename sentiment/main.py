import pandas as pd
import re
from textblob import TextBlob
from langdetect import detect, DetectorFactory
from googletrans import Translator
import emoji
import os

# Fix for consistent language detection
DetectorFactory.seed = 0
translator = Translator()


def clean_text(text):
    """Remove URLs, extra spaces, and some unwanted characters but keep emojis."""
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # remove URLs
    text = re.sub(r"\d+", "", text)  # remove numbers
    text = re.sub(r"[^\w\s" + emoji.get_emoji_regexp().pattern + "]", "", text)  # keep emojis
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_language(text):
    """Detect language safely."""
    try:
        return detect(text)
    except:
        return "unknown"


def translate_to_english(text, src_lang):
    """Translate non-English text to English."""
    try:
        if src_lang != "en" and src_lang != "unknown" and text.strip() != "":
            translated = translator.translate(text, src="auto", dest="en")
            return translated.text
        else:
            return text
    except:
        return text  # fallback if translation fails


def analyze_sentiment(text):
    """Return sentiment polarity (-1 to 1) and label."""
    if pd.isna(text) or str(text).strip() == "":
        return 0.0, "Neutral"
    blob = TextBlob(str(text))
    polarity = blob.sentiment.polarity
    if polarity > 0.05:
        label = "Positive"
    elif polarity < -0.05:
        label = "Negative"
    else:
        label = "Neutral"
    return polarity, label


def main():
    print("=== 📱 Multilingual App Review Sentiment Analyzer ===")
    input_path = input("Enter path to your CSV file: ").strip()

    if not os.path.exists(input_path):
        print("File not found.")
        return

    df = pd.read_csv(input_path)

    print("\nColumns found:", list(df.columns))
    text_col = input("\nEnter the column name containing review text: ").strip()

    if text_col not in df.columns:
        print("Column not found in CSV.")
        return

    print("\nCleaning and analyzing reviews... Please wait ⏳")

    # Step 1: Clean text
    df["Cleaned_Text"] = df[text_col].astype(str).apply(clean_text)

    # Step 2: Detect language
    df["Language"] = df["Cleaned_Text"].apply(detect_language)

    # Step 3: Translate non-English text
    df["Translated_Text"] = [
        translate_to_english(text, lang) for text, lang in zip(df["Cleaned_Text"], df["Language"])
    ]

    # Step 4: Sentiment analysis
    df["Polarity"], df["Sentiment"] = zip(*df["Translated_Text"].map(analyze_sentiment))

    # Create final dataframe with only 3 columns
    final_df = pd.DataFrame({
        'content': df['Translated_Text'],
        'polarity': df['Polarity'],
        'sentiment': df['Sentiment']
    })

    # Save results
    output_csv = "app_review_sentiment_results.csv"
    output_xlsx = "app_review_sentiment_results.xlsx"

    final_df.to_csv(output_csv, index=False)
    final_df.to_excel(output_xlsx, index=False)

    print(f"\nAnalysis complete!")
    print(f"📄 Saved CSV: {output_csv}")
    print(f"📊 Saved Excel: {output_xlsx}")
    print("\nColumns in output file: [content, polarity, sentiment]")


def analyze_sentiment_full(df, text_col):
    """Process dataframe for sentiment analysis and return results DataFrame."""
    # Step 1: Clean text
    df["Cleaned_Text"] = df[text_col].astype(str).apply(clean_text)

    # Step 2: Detect language
    df["Language"] = df["Cleaned_Text"].apply(detect_language)

    # Step 3: Translate non-English text
    df["Translated_Text"] = [
        translate_to_english(text, lang) for text, lang in zip(df["Cleaned_Text"], df["Language"])
    ]

    # Step 4: Sentiment analysis
    df["Polarity"], df["Sentiment"] = zip(*df["Translated_Text"].map(analyze_sentiment))

    # Create final dataframe
    final_df = pd.DataFrame({
        'original_content': df[text_col],
        'content': df['Translated_Text'],
        'polarity': df['Polarity'],
        'sentiment': df['Sentiment'],
        'language': df['Language']
    })

    return final_df


if __name__ == "__main__":
    main()
