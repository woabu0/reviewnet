import os
import pandas as pd
import nltk
from nltk.corpus import words
from nltk.tokenize import word_tokenize
import re

# Download NLTK resources
nltk.download('words')
nltk.download('punkt')

english_vocab = set(w.lower() for w in words.words())
bengali_pattern = re.compile(r'[\u0980-\u09FF]')

def is_strict_english(text):
    if not isinstance(text, str):
        return False

    # Remove any Bengali script → auto reject
    if bengali_pattern.search(text):
        return False

    # Tokenize English words
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha()]

    if len(tokens) == 0:
        return False

    english_matches = sum(1 for t in tokens if t in english_vocab)

    return english_matches / len(tokens) >= 0.8


# Folders
data_folder = "../data"
output_folder = "cleaned"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Process each CSV in /data
for file in os.listdir(data_folder):
    if file.endswith(".csv"):
        print(f"Processing {file} ...")

        df = pd.read_csv(os.path.join(data_folder, file))

        # change 'content' to your actual column name
        df["is_english"] = df["content"].apply(is_strict_english)

        english_df = df[df["is_english"] == True]

        output_path = os.path.join(output_folder, f"english_{file}")
        english_df.to_csv(output_path, index=False)

        print(f"Saved cleaned file → {output_path}")

print("\nAll files processed successfully!")
