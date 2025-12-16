# Sentiment-Specific Word Cloud Generation Script
# Generates unigram word clouds for each sentiment class (Positive, Neutral, Negative)
# where word size represents term frequency within each sentiment
import os
import sys
import pandas as pd
import numpy as np
import glob
import re
import warnings
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

warnings.filterwarnings("ignore")

# ============================================================================
# Configuration
# ============================================================================
class Config:
    DATA_DIR = '../dataset'
    OUTPUT_DIR = 'outputs'
    WORDCLOUD_DIR = os.path.join(OUTPUT_DIR, 'sentiment_wordclouds')
    
    # Word cloud settings
    WIDTH = 1200
    HEIGHT = 800
    MAX_WORDS = 100
    BACKGROUND_COLOR = 'white'
    
    # Color schemes for each sentiment
    COLORMAPS = {
        'Positive': 'Greens',      # Green for positive
        'Neutral': 'Blues',         # Blue for neutral
        'Negative': 'Reds'          # Red for negative
    }
    
    # Alternative: Use specific colors
    # COLORMAPS = {
    #     'Positive': 'viridis',
    #     'Neutral': 'plasma',
    #     'Negative': 'inferno'
    # }

# ============================================================================
# Text Preprocessing
# ============================================================================
def clean_text_advanced(text):
    """Advanced text cleaning with better handling"""
    if pd.isna(text) or not text:
        return ""
    
    text = str(text).strip()
    if len(text) == 0:
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove special characters but keep Bengali and basic punctuation
    text = re.sub(r'[^\w\s\u0980-\u09FF@#!?]', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text if len(text) > 0 else " "

def get_stopwords():
    """Get stopwords for English and common words"""
    try:
        english_stopwords = set(stopwords.words('english'))
    except:
        english_stopwords = set()
    
    # Add common food delivery app specific stopwords
    custom_stopwords = {
        'app', 'application', 'food', 'order', 'delivery', 'service', 'restaurant',
        'review', 'reviews', 'rating', 'ratings', 'star', 'stars', 'very', 'really', 'much',
        'also', 'would', 'could', 'should', 'will', 'can', 'may', 'might',
        'one', 'two', 'three', 'first', 'second', 'last', 'time', 'times',
        'day', 'days', 'week', 'weeks', 'month', 'months', 'year', 'years',
        'today', 'yesterday', 'tomorrow', 'now', 'then', 'when', 'where',
        'what', 'which', 'who', 'why', 'how', 'this', 'that', 'these', 'those',
        'the', 'a', 'an', 'and', 'or', 'but', 'if', 'because', 'as', 'until',
        'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
        'into', 'through', 'during', 'before', 'after', 'above', 'below', 'up',
        'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',
        'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
        'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
        'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
        'can', 'will', 'just', 'don', 'should', 'now', 'get', 'got', 'go',
        'come', 'came', 'see', 'saw', 'know', 'knew', 'think', 'thought',
        'take', 'took', 'give', 'gave', 'make', 'made', 'use', 'used', 'try',
        'tried', 'want', 'wanted', 'need', 'needed', 'like', 'liked', 'love',
        'loved', 'say', 'said', 'tell', 'told', 'ask', 'asked', 'work', 'worked',
        'call', 'called', 'call', 'calling', 'phone', 'number', 'email', 'address',
        'please', 'thank', 'thanks', 'sorry', 'ok', 'okay', 'yes', 'no', 'yeah',
        'yep', 'nope', 'maybe', 'sure', 'fine', 'well', 'oh', 'ah', 'um', 'uh',
        'hmm', 'haha', 'lol', 'lmao', 'rofl', 'omg', 'wtf'
    }
    
    return english_stopwords.union(custom_stopwords)

def tokenize_text(text, stopwords_set):
    """Tokenize text into unigrams and filter stopwords"""
    if not text or pd.isna(text):
        return []
    
    try:
        # Tokenize
        tokens = word_tokenize(text.lower())
        
        # Filter: keep only alphabetic words, length > 2, not in stopwords
        tokens = [
            token for token in tokens
            if token.isalpha() and len(token) > 2 and token not in stopwords_set
        ]
        
        return tokens
    except:
        # Fallback: simple split if NLTK fails
        tokens = text.lower().split()
        tokens = [
            token for token in tokens
            if token.isalpha() and len(token) > 2 and token not in stopwords_set
        ]
        return tokens

# ============================================================================
# Data Loading
# ============================================================================
def load_sentiment_data():
    """Load data with sentiment labels from various sources"""
    # Try to load from existing sentiment results (prioritize prescriptive analysis)
    sentiment_files = [
        os.path.join(Config.OUTPUT_DIR, 'prescriptive_analysis', 'prescriptive_analysis_results.csv'),
        os.path.join(Config.OUTPUT_DIR, 'qwen_sentiment_results', 'qwen_sentiment_results.csv'),
    ]
    
    # Also check for any CSV files in outputs with sentiment
    output_dir = Config.OUTPUT_DIR
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if 'sentiment' in file.lower() and file.endswith('.csv'):
                sentiment_files.append(os.path.join(root, file))
    
    df = None
    for file_path in sentiment_files:
        if os.path.exists(file_path):
            try:
                temp_df = pd.read_csv(file_path)
                # Check if it has sentiment labels
                if 'sentiment_label' in temp_df.columns or 'predicted_sentiment' in temp_df.columns:
                    df = temp_df
                    print(f"✅ Loaded sentiment data from: {file_path}")
                    break
            except:
                continue
    
    # If no sentiment data found, load raw data and classify
    if df is None:
        print("⚠️  No existing sentiment data found. Loading raw data...")
        df = load_raw_data()
        
        if df is None:
            return None
        
        # Try to use prescriptive analysis results which should have sentiment
        prescriptive_file = os.path.join(Config.OUTPUT_DIR, 'prescriptive_analysis', 'prescriptive_analysis_results.csv')
        if os.path.exists(prescriptive_file):
            try:
                prescriptive_df = pd.read_csv(prescriptive_file)
                if 'sentiment_label' in prescriptive_df.columns:
                    # Merge on content or cleaned_text
                    if 'cleaned_text' in prescriptive_df.columns and 'cleaned_text' in df.columns:
                        df = df.merge(
                            prescriptive_df[['cleaned_text', 'sentiment_label']].drop_duplicates(),
                            on='cleaned_text',
                            how='left'
                        )
                        print("✅ Merged sentiment labels from prescriptive analysis")
            except:
                pass
    
    # Ensure we have sentiment labels
    if df is not None:
        if 'sentiment_label' not in df.columns:
            if 'predicted_sentiment' in df.columns:
                df['sentiment_label'] = df['predicted_sentiment'].map({0: 'Negative', 1: 'Neutral', 2: 'Positive'})
            else:
                print("❌ No sentiment labels found in data")
                return None
        
        # Clean text if needed
        if 'cleaned_text' not in df.columns:
            if 'content' in df.columns:
                df['cleaned_text'] = df['content'].apply(clean_text_advanced)
            else:
                print("❌ No text column found")
                return None
        
        # Filter to valid sentiment labels
        df = df[df['sentiment_label'].isin(['Positive', 'Neutral', 'Negative'])].copy()
        
        return df
    
    return None

def load_raw_data():
    """Load raw data from data directory"""
    data_dir = Config.DATA_DIR
    all_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    if not all_files:
        print(f"❌ No CSV files found in {data_dir}")
        return None
    
    print(f"📂 Found {len(all_files)} CSV files")
    
    all_data = []
    for file_path in all_files:
        filename = os.path.basename(file_path)
        if filename.startswith('.'):
            continue
        
        try:
            df = pd.read_csv(file_path)
            
            if 'content' not in df.columns:
                for col in df.columns:
                    if 'review' in col.lower() or 'text' in col.lower() or 'comment' in col.lower():
                        df['content'] = df[col]
                        break
            
            if 'content' not in df.columns:
                continue
            
            df = df.dropna(subset=["content"])
            df['content'] = df['content'].astype(str)
            df['cleaned_text'] = df['content'].apply(clean_text_advanced)
            
            # Extract app name from filename
            app_name = os.path.basename(file_path).replace('.csv', '')
            df['app'] = app_name
            
            all_data.append(df)
            print(f"   📄 Loaded {filename}: {len(df)} rows")
        except Exception as e:
            print(f"   ⚠️  Error loading {filename}: {e}")
    
    if not all_data:
        print("❌ No valid data found")
        return None
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"✅ Combined dataset: {len(combined_df)} rows")
    return combined_df

# ============================================================================
# Word Cloud Generation
# ============================================================================
def generate_wordcloud_for_sentiment(sentiment_name, texts, output_dir, colormap='viridis'):
    """Generate word cloud for a specific sentiment"""
    try:
        # Combine all texts for this sentiment
        combined_text = ' '.join(texts)
        
        if not combined_text or len(combined_text.strip()) == 0:
            print(f"   ⚠️  No text available for sentiment: {sentiment_name}")
            return False
        
        # Get stopwords
        stopwords_set = get_stopwords()
        
        # Tokenize and count frequencies
        all_tokens = []
        for text in texts:
            tokens = tokenize_text(text, stopwords_set)
            all_tokens.extend(tokens)
        
        if not all_tokens:
            print(f"   ⚠️  No valid tokens for sentiment: {sentiment_name}")
            return False
        
        # Count word frequencies
        word_freq = Counter(all_tokens)
        
        # Create word cloud
        wordcloud = WordCloud(
            width=Config.WIDTH,
            height=Config.HEIGHT,
            max_words=Config.MAX_WORDS,
            background_color=Config.BACKGROUND_COLOR,
            colormap=colormap,
            relative_scaling=0.5,
            min_font_size=10,
            collocations=False  # Don't consider bigrams
        ).generate_from_frequencies(word_freq)
        
        # Create figure
        plt.figure(figsize=(16, 10))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud: {sentiment_name} Sentiment', fontsize=24, fontweight='bold', pad=20)
        plt.tight_layout(pad=0)
        
        # Save
        os.makedirs(output_dir, exist_ok=True)
        # Sanitize sentiment name for filename
        safe_sentiment_name = re.sub(r'[^\w\s-]', '', sentiment_name).strip().replace(' ', '_')
        filepath = os.path.join(output_dir, f'wordcloud_{safe_sentiment_name}.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"   ✅ Generated: {filepath}")
        return True
        
    except Exception as e:
        print(f"   ❌ Error generating word cloud for {sentiment_name}: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_all_sentiment_wordclouds(df, output_dir):
    """Generate word clouds for all sentiment classes"""
    print("\n" + "="*80)
    print("📊 GENERATING SENTIMENT-SPECIFIC WORD CLOUDS")
    print("="*80)
    
    # Group by sentiment
    sentiments = ['Positive', 'Neutral', 'Negative']
    
    print(f"\n📋 Generating word clouds for {len(sentiments)} sentiment classes\n")
    
    success_count = 0
    for sentiment in sentiments:
        print(f"🎨 Processing {sentiment} sentiment...")
        
        # Get all texts for this sentiment
        sentiment_texts = df[df['sentiment_label'] == sentiment]['cleaned_text'].tolist()
        sentiment_texts = [t for t in sentiment_texts if t and len(t.strip()) > 0]
        
        print(f"   📝 Found {len(sentiment_texts)} reviews")
        
        if len(sentiment_texts) == 0:
            print(f"   ⚠️  Skipping (no reviews)")
            continue
        
        # Get colormap for this sentiment
        colormap = Config.COLORMAPS.get(sentiment, 'viridis')
        
        # Generate word cloud
        if generate_wordcloud_for_sentiment(sentiment, sentiment_texts, output_dir, colormap):
            success_count += 1
        
        print()
    
    print("="*80)
    print(f"✅ Generated {success_count}/{len(sentiments)} word clouds")
    print("="*80)
    
    return success_count

# ============================================================================
# Main Function
# ============================================================================
def main():
    print("="*80)
    print("☁️  SENTIMENT-SPECIFIC WORD CLOUD GENERATION")
    print("="*80)
    
    # Create output directory
    os.makedirs(Config.WORDCLOUD_DIR, exist_ok=True)
    print(f"📁 Word clouds will be saved in: {Config.WORDCLOUD_DIR}\n")
    
    # Check if wordcloud library is available
    try:
        import wordcloud
        print("✅ WordCloud library available")
    except ImportError:
        print("❌ WordCloud library not available. Installing...")
        print("   Please run: pip install wordcloud")
        return
    
    # Load data with sentiment labels
    print("\n📂 Loading data with sentiment labels...")
    df = load_sentiment_data()
    
    if df is None:
        print("❌ Failed to load data. Cannot proceed.")
        return
    
    print(f"✅ Loaded {len(df)} reviews with sentiment labels")
    
    # Print sentiment distribution
    print("\n📊 Sentiment Distribution:")
    sentiment_dist = df['sentiment_label'].value_counts()
    for label, count in sentiment_dist.items():
        percentage = (count / len(df)) * 100
        print(f"   {label}: {count} ({percentage:.1f}%)")
    
    # Generate word clouds
    success_count = generate_all_sentiment_wordclouds(df, Config.WORDCLOUD_DIR)
    
    print("\n" + "="*80)
    print("✅ SENTIMENT WORD CLOUD GENERATION COMPLETE!")
    print("="*80)
    print(f"📁 All word clouds saved in: {Config.WORDCLOUD_DIR}")
    print(f"📊 Generated {success_count} word clouds")
    print("\n📋 Files generated:")
    print("   - wordcloud_Positive.png (a)")
    print("   - wordcloud_Neutral.png (b)")
    print("   - wordcloud_Negative.png (c)")

if __name__ == "__main__":
    main()

