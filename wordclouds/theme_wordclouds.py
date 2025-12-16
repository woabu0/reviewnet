# Theme-Specific Word Cloud Generation Script
# Generates unigram word clouds for each theme, where word size represents term frequency
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
    THEME_FILE = '../theme/theme.csv'
    OUTPUT_DIR = 'outputs'
    WORDCLOUD_DIR = os.path.join(OUTPUT_DIR, 'theme_wordclouds')
    
    # Word cloud settings
    WIDTH = 1200
    HEIGHT = 800
    MAX_WORDS = 100
    BACKGROUND_COLOR = 'white'
    COLORMAP = 'viridis'  # Can be changed to 'plasma', 'inferno', 'magma', etc.

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
        'review', 'reviews', 'rating', 'ratings', 'star', 'stars', 'good', 'bad',
        'great', 'excellent', 'poor', 'worst', 'best', 'very', 'really', 'much',
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
        'hmm', 'haha', 'lol', 'lmao', 'rofl', 'omg', 'wtf', 'fuck', 'shit',
        'damn', 'hell', 'crap', 'stupid', 'idiot', 'moron', 'ass', 'asshole',
        'bitch', 'bastard', 'fucking', 'shitty', 'damned', 'hellish', 'crap',
        'stupidly', 'idiotically', 'moronically', 'assholish', 'bitchy', 'bastardly'
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
# Theme Loading and Classification
# ============================================================================
def load_themes():
    """Load themes and sub-themes from CSV file"""
    try:
        theme_df = pd.read_csv(Config.THEME_FILE)
        return theme_df
    except Exception as e:
        print(f"❌ Error loading themes: {e}")
        return None

def classify_review_to_theme_subtheme(text, theme_df):
    """Classify a review into theme and sub-theme using keyword matching"""
    if not text or pd.isna(text):
        return None, None
    
    text_lower = text.lower()
    
    # Score each theme-subtheme pair
    scores = {}
    
    for _, row in theme_df.iterrows():
        theme = row['theme']
        subtheme = row['subtheme']
        
        # Create keywords from theme and subtheme
        theme_keywords = theme.lower().split()
        subtheme_keywords = subtheme.lower().split()
        
        # Score based on keyword matches
        score = 0
        for keyword in theme_keywords:
            if keyword in text_lower:
                score += 2
        for keyword in subtheme_keywords:
            if keyword in text_lower:
                score += 3
        
        # Check for specific phrases
        if subtheme.lower() in text_lower:
            score += 5
        
        if score > 0:
            key = (theme, subtheme)
            scores[key] = score
    
    if not scores:
        return None, None
    
    # Return the highest scoring theme-subtheme pair
    best_match = max(scores.items(), key=lambda x: x[1])
    return best_match[0]  # Returns (theme, subtheme) tuple

# ============================================================================
# Data Loading
# ============================================================================
def load_data_from_file(file_path):
    """Load data from a single CSV file"""
    try:
        df = pd.read_csv(file_path)
        
        if 'content' not in df.columns:
            for col in df.columns:
                if 'review' in col.lower() or 'text' in col.lower() or 'comment' in col.lower():
                    df['content'] = df[col]
                    break
        
        if 'content' not in df.columns:
            return None
        
        df = df.dropna(subset=["content"])
        df['content'] = df['content'].astype(str)
        df['cleaned_text'] = df['content'].apply(clean_text_advanced)
        
        # Extract app name from filename
        app_name = os.path.basename(file_path).replace('.csv', '')
        df['app'] = app_name
        
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def load_all_data():
    """Load all data from the data directory"""
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
        
        df = load_data_from_file(file_path)
        if df is not None and len(df) > 0:
            all_data.append(df)
            print(f"   📄 Loaded {filename}: {len(df)} rows")
    
    if not all_data:
        print("❌ No valid data found")
        return None
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"✅ Combined dataset: {len(combined_df)} rows")
    return combined_df

# ============================================================================
# Word Cloud Generation
# ============================================================================
def generate_wordcloud_for_theme(theme_name, texts, output_dir, colormap='viridis'):
    """Generate word cloud for a specific theme"""
    try:
        # Combine all texts for this theme
        combined_text = ' '.join(texts)
        
        if not combined_text or len(combined_text.strip()) == 0:
            print(f"   ⚠️  No text available for theme: {theme_name}")
            return False
        
        # Get stopwords
        stopwords_set = get_stopwords()
        
        # Tokenize and count frequencies
        all_tokens = []
        for text in texts:
            tokens = tokenize_text(text, stopwords_set)
            all_tokens.extend(tokens)
        
        if not all_tokens:
            print(f"   ⚠️  No valid tokens for theme: {theme_name}")
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
        plt.title(f'Word Cloud: {theme_name}', fontsize=24, fontweight='bold', pad=20)
        plt.tight_layout(pad=0)
        
        # Save
        os.makedirs(output_dir, exist_ok=True)
        # Sanitize theme name for filename
        safe_theme_name = re.sub(r'[^\w\s-]', '', theme_name).strip().replace(' ', '_')
        filepath = os.path.join(output_dir, f'wordcloud_{safe_theme_name}.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"   ✅ Generated: {filepath}")
        return True
        
    except Exception as e:
        print(f"   ❌ Error generating word cloud for {theme_name}: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_all_wordclouds(df, output_dir):
    """Generate word clouds for all themes"""
    print("\n" + "="*80)
    print("📊 GENERATING THEME-SPECIFIC WORD CLOUDS")
    print("="*80)
    
    # Group by theme
    themes = df['theme'].unique()
    themes = [t for t in themes if pd.notna(t)]
    
    print(f"\n📋 Found {len(themes)} themes to process\n")
    
    # Color maps for different themes (cycling through)
    colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'coolwarm', 'RdYlGn', 'Set2', 'tab20']
    
    success_count = 0
    for i, theme in enumerate(sorted(themes)):
        print(f"🎨 Processing theme {i+1}/{len(themes)}: {theme}")
        
        # Get all texts for this theme
        theme_texts = df[df['theme'] == theme]['cleaned_text'].tolist()
        theme_texts = [t for t in theme_texts if t and len(t.strip()) > 0]
        
        print(f"   📝 Found {len(theme_texts)} reviews")
        
        if len(theme_texts) == 0:
            print(f"   ⚠️  Skipping (no reviews)")
            continue
        
        # Select colormap
        colormap = colormaps[i % len(colormaps)]
        
        # Generate word cloud
        if generate_wordcloud_for_theme(theme, theme_texts, output_dir, colormap):
            success_count += 1
        
        print()
    
    print("="*80)
    print(f"✅ Generated {success_count}/{len(themes)} word clouds")
    print("="*80)
    
    return success_count

# ============================================================================
# Main Function
# ============================================================================
def main():
    print("="*80)
    print("☁️  THEME-SPECIFIC WORD CLOUD GENERATION")
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
    
    # Load themes
    print("\n📂 Loading themes...")
    theme_df = load_themes()
    if theme_df is None:
        print("❌ Failed to load themes. Cannot proceed.")
        return
    
    # Try to load existing classified data first
    existing_results = os.path.join(Config.OUTPUT_DIR, 'prescriptive_analysis', 'prescriptive_analysis_results.csv')
    
    if os.path.exists(existing_results):
        print(f"\n📂 Loading existing classified data from: {existing_results}")
        df = pd.read_csv(existing_results)
        
        # Check if we have theme column
        if 'theme' not in df.columns:
            print("❌ Theme column not found in existing data. Will classify reviews...")
            df = None
        else:
            print(f"✅ Loaded {len(df)} classified reviews")
            # We need cleaned_text, so let's add it
            if 'cleaned_text' not in df.columns:
                df['cleaned_text'] = df['content'].apply(clean_text_advanced)
    else:
        df = None
    
    # If no existing data, load and classify
    if df is None:
        print("\n📂 Loading data...")
        df = load_all_data()
        if df is None:
            return
        
        # Classify reviews into themes
        print("\n🔍 Classifying reviews into themes and sub-themes...")
        theme_subtheme_pairs = []
        for text in df['cleaned_text']:
            pair = classify_review_to_theme_subtheme(text, theme_df)
            theme_subtheme_pairs.append(pair)
        
        df['theme'] = [pair[0] if pair else None for pair in theme_subtheme_pairs]
        df['subtheme'] = [pair[1] if pair else None for pair in theme_subtheme_pairs]
        
        # Filter to classified reviews
        total_reviews = len(df)
        df = df[df['theme'].notna()].copy()
        print(f"✅ Classified {len(df)} reviews ({len(df)/total_reviews*100:.1f}% of total)")
    
    # Generate word clouds
    success_count = generate_all_wordclouds(df, Config.WORDCLOUD_DIR)
    
    print("\n" + "="*80)
    print("✅ WORD CLOUD GENERATION COMPLETE!")
    print("="*80)
    print(f"📁 All word clouds saved in: {Config.WORDCLOUD_DIR}")
    print(f"📊 Generated {success_count} word clouds")

if __name__ == "__main__":
    main()

