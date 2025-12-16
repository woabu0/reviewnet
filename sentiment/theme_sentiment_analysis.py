# Theme-Based Sentiment Analysis Script
# Creates sentiment heatmap by theme from theme.csv
import os
import sys
import pandas as pd
import numpy as np
import glob
import re
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ============================================================================
# Configuration
# ============================================================================
class Config:
    DATA_DIR = '../dataset'
    THEME_FILE = '../theme/theme.csv'
    OUTPUT_DIR = 'outputs'
    RESULTS_DIR = os.path.join(OUTPUT_DIR, 'theme_sentiment_analysis')

# ============================================================================
# Theme Loading and Classification
# ============================================================================
def load_themes():
    """Load themes and sub-themes from CSV file"""
    try:
        theme_df = pd.read_csv(Config.THEME_FILE)
        print(f"📂 Loaded {len(theme_df)} theme-subtheme pairs from {Config.THEME_FILE}")
        
        # Group by theme
        theme_dict = defaultdict(list)
        for _, row in theme_df.iterrows():
            theme = row['theme']
            subtheme = row['subtheme']
            theme_dict[theme].append(subtheme)
        
        return theme_dict, theme_df
    except Exception as e:
        print(f"❌ Error loading themes: {e}")
        return None, None

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
def load_data_with_sentiment():
    """Load data with sentiment labels from prescriptive analysis or raw data"""
    # Try to load from prescriptive analysis results first
    if os.path.exists(Config.PRESCRIPTIVE_RESULTS):
        print(f"📂 Loading existing sentiment data from: {Config.PRESCRIPTIVE_RESULTS}")
        try:
            df = pd.read_csv(Config.PRESCRIPTIVE_RESULTS)
            
            # Check if we have required columns
            if 'sentiment_label' in df.columns and 'theme' in df.columns:
                print(f"✅ Loaded {len(df)} reviews with sentiment and theme labels")
                return df
            elif 'sentiment_label' in df.columns:
                print("⚠️  Found sentiment labels but no theme. Will classify themes...")
                # We'll classify themes later
                return df
            else:
                print("⚠️  No sentiment labels found. Will need to run sentiment analysis...")
        except Exception as e:
            print(f"⚠️  Error loading prescriptive results: {e}")
    
    # If no prescriptive results, load raw data
    print("📂 Loading raw data...")
    return load_raw_data()

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
# Visualization Functions
# ============================================================================
def create_sentiment_heatmap_by_theme(df, output_dir):
    """Create heatmap showing sentiment distribution by theme"""
    try:
        # Filter to reviews with both theme and sentiment
        df_filtered = df[(df['theme'].notna()) & (df['sentiment_label'].notna())].copy()
        
        if len(df_filtered) == 0:
            print("   ⚠️  No data with both theme and sentiment labels")
            return False
        
        # Create pivot table: theme vs sentiment
        sentiment_theme = pd.crosstab(df_filtered['theme'], df_filtered['sentiment_label'])
        
        # Normalize to percentages (row-wise)
        sentiment_theme_pct = sentiment_theme.div(sentiment_theme.sum(axis=1), axis=0) * 100
        
        # Ensure we have all sentiment labels
        for label in ['Positive', 'Neutral', 'Negative']:
            if label not in sentiment_theme_pct.columns:
                sentiment_theme_pct[label] = 0
        
        # Reorder columns
        sentiment_theme_pct = sentiment_theme_pct[['Positive', 'Neutral', 'Negative']]
        
        plt.figure(figsize=(14, 10))
        sns.heatmap(sentiment_theme_pct, annot=True, fmt='.1f', cmap='Blues', 
                   cbar_kws={'label': 'Percentage (%)'}, linewidths=0, linecolor='none')
        plt.title('Sentiment Distribution Heatmap by Theme', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Sentiment', fontsize=12, fontweight='bold')
        plt.ylabel('Theme', fontsize=12, fontweight='bold')
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, 'sentiment_heatmap_by_theme.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"   ✅ Generated: {filepath}")
        return True
    except Exception as e:
        print(f"   ❌ Error creating sentiment heatmap by theme: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_sentiment_by_theme_chart(df, output_dir):
    """Create stacked bar chart showing sentiment distribution by theme"""
    try:
        df_filtered = df[(df['theme'].notna()) & (df['sentiment_label'].notna())].copy()
        
        if len(df_filtered) == 0:
            return False
        
        # Create pivot table
        sentiment_theme = pd.crosstab(df_filtered['theme'], df_filtered['sentiment_label'])
        
        # Normalize to percentages
        sentiment_theme_pct = sentiment_theme.div(sentiment_theme.sum(axis=1), axis=0) * 100
        
        # Ensure all sentiment labels exist
        for label in ['Positive', 'Neutral', 'Negative']:
            if label not in sentiment_theme_pct.columns:
                sentiment_theme_pct[label] = 0
        
        # Reorder columns
        sentiment_theme_pct = sentiment_theme_pct[['Positive', 'Neutral', 'Negative']]
        
        # Color palette for sentiments
        sentiment_colors = {
            'Positive': '#90EE90',  # Light green
            'Neutral': '#87CEEB',   # Sky blue
            'Negative': '#FF6347'    # Tomato red
        }
        
        colors = [sentiment_colors.get(sentiment, '#808080') for sentiment in sentiment_theme_pct.columns]
        
        plt.figure(figsize=(14, 8))
        sentiment_theme_pct.plot(kind='bar', stacked=True, color=colors, edgecolor='none', linewidth=0, ax=plt.gca())
        plt.xlabel('Theme', fontsize=12, fontweight='bold')
        plt.ylabel('Percentage (%)', fontsize=12, fontweight='bold')
        plt.title('Sentiment Distribution by Theme', fontsize=16, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Sentiment', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, 'sentiment_by_theme.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"   ✅ Generated: {filepath}")
        return True
    except Exception as e:
        print(f"   ❌ Error creating sentiment by theme chart: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_sentiment_distribution_by_theme(df, output_dir):
    """Create chart showing sentiment distribution (counts) by theme"""
    try:
        df_filtered = df[(df['theme'].notna()) & (df['sentiment_label'].notna())].copy()
        
        if len(df_filtered) == 0:
            return False
        
        sentiment_counts_by_theme = df_filtered.groupby(['theme', 'sentiment_label']).size().unstack(fill_value=0)
        
        # Ensure all sentiment labels exist
        for label in ['Positive', 'Neutral', 'Negative']:
            if label not in sentiment_counts_by_theme.columns:
                sentiment_counts_by_theme[label] = 0
        
        # Reorder columns
        sentiment_counts_by_theme = sentiment_counts_by_theme[['Positive', 'Neutral', 'Negative']]
        
        # Color palette
        sentiment_colors = {
            'Positive': '#90EE90',
            'Neutral': '#87CEEB',
            'Negative': '#FF6347'
        }
        
        colors = [sentiment_colors.get(sentiment, '#808080') for sentiment in sentiment_counts_by_theme.columns]
        
        plt.figure(figsize=(14, 8))
        sentiment_counts_by_theme.plot(kind='bar', color=colors, edgecolor='none', linewidth=0, ax=plt.gca())
        plt.xlabel('Theme', fontsize=12, fontweight='bold')
        plt.ylabel('Count', fontsize=12, fontweight='bold')
        plt.title('Sentiment Count Distribution by Theme', fontsize=16, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Sentiment', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, 'sentiment_distribution_by_theme.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"   ✅ Generated: {filepath}")
        return True
    except Exception as e:
        print(f"   ❌ Error creating sentiment distribution by theme: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# Main Function
# ============================================================================
def main():
    print("="*80)
    print("📊 THEME-BASED SENTIMENT ANALYSIS")
    print("="*80)
    
    # Create output directories
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    print(f"📁 Results will be saved in: {Config.RESULTS_DIR}\n")
    
    # Load themes
    print("📂 Loading themes and sub-themes...")
    theme_dict, theme_df = load_themes()
    if theme_dict is None or theme_df is None:
        print("❌ Failed to load themes. Cannot proceed.")
        return
    
    # Load data with sentiment
    print("\n📂 Loading data with sentiment labels...")
    df = load_data_with_sentiment()
    if df is None:
        return
    
    # Check if we need to classify themes
    if 'theme' not in df.columns or df['theme'].isna().all():
        print("\n🔍 Classifying reviews into themes and sub-themes...")
        theme_subtheme_pairs = []
        for text in tqdm(df['content'] if 'content' in df.columns else df['cleaned_text'], desc="   Classifying"):
            pair = classify_review_to_theme_subtheme(text, theme_df)
            theme_subtheme_pairs.append(pair)
        
        df['theme'] = [pair[0] if pair else None for pair in theme_subtheme_pairs]
        df['subtheme'] = [pair[1] if pair else None for pair in theme_subtheme_pairs]
    
    # Check if we need to get sentiment labels
    if 'sentiment_label' not in df.columns or df['sentiment_label'].isna().all():
        print("\n⚠️  No sentiment labels found in data.")
        print("   Please run prescriptive_analysis.py first to generate sentiment labels.")
        print("   Or the script will use existing sentiment data if available.")
        return
    
    # Filter to reviews with both theme and sentiment
    df_classified = df[(df['theme'].notna()) & (df['sentiment_label'].notna())].copy()
    print(f"\n✅ Found {len(df_classified)} reviews with both theme and sentiment labels")
    
    # Print sentiment distribution
    print("\n📊 Overall Sentiment Distribution:")
    sentiment_dist = df_classified['sentiment_label'].value_counts()
    for sentiment, count in sentiment_dist.items():
        percentage = (count / len(df_classified)) * 100
        print(f"   {sentiment}: {count} ({percentage:.1f}%)")
    
    # Print sentiment by theme
    print("\n📊 Sentiment Distribution by Theme:")
    for theme in sorted(df_classified['theme'].unique()):
        theme_df_subset = df_classified[df_classified['theme'] == theme]
        print(f"\n   {theme}:")
        sentiment_counts = theme_df_subset['sentiment_label'].value_counts()
        for sentiment, count in sentiment_counts.items():
            percentage = (count / len(theme_df_subset)) * 100
            print(f"      {sentiment}: {count} ({percentage:.1f}%)")
    
    # Generate visualizations
    print("\n📊 Generating visualizations...")
    create_sentiment_heatmap_by_theme(df_classified, Config.RESULTS_DIR)
    create_sentiment_by_theme_chart(df_classified, Config.RESULTS_DIR)
    create_sentiment_distribution_by_theme(df_classified, Config.RESULTS_DIR)
    
    print("\n" + "="*80)
    print("✅ THEME-BASED SENTIMENT ANALYSIS COMPLETE!")
    print("="*80)
    print(f"📊 Graphs saved in: {Config.RESULTS_DIR}")

if __name__ == "__main__":
    main()

