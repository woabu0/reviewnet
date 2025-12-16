# RoBERTa-based Sentiment Analysis with Visualization
import os
import sys
import pandas as pd
import numpy as np
import glob
import re
import warnings
import torch
import gc
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Disable multiprocessing to avoid segmentation faults
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("❌ Transformers library not available. Please install: pip install transformers")

warnings.filterwarnings("ignore")

# ============================================================================
# Configuration
# ============================================================================
class Config:
    DATA_DIR = '../dataset'
    OUTPUT_DIR = 'outputs'
    NEGATIVE_DATA_DIR = '../negative_dataset'

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
    return text if len(text) > 0 else " "  # Return space if empty to keep the row

# ============================================================================
# RoBERTa Model Loading
# ============================================================================
def get_sentiment_model(model_name):
    """Load sentiment model with error handling"""
    if not TRANSFORMERS_AVAILABLE:
        return None, None
    try:
        gc.collect()
        time.sleep(0.2)
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            local_files_only=False
        )
        
        time.sleep(0.1)
        
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                dtype=torch.float32,
                low_cpu_mem_usage=True,
                local_files_only=False
            )
        except (TypeError, ValueError):
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                low_cpu_mem_usage=True,
                local_files_only=False
            )
        
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        
        model = model.cpu()
        return tokenizer, model
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return None, None

def predict_sentiment_batch(texts, tokenizer, model, model_name, device="cpu", batch_size=32):
    """Predict sentiment for a batch of texts using RoBERTa"""
    if not TRANSFORMERS_AVAILABLE or tokenizer is None or model is None:
        return [-1] * len(texts)
    
    try:
        # Map model outputs to sentiment labels
        model_label_map = {
            "cardiffnlp/twitter-roberta-base-sentiment-latest": {0: "negative", 1: "neutral", 2: "positive"},
        }
        
        all_sentiments = []
        
        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="   Processing batches", leave=False):
            batch_texts = texts[i:i + batch_size]
            
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=256,
                add_special_tokens=True
            )
            
            # Keep everything on CPU
            model = model.cpu()
            
            with torch.no_grad():
                logits = model(**inputs).logits
            
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
            
            # Map to sentiment labels
            label_map = model_label_map.get(model_name, {0: "negative", 1: "positive"})
            batch_sentiments = []
            for pred in predictions:
                if pred in label_map:
                    sentiment = label_map[pred]
                    # Convert to numeric: negative=0, neutral=1, positive=2
                    if sentiment == "negative":
                        batch_sentiments.append(0)
                    elif sentiment == "neutral":
                        batch_sentiments.append(1)
                    else:
                        batch_sentiments.append(2)
                else:
                    batch_sentiments.append(-1)
            
            all_sentiments.extend(batch_sentiments)
            
            # Clear cache
            gc.collect()
        
        return all_sentiments
    except Exception as e:
        print(f"Error in batch sentiment prediction for {model_name}: {e}")
        return [-1] * len(texts)

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
            return None, None
            
        df = df.dropna(subset=["content"])
        df['content'] = df['content'].astype(str)
        df['cleaned_text'] = df['content'].apply(clean_text_advanced)
        
        # Extract app name from filename (remove .csv extension)
        app_name = os.path.basename(file_path).replace('.csv', '')
        df['app'] = app_name
        
        return df, app_name
    except Exception as e:
        print(f"   ❌ Error loading {os.path.basename(file_path)}: {e}")
        return None, None

# ============================================================================
# Visualization Functions
# ============================================================================
def save_negative_reviews(df_dict, output_dir):
    """Save negative reviews to CSV files"""
    os.makedirs(output_dir, exist_ok=True)
    
    total_negative = 0
    total_processed = 0
    
    for filename, df in df_dict.items():
        # Filter negative reviews (sentiment == 0)
        negative_df = df[df['sentiment'] == 0].copy()
        
        # Remove temporary columns
        columns_to_drop = ['cleaned_text', 'app', 'sentiment', 'sentiment_label']
        for col in columns_to_drop:
            if col in negative_df.columns:
                negative_df = negative_df.drop(columns=[col])
        
        # Save negative reviews to output directory
        output_path = os.path.join(output_dir, filename)
        negative_df.to_csv(output_path, index=False)
        
        num_negative = len(negative_df)
        total_negative += num_negative
        total_processed += len(df)
        
        print(f"   💾 Saved {num_negative} negative reviews to {output_path}")
    
    print(f"\n   📊 Total negative reviews saved: {total_negative} ({total_negative/total_processed*100:.1f}%)")
    return total_negative, total_processed

def create_sentiment_distribution_chart(df, output_dir):
    """Create overall sentiment distribution bar chart with individual bars"""
    # Count sentiments by numeric value first to ensure correct order
    sentiment_counts_numeric = df['sentiment'].value_counts().sort_index()
    
    # Map numeric to labels in correct order: Negative (0), Neutral (1), Positive (2)
    sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    sentiment_order = ['Negative', 'Neutral', 'Positive']
    
    # Build ordered lists
    sentiment_labels = []
    counts = []
    for sentiment_label in sentiment_order:
        # Find the numeric value for this label
        numeric_val = [k for k, v in sentiment_map.items() if v == sentiment_label][0]
        if numeric_val in sentiment_counts_numeric.index:
            sentiment_labels.append(sentiment_label)
            counts.append(sentiment_counts_numeric[numeric_val])
    
    # Calculate percentages
    total = len(df)
    percentages = [(count / total) * 100 for count in counts]
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Define colors for each sentiment
    colors = {'Negative': '#e74c3c', 'Neutral': '#95a5a6', 'Positive': '#2ecc71'}
    bar_colors = [colors.get(label, '#3498db') for label in sentiment_labels]
    
    # Create individual bars (not stacked)
    bars = plt.bar(sentiment_labels, counts, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for i, (bar, count, pct) in enumerate(zip(bars, counts, percentages)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Customize plot
    plt.title('Sentiment Distribution of Food-Delivery Reviews', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Sentiment', fontsize=12, fontweight='bold')
    plt.ylabel('Count', fontsize=12, fontweight='bold')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    
    # Set y-axis to show appropriate range
    if counts:
        max_count = max(counts)
        plt.ylim(0, max_count * 1.15)
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'sentiment_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved sentiment distribution chart to {output_path}")
    return output_path

def create_sentiment_by_app_chart(df, output_dir):
    """Create sentiment distribution by app grouped bar chart"""
    # Group by app and numeric sentiment first to ensure correct order
    app_sentiment_counts = df.groupby(['app', 'sentiment']).size().unstack(fill_value=0)
    
    # Map numeric to labels
    sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    
    # Rename columns using sentiment map
    app_sentiment_counts.columns = [sentiment_map.get(col, 'Unknown') for col in app_sentiment_counts.columns]
    
    # Reorder columns to match: Negative, Neutral, Positive
    column_order = ['Negative', 'Neutral', 'Positive']
    app_sentiment_counts = app_sentiment_counts.reindex(columns=[col for col in column_order if col in app_sentiment_counts.columns])
    
    # Sort apps by total reviews (descending)
    app_sentiment_counts['Total'] = app_sentiment_counts.sum(axis=1)
    app_sentiment_counts = app_sentiment_counts.sort_values('Total', ascending=False)
    app_sentiment_counts = app_sentiment_counts.drop(columns=['Total'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Define colors
    colors = {'Negative': '#e74c3c', 'Neutral': '#95a5a6', 'Positive': '#2ecc71'}
    bar_colors = [colors.get(col, '#3498db') for col in app_sentiment_counts.columns]
    
    # Create grouped bar chart
    x = np.arange(len(app_sentiment_counts.index))
    width = 0.25
    multiplier = 0
    
    for sentiment in app_sentiment_counts.columns:
        offset = width * multiplier
        bars = ax.bar(x + offset, app_sentiment_counts[sentiment], width, 
                      label=sentiment, color=colors.get(sentiment, '#3498db'), 
                      alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        multiplier += 1
    
    # Customize plot
    ax.set_title('Sentiment distribution by App', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('App', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_xticks(x + width, app_sentiment_counts.index)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Rotate x-axis labels if needed
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'sentiment_by_app.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved sentiment by app chart to {output_path}")
    return output_path

# ============================================================================
# Main Processing Function
# ============================================================================
def main():
    print("="*80)
    print("🎯 SENTIMENT ANALYSIS WITH VISUALIZATION")
    print("="*80)
    
    if not TRANSFORMERS_AVAILABLE:
        print("❌ Transformers library not available.")
        return
    
    # Force CPU
    device = "cpu"
    print(f"Using device: {device}\n")
    
    # Create output directories
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(Config.NEGATIVE_DATA_DIR, exist_ok=True)
    print(f"📁 Graphs will be saved in: {Config.OUTPUT_DIR}")
    print(f"📁 Negative reviews will be saved in: {Config.NEGATIVE_DATA_DIR}\n")
    
    # Load RoBERTa model
    roberta_model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    print("🔍 Loading RoBERTa model for sentiment analysis...")
    print(f"   📥 Loading {roberta_model_name}...")
    roberta_tokenizer, roberta_model = get_sentiment_model(roberta_model_name)
    
    if roberta_tokenizer is None or roberta_model is None:
        print("❌ Failed to load RoBERTa model. Please check your Hugging Face authentication.")
        print("   Run: huggingface-cli login")
        return
    
    print("✅ RoBERTa model loaded successfully\n")
    
    # Find all CSV files in data directory
    data_dir = Config.DATA_DIR
    all_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    if not all_files:
        print(f"❌ No CSV files found in {data_dir}")
        return
    
    print(f"📂 Found {len(all_files)} CSV files")
    print()
    
    all_dataframes = []
    negative_reviews_dict = {}  # Store dataframes for negative review extraction
    
    # Process each file
    for file_path in all_files:
        filename = os.path.basename(file_path)
        
        # Skip .DS_Store and other non-data files
        if filename.startswith('.'):
            continue
        
        print(f"📄 Processing {filename}...")
        
        # Load data
        df, app_name = load_data_from_file(file_path)
        if df is None or len(df) == 0:
            print(f"   ⚠️  No valid data found in {filename}")
            print()
            continue
        
        print(f"   📊 Loaded {len(df)} reviews")
        
        # Get cleaned texts
        texts = df['cleaned_text'].tolist()
        
        # Predict sentiment using RoBERTa
        print(f"   🔍 Analyzing sentiment with RoBERTa...")
        sentiments = predict_sentiment_batch(
            texts, 
            roberta_tokenizer, 
            roberta_model, 
            roberta_model_name,
            device=device,
            batch_size=32
        )
        
        # Add sentiment columns
        df['sentiment'] = sentiments
        df['sentiment_label'] = df['sentiment'].map({0: 'Negative', 1: 'Neutral', 2: 'Positive'})
        
        all_dataframes.append(df)
        negative_reviews_dict[filename] = df  # Store for negative review extraction
        
        # Show sentiment distribution for this file
        sentiment_counts = df['sentiment_label'].value_counts()
        print(f"   📈 Sentiment distribution:")
        for sentiment, count in sentiment_counts.items():
            pct = (count / len(df)) * 100
            print(f"      {sentiment}: {count} ({pct:.1f}%)")
        print()
    
    if not all_dataframes:
        print("❌ No data processed")
        return
    
    # Combine all dataframes
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    print(f"✅ Combined dataset: {len(combined_df)} reviews from {len(all_dataframes)} apps\n")
    
    # Save negative reviews
    print("💾 Saving negative reviews...")
    print()
    total_negative, total_processed = save_negative_reviews(negative_reviews_dict, Config.NEGATIVE_DATA_DIR)
    print()
    
    # Create visualizations
    print("📊 Creating visualizations...")
    print()
    
    # 1. Overall sentiment distribution
    print("   📈 Creating overall sentiment distribution chart...")
    create_sentiment_distribution_chart(combined_df, Config.OUTPUT_DIR)
    print()
    
    # 2. Sentiment distribution by app
    print("   📈 Creating sentiment distribution by app chart...")
    create_sentiment_by_app_chart(combined_df, Config.OUTPUT_DIR)
    print()
    
    # Summary statistics
    print("="*80)
    print("📊 SUMMARY STATISTICS")
    print("="*80)
    print(f"Total reviews analyzed: {len(combined_df)}")
    print(f"Number of apps: {len(combined_df['app'].unique())}")
    print()
    print("Overall sentiment distribution:")
    overall_counts = combined_df['sentiment_label'].value_counts().sort_index()
    for sentiment, count in overall_counts.items():
        pct = (count / len(combined_df)) * 100
        print(f"  {sentiment}: {count} ({pct:.1f}%)")
    print()
    print("Sentiment by app:")
    app_sentiment = combined_df.groupby(['app', 'sentiment_label']).size().unstack(fill_value=0)
    for app in app_sentiment.index:
        print(f"  {app}:")
        for sentiment in ['Negative', 'Neutral', 'Positive']:
            if sentiment in app_sentiment.columns:
                count = app_sentiment.loc[app, sentiment]
                total = app_sentiment.loc[app].sum()
                pct = (count / total) * 100 if total > 0 else 0
                print(f"    {sentiment}: {count} ({pct:.1f}%)")
    print()
    print(f"✅ Graphs saved to: {Config.OUTPUT_DIR}")
    print(f"✅ Negative reviews saved to: {Config.NEGATIVE_DATA_DIR}")
    print("="*80)
    print("✅ ANALYSIS COMPLETE!")

if __name__ == "__main__":
    main()

