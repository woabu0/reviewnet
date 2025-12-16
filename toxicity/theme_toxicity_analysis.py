# Theme-Based Toxicity Analysis Script
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
from collections import defaultdict

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
    THEME_FILE = '../theme/theme.csv'
    OUTPUT_DIR = 'outputs'
    GRAPHS_DIR = os.path.join(OUTPUT_DIR, 'theme_toxicity_graphs')

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
        
        print(f"📊 Found {len(theme_dict)} themes:")
        for theme, subthemes in theme_dict.items():
            print(f"   {theme}: {len(subthemes)} sub-themes")
        
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
# Toxicity Model Loading
# ============================================================================
def get_toxicity_model(model_name):
    """Load toxicity detection model with error handling"""
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

def predict_toxicity_batch(texts, tokenizer, model, device="cpu"):
    """Predict toxicity for a batch of texts"""
    if tokenizer is None or model is None:
        return [0.0] * len(texts)
    
    try:
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256,
            add_special_tokens=True
        )
        
        model = model.cpu()
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Get probabilities using softmax
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            # Get toxic probability (assuming label 1 is toxic)
            toxic_probs = probs[:, 1].cpu().numpy()
        
        return toxic_probs.tolist()
    except Exception as e:
        print(f"Error in batch toxicity prediction: {e}")
        return [0.0] * len(texts)

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
        df = load_data_from_file(file_path)
        if df is not None and len(df) > 0:
            all_data.append(df)
            print(f"   📄 Loaded {os.path.basename(file_path)}: {len(df)} rows")
    
    if not all_data:
        print("❌ No valid data found")
        return None
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"✅ Combined dataset: {len(combined_df)} rows")
    return combined_df

# ============================================================================
# Visualization Functions
# ============================================================================
def create_toxicity_by_theme_chart(df, output_dir):
    """Create chart showing average toxicity by theme"""
    try:
        # Calculate average toxicity per theme
        theme_toxicity = df.groupby('theme')['toxicity_score'].mean().sort_values(ascending=False)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Use different colors for each theme
        color_palette = [
            '#90EE90', '#87CEEB', '#FFD700', '#D3D3D3', '#FFB6C1', 
            '#DDA0DD', '#98FB98', '#F0E68C', '#FFA07A', '#20B2AA',
            '#87CEFA', '#FFE4B5'
        ]
        colors = [color_palette[i % len(color_palette)] for i in range(len(theme_toxicity))]
        
        bars = ax.bar(
            theme_toxicity.index,
            theme_toxicity.values,
            color=colors,
            edgecolor='none',
            linewidth=0,
            alpha=0.8
        )
        
        # Add value labels
        for i, (theme, score) in enumerate(theme_toxicity.items()):
            ax.text(
                i,
                score + theme_toxicity.max() * 0.01,
                f'{score:.3f}',
                ha='center',
                va='bottom',
                fontsize=11,
                fontweight='bold'
            )
        
        ax.set_title('Average Toxicity Score by Theme', fontsize=18, fontweight='bold', pad=20)
        ax.set_xlabel('Theme', fontsize=14, fontweight='bold')
        ax.set_ylabel('Average Toxicity Score', fontsize=14, fontweight='bold')
        ax.set_ylim(0, theme_toxicity.max() * 1.2)
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#CCCCCC')
        ax.spines['bottom'].set_color('#CCCCCC')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, 'toxicity_by_theme.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"📊 Toxicity by theme chart saved: {filepath}")
        return True
    except Exception as e:
        print(f"❌ Error creating toxicity by theme chart: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_toxicity_by_subtheme_chart(df, output_dir):
    """Create chart showing average toxicity by sub-theme"""
    try:
        # Calculate average toxicity per sub-theme
        subtheme_toxicity = df.groupby('subtheme')['toxicity_score'].mean().sort_values(ascending=False)
        
        # Limit to top 15 for readability
        subtheme_toxicity = subtheme_toxicity.head(15)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Use different colors for each sub-theme
        color_palette = [
            '#90EE90', '#87CEEB', '#FFD700', '#D3D3D3', '#FFB6C1', 
            '#DDA0DD', '#98FB98', '#F0E68C', '#FFA07A', '#20B2AA',
            '#87CEFA', '#FFE4B5', '#FF69B4', '#00CED1', '#FFDAB9'
        ]
        colors = [color_palette[i % len(color_palette)] for i in range(len(subtheme_toxicity))]
        
        bars = ax.barh(
            subtheme_toxicity.index,
            subtheme_toxicity.values,
            color=colors,
            edgecolor='none',
            linewidth=0,
            alpha=0.8
        )
        
        # Add value labels
        for i, (subtheme, score) in enumerate(subtheme_toxicity.items()):
            ax.text(
                score + subtheme_toxicity.max() * 0.01,
                i,
                f'{score:.3f}',
                va='center',
                ha='left',
                fontsize=10,
                fontweight='bold'
            )
        
        ax.set_title('Average Toxicity Score by Sub-Theme (Top 15)', fontsize=18, fontweight='bold', pad=20)
        ax.set_xlabel('Average Toxicity Score', fontsize=14, fontweight='bold')
        ax.set_ylabel('Sub-Theme', fontsize=14, fontweight='bold')
        ax.set_xlim(0, subtheme_toxicity.max() * 1.2)
        ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#CCCCCC')
        ax.spines['bottom'].set_color('#CCCCCC')
        
        plt.tight_layout()
        
        # Save
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, 'toxicity_by_subtheme.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"📊 Toxicity by sub-theme chart saved: {filepath}")
        return True
    except Exception as e:
        print(f"❌ Error creating toxicity by sub-theme chart: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_toxicity_heatmap_theme(df, output_dir):
    """Create heatmap showing toxicity distribution by theme only"""
    try:
        # Categorize toxicity scores
        df['toxicity_category'] = pd.cut(
            df['toxicity_score'],
            bins=[0, 0.3, 0.7, 1.0],
            labels=['Low', 'Medium', 'High'],
            include_lowest=True
        )
        
        # Create pivot table: theme vs toxicity category (percentage)
        pivot_table = pd.crosstab(df['theme'], df['toxicity_category'], normalize='index') * 100
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create heatmap with blue gradient
        sns.heatmap(
            pivot_table,
            annot=True,
            fmt='.1f',
            cmap='Blues',
            cbar_kws={'label': 'Percentage (%)'},
            linewidths=0,
            linecolor='none',
            ax=ax,
            vmin=0,
            vmax=100
        )
        
        ax.set_title('Toxicity Distribution within Each Theme', 
                    fontsize=18, fontweight='bold', pad=20)
        ax.set_xlabel('Toxicity Level', fontsize=14, fontweight='bold')
        ax.set_ylabel('Theme', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, 'toxicity_heatmap_theme.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"📊 Theme-only toxicity heatmap saved: {filepath}")
        return True
    except Exception as e:
        print(f"❌ Error creating theme-only heatmap: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_toxicity_distribution_by_theme(df, output_dir):
    """Create chart showing toxicity distribution (Low/Medium/High) by theme"""
    try:
        # Categorize toxicity scores
        df['toxicity_category'] = pd.cut(
            df['toxicity_score'],
            bins=[0, 0.3, 0.7, 1.0],
            labels=['Low', 'Medium', 'High'],
            include_lowest=True
        )
        
        # Create pivot table
        pivot_table = pd.crosstab(df['theme'], df['toxicity_category'], normalize='index') * 100
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Create stacked bar chart
        pivot_table.plot(kind='bar', stacked=True, ax=ax,
                        color={'Low': '#90EE90', 'Medium': '#87CEEB', 'High': '#FFB6C1'},
                        edgecolor='none', linewidth=0, alpha=0.8)
        
        ax.set_title('Toxicity Distribution by Theme', fontsize=18, fontweight='bold', pad=20)
        ax.set_xlabel('Theme', fontsize=14, fontweight='bold')
        ax.set_ylabel('Percentage (%)', fontsize=14, fontweight='bold')
        ax.legend(title='Toxicity Level', fontsize=12)
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#CCCCCC')
        ax.spines['bottom'].set_color('#CCCCCC')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, 'toxicity_distribution_by_theme.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"📊 Toxicity distribution by theme chart saved: {filepath}")
        return True
    except Exception as e:
        print(f"❌ Error creating toxicity distribution by theme chart: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# Main Function
# ============================================================================
def main():
    print("="*80)
    print("🔍 THEME-BASED TOXICITY ANALYSIS")
    print("="*80)
    
    # Create output directory
    os.makedirs(Config.GRAPHS_DIR, exist_ok=True)
    print(f"📁 Graphs will be saved in: {Config.GRAPHS_DIR}\n")
    
    # Load themes
    print("📂 Loading themes and sub-themes...")
    theme_dict, theme_df = load_themes()
    if theme_dict is None or theme_df is None:
        print("❌ Failed to load themes. Cannot proceed.")
        return
    
    # Load data
    print("\n📂 Loading data...")
    df = load_all_data()
    if df is None:
        return
    
    # Classify reviews into themes and sub-themes
    print("\n🔍 Classifying reviews into themes and sub-themes...")
    theme_subtheme_pairs = []
    for text in tqdm(df['cleaned_text'], desc="   Classifying"):
        pair = classify_review_to_theme_subtheme(text, theme_df)
        theme_subtheme_pairs.append(pair)
    
    # Add theme and subtheme columns
    df['theme'] = [pair[0] if pair else None for pair in theme_subtheme_pairs]
    df['subtheme'] = [pair[1] if pair else None for pair in theme_subtheme_pairs]
    
    # Filter to reviews with classified themes
    df_classified = df[df['theme'].notna()].copy()
    print(f"✅ Classified {len(df_classified)} reviews ({len(df_classified)/len(df)*100:.1f}% of total)")
    
    # Print theme distribution
    print("\n📊 Theme Distribution:")
    theme_counts = df_classified['theme'].value_counts()
    for theme, count in theme_counts.items():
        percentage = (count / len(df_classified)) * 100
        print(f"   {theme}: {count} ({percentage:.1f}%)")
    
    # Load toxicity model
    print("\n🤖 Loading toxicity detection model...")
    toxicity_model_name = "martin-ha/toxic-comment-model"
    print(f"   📥 Loading model: {toxicity_model_name}")
    tokenizer, model = get_toxicity_model(toxicity_model_name)
    
    if tokenizer is None or model is None:
        print("   ❌ Failed to load toxicity model. Trying alternative model...")
        toxicity_model_name = "unitary/toxic-bert"
        print(f"   📥 Trying alternative model: {toxicity_model_name}")
        tokenizer, model = get_toxicity_model(toxicity_model_name)
        
        if tokenizer is None or model is None:
            print("   ❌ Failed to load toxicity models. Cannot proceed.")
            return
    
    # Predict toxicity
    print("\n🔮 Predicting toxicity scores...")
    texts = df_classified['cleaned_text'].tolist()
    toxicity_scores = []
    
    device = "cpu"
    batch_size = 8
    
    for i in tqdm(range(0, len(texts), batch_size), desc="   Processing"):
        try:
            batch_texts = texts[i:i+batch_size]
            batch_scores = predict_toxicity_batch(batch_texts, tokenizer, model, device)
            toxicity_scores.extend(batch_scores)
            gc.collect()
        except Exception as e:
            print(f"\n      ⚠️  Error in batch {i//batch_size + 1}: {e}")
            toxicity_scores.extend([0.0] * len(batch_texts))
            gc.collect()
            continue
    
    # Ensure we have scores for all texts
    while len(toxicity_scores) < len(texts):
        toxicity_scores.append(0.0)
    
    df_classified['toxicity_score'] = toxicity_scores
    
    # Print statistics
    print("\n📊 Toxicity Statistics by Theme:")
    for theme in df_classified['theme'].unique():
        theme_df = df_classified[df_classified['theme'] == theme]
        avg_toxicity = theme_df['toxicity_score'].mean()
        print(f"   {theme}: {avg_toxicity:.4f}")
    
    print("\n📊 Toxicity Statistics by Sub-Theme (Top 10):")
    subtheme_stats = df_classified.groupby('subtheme')['toxicity_score'].agg(['mean', 'count']).sort_values('mean', ascending=False)
    for subtheme, row in subtheme_stats.head(10).iterrows():
        print(f"   {subtheme}: {row['mean']:.4f} (n={int(row['count'])})")
    
    # Create visualizations
    print("\n" + "="*80)
    print("📊 GENERATING VISUALIZATIONS")
    print("="*80)
    
    create_toxicity_by_theme_chart(df_classified, Config.GRAPHS_DIR)
    create_toxicity_by_subtheme_chart(df_classified, Config.GRAPHS_DIR)
    create_toxicity_heatmap_theme(df_classified, Config.GRAPHS_DIR)
    create_toxicity_distribution_by_theme(df_classified, Config.GRAPHS_DIR)
    
    # Cleanup
    del model
    del tokenizer
    gc.collect()
    time.sleep(1.0)
    
    print("\n" + "="*80)
    print("✅ THEME-BASED TOXICITY ANALYSIS COMPLETE!")
    print("="*80)
    print(f"📁 All graphs saved in: {Config.GRAPHS_DIR}")

if __name__ == "__main__":
    main()

