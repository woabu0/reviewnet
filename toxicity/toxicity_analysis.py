# Toxicity Analysis Script
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
    GRAPHS_DIR = os.path.join(OUTPUT_DIR, 'toxicity_graphs')

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
# Visualization Functions (Minimalistic)
# ============================================================================
def create_toxicity_distribution_chart(df, output_dir):
    """Create minimalistic toxicity distribution chart"""
    try:
        # Categorize toxicity scores
        df['toxicity_category'] = pd.cut(
            df['toxicity_score'],
            bins=[0, 0.3, 0.7, 1.0],
            labels=['Low', 'Medium', 'High'],
            include_lowest=True
        )
        
        # Count by category
        category_counts = df['toxicity_category'].value_counts().sort_index()
        
        # Create figure with minimalistic style
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Use colors like the sentiment distribution chart (2nd image)
        # Light green for Low, light blue for Medium, light red/pink for High
        color_map = {
            'Low': '#90EE90',      # Light green (like positive sentiment)
            'Medium': '#87CEEB',    # Light blue (like neutral sentiment)
            'High': '#FFB6C1'       # Light pink/red (like negative sentiment)
        }
        
        # Map categories to colors
        colors = [color_map.get(cat, '#E8E8E8') for cat in category_counts.index.astype(str)]
        
        bars = ax.bar(
            category_counts.index.astype(str),
            category_counts.values,
            color=colors,
            edgecolor='none',
            linewidth=0,
            alpha=0.9
        )
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{int(height)}',
                ha='center',
                va='bottom',
                fontsize=14,
                fontweight='bold'
            )
        
        # Minimalistic styling
        ax.set_title('Toxicity Distribution', fontsize=18, fontweight='bold', pad=20)
        ax.set_xlabel('Toxicity Level', fontsize=14, fontweight='bold')
        ax.set_ylabel('Number of Reviews', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#CCCCCC')
        ax.spines['bottom'].set_color('#CCCCCC')
        
        plt.tight_layout()
        
        # Save
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, 'toxicity_distribution.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"📊 Toxicity distribution chart saved: {filepath}")
        return True
    except Exception as e:
        print(f"❌ Error creating toxicity distribution chart: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_toxicity_by_app_chart(df, output_dir):
    """Create minimalistic toxicity distribution by app chart"""
    try:
        # Calculate average toxicity per app
        app_toxicity = df.groupby('app')['toxicity_score'].mean().sort_values(ascending=False)
        
        # Create figure with minimalistic style
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Use different color for each app - expanded color palette
        # Light, pastel colors that are distinct from each other
        color_palette = [
            '#90EE90',  # Light green
            '#87CEEB',  # Light blue
            '#FFD700',  # Light yellow/gold
            '#D3D3D3',  # Light gray
            '#FFB6C1',  # Light pink
            '#DDA0DD',  # Light plum
            '#98FB98',  # Light green (pale)
            '#F0E68C',  # Light khaki
            '#FFA07A',  # Light salmon
            '#20B2AA',  # Light sea green
            '#87CEFA',  # Light sky blue
            '#FFE4B5',  # Moccasin
        ]
        
        # Assign unique color to each app
        colors = [color_palette[i % len(color_palette)] for i in range(len(app_toxicity))]
        
        bars = ax.bar(
            app_toxicity.index,
            app_toxicity.values,
            color=colors,
            edgecolor='none',
            linewidth=0,
            alpha=0.8
        )
        
        # Add value labels on top of bars
        for i, (app, score) in enumerate(app_toxicity.items()):
            ax.text(
                i,
                score + app_toxicity.max() * 0.01,
                f'{score:.3f}',
                ha='center',
                va='bottom',
                fontsize=11,
                fontweight='bold'
            )
        
        # Minimalistic styling
        ax.set_title('Average Toxicity Score by App', fontsize=18, fontweight='bold', pad=20)
        ax.set_xlabel('App', fontsize=14, fontweight='bold')
        ax.set_ylabel('Average Toxicity Score', fontsize=14, fontweight='bold')
        ax.set_ylim(0, app_toxicity.max() * 1.2)
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#CCCCCC')
        ax.spines['bottom'].set_color('#CCCCCC')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, 'toxicity_by_app.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"📊 Toxicity by app chart saved: {filepath}")
        return True
    except Exception as e:
        print(f"❌ Error creating toxicity by app chart: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_toxicity_heatmap(df, output_dir):
    """Create minimalistic toxicity heatmap by app and category"""
    try:
        # Create pivot table: app vs toxicity category
        df['toxicity_category'] = pd.cut(
            df['toxicity_score'],
            bins=[0, 0.3, 0.7, 1.0],
            labels=['Low', 'Medium', 'High'],
            include_lowest=True
        )
        
        pivot_table = pd.crosstab(df['app'], df['toxicity_category'], normalize='index') * 100
        
        # Create figure with minimalistic style
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Use blue gradient colormap like the 3rd image (Sentiment Distribution within Each Theme)
        # Light blue to dark blue gradient
        sns.heatmap(
            pivot_table,
            annot=True,
            fmt='.1f',
            cmap='Blues',  # Blue gradient from light to dark
            cbar_kws={'label': 'Percentage (%)'},
            linewidths=0,
            linecolor='none',
            ax=ax,
            vmin=0,
            vmax=100
        )
        
        # Minimalistic styling
        ax.set_title('Toxicity Distribution by App (Percentage)', fontsize=18, fontweight='bold', pad=20)
        ax.set_xlabel('Toxicity Level', fontsize=14, fontweight='bold')
        ax.set_ylabel('App', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, 'toxicity_heatmap.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"📊 Toxicity heatmap saved: {filepath}")
        return True
    except Exception as e:
        print(f"❌ Error creating toxicity heatmap: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# Main Function
# ============================================================================
def main():
    print("="*80)
    print("🔍 TOXICITY ANALYSIS")
    print("="*80)
    
    # Create output directory
    os.makedirs(Config.GRAPHS_DIR, exist_ok=True)
    print(f"📁 Graphs will be saved in: {Config.GRAPHS_DIR}\n")
    
    # Load data
    print("📂 Loading data...")
    df = load_all_data()
    if df is None:
        return
    
    # Load toxicity model
    print("\n🤖 Loading toxicity detection model...")
    # Using martin-ha/toxic-comment-model which is a good toxicity classifier
    toxicity_model_name = "martin-ha/toxic-comment-model"
    print(f"   📥 Loading model: {toxicity_model_name}")
    tokenizer, model = get_toxicity_model(toxicity_model_name)
    
    if tokenizer is None or model is None:
        print("   ❌ Failed to load toxicity model. Trying alternative model...")
        # Try alternative model
        toxicity_model_name = "unitary/toxic-bert"
        print(f"   📥 Trying alternative model: {toxicity_model_name}")
        tokenizer, model = get_toxicity_model(toxicity_model_name)
        
        if tokenizer is None or model is None:
            print("   ❌ Failed to load toxicity models. Cannot proceed.")
            return
    
    # Predict toxicity
    print("\n🔮 Predicting toxicity scores...")
    texts = df['cleaned_text'].tolist()
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
    
    df['toxicity_score'] = toxicity_scores
    
    # Print statistics
    print("\n📊 Toxicity Statistics:")
    print(f"   Average toxicity score: {df['toxicity_score'].mean():.4f}")
    print(f"   Median toxicity score: {df['toxicity_score'].median():.4f}")
    print(f"   Max toxicity score: {df['toxicity_score'].max():.4f}")
    print(f"   Min toxicity score: {df['toxicity_score'].min():.4f}")
    
    # Categorize
    df['toxicity_category'] = pd.cut(
        df['toxicity_score'],
        bins=[0, 0.3, 0.7, 1.0],
        labels=['Low', 'Medium', 'High'],
        include_lowest=True
    )
    
    category_counts = df['toxicity_category'].value_counts().sort_index()
    print("\n📊 Toxicity Distribution:")
    for category, count in category_counts.items():
        percentage = (count / len(df)) * 100
        print(f"   {category}: {count} ({percentage:.1f}%)")
    
    # Create visualizations
    print("\n" + "="*80)
    print("📊 GENERATING VISUALIZATIONS")
    print("="*80)
    
    create_toxicity_distribution_chart(df, Config.GRAPHS_DIR)
    create_toxicity_by_app_chart(df, Config.GRAPHS_DIR)
    create_toxicity_heatmap(df, Config.GRAPHS_DIR)
    
    # Cleanup
    del model
    del tokenizer
    gc.collect()
    time.sleep(1.0)
    
    print("\n" + "="*80)
    print("✅ TOXICITY ANALYSIS COMPLETE!")
    print("="*80)
    print(f"📁 All graphs saved in: {Config.GRAPHS_DIR}")

if __name__ == "__main__":
    main()

