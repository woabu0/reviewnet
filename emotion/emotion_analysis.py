# Emotion Analysis Script
# Performs emotion analysis on user reviews using transformer models
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
from collections import Counter

# Disable multiprocessing to avoid segmentation faults
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
    PIPELINE_AVAILABLE = True
    try:
        from transformers.pipelines import pipeline
    except:
        try:
            from transformers import pipeline
        except:
            PIPELINE_AVAILABLE = False
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    PIPELINE_AVAILABLE = False
    print(f"❌ Transformers library not available: {e}")
    print("   Please install: pip install transformers")

warnings.filterwarnings("ignore")

# ============================================================================
# Configuration
# ============================================================================
class Config:
    DATA_DIR = 'data'
    OUTPUT_DIR = 'outputs'
    RESULTS_DIR = os.path.join(OUTPUT_DIR, 'emotion_analysis')
    GRAPHS_DIR = os.path.join(RESULTS_DIR, 'graphs')
    
    # Emotion model options
    # Primary: j-hartmann/emotion-english-distilroberta-base (6 emotions: joy, sadness, anger, fear, surprise, love)
    EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"
    # Alternative: cardiffnlp/twitter-roberta-base-emotion (4 emotions: anger, joy, optimism, sadness)
    # EMOTION_MODEL = "cardiffnlp/twitter-roberta-base-emotion"
    
    # Use pipeline for easier inference
    USE_PIPELINE = True
    MAX_LENGTH = 256
    BATCH_SIZE = 8

# ============================================================================
# Text Preprocessing
# ============================================================================
def clean_text_advanced(text):
    """Advanced text cleaning"""
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
# Model Loading
# ============================================================================
def get_emotion_model(model_name):
    """Load emotion detection model"""
    if not TRANSFORMERS_AVAILABLE:
        return None, None
    
    try:
        gc.collect()
        time.sleep(0.2)
        
        print(f"   📥 Loading tokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=False
        )
        
        time.sleep(0.1)
        
        print(f"   📥 Loading model: {model_name}")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            local_files_only=False
        )
        
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        
        model = model.cpu()
        print(f"   ✅ Model loaded successfully")
        return tokenizer, model
        
    except Exception as e:
        print(f"   ❌ Error loading model {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def get_emotion_pipeline(model_name):
    """Get emotion analysis pipeline"""
    if not PIPELINE_AVAILABLE:
        return None
    try:
        from transformers.pipelines import pipeline as create_pipeline
        pipe = create_pipeline(
            "text-classification",
            model=model_name,
            tokenizer=model_name,
            device=-1,  # CPU
            return_all_scores=True  # Get all emotion scores
        )
        return pipe
    except Exception as e:
        try:
            from transformers import pipeline as create_pipeline
            pipe = create_pipeline(
                "text-classification",
                model=model_name,
                tokenizer=model_name,
                device=-1,  # CPU
                return_all_scores=True  # Get all emotion scores
            )
            return pipe
        except Exception as e2:
            print(f"   ❌ Error creating pipeline: {e2}")
            return None

# ============================================================================
# Emotion Prediction
# ============================================================================
def predict_emotion_batch(texts, tokenizer, model, batch_size=8):
    """Predict emotions for a batch of texts"""
    if tokenizer is None or model is None:
        return []
    
    all_emotions = []
    
    try:
        for i in tqdm(range(0, len(texts), batch_size), desc="   Processing batches", leave=False):
            batch_texts = texts[i:i + batch_size]
            
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=Config.MAX_LENGTH,
                add_special_tokens=True
            )
            
            model = model.cpu()
            
            with torch.no_grad():
                outputs = model(**inputs)
                # Get probabilities using softmax
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                # Get predicted class
                predictions = torch.argmax(probs, dim=-1).cpu().numpy()
                # Get probabilities
                probabilities = probs.cpu().numpy()
            
            # Get emotion labels (model-specific)
            emotion_labels = model.config.id2label
            
            batch_emotions = []
            for pred, prob in zip(predictions, probabilities):
                emotion_label = emotion_labels[pred]
                confidence = prob[pred]
                batch_emotions.append({
                    'emotion': emotion_label,
                    'confidence': float(confidence),
                    'all_scores': {emotion_labels[i]: float(prob[i]) for i in range(len(prob))}
                })
            
            all_emotions.extend(batch_emotions)
            gc.collect()
        
        return all_emotions
        
    except Exception as e:
        print(f"Error in batch emotion prediction: {e}")
        return []

def predict_emotion_with_pipeline(texts, pipe, batch_size=8):
    """Predict emotions using pipeline"""
    if pipe is None:
        return []
    
    all_emotions = []
    
    try:
        for i in tqdm(range(0, len(texts), batch_size), desc="   Processing batches", leave=False):
            batch_texts = texts[i:i + batch_size]
            results = pipe(batch_texts)
            
            batch_emotions = []
            for result in results:
                if isinstance(result, list):
                    # Get the emotion with highest score
                    top_emotion = max(result, key=lambda x: x['score'])
                    emotion_label = top_emotion['label']
                    confidence = top_emotion['score']
                    # Get all scores
                    all_scores = {item['label']: item['score'] for item in result}
                else:
                    emotion_label = result['label']
                    confidence = result['score']
                    all_scores = {emotion_label: confidence}
                
                batch_emotions.append({
                    'emotion': emotion_label,
                    'confidence': float(confidence),
                    'all_scores': all_scores
                })
            
            all_emotions.extend(batch_emotions)
            gc.collect()
        
        return all_emotions
        
    except Exception as e:
        print(f"Error in pipeline emotion prediction: {e}")
        return []

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
# Visualization Functions
# ============================================================================
def create_emotion_distribution_chart(df, output_dir):
    """Create chart showing emotion distribution"""
    try:
        emotion_counts = df['emotion'].value_counts()
        
        # Color palette for emotions
        emotion_colors = {
            'joy': '#90EE90',      # Light green
            'love': '#FFB6C1',     # Light pink
            'surprise': '#FFD700', # Gold
            'anger': '#FF6347',    # Tomato red
            'sadness': '#87CEEB',  # Sky blue
            'fear': '#DDA0DD',     # Plum
            'optimism': '#98FB98', # Pale green
            'disgust': '#FFA07A',  # Light salmon
            'neutral': '#D3D3D3'   # Light gray
        }
        
        colors = [emotion_colors.get(emotion.lower(), '#808080') for emotion in emotion_counts.index]
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(range(len(emotion_counts)), emotion_counts.values, color=colors, edgecolor='none', linewidth=0)
        plt.xlabel('Emotion', fontsize=12, fontweight='bold')
        plt.ylabel('Count', fontsize=12, fontweight='bold')
        plt.title('Emotion Distribution in User Reviews', fontsize=16, fontweight='bold', pad=20)
        plt.xticks(range(len(emotion_counts)), emotion_counts.index, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for i, (emotion, count) in enumerate(emotion_counts.items()):
            plt.text(i, count + max(emotion_counts.values) * 0.01, str(count), 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, 'emotion_distribution.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"   ✅ Generated: {filepath}")
        return True
    except Exception as e:
        print(f"   ❌ Error creating emotion distribution chart: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_emotion_by_app_chart(df, output_dir):
    """Create chart showing emotion distribution by app"""
    try:
        # Create pivot table
        emotion_app = pd.crosstab(df['app'], df['emotion'])
        
        # Normalize to percentages
        emotion_app_pct = emotion_app.div(emotion_app.sum(axis=1), axis=0) * 100
        
        # Color palette
        emotion_colors = {
            'joy': '#90EE90',
            'love': '#FFB6C1',
            'surprise': '#FFD700',
            'anger': '#FF6347',
            'sadness': '#87CEEB',
            'fear': '#DDA0DD',
            'optimism': '#98FB98',
            'disgust': '#FFA07A',
            'neutral': '#D3D3D3'
        }
        
        colors = [emotion_colors.get(emotion.lower(), '#808080') for emotion in emotion_app_pct.columns]
        
        plt.figure(figsize=(14, 8))
        emotion_app_pct.plot(kind='bar', stacked=True, color=colors, edgecolor='none', linewidth=0, ax=plt.gca())
        plt.xlabel('App', fontsize=12, fontweight='bold')
        plt.ylabel('Percentage (%)', fontsize=12, fontweight='bold')
        plt.title('Emotion Distribution by App', fontsize=16, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Emotion', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, 'emotion_by_app.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"   ✅ Generated: {filepath}")
        return True
    except Exception as e:
        print(f"   ❌ Error creating emotion by app chart: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_emotion_heatmap(df, output_dir):
    """Create heatmap showing emotion distribution by app"""
    try:
        # Create pivot table
        emotion_app = pd.crosstab(df['app'], df['emotion'])
        
        # Normalize to percentages
        emotion_app_pct = emotion_app.div(emotion_app.sum(axis=1), axis=0) * 100
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(emotion_app_pct, annot=True, fmt='.1f', cmap='Blues', 
                   cbar_kws={'label': 'Percentage (%)'}, linewidths=0, linecolor='none')
        plt.title('Emotion Distribution Heatmap by App', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Emotion', fontsize=12, fontweight='bold')
        plt.ylabel('App', fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, 'emotion_heatmap.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"   ✅ Generated: {filepath}")
        return True
    except Exception as e:
        print(f"   ❌ Error creating emotion heatmap: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_emotion_confidence_distribution(df, output_dir):
    """Create chart showing confidence score distribution for emotions"""
    try:
        plt.figure(figsize=(12, 8))
        
        emotions = df['emotion'].unique()
        for emotion in emotions:
            emotion_data = df[df['emotion'] == emotion]['emotion_confidence']
            plt.hist(emotion_data, alpha=0.6, label=emotion, bins=20)
        
        plt.xlabel('Confidence Score', fontsize=12, fontweight='bold')
        plt.ylabel('Frequency', fontsize=12, fontweight='bold')
        plt.title('Emotion Prediction Confidence Distribution', fontsize=16, fontweight='bold', pad=20)
        plt.legend()
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, 'emotion_confidence_distribution.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"   ✅ Generated: {filepath}")
        return True
    except Exception as e:
        print(f"   ❌ Error creating confidence distribution chart: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# Main Function
# ============================================================================
def main():
    print("="*80)
    print("😊 EMOTION ANALYSIS")
    print("="*80)
    
    # Create output directories
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    os.makedirs(Config.GRAPHS_DIR, exist_ok=True)
    print(f"📁 Results will be saved in: {Config.RESULTS_DIR}\n")
    
    # Load data
    print("📂 Loading data...")
    df = load_all_data()
    if df is None:
        return
    
    print(f"✅ Using all {len(df)} rows for analysis\n")
    
    # Load model
    print(f"🤖 Loading emotion detection model: {Config.EMOTION_MODEL}")
    
    if Config.USE_PIPELINE:
        print("   Using pipeline approach...")
        pipe = get_emotion_pipeline(Config.EMOTION_MODEL)
        if pipe is None:
            print("   ❌ Failed to create pipeline. Trying direct model loading...")
            tokenizer, model = get_emotion_model(Config.EMOTION_MODEL)
            use_pipeline = False
        else:
            tokenizer, model = None, None
            use_pipeline = True
    else:
        tokenizer, model = get_emotion_model(Config.EMOTION_MODEL)
        pipe = None
        use_pipeline = False
    
    if not use_pipeline and (tokenizer is None or model is None):
        print("❌ Failed to load model. Cannot proceed.")
        return
    
    # Predict emotions
    print("\n🔮 Predicting emotions...")
    texts = df['cleaned_text'].tolist()
    
    if use_pipeline:
        emotions = predict_emotion_with_pipeline(texts, pipe, Config.BATCH_SIZE)
    else:
        emotions = predict_emotion_batch(texts, tokenizer, model, Config.BATCH_SIZE)
    
    if not emotions:
        print("❌ Failed to predict emotions")
        return
    
    # Add emotion columns to dataframe
    df['emotion'] = [e['emotion'] for e in emotions]
    df['emotion_confidence'] = [e['confidence'] for e in emotions]
    
    print(f"✅ Predicted emotions for {len(df)} reviews")
    
    # Print emotion distribution
    print("\n📊 Emotion Distribution:")
    emotion_dist = df['emotion'].value_counts()
    for emotion, count in emotion_dist.items():
        percentage = (count / len(df)) * 100
        print(f"   {emotion}: {count} ({percentage:.1f}%)")
    
    # Print average confidence
    print("\n📈 Average Confidence by Emotion:")
    for emotion in df['emotion'].unique():
        avg_conf = df[df['emotion'] == emotion]['emotion_confidence'].mean()
        print(f"   {emotion}: {avg_conf:.4f}")
    
    # Save results
    output_file = os.path.join(Config.RESULTS_DIR, 'emotion_analysis_results.csv')
    df[['app', 'content', 'cleaned_text', 'emotion', 'emotion_confidence']].to_csv(
        output_file, index=False
    )
    print(f"\n💾 Results saved: {output_file}")
    
    # Generate visualizations
    print("\n📊 Generating visualizations...")
    create_emotion_distribution_chart(df, Config.GRAPHS_DIR)
    create_emotion_by_app_chart(df, Config.GRAPHS_DIR)
    create_emotion_heatmap(df, Config.GRAPHS_DIR)
    create_emotion_confidence_distribution(df, Config.GRAPHS_DIR)
    
    print("\n" + "="*80)
    print("✅ EMOTION ANALYSIS COMPLETE!")
    print("="*80)
    print(f"📁 Results saved in: {Config.RESULTS_DIR}")
    print(f"📊 Graphs saved in: {Config.GRAPHS_DIR}")

if __name__ == "__main__":
    main()

