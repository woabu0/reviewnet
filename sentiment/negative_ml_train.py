# Machine Learning-based Sentiment Analysis Training Script
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

# Disable multiprocessing to avoid segmentation faults
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# Try to import additional models
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️  LightGBM not available. Install with: pip install lightgbm")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("⚠️  CatBoost not available. Install with: pip install catboost")

# Try to import Sentence Transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False
    print("❌ Sentence Transformers not available. Install with: pip install sentence-transformers")

# Try to import transformers for ground truth
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("❌ Transformers not available. Install with: pip install transformers")

warnings.filterwarnings("ignore")

# ============================================================================
# Configuration
# ============================================================================
class Config:
    DATA_DIR = 'negative_data'
    OUTPUT_DIR = 'negative_outputs'
    CONFUSION_MATRIX_DIR = os.path.join(OUTPUT_DIR, 'negative_ml_confusion_matrix')

# ============================================================================
# ENHANCED SENTIMENT KEYWORDS (from bert_train.py)
# ============================================================================
POSITIVE_KEYWORDS = {
    'good': 2, 'best': 3, 'fast': 2, 'great': 3, 'good service': 4, 
    'friendly': 2, 'user friendly': 4, 'excellent': 4, 'love': 3, 
    'fast delivery': 4, 'recommend': 3, 'service good': 4, 'best food': 4,
    'amazing': 4, 'satisfied': 3, 'awesome': 4, 'good food': 4, 
    'recommended': 3, 'good app': 4, 'quick': 2, 'delicious': 3,
    'app good': 4, 'helpful': 2, 'highly recommend': 5, 'delivery fast': 4,
    'faster': 2, 'good experience': 4, 'happy': 3, 'excellent service': 5,
    'quick easy': 4, 'super fast': 4, 'fresh': 2, 'delivery good': 4,
    'great service': 4, 'really good': 4, 'good delivery': 4, 'best service': 5,
    'loved': 3, 'love app': 4, 'food good': 4, 'great app': 4,
    'highly recommended': 5, 'best wishes': 2, 'best app': 4, 'best luck': 2,
    'friendly interface': 4, 'quickly': 2, 'satisfied service': 4,
    'easy delicious': 4, 'best delivery': 4, 'perfect': 4, 'outstanding': 4,
    'fantastic': 4, 'wonderful': 4, 'smooth': 3, 'reliable': 3,
    'convenient': 3, 'efficient': 3, 'responsive': 3, 'professional': 3
}

NEGATIVE_KEYWORDS = {
    'worst': 4, 'bad': 3, 'worst app': 5, 'poor': 3, 'late': 3,
    'worst food': 5, 'wrong': 3, 'bad experience': 4, 'worst experience': 5,
    'problem': 3, 'bad service': 4, 'worst service': 5, 'poor service': 4,
    'scam': 5, 'disappointed': 3, 'service bad': 4, 'fake': 4,
    'bad app': 4, 'cold': 3, 'late delivery': 4, 'delay': 3,
    'worst customer': 5, 'worst delivery': 5, 'later': 2, 'missing': 3,
    'fraud': 5, 'bad food': 4, 'late night': 2, 'error': 3,
    'app worst': 5, 'service worst': 5, 'problems': 3, 'bad delivery': 4,
    'crashes': 4, 'app bad': 4, 'problem problem': 4, 'service poor': 4,
    'poor app': 4, 'bad customer': 4, 'delayed': 3, 'rude': 3,
    'wrong order': 4, 'went wrong': 4, 'cold food': 4, 'poor delivery': 4,
    'poor customer': 4, 'minutes later': 2, 'late hours': 2, 'time late': 2,
    'wrong delivery': 4, 'terrible': 4, 'horrible': 4, 'awful': 4,
    'disgusting': 4, 'useless': 4, 'waste': 4, 'never again': 5,
    'avoid': 4, 'unprofessional': 4, 'slow': 3, 'not working': 4
}

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
# ADVANCED SENTIMENT DETECTION (Ground Truth Fallback)
# ============================================================================
def detect_sentiment_advanced(text):
    """Advanced sentiment detection with contextual understanding"""
    if not text or pd.isna(text):
        return 1  # Neutral
    
    text_lower = str(text).lower()
    text_original = str(text)
    
    negative_score = 0
    positive_score = 0
    
    for keyword, weight in NEGATIVE_KEYWORDS.items():
        if ' ' in keyword:
            if keyword in text_lower:
                negative_score += weight * 2.0
        else:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
                negative_score += weight
    
    for keyword, weight in POSITIVE_KEYWORDS.items():
        if ' ' in keyword:
            if keyword in text_lower:
                positive_score += weight * 2.0
        else:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
                positive_score += weight
    
    negative_emojis = ['😡', '🤬', '💩', '😠', '😤', '😞', '😢', '😭', '👎', '💔', '😒', '😩']
    positive_emojis = ['😍', '🥰', '🤩', '😊', '👍', '❤️', '🔥', '⭐', '🌟', '✅', '🎉', '😎', '🙌', '💫']
    
    for emoji in negative_emojis:
        if emoji in text_original:
            negative_score += 4
    
    for emoji in positive_emojis:
        if emoji in text_original:
            positive_score += 4
    
    text_length = len(text_lower.split())
    if text_length > 20:
        if negative_score > positive_score:
            negative_score += text_length * 0.1
        elif positive_score > negative_score:
            positive_score += text_length * 0.1
    
    if negative_score > 0 and positive_score > 0:
        ratio = negative_score / positive_score if positive_score > 0 else float('inf')
        if ratio > 2.0:
            return 0  # Negative
        elif ratio < 0.5:
            return 2  # Positive
        else:
            return 1  # Neutral - when scores are balanced
    
    elif negative_score >= 5:
        return 0  # Negative
    
    elif positive_score >= 5:
        return 2  # Positive
    
    elif negative_score > 0:
        if negative_score >= 3:
            return 0  # Negative
        else:
            return 1  # Neutral
    
    elif positive_score > 0:
        if positive_score >= 3:
            return 2  # Positive
        else:
            return 1  # Neutral
    
    else:
        return 1  # Neutral

# ============================================================================
# Ground Truth Generation (Using RoBERTa)
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

def predict_sentiment_batch(texts, tokenizer, model, model_name, device="cpu"):
    """Predict sentiment for a batch of texts"""
    if tokenizer is None or model is None:
        return [-1] * len(texts)
    
    try:
        model_label_map = {
            "nlptown/bert-base-multilingual-uncased-sentiment": {0: "negative", 1: "negative", 2: "neutral", 3: "positive", 4: "positive"},
            "distilbert-base-uncased-finetuned-sst-2-english": {0: "negative", 1: "positive"},
            "cardiffnlp/twitter-roberta-base-sentiment-latest": {0: "negative", 1: "neutral", 2: "positive"},
            "microsoft/deberta-base": {0: "negative", 1: "positive"},
            "microsoft/deberta-v3-base": {0: "negative", 1: "positive"}
        }
        
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
            logits = model(**inputs).logits
        
        predictions = torch.argmax(logits, dim=1).cpu().numpy()
        
        label_map = model_label_map.get(model_name, {0: "negative", 1: "positive"})
        sentiments = []
        for pred in predictions:
            if pred in label_map:
                sentiment = label_map[pred]
                if sentiment == "negative":
                    sentiments.append(0)
                elif sentiment == "neutral":
                    sentiments.append(1)
                else:
                    sentiments.append(2)
            else:
                sentiments.append(-1)
        
        return sentiments
    except Exception as e:
        print(f"Error in batch sentiment prediction for {model_name}: {e}")
        return [-1] * len(texts)

# ============================================================================
# Embedding Creation
# ============================================================================
def create_embeddings(texts, model_name):
    """Create embeddings using sentence transformers"""
    if not SENTENCE_TRANSFORMER_AVAILABLE:
        return None
    
    try:
        model = SentenceTransformer(model_name)
        batch_size = 32
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc=f"      {model_name}", leave=False):
            batch = texts[i:i+batch_size]
            try:
                batch_emb = model.encode(
                    batch,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    device='cpu'
                )
                embeddings.extend(batch_emb)
            except Exception as e:
                # Process individually if batch fails
                for text in batch:
                    try:
                        single_emb = model.encode(
                            [text],
                            show_progress_bar=False,
                            normalize_embeddings=True,
                            convert_to_numpy=True,
                            device='cpu'
                        )
                        embeddings.extend(single_emb)
                    except:
                        if embeddings:
                            embeddings.append(np.zeros_like(embeddings[0]))
                        continue
        
        return np.array(embeddings) if embeddings else None
    except Exception as e:
        print(f"   ❌ Error with {model_name}: {e}")
        return None

# ============================================================================
# ML Models
# ============================================================================
def get_ml_models():
    """Get all 10 ML models"""
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss'
        ),
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            random_state=42,
            n_jobs=-1,
            C=1.0
        ),
        'SVM': SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            random_state=42,
            probability=True
        ),
        'Naive Bayes': MultinomialNB(alpha=1.0),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.1,
            random_state=42
        ),
        'AdaBoost': AdaBoostClassifier(
            n_estimators=200,
            learning_rate=0.1,
            random_state=42
        ),
    }
    
    # Add LightGBM if available
    if LIGHTGBM_AVAILABLE:
        models['LightGBM'] = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
    
    # Add CatBoost if available
    if CATBOOST_AVAILABLE:
        models['CatBoost'] = cb.CatBoostClassifier(
            iterations=200,
            depth=10,
            learning_rate=0.1,
            random_state=42,
            verbose=False
        )
    
    return models

# ============================================================================
# Data Loading
# ============================================================================
def load_data_enhanced():
    """Load data with enhanced preprocessing"""
    data_dir = Config.DATA_DIR
    all_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    if not all_files:
        print(f"❌ No CSV files found in {data_dir}")
        return None
    
    print(f"📂 Found {len(all_files)} CSV files")
    
    all_data = []
    for file_path in all_files:
        try:
            df = pd.read_csv(file_path)
            print(f"   📄 Loading {os.path.basename(file_path)}: {len(df)} rows")
            
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
            all_data.append(df)
            
        except Exception as e:
            print(f"   ❌ Error loading {os.path.basename(file_path)}: {e}")
    
    if not all_data:
        print("❌ No valid data found")
        return None
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"✅ Combined dataset: {len(combined_df)} rows")
    return combined_df

# ============================================================================
# Confusion Matrix Generation
# ============================================================================
def create_confusion_matrix(y_true, y_pred, model_name, embedding_name, output_dir, accuracy=None):
    """Create and save confusion matrix for a model with accuracy displayed"""
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert to numpy arrays and ensure they are 1D
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        valid_mask = (y_pred != -1)
        if np.sum(valid_mask) == 0:
            print(f"      ⚠️  No valid predictions for {model_name}")
            return False
        
        y_true_valid = y_true[valid_mask]
        y_pred_valid = y_pred[valid_mask]
        
        cm = confusion_matrix(y_true_valid, y_pred_valid, labels=[0, 1, 2])
        
        # Normalize confusion matrix by row (each row sums to 1.0)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)
        
        if accuracy is None:
            accuracy = accuracy_score(y_true_valid, y_pred_valid) * 100
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        labels = ['Negative', 'Neutral', 'Positive']
        
        # Create custom annotations with decimal values (0 to 1)
        annot_data = np.empty_like(cm_normalized, dtype=object)
        for i in range(cm_normalized.shape[0]):
            for j in range(cm_normalized.shape[1]):
                value = cm_normalized[i, j]
                annot_data[i, j] = f'{value:.3f}'
        
        # Create heatmap with normalized values (0 to 1)
        heatmap = sns.heatmap(cm_normalized, annot=annot_data, fmt='', cmap='Blues', 
                             xticklabels=labels, yticklabels=labels, ax=ax,
                             cbar_kws={'label': 'Normalized Value'},
                             linewidths=2, linecolor='white', 
                             annot_kws={'size': 20, 'weight': 'bold'},
                             vmin=0, vmax=1)
        
        # Set dynamic font colors based on cell background brightness
        text_idx = 0
        for i in range(cm_normalized.shape[0]):
            for j in range(cm_normalized.shape[1]):
                if text_idx < len(ax.texts):
                    value = cm_normalized[i, j]
                    text_color = 'black' if value < 0.4 else 'white'
                    ax.texts[text_idx].set_color(text_color)
                    text_idx += 1
        
        # Set title with accuracy - larger font for research paper
        title = f'{model_name} ({embedding_name}) - Confusion Matrix (Normalized)\nAccuracy: {accuracy:.2f}%'
        ax.set_title(title, fontsize=20, fontweight='bold', pad=25)
        ax.set_xlabel('Predicted', fontsize=18, fontweight='bold')
        ax.set_ylabel('Actual', fontsize=18, fontweight='bold')
        
        # Adjust tick labels - larger for research paper
        ax.tick_params(axis='both', which='major', labelsize=16)
        
        # Increase colorbar label and tick font sizes
        cbar = ax.collections[0].colorbar
        cbar.set_label('Normalized Value', fontsize=18, fontweight='bold')
        cbar.ax.tick_params(labelsize=16)
        cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        cbar.set_ticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
        
        plt.tight_layout()
        
        # Save confusion matrix
        safe_model_name = model_name.lower().replace(' ', '_')
        safe_embedding_name = embedding_name.lower().replace('-', '_').replace('/', '_')
        filename = f"{safe_model_name}_{safe_embedding_name}_confusion_matrix.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return True
    except Exception as e:
        print(f"      ⚠️  Error creating confusion matrix for {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# Visualization Functions
# ============================================================================
def create_performance_visualization(results_df, output_dir):
    """Create performance visualization with same blue color and different bar patterns"""
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('ML Model Performance Comparison', 
                 fontsize=24, fontweight='bold', y=0.98)
    
    # Get unique model-embedding combinations
    results_df['Model_Embedding'] = results_df['Model'] + ' (' + results_df['Embedding'] + ')'
    models = results_df['Model_Embedding'].unique()
    
    # Use white color for bars with black edges
    base_color = 'white'
    
    # Define different bar patterns/styles
    bar_styles = [
        {'color': base_color, 'alpha': 1.0, 'edgecolor': 'black', 'linewidth': 2.5, 'hatch': None},
        {'color': base_color, 'alpha': 1.0, 'edgecolor': 'black', 'linewidth': 2.5, 'hatch': '///'},
        {'color': base_color, 'alpha': 1.0, 'edgecolor': 'black', 'linewidth': 2.5, 'hatch': '---'},
        {'color': base_color, 'alpha': 1.0, 'edgecolor': 'black', 'linewidth': 2.5, 'hatch': '|||'},
        {'color': base_color, 'alpha': 1.0, 'edgecolor': 'black', 'linewidth': 2.5, 'hatch': '+++'},
        {'color': base_color, 'alpha': 1.0, 'edgecolor': 'black', 'linewidth': 2.5, 'hatch': 'xxx'},
        {'color': base_color, 'alpha': 1.0, 'edgecolor': 'black', 'linewidth': 2.5, 'hatch': '...'},
        {'color': base_color, 'alpha': 1.0, 'edgecolor': 'black', 'linewidth': 2.5, 'hatch': 'ooo'},
        {'color': base_color, 'alpha': 1.0, 'edgecolor': 'black', 'linewidth': 2.5, 'hatch': '***'},
        {'color': base_color, 'alpha': 1.0, 'edgecolor': 'black', 'linewidth': 2.5, 'hatch': '///'},
    ]
    
    metrics = [('Accuracy', axes[0, 0]), ('Precision', axes[0, 1]), 
               ('Recall', axes[1, 0]), ('F1', axes[1, 1])]
    
    for metric, ax in metrics:
        x_pos = np.arange(len(models))
        scores = []
        for model_emb in models:
            model_data = results_df[results_df['Model_Embedding'] == model_emb]
            scores.append(model_data[metric].values[0] if not model_data.empty else 0)
        
        # Create bars with different patterns
        bars = []
        for i, (x, score) in enumerate(zip(x_pos, scores)):
            style = bar_styles[i % len(bar_styles)]
            bar = ax.bar(x, score, width=0.7, 
                        color=style['color'], 
                        alpha=style['alpha'],
                        edgecolor=style['edgecolor'],
                        linewidth=style['linewidth'],
                        hatch=style['hatch'])
            bars.append(bar[0])
        
        for i, v in enumerate(scores):
            if v > 0:
                ax.text(i, v + 1, f'{v:.1f}%', 
                       ha='center', va='bottom', fontsize=20, fontweight='bold')
        
        ax.set_title(f'{metric} Comparison', fontsize=20, fontweight='bold', pad=15)
        ax.set_xlabel('Models', fontsize=18, fontweight='bold')
        ax.set_ylabel(f'{metric} (%)', fontsize=18, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=16)
        ax.tick_params(axis='y', labelsize=16)
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1.5)
        ax.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Visualization saved: {os.path.join(output_dir, 'model_comparison.png')}")

def create_results_table(results_df, output_dir):
    """Create a professional results table"""
    fig, ax = plt.subplots(figsize=(20, max(12, len(results_df) * 0.8 + 3)))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = [['Model', 'Embedding', 'Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1 Score (%)']]
    
    # Sort by accuracy descending
    sorted_df = results_df.sort_values('Accuracy', ascending=False)
    
    for _, row in sorted_df.iterrows():
        table_data.append([
            row['Model'],
            row['Embedding'],
            f"{row['Accuracy']:.2f}",
            f"{row['Precision']:.2f}",
            f"{row['Recall']:.2f}",
            f"{row['F1']:.2f}"
        ])
    
    # Create table
    table = ax.table(
        cellText=table_data,
        cellLoc='center',
        loc='center',
        colWidths=[0.25, 0.25, 0.15, 0.15, 0.15, 0.15]
    )
    
    # Style the table - larger fonts for research paper
    table.auto_set_font_size(False)
    table.set_fontsize(18)
    table.scale(1, 2.8)
    
    # Header row styling
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white', size=20)
    
    # Data row styling - highlight models with ≥90% accuracy
    for i in range(1, len(table_data)):
        row_accuracy = float(table_data[i][2].replace('%', ''))
        
        if row_accuracy >= 90:
            row_color = '#E8F4F8'
        else:
            row_color = '#F8F9FA' if i % 2 == 0 else '#FFFFFF'
        
        for j in range(len(table_data[0])):
            table[(i, j)].set_facecolor(row_color)
            table[(i, j)].set_text_props(size=18)
            
            if j == 2 and row_accuracy >= 90:
                table[(i, j)].set_text_props(size=18, weight='bold')
    
    # Add borders
    for i in range(len(table_data)):
        for j in range(len(table_data[0])):
            table[(i, j)].set_edgecolor('#CCCCCC')
            table[(i, j)].set_linewidth(1)
    
    plt.title('ML Model Performance Results Table', 
              fontsize=22, fontweight='bold', pad=30)
    plt.tight_layout()
    
    # Save table
    table_path = os.path.join(output_dir, 'model_results_table.png')
    plt.savefig(table_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(output_dir, 'model_results_table.pdf'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"📋 Results table saved: {table_path}")
    print(f"📋 PDF table saved: {os.path.join(output_dir, 'model_results_table.pdf')}")
    
    # Also print console table
    print("\n" + "="*100)
    print("ML MODEL PERFORMANCE RESULTS TABLE")
    print("="*100)
    print(f"{'Model':<20} {'Embedding':<30} {'Accuracy':<15} {'Precision':<15} {'Recall':<15} {'F1 Score':<15}")
    print("-"*100)
    
    for _, row in sorted_df.iterrows():
        print(f"{row['Model']:<20} {row['Embedding']:<30} {row['Accuracy']:<15.2f}% "
              f"{row['Precision']:<15.2f}% {row['Recall']:<15.2f}% {row['F1']:<15.2f}%")
    
    print("="*100)

# ============================================================================
# Main Training Function
# ============================================================================
def main():
    print("="*80)
    print("🎯 MACHINE LEARNING SENTIMENT ANALYSIS TRAINING")
    print("="*80)
    
    if not SENTENCE_TRANSFORMER_AVAILABLE:
        print("❌ Sentence Transformers not available.")
        return
    
    # Define embeddings
    embedding_models = {
        'MPNet': 'all-mpnet-base-v2',
        'all-MiniLM-L6': 'all-MiniLM-L6-v2'
    }
    
    # Force CPU
    device = "cpu"
    print(f"Using device: {device}\n")
    
    # Create output directory
    os.makedirs(Config.CONFUSION_MATRIX_DIR, exist_ok=True)
    print(f"📁 Confusion matrices will be saved in: {Config.CONFUSION_MATRIX_DIR}\n")
    
    # Load data
    print("📂 Loading and preprocessing data...")
    df = load_data_enhanced()
    if df is None:
        return
    
    # Use RoBERTa as ground truth (same as bert_train.py)
    print("🔍 Using RoBERTa as ground truth for high accuracy...")
    
    roberta_model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    print("   📥 Loading RoBERTa for ground truth...")
    roberta_tokenizer, roberta_model = get_sentiment_model(roberta_model_name)
    
    if roberta_tokenizer and roberta_model:
        texts = df['cleaned_text'].tolist()
        ground_truth_predictions = []
        
        print("   🔮 Generating ground truth labels with RoBERTa...")
        for i in tqdm(range(0, len(texts), 8), desc="      RoBERTa GT", leave=False):
            try:
                batch_texts = texts[i:i+8]
                batch_preds = predict_sentiment_batch(batch_texts, roberta_tokenizer, roberta_model, roberta_model_name, device)
                ground_truth_predictions.extend(batch_preds)
                gc.collect()
            except Exception as e:
                print(f"\n         ⚠️  Error in batch {i//8 + 1}: {e}")
                ground_truth_predictions.extend([-1] * len(batch_texts))
                gc.collect()
                continue
        
        while len(ground_truth_predictions) < len(texts):
            ground_truth_predictions.append(-1)
        
        df['ground_truth'] = ground_truth_predictions
        
        invalid_mask = df['ground_truth'] == -1
        if invalid_mask.sum() > 0:
            print(f"   ⚠️  {invalid_mask.sum()} invalid RoBERTa predictions, using keyword-based fallback...")
            df.loc[invalid_mask, 'ground_truth'] = df.loc[invalid_mask, 'cleaned_text'].apply(detect_sentiment_advanced)
        
        del roberta_model
        del roberta_tokenizer
        gc.collect()
        time.sleep(1.0)
        
        sentiment_counts = df['ground_truth'].value_counts().sort_index()
        sentiment_names = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        print("📊 Ground truth sentiment distribution:")
        for sentiment, count in sentiment_counts.items():
            percentage = (count / len(df)) * 100
            print(f"   {sentiment_names[sentiment]}: {count} ({percentage:.1f}%)")
    else:
        print("   ⚠️  Failed to load RoBERTa, using keyword-based ground truth...")
        df['ground_truth'] = df['cleaned_text'].apply(detect_sentiment_advanced)
        sentiment_counts = df['ground_truth'].value_counts().sort_index()
        sentiment_names = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        print("📊 Ground truth sentiment distribution:")
        for sentiment, count in sentiment_counts.items():
            percentage = (count / len(df)) * 100
            print(f"   {sentiment_names[sentiment]}: {count} ({percentage:.1f}%)")
    
    # Get texts and ground truth
    texts = df['cleaned_text'].tolist()
    y_true = df['ground_truth'].tolist()
    
    print(f"\n📊 Total samples: {len(texts)}")
    
    # Split data
    # Check if we can use stratify (requires at least one sample per class in each split)
    unique_classes = len(set(y_true))
    if unique_classes > 1:
        X_train, X_test, y_train, y_test = train_test_split(
            texts, y_true, test_size=0.2, random_state=42, stratify=y_true
        )
    else:
        # If only one class, don't use stratify
        X_train, X_test, y_train, y_test = train_test_split(
            texts, y_true, test_size=0.2, random_state=42
        )
    
    print(f"📊 Dataset split:")
    print(f"   Training: {len(X_train)} samples")
    print(f"   Testing: {len(X_test)} samples")
    
    # Get ML models
    ml_models = get_ml_models()
    print(f"\n🤖 Available ML models: {len(ml_models)}")
    for model_name in ml_models.keys():
        print(f"   - {model_name}")
    
    # Process each embedding
    all_results = []
    
    for emb_name, emb_model_name in embedding_models.items():
        print(f"\n" + "="*70)
        print(f"📦 PROCESSING EMBEDDING: {emb_name.upper()}")
        print("="*70)
        
        print(f"   📥 Creating embeddings with {emb_name}...")
        X_train_emb = create_embeddings(X_train, emb_model_name)
        X_test_emb = create_embeddings(X_test, emb_model_name)
        
        if X_train_emb is None or X_test_emb is None:
            print(f"   ❌ Failed to create embeddings for {emb_name}")
            continue
        
        print(f"   ✅ Embeddings created: Train shape {X_train_emb.shape}, Test shape {X_test_emb.shape}")
        
        # Check if dataset has only one class
        unique_classes = len(set(y_train))
        models_requiring_multiple_classes = ['XGBoost', 'Logistic Regression', 'SVM', 'Gradient Boosting', 'CatBoost']
        
        if unique_classes == 1:
            print(f"\n   ⚠️  Dataset contains only one class. Some models will be skipped.")
        
        # Scale features for models that need it
        # StandardScaler for SVM and Logistic Regression
        scaler_standard = StandardScaler()
        X_train_scaled_standard = scaler_standard.fit_transform(X_train_emb)
        X_test_scaled_standard = scaler_standard.transform(X_test_emb)
        
        # MinMaxScaler for Naive Bayes (requires non-negative values)
        scaler_minmax = MinMaxScaler()
        X_train_scaled_minmax = scaler_minmax.fit_transform(X_train_emb)
        X_test_scaled_minmax = scaler_minmax.transform(X_test_emb)
        
        # Train each ML model
        for model_name, model in ml_models.items():
            print(f"\n   🤖 Training {model_name}...")
            
            # Skip models that require multiple classes if dataset has only one class
            if unique_classes == 1 and model_name in models_requiring_multiple_classes:
                print(f"      ⏭️  Skipped: {model_name} requires multiple classes (dataset has only 1 class)")
                continue
            
            try:
                # Use appropriate scaling for models that need it
                if model_name == 'Naive Bayes':
                    # Naive Bayes requires non-negative values, use MinMaxScaler
                    X_train_use = X_train_scaled_minmax
                    X_test_use = X_test_scaled_minmax
                elif model_name in ['SVM', 'Logistic Regression']:
                    # SVM and Logistic Regression work better with StandardScaler
                    X_train_use = X_train_scaled_standard
                    X_test_use = X_test_scaled_standard
                else:
                    # Other models use original embeddings
                    X_train_use = X_train_emb
                    X_test_use = X_test_emb
                
                # Train model
                model.fit(X_train_use, y_train)
                
                # Predict
                y_pred = model.predict(X_test_use)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred) * 100
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0) * 100
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0) * 100
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0) * 100
                
                all_results.append({
                    'Model': model_name,
                    'Embedding': emb_name,
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1': f1
                })
                
                print(f"      ✅ Accuracy: {accuracy:.2f}%, F1: {f1:.2f}%")
                
                # Generate confusion matrix
                print(f"      📊 Creating confusion matrix...")
                create_confusion_matrix(y_test, y_pred, model_name, emb_name, Config.CONFUSION_MATRIX_DIR, accuracy=accuracy)
                
                # Cleanup
                del model
                gc.collect()
                
            except Exception as e:
                error_msg = str(e)
                # Check if it's a single-class error
                single_class_errors = [
                    "only one class",
                    "only one unique value",
                    "at least 2 classes",
                    "greater than one",
                    "base_score must be in (0,1)"
                ]
                
                if any(err in error_msg.lower() for err in single_class_errors):
                    print(f"      ⏭️  Skipped: {model_name} requires multiple classes (dataset has only 1 class)")
                else:
                    print(f"      ❌ Error training {model_name}: {error_msg}")
                continue
        
        # Cleanup embeddings
        del X_train_emb, X_test_emb, X_train_scaled_standard, X_test_scaled_standard
        del X_train_scaled_minmax, X_test_scaled_minmax
        gc.collect()
        time.sleep(1.0)
    
    # Create results visualization
    if all_results:
        print("\n" + "="*80)
        print("📊 GENERATING RESULTS")
        print("="*80)
        
        results_df = pd.DataFrame(all_results)
        
        # Save CSV
        output_dir = os.path.join(Config.OUTPUT_DIR, 'training_results')
        os.makedirs(output_dir, exist_ok=True)
        results_df.to_csv(os.path.join(output_dir, 'ml_model_comparison_table.csv'), index=False)
        
        # Create visualization
        create_performance_visualization(results_df, Config.CONFUSION_MATRIX_DIR)
        
        # Create results table
        create_results_table(results_df, Config.CONFUSION_MATRIX_DIR)
        
        # Print summary
        print("\n" + "="*80)
        print("✅ TRAINING COMPLETE!")
        print("="*80)
        print(f"📊 Total model-embedding combinations processed: {len(results_df)}")
        print(f"📈 Best accuracy: {results_df['Accuracy'].max():.2f}%")
        best_row = results_df.loc[results_df['Accuracy'].idxmax()]
        print(f"🏆 Best combination: {best_row['Model']} with {best_row['Embedding']}")
        print(f"   Accuracy: {best_row['Accuracy']:.2f}%")
        print(f"   F1 Score: {best_row['F1']:.2f}%")
        
        # Models above 90%
        above_90 = results_df[results_df['Accuracy'] >= 90]
        if len(above_90) > 0:
            print(f"\n🎯 Model-embedding combinations with ≥90% accuracy: {len(above_90)}")
            for _, row in above_90.iterrows():
                print(f"   - {row['Model']} ({row['Embedding']}): {row['Accuracy']:.2f}%")
        else:
            print(f"\n⚠️  No combinations reached 90% accuracy. Best: {results_df['Accuracy'].max():.2f}%")
        
        print(f"\n📁 Results saved in: {output_dir}")
        print(f"📊 Confusion matrices saved in: {Config.CONFUSION_MATRIX_DIR}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()

