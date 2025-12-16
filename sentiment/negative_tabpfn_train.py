# TabPFN-based Sentiment Analysis Training Script
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

# Fix einops TensorFlow detection issue
# Patch einops to skip TensorFlow backend detection
try:
    import einops._backends
    # Monkey patch to prevent TensorFlow detection errors
    original_get_backend = einops._backends.get_backend
    def patched_get_backend(tensor):
        try:
            return original_get_backend(tensor)
        except (AttributeError, ImportError):
            # If TensorFlow detection fails, use PyTorch backend
            import torch
            if isinstance(tensor, torch.Tensor):
                return einops._backends.TorchBackend()
            raise
    einops._backends.get_backend = patched_get_backend
except Exception:
    pass  # If patching fails, continue anyway

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Try to import TabPFN with compatibility for newer versions
try:
    from tabpfn import TabPFNClassifier
    TABPFN_AVAILABLE = True
    print("✅ Using TabPFN Classifier")
except ImportError:
    try:
        from tabpfn.models import TabPFNClassifier
        TABPFN_AVAILABLE = True
        print("✅ Using TabPFN Classifier from models module")
    except ImportError:
        TABPFN_AVAILABLE = False
        print("⚠️  TabPFN not available. Install with: pip install tabpfn")

# Try to import transformers for BERT ground truth
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
    DATA_DIR = '../negative_dataset'
    OUTPUT_DIR = 'outputs'
    CONFUSION_MATRIX_DIR = os.path.join(OUTPUT_DIR, 'negative_tabpfn_confusion_matrix')

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
# Ground Truth Generation (Using BERT)
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
# Feature Extraction (TF-IDF)
# ============================================================================
def create_features(texts_train, texts_test, max_features=100):
    """Create TF-IDF features from text
    Note: TabPFN is restricted to max 100 features, so we limit to 100
    """
    try:
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),  # Use unigrams and bigrams
            min_df=2,  # Minimum document frequency
            max_df=0.95,  # Maximum document frequency
            stop_words='english'  # Remove English stop words
        )
        
        X_train_features = vectorizer.fit_transform(texts_train)
        X_test_features = vectorizer.transform(texts_test)
        
        # Convert to dense array (TabPFN works better with dense arrays)
        X_train_features = X_train_features.toarray()
        X_test_features = X_test_features.toarray()
        
        return X_train_features, X_test_features, vectorizer
    except Exception as e:
        print(f"   ❌ Error creating TF-IDF features: {e}")
        return None, None, None

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
def create_confusion_matrix(y_true, y_pred, model_name, output_dir, accuracy=None):
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
        title = f'{model_name} - Confusion Matrix (Normalized)\nAccuracy: {accuracy:.2f}%'
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
        filename = f"{safe_model_name}_confusion_matrix.png"
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
def create_performance_visualization(result, output_dir):
    """Create performance visualization with same blue color and different bar patterns"""
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('TabPFN Model Performance', 
                 fontsize=24, fontweight='bold', y=0.98)
    
    # Use white color for bars with black edges
    base_color = 'white'
    
    metrics = [('Accuracy', axes[0, 0]), ('Precision', axes[0, 1]), 
               ('Recall', axes[1, 0]), ('F1', axes[1, 1])]
    
    for metric, ax in metrics:
        score = result[metric]
        
        # Create bar
        bar = ax.bar(0, score, width=0.7, 
                    color=base_color, 
                    alpha=1.0,
                    edgecolor='black',
                    linewidth=2.5)
        
        # Add value label
        if score > 0:
            ax.text(0, score + 1, f'{score:.1f}%', 
                   ha='center', va='bottom', fontsize=20, fontweight='bold')
        
        ax.set_title(f'{metric}', fontsize=20, fontweight='bold', pad=15)
        ax.set_ylabel(f'{metric} (%)', fontsize=18, fontweight='bold')
        ax.set_xticks([0])
        ax.set_xticklabels([result['Model']], fontsize=16)
        ax.tick_params(axis='y', labelsize=16)
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1.5)
        ax.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Visualization saved: {os.path.join(output_dir, 'model_comparison.png')}")

def create_results_table(result, output_dir):
    """Create a professional results table"""
    fig, ax = plt.subplots(figsize=(18, max(10, 3)))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = [['Model', 'Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1 Score (%)']]
    
    table_data.append([
        result['Model'],
        f"{result['Accuracy']:.2f}",
        f"{result['Precision']:.2f}",
        f"{result['Recall']:.2f}",
        f"{result['F1']:.2f}"
    ])
    
    # Create table
    table = ax.table(
        cellText=table_data,
        cellLoc='center',
        loc='center',
        colWidths=[0.30, 0.20, 0.20, 0.20, 0.20]
    )
    
    # Style the table - larger fonts for research paper
    table.auto_set_font_size(False)
    table.set_fontsize(18)
    table.scale(1, 2.8)
    
    # Header row styling
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white', size=20)
    
    # Data row styling - highlight if ≥90% accuracy
    row_accuracy = result['Accuracy']
    
    if row_accuracy >= 90:
        row_color = '#E8F4F8'  # Light blue for high performers
    else:
        row_color = '#FFFFFF'
    
    for j in range(len(table_data[0])):
        table[(1, j)].set_facecolor(row_color)
        table[(1, j)].set_text_props(size=18)
        
        # Bold accuracy if ≥90%
        if j == 1 and row_accuracy >= 90:
            table[(1, j)].set_text_props(size=18, weight='bold')
    
    # Add borders
    for i in range(len(table_data)):
        for j in range(len(table_data[0])):
            table[(i, j)].set_edgecolor('#CCCCCC')
            table[(i, j)].set_linewidth(1)
    
    plt.title('TabPFN Model Performance Results Table', 
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
    print("\n" + "="*80)
    print("TABPFN MODEL PERFORMANCE RESULTS TABLE")
    print("="*80)
    print(f"{'Model':<20} {'Accuracy':<15} {'Precision':<15} {'Recall':<15} {'F1 Score':<15}")
    print("-"*80)
    print(f"{result['Model']:<20} {result['Accuracy']:<15.2f}% "
          f"{result['Precision']:<15.2f}% {result['Recall']:<15.2f}% {result['F1']:<15.2f}%")
    print("="*80)

# ============================================================================
# Main Training Function
# ============================================================================
def main():
    print("="*80)
    print("🎯 TABPFN SENTIMENT ANALYSIS TRAINING (Negative Data)")
    print("="*80)
    
    if not TABPFN_AVAILABLE:
        print("❌ TabPFN not available.")
        return
    
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
    
    # Use BERT as ground truth (multi-class BERT model)
    print("🔍 Using BERT as ground truth for sentiment classification...")
    
    bert_model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    print("   📥 Loading BERT for ground truth...")
    bert_tokenizer, bert_model = get_sentiment_model(bert_model_name)
    
    if bert_tokenizer and bert_model:
        texts = df['cleaned_text'].tolist()
        ground_truth_predictions = []
        
        print("   🔮 Generating ground truth labels with BERT...")
        for i in tqdm(range(0, len(texts), 8), desc="      BERT GT", leave=False):
            try:
                batch_texts = texts[i:i+8]
                batch_preds = predict_sentiment_batch(batch_texts, bert_tokenizer, bert_model, bert_model_name, device)
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
            print(f"   ⚠️  {invalid_mask.sum()} invalid BERT predictions, using keyword-based fallback...")
            df.loc[invalid_mask, 'ground_truth'] = df.loc[invalid_mask, 'cleaned_text'].apply(detect_sentiment_advanced)
        
        del bert_model
        del bert_tokenizer
        gc.collect()
        time.sleep(1.0)
        
        sentiment_counts = df['ground_truth'].value_counts().sort_index()
        sentiment_names = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        print("📊 Ground truth sentiment distribution (from BERT):")
        for sentiment, count in sentiment_counts.items():
            percentage = (count / len(df)) * 100
            print(f"   {sentiment_names[sentiment]}: {count} ({percentage:.1f}%)")
    else:
        print("   ⚠️  Failed to load BERT, using keyword-based ground truth...")
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
    
    # TabPFN can handle larger datasets with ignore_pretraining_limits=True
    # We set TABPFN_ALLOW_CPU_LARGE_DATASET=1, so use all training samples
    # Note: TabPFN is optimized for <= 1000 samples, but can work with more
    # Performance may be slower with larger datasets on CPU
    X_train_sampled, y_train_sampled = X_train, y_train
    if len(X_train) > 1000:
        print(f"\n⚠️  Training set size ({len(X_train)}) exceeds TabPFN's recommended limit (1000)")
        print(f"   Using all {len(X_train)} samples (ignore_pretraining_limits=True)")
        print(f"   Note: Training may take longer on CPU with larger datasets")
    
    # Create TF-IDF features
    print(f"\n📥 Creating TF-IDF features...")
    X_train_features, X_test_features, vectorizer = create_features(X_train_sampled, X_test)
    
    if X_train_features is None or X_test_features is None:
        print(f"❌ Failed to create features")
        return
    
    print(f"✅ Features created: Train shape {X_train_features.shape}, Test shape {X_test_features.shape}")
    
    # Scale features for TabPFN (TabPFN works better with standardized features)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_features)
    X_test_scaled = scaler.transform(X_test_features)
    
    # Train TabPFN
    print(f"\n🤖 Training TabPFN...")
    
    result = None
    
    if not TABPFN_AVAILABLE:
        print("❌ TabPFN is not available. Please install it with: pip install tabpfn")
        print("   Note: TabPFN 2.5+ requires Hugging Face authentication.")
        print("   Visit: https://huggingface.co/Prior-Labs/tabpfn_2_5 to accept terms")
        print("   Then run: hf auth login")
        return
    
    try:
        # TabPFN works best with smaller datasets, but can handle larger ones
        # It's a prior-data fitted network, so it doesn't need traditional training
        # Try different initialization methods based on version
        model = None
        init_error = None
        
        # Try with N_ensemble_configurations (older versions)
        try:
            model = TabPFNClassifier(device='cpu', N_ensemble_configurations=32, ignore_pretraining_limits=True)
        except TypeError:
            # Fallback to basic initialization (newer versions)
            try:
                model = TabPFNClassifier(device='cpu', ignore_pretraining_limits=True)
            except TypeError:
                # Try without ignore_pretraining_limits if parameter doesn't exist
                try:
                    model = TabPFNClassifier(device='cpu')
                except Exception as e:
                    init_error = e
                    model = None
            except Exception as e:
                init_error = e
                model = None
        
        if model is None:
            raise Exception(f"TabPFN initialization failed: {init_error}")
        
        # Fit the model (TabPFN uses prior data, so this is fast)
        print("   Fitting TabPFN model...")
        
        # Patch the size validation in transformer_prediction_interface before fitting
        original_validation = None
        try:
            import tabpfn.scripts.transformer_prediction_interface as tpi
            # Find and patch the validation that raises the ValueError
            # The check is at line 239 in transformer_prediction_interface.py
            # We'll patch the fit method to skip the size check
            if hasattr(tpi, 'TabPFNTransformer'):
                original_fit_method = tpi.TabPFNTransformer.fit
                
                def patched_fit_method(self, X, y, **kwargs):
                    # Temporarily disable the size check by patching the internal validation
                    import tabpfn.base
                    original_check = getattr(tabpfn.base, 'check_cpu_warning', None)
                    
                    def bypass_check(*args, **kwargs):
                        pass
                    
                    if original_check:
                        tabpfn.base.check_cpu_warning = bypass_check
                    
                    try:
                        # Also need to bypass the check in transformer_prediction_interface
                        # The ValueError is raised directly in the fit method
                        # We'll catch it and continue
                        try:
                            result = original_fit_method(self, X, y, **kwargs)
                            return result
                        except ValueError as ve:
                            error_str = str(ve).lower()
                            if 'trainingsize' in error_str or '1024' in error_str:
                                # Bypass by calling the parent implementation
                                # This is a workaround - we'll call the actual fit logic
                                return super(tpi.TabPFNTransformer, self).fit(X, y, **kwargs)
                            raise
                    finally:
                        if original_check:
                            tabpfn.base.check_cpu_warning = original_check
                
                # Apply the patch
                tpi.TabPFNTransformer.fit = patched_fit_method
                original_validation = original_fit_method
        except Exception as patch_err:
            # If patching fails, we'll handle it in the exception handler below
            pass
        
        try:
            # Try to fit with all samples
            model.fit(X_train_scaled, y_train_sampled)
        except ValueError as fit_error:
            # Check if it's the dataset size warning
            error_str = str(fit_error).lower()
            if 'trainingsize' in error_str or '1024' in error_str or 'overwrite_warning' in error_str:
                print(f"   ⚠️  TabPFN hard limit: This version requires <= 1024 training samples.")
                print(f"   📊 Using stratified sampling to 1024 samples to proceed...")
                print(f"   💡 Note: To use all {len(X_train_sampled)} samples, you may need a different TabPFN version or GPU.")
                from sklearn.utils import resample
                X_train_fit, y_train_fit = resample(
                    X_train_scaled, y_train_sampled,
                    n_samples=1024,
                    random_state=42,
                    stratify=y_train_sampled
                )
                model.fit(X_train_fit, y_train_fit)
            elif '403' in error_str or 'forbidden' in error_str or 'authentication' in error_str or 'gated' in error_str:
                raise Exception("TabPFN authentication required")
            else:
                raise
        except Exception as fit_error:
            # Check if it's an authentication/download error
            error_str = str(fit_error).lower()
            if '403' in error_str or 'forbidden' in error_str or 'authentication' in error_str or 'gated' in error_str:
                raise Exception("TabPFN authentication required")
            else:
                raise
        finally:
            # Restore original method if we patched it
            if original_validation:
                try:
                    import tabpfn.scripts.transformer_prediction_interface as tpi
                    tpi.TabPFNTransformer.fit = original_validation
                except Exception:
                    pass
        
        # Predict
        print("   Making predictions...")
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred) * 100
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0) * 100
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0) * 100
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0) * 100
        
        result = {
            'Model': 'TabPFN',
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1
        }
        
        print(f"✅ Accuracy: {accuracy:.2f}%, F1: {f1:.2f}%")
        
        # Generate confusion matrix
        print(f"📊 Creating confusion matrix...")
        create_confusion_matrix(y_test, y_pred, 'TabPFN', Config.CONFUSION_MATRIX_DIR, accuracy=accuracy)
        
        # Cleanup
        del model
        gc.collect()
        
    except Exception as e:
        error_msg = str(e)
        error_str_lower = error_msg.lower()
        
        # Check if it's an authentication/download error
        is_auth_error = (
            "huggingface" in error_str_lower or 
            "gated" in error_str_lower or 
            "authentication" in error_str_lower or
            "403" in error_msg or
            "forbidden" in error_str_lower or
            "tabpfn authentication required" in error_str_lower
        )
        
        if is_auth_error:
            print("\n" + "="*80)
            print("⚠️  TABPFN AUTHENTICATION REQUIRED")
            print("="*80)
            print("TabPFN 2.5+ requires Hugging Face authentication and is a gated model.")
            print("\n📋 SOLUTION OPTIONS:")
            print("\n1. SET UP HUGGING FACE AUTHENTICATION:")
            print("   a. Visit https://huggingface.co/Prior-Labs/tabpfn_2_5")
            print("      and accept the terms of use")
            print("   b. Log in via command line:")
            print("      hf auth login")
            print("   c. Make sure your token has 'read' access to gated repositories")
            print("      (check fine-grained token settings if using tokens)")
            print("\n2. USE OLDER VERSION (NO AUTH REQUIRED):")
            print("   Install TabPFN 0.1.x which doesn't require authentication:")
            print("   pip install 'tabpfn<0.2'")
            print("\n3. FOR COMMERCIAL USE:")
            print("   Contact sales@priorlabs.ai for alternative download options")
            print("\nFor detailed instructions, see:")
            print("https://docs.priorlabs.ai/how-to-access-gated-models")
            print("="*80)
        else:
            print(f"❌ Error training TabPFN: {error_msg}")
            import traceback
            traceback.print_exc()
        return
    
    # Cleanup features
    del X_train_features, X_test_features, X_train_scaled, X_test_scaled, vectorizer
    gc.collect()
    time.sleep(1.0)
    
    # Create results visualization
    if result:
        print("\n" + "="*80)
        print("📊 GENERATING RESULTS")
        print("="*80)
        
        # Create visualization
        create_performance_visualization(result, Config.CONFUSION_MATRIX_DIR)
        
        # Create results table
        create_results_table(result, Config.CONFUSION_MATRIX_DIR)
        
        # Print summary
        print("\n" + "="*80)
        print("✅ TRAINING COMPLETE!")
        print("="*80)
        print(f"🏆 Model: {result['Model']}")
        print(f"   Accuracy: {result['Accuracy']:.2f}%")
        print(f"   Precision: {result['Precision']:.2f}%")
        print(f"   Recall: {result['Recall']:.2f}%")
        print(f"   F1 Score: {result['F1']:.2f}%")
        
        if result['Accuracy'] >= 90:
            print(f"\n🎯 Model reached ≥90% accuracy!")
        else:
            print(f"\n⚠️  Model accuracy: {result['Accuracy']:.2f}%")
        
        print(f"\n📁 Results saved in: {output_dir}")
        print(f"📊 Confusion matrices saved in: {Config.CONFUSION_MATRIX_DIR}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()

