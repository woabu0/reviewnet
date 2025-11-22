# -*- coding: utf-8 -*-
import sys
import io

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Try importing TabPFN with compatibility for newer versions
try:
    from tabpfn import TabPFNClassifier
    TABPFN_AVAILABLE = True
    print("Using TabPFN Classifier")
except ImportError:
    try:
        from tabpfn.models import TabPFNClassifier
        TABPFN_AVAILABLE = True
        print("Using TabPFN Classifier from models module")
    except ImportError:
        from sklearn.ensemble import RandomForestClassifier
        TABPFN_AVAILABLE = False
        print("TabPFN not available, using Random Forest as fallback")

print("="*80)
print("THEME CLASSIFICATION FOR BLIND APP REVIEWS")
print("="*80 + "\n")

# Configuration
CSV_FILE = "proyojon.csv"  # Update path as needed
REVIEW_COLUMN = 'review'

# ============================================================================
# STEP 1: Load Dataset
# ============================================================================
print("Step 1: Loading dataset...")
try:
    df = pd.read_csv(CSV_FILE, encoding='utf-8')
    print(f"[OK] Loaded {len(df)} reviews\n")
except Exception as e:
    print(f"[ERROR] Could not load CSV: {e}")
    print("Please update CSV_FILE path in the script")
    exit(1)

# Check if review column exists
if REVIEW_COLUMN not in df.columns:
    print(f"[ERROR] Column '{REVIEW_COLUMN}' not found")
    print(f"Available columns: {list(df.columns)}")
    exit(1)

# Filter out empty reviews
df = df[df[REVIEW_COLUMN].notna()]
df = df[df[REVIEW_COLUMN].str.len() > 10]
print(f"After filtering: {len(df)} valid reviews\n")

# ============================================================================
# STEP 2: Generate Sentence Embeddings
# ============================================================================
print("Step 2: Generating sentence embeddings...")
print("Loading sentence transformer model...\n")

try:
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("[OK] Model loaded\n")
    
    print("Creating embeddings for all reviews...")
    embeddings = embedding_model.encode(
        df[REVIEW_COLUMN].tolist(), 
        show_progress_bar=True,
        batch_size=32
    )
    embeddings = np.array(embeddings)
    print(f"[OK] Embedding shape: {embeddings.shape}\n")
    
except Exception as e:
    print(f"[ERROR] Could not create embeddings: {e}")
    # Fallback to TF-IDF
    from sklearn.feature_extraction.text import TfidfVectorizer
    print("Falling back to TF-IDF features...")
    vectorizer = TfidfVectorizer(max_features=1000)
    embeddings = vectorizer.fit_transform(df[REVIEW_COLUMN]).toarray()
    print(f"[OK] TF-IDF features shape: {embeddings.shape}\n")

# ============================================================================
# STEP 3: Define Themes with Training Examples
# ============================================================================
print("Step 3: Setting up theme classification...")

THEMES = {
    "Accessibility": [
        "Works great with TalkBack screen reader",
        "Voice feedback is clear and helpful",
        "Screen reader support is excellent",
        "Audio instructions are very clear",
        "Accessible interface for blind users",
        "VoiceOver works perfectly with this app",
        "Great for visually impaired people",
        "Easy to use without seeing the screen"
    ],
    "Functionality and Recognition": [
        "Accurately identifies currency notes",
        "Recognition is fast and reliable",
        "Correctly detected all my bills",
        "Object recognition works well",
        "Text reading is accurate",
        "Identifies money perfectly every time",
        "Recognition accuracy is impressive",
        "Detects currency quickly and correctly"
    ],
    "Independence and Support": [
        "Helps me be more independent",
        "Life-changing app for blind people",
        "No longer need help reading money",
        "Gives me freedom to shop alone",
        "Empowering for visually impaired users",
        "Feel more confident using this app",
        "Don't need to ask for help anymore",
        "Makes daily tasks much easier"
    ],
    "Technical Issues": [
        "App crashes frequently",
        "Very slow to load",
        "Runs smoothly on my phone",
        "Battery drain is high",
        "Freezes during use",
        "Fast and responsive",
        "Needs performance improvements",
        "Works without lag"
    ],
    "User Experience": [
        "Easy to navigate",
        "Interface is intuitive",
        "Confusing button layout",
        "Simple and straightforward",
        "Hard to find settings",
        "User-friendly design",
        "Navigation is difficult",
        "Clean and easy interface"
    ]
}

print("Themes defined:")
for i, theme in enumerate(THEMES.keys(), 1):
    print(f"  {i}. {theme}")
print()

# Create training data from theme examples
print("Creating training examples from theme definitions...")
training_texts = []
training_labels = []

for theme, examples in THEMES.items():
    training_texts.extend(examples)
    training_labels.extend([theme] * len(examples))

print(f"[OK] Created {len(training_texts)} training examples\n")

# Encode training examples
if 'embedding_model' in locals():
    training_embeddings = embedding_model.encode(training_texts, show_progress_bar=False)
else:
    training_embeddings = vectorizer.transform(training_texts).toarray()
training_embeddings = np.array(training_embeddings)

# Encode labels
le = LabelEncoder()
training_labels_encoded = le.fit_transform(training_labels)

print(f"[OK] Training data prepared\n")

# ============================================================================
# STEP 4: Train Classifier
# ============================================================================
print("Step 4: Training classifier...")

if TABPFN_AVAILABLE:
    print("Using TabPFN Classifier...")
    try:
        # Initialize TabPFN - newer versions may have different parameters
        clf = TabPFNClassifier(device='cpu', N_ensemble_configurations=32)
        
        print("Fitting classifier to training examples...")
        clf.fit(training_embeddings, training_labels_encoded)
        print("[OK] TabPFN Classifier trained successfully\n")
        
    except Exception as e:
        print(f"[WARNING] TabPFN failed: {e}")
        print("Falling back to Random Forest...")
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(training_embeddings, training_labels_encoded)
        print("[OK] Random Forest Classifier trained successfully\n")
else:
    print("Using Random Forest Classifier...")
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(training_embeddings, training_labels_encoded)
    print("[OK] Random Forest Classifier trained successfully\n")

# ============================================================================
# STEP 5: Classify All Reviews
# ============================================================================
print("Step 5: Classifying all reviews...")
print(f"Processing {len(df)} reviews...\n")

try:
    # Predict themes for all reviews
    predictions_encoded = clf.predict(embeddings)
    predicted_themes = le.inverse_transform(predictions_encoded)
    
    # Add predictions to dataframe
    df['predicted_theme'] = predicted_themes
    
    # Get prediction probabilities if available
    try:
        probabilities = clf.predict_proba(embeddings)
        confidence_scores = np.max(probabilities, axis=1)
        df['confidence'] = confidence_scores
    except:
        df['confidence'] = 1.0  # Default confidence
    
    print("[OK] Classification complete!\n")
    
except Exception as e:
    print(f"[ERROR] Classification failed: {e}")
    exit(1)

# ============================================================================
# STEP 6: Generate Analysis Report
# ============================================================================
print("="*80)
print("THEMATIC ANALYSIS REPORT")
print("="*80 + "\n")

# Overall theme distribution
print("1. THEME DISTRIBUTION ACROSS ALL REVIEWS")
print("-"*80)
theme_counts = df['predicted_theme'].value_counts()
for theme, count in theme_counts.items():
    percentage = (count / len(df)) * 100
    bar = '#' * int(percentage / 2)
    print(f"  {theme:35} : {count:4} ({percentage:5.1f}%) {bar}")

# Average confidence
if 'confidence' in df.columns:
    avg_confidence = df['confidence'].mean()
    print(f"\n  Average Classification Confidence: {avg_confidence:.3f}\n")

# Sample reviews per theme
print("\n2. SAMPLE REVIEWS FOR EACH THEME")
print("-"*80)

for theme in THEMES.keys():
    theme_df = df[df['predicted_theme'] == theme]
    if len(theme_df) > 0:
        print(f"\n  {theme} ({len(theme_df)} reviews):")
        samples = theme_df.sample(min(2, len(theme_df)))[REVIEW_COLUMN]
        for i, review in enumerate(samples, 1):
            preview = review[:100] + "..." if len(review) > 100 else review
            print(f"    {i}. \"{preview}\"")

# ============================================================================
# STEP 7: Save Results
# ============================================================================
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

csv_output = "quantitative.csv"

try:
    df.to_csv(csv_output, index=False, encoding='utf-8')
    print(f"[OK] CSV saved: {csv_output}")
except Exception as e:
    print(f"[ERROR] Could not save CSV: {e}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)


def predict_themes_df(reviews_df, review_column='content'):
    """Predict themes for a DataFrame of reviews."""
    # Use global clf, embedding_model if available, else fallback

    # Prepare texts
    texts = reviews_df[review_column].astype(str).tolist()

    # Generate embeddings
    try:
        embeddings = embedding_model.encode(texts, show_progress_bar=False)
    except:
        # Fallback to TF-IDF
        embeddings = vectorizer.transform(texts).toarray()

    # Predict
    predictions_encoded = clf.predict(embeddings)
    predicted_themes = le.inverse_transform(predictions_encoded)

    # Add to df
    reviews_df['predicted_theme'] = predicted_themes

    try:
        probabilities = clf.predict_proba(embeddings)
        confidence_scores = np.max(probabilities, axis=1)
        reviews_df['confidence'] = confidence_scores
    except:
        reviews_df['confidence'] = 1.0

    return reviews_df[[review_column, 'predicted_theme', 'confidence']]


# Main execution happens at module import
