# train.py - Train DistilBERT model with improved multilingual sentiment labeling
import pandas as pd
import numpy as np
import os
import sys
import torch
from torch.utils.data import Dataset
from transformers import (
    DistilBertTokenizer, 
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import re
import matplotlib.pyplot as plt
import json

# Disable TensorFlow
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['USE_TF'] = '0'

class SentimentDataset(Dataset):
    """Dataset for DistilBERT sentiment classification"""
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors=None
        )
        
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels': label
        }

def clean_text(text):
    """Clean text while preserving Bengali characters"""
    if pd.isna(text) or text is None:
        return ""
    text = str(text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:250]

def detect_sentiment_improved(text):
    """
    Improved sentiment detection that properly handles Bengali, Banglish, and English
    """
    if not text or len(str(text).strip()) == 0:
        return 2  # Neutral
    
    text_str = str(text)
    text_lower = text_str.lower()
    
    # Bengali negative words and phrases (Unicode)
    bengali_negative = [
        'বাজে', 'খুব বাজে', 'অনেক বাজে', 'জঘন্য', 'ফালতু', 'খুব ফালতু',
        'বাজে সার্ভিস', 'জঘন্য সার্ভিস', 'ফালতু অ্যাপ', 'বাজে অভিজ্ঞতা',
        'খুব বাজে অভিজ্ঞতা', 'সমস্যা', 'অনেক সমস্যা', 'খুব সমস্যা',
        'নষ্ট', 'খারাপ', 'ভালো না', 'কাজ করে না',
        # Specific phrases from the data
        'ঢুকতে পারতেছি না', 'পারতেছি না', 'পারছি না',  # can't enter/get in
        'সার্ভিস এখন বন্ধ', 'সার্ভিস বন্ধ', 'বন্ধ করে দিয়েছে',  # service closed
        'ব্যবহার করে না', 'ব্যবহার করবে না',  # don't use
        'ডেলিভারিতে সমস্যা', 'ডেলিভারি সমস্যা',  # delivery problem
        'অনেক দেরি', 'খুব দেরি', 'দেরি হবে',  # very late, will be late
        'কাজ করে না', 'কাজ করছে না',  # doesn't work
        'পারছি না', 'পারতেছি না', 'পারব না'  # can't (do something)
    ]
    
    # Bengali positive words and phrases (Unicode)
    bengali_positive = [
        'ভালো', 'ভাল', 'জোস', 'সুন্দর', 'খুব ভালো', 'অনেক ভালো',
        'ভালো লাগল', 'ভালো সার্ভিস', 'জোস সার্ভিস', 'সুন্দর সার্ভিস',
        'ভালো কাজ', 'সন্তুষ্ট', 'খুব সন্তুষ্ট', 'আস্থার স্থান',
        # Specific phrases from the data
        'ভালো অ্যাপ', 'ভালো অ্যাপ পাইলাম',  # good app, found good app
        'সুস্বাদু', 'খুব সুস্বাদু',  # tasty, very tasty
        'ধন্যবাদ', 'ধন্যবাদ জানাই',  # thanks, thank you
        'গরম ছিল', 'গরম',  # was hot (positive for food)
        'আলহামদুলিল্লাহ',  # Alhamdulillah (positive expression)
        'পাইলাম', 'পেয়েছি'  # found, got (often positive context)
    ]
    
    # Banglish negative words (romanized Bengali)
    banglish_negative = [
        'baler', 'bal', 'baje', 'khub baje', 'onek baje', 'joghonno',
        'faltu', 'fokinni', 'khub faltu', 'onek faltu', 'faltu app',
        'baje service', 'khub baje service', 'joghonno service', 'baler app',
        'baje experience', 'khub baje experience', 'problem', 'onek problem',
        'khub problem', 'kaj kore na', 'thik moto kaj kore na', 'slow',
        'onk slow', 'kaj kore na onk slow', 'provide kore',  # "Baler service provide kore" = bad service
        'hudai', 'hudai dia', 'ta hudai',  # useless, waste
        'bolda app', 'bolda',  # bad app, bad
        'otp ashena', 'otp ase na', 'otp nei',  # OTP not coming
        'account khulbo', 'khulbo kivabe'  # how to open account (often negative context)
    ]
    
    # Banglish positive words
    banglish_positive = [
        'valo', 'bhalo', 'joss', 'khub valo', 'onek valo', 'valo laglo',
        'sundor', 'bhalo service', 'valo service', 'joss service', 'khub bhalo',
        'onek bhalo', 'valo kaj', 'bhalo kaj', 'khub satisfied', 'shubidha'
    ]
    
    # English negative words
    english_negative = [
        'bad', 'worst', 'horrible', 'terrible', 'slow', 'fake', 'scam',
        'waste', 'poor', 'rubbish', 'disappoint', 'disgusting', 'awful',
        'pathetic', 'useless', 'worthless', 'garbage', 'trash',
        'not working', 'does not work', 'not good', 'never again',
        'stay away', 'avoid', 'beware', 'cheat', 'fraud', 'problem',
        'issue', 'error', 'broken', 'bug', 'unavailable', 'unprofessional',
        'irresponsible', 'cannot', 'could not', 'can not', 'unable',
        'failed', 'failure', 'no response', 'no reply', 'never delivered',
        'dropped the call', 'no one picked up', 'left in the dark',
        'absurd', 'garbage app', 'trash app', 'miserable', 'total waste'
    ]
    
    # English positive words
    english_positive = [
        'good', 'great', 'excellent', 'awesome', 'best', 'love', 'amazing',
        'fast', 'nice', 'wonderful', 'perfect', 'fantastic', 'outstanding',
        'satisfied', 'recommend', 'delicious', 'quality', 'friendly', 'helpful',
        'convenient', 'easy', 'smooth', 'quick', 'reliable', 'trustworthy'
    ]
    
    # Calculate scores
    pos_score = 0
    neg_score = 0
    
    # Check Bengali (Unicode) - check original text
    # Check for longer phrases first (more specific)
    for phrase in sorted(bengali_negative, key=len, reverse=True):
        if phrase in text_str:
            neg_score += 4  # Higher weight for longer, more specific phrases
            break  # Only count once per phrase
    
    for phrase in sorted(bengali_positive, key=len, reverse=True):
        if phrase in text_str:
            pos_score += 4
            break
    
    # Also check individual words if phrases didn't match
    for word in bengali_negative:
        if word in text_str and word not in [p for p in bengali_negative if len(p) > len(word)]:
            neg_score += 3
    for word in bengali_positive:
        if word in text_str and word not in [p for p in bengali_positive if len(p) > len(word)]:
            pos_score += 3
    
    # Check Banglish (romanized) - check lowercase
    for word in banglish_negative:
        if word in text_lower:
            neg_score += 2.5  # High weight for Banglish
    for word in banglish_positive:
        if word in text_lower:
            pos_score += 2.5
    
    # Check English
    for word in english_negative:
        if word in text_lower:
            if word in ['worst', 'horrible', 'terrible', 'disgusting', 'pathetic', 'awful']:
                neg_score += 2
            else:
                neg_score += 1.5
    
    for word in english_positive:
        if word in text_lower:
            if word in ['excellent', 'outstanding', 'perfect', 'fantastic', 'amazing', 'wonderful']:
                pos_score += 2
            else:
                pos_score += 1.5
    
    # Check for specific negative patterns
    negative_patterns = [
        'kaj kore na',  # doesn't work
        'thik moto kaj kore na',  # doesn't work properly
        'onk slow',  # very slow
        'baler service',  # bad service
        'provide kore',  # provides (often negative context in Banglish)
        'not working', 'does not work', 'not good', 'never again',
        'waste of time', 'west of time',  # waste of time (typo)
        'could not', 'cannot', 'can not', 'unable to',
        'no verification', 'never received', 'no message',
        'unprofessional', 'irresponsible', 'never delivered',
        'dropped the call', 'no one picked up', 'left in the dark',
        'everything unavailable', 'unavailable', 'could not even',
        'could not log', 'cannot log', 'unable to log'
    ]
    
    for pattern in negative_patterns:
        if pattern in text_lower:
            neg_score += 2
    
    # Check for positive patterns
    positive_patterns = [
        'very good', 'very nice', 'highly recommend', 'great experience',
        'khub valo', 'onek valo', 'valo laglo', 'bhalo laglo',
        'its better', "it's better", 'much better', 'way better'
    ]
    
    # Check for "better" in negative context
    if 'better' in text_lower:
        # Check if it's comparing negatively (e.g., "X is better than this")
        better_context = text_lower
        if any(phrase in better_context for phrase in ['than this', 'than it', 'than app', 'dominos better', 'foodpanda better']):
            neg_score += 2  # Negative comparison
        elif 'its better' in better_context or "it's better" in better_context or 'much better' in better_context:
            pos_score += 2  # Positive standalone
    
    for pattern in positive_patterns:
        if pattern in text_lower:
            pos_score += 2.5
    
    # Special case: "Baler service provide kore" = negative
    if 'baler' in text_lower and ('service' in text_lower or 'provide' in text_lower):
        neg_score += 3
    
    # Special case: "thik moto kaj kore na" = negative (doesn't work properly)
    if 'thik moto kaj kore na' in text_lower or ('kaj kore na' in text_lower and 'slow' in text_lower):
        neg_score += 3
    
    # Special cases for specific negative phrases
    if 'unprofessional' in text_lower or 'irresponsible' in text_lower:
        neg_score += 3
    if 'never delivered' in text_lower or 'food was never delivered' in text_lower:
        neg_score += 3
    if 'dropped the call' in text_lower or 'no one picked up' in text_lower:
        neg_score += 2
    if 'left in the dark' in text_lower or 'no response' in text_lower:
        neg_score += 2
    if 'could not even log' in text_lower or 'cannot log' in text_lower:
        neg_score += 2
    if 'no verification message' in text_lower or 'never received' in text_lower:
        neg_score += 2
    if 'everything unavailable' in text_lower or 'unavailable' in text_lower:
        neg_score += 1.5
    
    # "wow.. its better" = positive (standalone better)
    if 'wow' in text_lower and 'better' in text_lower and len(text_lower.split()) <= 5:
        pos_score += 2
    
    # Decision logic - prioritize Bengali/Banglish detection
    word_count = len(text_lower.split())
    
    # If we have Bengali/Banglish indicators, they take priority
    has_bengali_negative = any(phrase in text_str for phrase in bengali_negative)
    has_bengali_positive = any(phrase in text_str for phrase in bengali_positive)
    has_banglish_negative = any(word in text_lower for word in banglish_negative)
    has_banglish_positive = any(word in text_lower for word in banglish_positive)
    
    # Strong Bengali/Banglish indicators override other logic
    if has_bengali_negative or has_banglish_negative:
        if neg_score >= 1.5:  # Lower threshold for Bengali/Banglish
            return 0  # Negative
    
    if has_bengali_positive or has_banglish_positive:
        if pos_score >= 1.5:  # Lower threshold for Bengali/Banglish
            return 1  # Positive
    
    # For very short texts, be more lenient
    if word_count <= 3:
        min_threshold = 1.5
        min_diff = 0.5
    else:
        min_threshold = 1.0
        min_diff = 0.3
    
    # If we have strong indicators, use them
    if neg_score >= 2.0 and neg_score > pos_score:
        return 0  # Negative
    elif pos_score >= 2.0 and pos_score > neg_score:
        return 1  # Positive
    elif neg_score > pos_score and neg_score >= min_threshold:
        return 0  # Negative
    elif pos_score > neg_score and pos_score >= min_threshold:
        return 1  # Positive
    else:
        return 2  # Neutral

def add_sentiment_labels(df):
    """Add sentiment labels using improved multilingual detection"""
    print("\n🎯 Adding sentiment labels with improved multilingual detection...")
    df['sentiment'] = df['cleaned_text'].apply(detect_sentiment_improved)
    return df

def main():
    print("=== 🚀 Training DistilBERT Sentiment Model ===")
    
    # Load data
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    file_names = [
        'khabarkoi.csv', 'khaodao.csv', 'coopers.csv',
        'pizzahut.csv', 'proyojon.csv', 'sultansdine.csv'
    ]
    
    all_data = []
    for file_name in file_names:
        file_path = os.path.join(data_dir, file_name)
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                df['cleaned_text'] = df['content'].apply(clean_text)
                df = df[df['cleaned_text'].str.len() > 0]
                all_data.append(df)
                print(f"✓ Loaded {file_name}: {len(df)} reviews")
            except Exception as e:
                print(f"✗ Error loading {file_name}: {e}")
    
    if not all_data:
        print("❌ No data files found!")
        return
    
    # Combine data
    df = pd.concat(all_data, ignore_index=True)
    print(f"\n✓ Total reviews: {len(df)}")
    
    # Add labels with improved detection
    df = add_sentiment_labels(df)
    
    sentiment_counts = df['sentiment'].value_counts().sort_index()
    print(f"\n📊 Sentiment Distribution:")
    print(f"Positive: {sentiment_counts.get(1, 0)}")
    print(f"Negative: {sentiment_counts.get(0, 0)}")
    print(f"Neutral:  {sentiment_counts.get(2, 0)}")
    
    # Show some examples
    print("\n📝 Sample Labeled Reviews:")
    print("-" * 80)
    negative_samples = df[df['sentiment'] == 0].head(3)
    for idx, row in negative_samples.iterrows():
        print(f"Negative: {row['cleaned_text'][:70]}...")
    positive_samples = df[df['sentiment'] == 1].head(3)
    for idx, row in positive_samples.iterrows():
        print(f"Positive: {row['cleaned_text'][:70]}...")
    print("-" * 80)
    
    # Prepare data
    X = df['cleaned_text'].tolist()
    y = df['sentiment'].tolist()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📊 Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Initialize tokenizer and model
    print("\n🤖 Initializing DistilBERT...")
    model_name = 'distilbert-base-uncased'
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3
    )
    
    # Create datasets
    train_dataset = SentimentDataset(X_train, y_train, tokenizer)
    test_dataset = SentimentDataset(X_test, y_test, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir='./distilbert_sentiment_model',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=2,
        report_to="none",  # Disable wandb/tensorboard
    )
    
    # Compute metrics
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return {'accuracy': accuracy_score(labels, predictions)}
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Train
    print("\n🚀 Starting training...")
    print("This may take 15-25 minutes. Please wait...\n")
    
    trainer.train()
    
    # Plot training history
    try:
        # Get history from trainer state
        if hasattr(trainer.state, 'log_history') and trainer.state.log_history:
            # Extract data
            train_losses = []
            eval_losses = []
            eval_accs = []
            epochs = []
            
            for log in trainer.state.log_history:
                if 'loss' in log and 'eval_loss' not in log and 'epoch' in log:
                    train_losses.append(log['loss'])
                    epochs.append(int(log['epoch']))
                if 'eval_loss' in log:
                    eval_losses.append(log['eval_loss'])
                if 'eval_accuracy' in log:
                    eval_accs.append(log['eval_accuracy'])
            
            if train_losses or eval_losses:
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                
                # Plot losses
                if train_losses and epochs:
                    unique_epochs = sorted(set(epochs))
                    if len(train_losses) >= len(unique_epochs):
                        axes[0].plot(unique_epochs[:len(train_losses)], train_losses[:len(unique_epochs)], 
                                   label='Train Loss', marker='o', linewidth=2)
                if eval_losses:
                    eval_epochs = list(range(1, len(eval_losses) + 1))
                    axes[0].plot(eval_epochs, eval_losses, label='Eval Loss', marker='s', linewidth=2)
                
                axes[0].set_xlabel('Epoch', fontsize=12)
                axes[0].set_ylabel('Loss', fontsize=12)
                axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
                axes[0].legend(fontsize=11)
                axes[0].grid(True, alpha=0.3)
                
                # Plot accuracy
                if eval_accs:
                    acc_epochs = list(range(1, len(eval_accs) + 1))
                    axes[1].plot(acc_epochs, eval_accs, label='Validation Accuracy', marker='o', 
                               color='green', linewidth=2, markersize=8)
                    axes[1].set_xlabel('Epoch', fontsize=12)
                    axes[1].set_ylabel('Accuracy', fontsize=12)
                    axes[1].set_title('Validation Accuracy', fontsize=14, fontweight='bold')
                    axes[1].legend(fontsize=11)
                    axes[1].grid(True, alpha=0.3)
                    axes[1].set_ylim([0, 1])
                
                plt.tight_layout()
                plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
                print("✓ Training history saved to: training_history.png")
                plt.close()
    except Exception as e:
        print(f"⚠️  Could not create training plot: {e}")
        import traceback
        traceback.print_exc()
    
    # Evaluate
    print("\n📊 Evaluating model...")
    eval_results = trainer.evaluate()
    print(f"Test Accuracy: {eval_results['eval_accuracy']:.3f}")
    
    # Make predictions
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive', 'Neutral']))
    
    # Save model
    model_dir = './distilbert_sentiment_model_final'
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)
    
    # Save model info
    joblib.dump({
        'model_path': model_dir,
        'num_labels': 3,
        'label_map': {0: 'Negative', 1: 'Positive', 2: 'Neutral'}
    }, 'model_info.pkl')
    
    print(f"\n✅ Model saved to: {model_dir}")
    print("✅ Model info saved to: model_info.pkl")
    print("\n🎉 Training complete!")


if __name__ == "__main__":
    main()

