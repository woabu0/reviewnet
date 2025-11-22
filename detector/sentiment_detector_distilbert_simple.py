# sentiment_detector_distilbert_simple.py - Simplified DistilBERT using pipeline
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import os

# Disable TensorFlow
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['USE_TF'] = '0'

try:
    from transformers import pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  transformers not available. Please install: pip install transformers torch")

class SentimentDetectorDistilBERTSimple:
    def __init__(self, model_name='distilbert-base-uncased-finetuned-sst-2-english'):
        """
        Initialize DistilBERT-based sentiment detector using pipeline
        
        Args:
            model_name: Hugging Face model name
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library not available")
        
        self.device = 0 if torch.cuda.is_available() else -1
        print(f"Using device: {'CUDA' if self.device == 0 else 'CPU'}")
        
        # Use a pre-trained sentiment analysis model
        # This model outputs positive/negative, we'll map to 3 classes
        try:
            self.classifier = pipeline(
                "sentiment-analysis",
                model=model_name,
                device=self.device
            )
            print(f"✓ Loaded model: {model_name}")
        except Exception as e:
            print(f"⚠️  Error loading {model_name}, trying default model...")
            self.classifier = pipeline(
                "sentiment-analysis",
                device=self.device
            )
        
        self.label_map = {0: 'Negative', 1: 'Positive', 2: 'Neutral'}
    
    def clean_text(self, text):
        """Clean and preprocess text data"""
        if isinstance(text, float) or text is None:
            return ""
        
        text = str(text)
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def load_data(self):
        """Load all CSV files"""
        file_names = [
            'khabarkoi.csv',
            'khaodao.csv', 
            'coopers.csv',
            'pizzahut.csv',
            'proyojon.csv',
            'sultansdine.csv'
        ]
        
        all_data = []
        for file_name in file_names:
            try:
                df = pd.read_csv(f'data/{file_name}')
                print(f"✓ Loaded {file_name}: {len(df)} reviews")
                all_data.append(df)
            except Exception as e:
                print(f"✗ Error loading {file_name}: {e}")
        
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df['cleaned_text'] = combined_df['content'].apply(self.clean_text)
        combined_df = combined_df[combined_df['cleaned_text'].str.len() > 0]
        
        return combined_df
    
    def predict_sentiment(self, text, return_confidence=True):
        """
        Predict sentiment for a single text
        
        The model returns POSITIVE/NEGATIVE, we map to 3 classes:
        - If confidence > 0.7: use the label directly
        - If confidence < 0.7: classify as Neutral
        """
        cleaned_text = self.clean_text(text)
        
        if len(cleaned_text) == 0:
            return {
                'text': text,
                'sentiment': 'Neutral',
                'confidence': 0.5,
                'probabilities': {'negative': 0.33, 'positive': 0.33, 'neutral': 0.34}
            }
        
        try:
            # Truncate to 512 tokens (model limit)
            if len(cleaned_text) > 500:
                cleaned_text = cleaned_text[:500]
            
            result = self.classifier(cleaned_text)[0]
            label = result['label']
            score = result['score']
            
            # Map to 3 classes
            if score < 0.6:  # Low confidence -> Neutral
                sentiment = 'Neutral'
                confidence = 1.0 - score  # Invert for neutral
                probabilities = {
                    'negative': (1 - score) / 3 if label == 'NEGATIVE' else (1 - score) / 3,
                    'positive': (1 - score) / 3 if label == 'POSITIVE' else (1 - score) / 3,
                    'neutral': score
                }
            else:
                # High confidence -> use the label
                if label == 'POSITIVE' or label == 'LABEL_1':
                    sentiment = 'Positive'
                    probabilities = {
                        'negative': (1 - score) / 2,
                        'positive': score,
                        'neutral': (1 - score) / 2
                    }
                else:  # NEGATIVE or LABEL_0
                    sentiment = 'Negative'
                    probabilities = {
                        'negative': score,
                        'positive': (1 - score) / 2,
                        'neutral': (1 - score) / 2
                    }
                confidence = score
            
            return {
                'text': text,
                'sentiment': sentiment,
                'confidence': confidence,
                'probabilities': probabilities
            }
        except Exception as e:
            print(f"Error predicting sentiment: {e}")
            return {
                'text': text,
                'sentiment': 'Neutral',
                'confidence': 0.5,
                'probabilities': {'negative': 0.33, 'positive': 0.33, 'neutral': 0.34}
            }
    
    def predict_batch(self, texts, batch_size=32):
        """Predict sentiment for multiple texts"""
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_results = [self.predict_sentiment(text) for text in batch]
            results.extend(batch_results)
        return results

