# detector.py - Use trained DistilBERT model for sentiment analysis
import pandas as pd
import numpy as np
import os
import sys
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import joblib
import re

# Disable TensorFlow
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['USE_TF'] = '0'

class SentimentDetector:
    def __init__(self, model_dir='./distilbert_sentiment_model_final'):
        """Initialize with trained DistilBERT model"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model info
        if os.path.exists('model_info.pkl'):
            model_info = joblib.load('model_info.pkl')
            model_dir = model_info.get('model_path', model_dir)
            self.label_map = model_info.get('label_map', {0: 'Negative', 1: 'Positive', 2: 'Neutral'})
        else:
            self.label_map = {0: 'Negative', 1: 'Positive', 2: 'Neutral'}
        
        # Load tokenizer and model
        if not os.path.exists(model_dir):
            raise FileNotFoundError(
                f"Model not found at {model_dir}. Please train the model first using train.py"
            )
        
        print(f"Loading model from {model_dir}...")
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()
        print("✓ Model loaded successfully")
    
    def clean_text(self, text):
        """Clean text while preserving Bengali characters"""
        if isinstance(text, float) or text is None:
            return ""
        text = str(text)
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:250]
    
    def load_data(self, file_path=None):
        """Load CSV file(s) from data folder"""
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        
        if file_path:
            if os.path.isabs(file_path):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_csv(os.path.join(data_dir, file_path))
            return df
        else:
            file_names = [
                'khabarkoi.csv', 'khaodao.csv', 'coopers.csv',
                'pizzahut.csv', 'proyojon.csv', 'sultansdine.csv'
            ]
            
            all_data = []
            for file_name in file_names:
                try:
                    file_path = os.path.join(data_dir, file_name)
                    if os.path.exists(file_path):
                        df = pd.read_csv(file_path)
                        print(f"✓ Loaded {file_name}: {len(df)} reviews")
                        all_data.append(df)
                except Exception as e:
                    print(f"✗ Error loading {file_name}: {e}")
            
            if all_data:
                combined_df = pd.concat(all_data, ignore_index=True)
                combined_df['cleaned_text'] = combined_df['content'].apply(self.clean_text)
                combined_df = combined_df[combined_df['cleaned_text'].str.len() > 0]
                return combined_df
            else:
                raise FileNotFoundError("No data files found in data folder")
    
    def predict_sentiment(self, text):
        """Predict sentiment for a single text"""
        cleaned_text = self.clean_text(text)
        
        if len(cleaned_text) == 0:
            return {
                'text': text,
                'sentiment': 'Neutral',
                'confidence': 0.5,
                'polarity': 0.0
            }
        
        try:
            # Tokenize
            encoding = self.tokenizer(
                cleaned_text,
                truncation=True,
                padding='max_length',
                max_length=128,
                return_tensors='pt'
            )
            
            # Move to device
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                prediction = torch.argmax(probabilities, dim=-1).item()
                confidence = probabilities[0][prediction].item()
            
            sentiment = self.label_map[prediction]
            
            return {
                'text': text,
                'sentiment': sentiment,
                'confidence': confidence,
                'polarity': 1.0 if sentiment == 'Positive' else (-1.0 if sentiment == 'Negative' else 0.0),
                'probabilities': {
                    'negative': probabilities[0][0].item(),
                    'positive': probabilities[0][1].item(),
                    'neutral': probabilities[0][2].item()
                }
            }
        except Exception as e:
            return {
                'text': str(text)[:200] if text else '',
                'sentiment': 'Neutral',
                'confidence': 0.5,
                'polarity': 0.0
            }
    
    def analyze_dataframe(self, df, text_column='content'):
        """Analyze sentiment for entire dataframe"""
        import gc
        
        print(f"\n📊 Analyzing {len(df)} reviews...")
        
        results = []
        texts = df[text_column].tolist() if text_column in df.columns else df.iloc[:, 0].tolist()
        
        for i, text in enumerate(texts):
            result = self.predict_sentiment(text)
            results.append({
                'content': result['text'][:200] if isinstance(result['text'], str) else str(result['text'])[:200],
                'sentiment': result['sentiment'],
                'confidence': result['confidence'],
                'polarity': result['polarity']
            })
            
            # Progress
            if (i + 1) % 50 == 0 or (i + 1) == len(texts):
                print(f"  Processed {i + 1}/{len(texts)} reviews... ({(i+1)*100//len(texts)}%)")
            
            # Cleanup every 100 reviews
            if (i + 1) % 100 == 0:
                gc.collect()
        
        result_df = pd.DataFrame(results)
        print(f"\n✓ Analysis complete!")
        
        gc.collect()
        return result_df

