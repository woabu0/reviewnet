# sentiment_detector.py
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import joblib

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

class SentimentDetector:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """Clean and preprocess text data - supports English and Bengali"""
        if isinstance(text, float) or text is None:
            return ""
        
        text = str(text).lower()
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        # Add space before common words that might be concatenated
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Handle camelCase
        # Replace common concatenated patterns
        text = re.sub(r'(\w)(friendly|good|bad|nice|great|best|worst)', r'\1 \2', text)
        # Keep Bengali characters (Unicode range for Bengali: \u0980-\u09FF)
        # Remove special characters but keep Bengali, English letters, and spaces
        text = re.sub(r'[^\u0980-\u09FFa-zA-Z\s]', ' ', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        tokens = text.split()
        # Only lemmatize English words (NLTK doesn't support Bengali)
        processed_tokens = []
        for word in tokens:
            # Check if word is English (contains only ASCII letters)
            if word.isascii() and word.isalpha():
                if word not in self.stop_words and len(word) > 2:
                    lemmatized = self.lemmatizer.lemmatize(word)
                    processed_tokens.append(lemmatized)
            else:
                # Bengali or mixed - keep as is if length > 1
                if len(word) > 1:
                    processed_tokens.append(word)
        
        return ' '.join(processed_tokens)
    
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
    
    def add_sentiment_labels(self, df):
        """Improved sentiment labeling with expanded word lists and emoji detection"""
        # Expanded positive words and phrases (English + Bengali/Banglish)
        positive_words = [
            # English
            'good', 'great', 'excellent', 'awesome', 'best', 'love', 'amazing', 
            'fast', 'nice', 'wonderful', 'perfect', 'fantastic', 'outstanding',
            'satisfied', 'satisfaction', 'recommend', 'recommended', 'appreciate',
            'delicious', 'fresh', 'quality', 'friendly', 'helpful', 'convenient',
            'easy', 'smooth', 'quick', 'prompt', 'reliable', 'trustworthy',
            'superb', 'brilliant', 'marvelous', 'fabulous', 'super', 'top',
            'exceed', 'impress', 'pleased', 'happy', 'glad', 'enjoy', 'enjoyed',
            'delight', 'delighted', 'perfect', 'flawless', 'seamless', 'smooth',
            'user friendly', 'well done', 'thumbs up', 'five star', 'highly recommend',
            'timesaver', 'time saver', 'without stress', 'stress free', 'affordable',
            'value', 'worth', 'favorite', 'go to', 'pleased', 'satisfy', 'satisfying',
            # Bengali/Banglish positive words
            'valo', 'bhalo', 'joss', 'khub valo', 'onek valo', 'valo laglo',
            'sundor', 'bhalo service', 'valo service', 'joss service', 'khub bhalo',
            'onek bhalo', 'valo kaj', 'bhalo kaj', 'satisfied', 'khub satisfied',
            'shubidha', 'valo shubidha', 'bhalo shubidha', 'perfect', 'joss ui'
        ]
        
        # Expanded negative words and phrases (English + Bengali/Banglish)
        negative_words = [
            # English
            'bad', 'worst', 'horrible', 'terrible', 'slow', 'fake', 'scam', 
            'waste', 'poor', 'rubbish', 'disappoint', 'disgusting', 'awful',
            'pathetic', 'useless', 'worthless', 'garbage', 'trash', 'faltu',
            'fokinni', 'third class', 'unprofessional', 'irresponsible',
            'unavailable', 'broken', 'bug', 'error', 'issue', 'problem',
            'cheat', 'fraud', 'lie', 'false', 'mislead', 'unreliable',
            'incompetent', 'inadequate', 'unsatisfactory', 'unacceptable',
            'frustrat', 'annoy', 'angry', 'hate', 'dislike', 'regret',
            'never again', 'not recommend', 'stay away', 'avoid', 'beware',
            # Bengali/Banglish negative words
            'baje', 'khub baje', 'onek baje', 'joghonno', 'baler', 'bal',
            'faltu', 'fokinni', 'khub faltu', 'onek faltu', 'faltu app',
            'baje service', 'khub baje service', 'joghonno service', 'baler app',
            'waste of time', 'time waste', 'nasta', 'khub nasta', 'baje experience',
            'khub baje experience', 'problem', 'onek problem', 'khub problem',
            'scam', 'fraud', 'cheat', 'baje kaj', 'khub baje kaj'
        ]
        
        # Positive emojis
        positive_emojis = ['❤️', '❤', '👍', '👌', '😊', '😍', '🥰', '😁', '✅', '⭐', '🌟', '💯']
        
        # Negative emojis
        negative_emojis = ['😡', '👎', '😞', '😢', '😠', '🤬', '❌', '💔']
        
        def detect_sentiment(text):
            """Improved sentiment detection with scoring system"""
            if not text or len(text.strip()) == 0:
                return 2  # Neutral for empty text
            
            text_lower = text.lower()
            original_text = text  # Keep original for emoji detection
            
            # For very short texts (less than 3 words), require stronger evidence
            word_count = len(text_lower.split())
            is_short_text = word_count < 3
            
            # Bengali positive words (Unicode)
            bengali_positive = ['ভালো', 'ভাল', 'জোস', 'সুন্দর', 'খুব ভালো', 'অনেক ভালো', 
                              'ভালো লাগল', 'ভালো সার্ভিস', 'জোস সার্ভিস', 'সুন্দর সার্ভিস',
                              'খুব ভালো', 'অনেক ভালো', 'ভালো কাজ', 'সন্তুষ্ট', 'খুব সন্তুষ্ট']
            
            # Bengali negative words (Unicode)
            bengali_negative = ['বাজে', 'খুব বাজে', 'অনেক বাজে', 'জঘন্য', 'ফালতু', 'খুব ফালতু',
                              'বাজে সার্ভিস', 'জঘন্য সার্ভিস', 'ফালতু অ্যাপ', 'বাজে অভিজ্ঞতা',
                              'খুব বাজে অভিজ্ঞতা', 'সমস্যা', 'অনেক সমস্যা', 'খুব সমস্যা']
            
            # Calculate positive score
            pos_score = 0
            for word in positive_words:
                if word in text_lower:
                    # Give more weight to stronger positive words
                    if word in ['excellent', 'outstanding', 'perfect', 'fantastic', 'amazing', 'wonderful']:
                        pos_score += 2
                    elif word in ['great', 'awesome', 'best', 'love', 'superb', 'brilliant']:
                        pos_score += 1.5
                    else:
                        pos_score += 1
            
            # Check for Bengali positive words
            for word in bengali_positive:
                if word in text:
                    pos_score += 2  # Bengali words get higher weight
            
            # Check for positive phrases (English + Bengali/Banglish)
            positive_phrases = [
                # English
                'very good', 'very nice', 'very well', 'so good', 'really good',
                'highly recommend', 'strongly recommend', 'well done', 'good job',
                'keep it up', 'thumbs up', 'five star', 'top notch', 'time saver',
                'timesaver', 'without stress', 'stress free', 'great experience',
                'best experience', 'love it', 'love this', 'my favorite', 'go to app',
                # Bengali/Banglish
                'khub valo', 'onek valo', 'khub bhalo', 'onek bhalo', 'valo laglo',
                'bhalo laglo', 'khub sundor', 'valo service', 'bhalo service',
                'joss service', 'valo kaj', 'bhalo kaj', 'khub shubidha', 'valo shubidha'
            ]
            for phrase in positive_phrases:
                if phrase in text_lower:
                    pos_score += 2.5  # Increased weight for phrases
            
            # Calculate negative score
            neg_score = 0
            for word in negative_words:
                if word in text_lower:
                    # Give more weight to stronger negative words
                    if word in ['worst', 'horrible', 'terrible', 'disgusting', 'pathetic', 'awful']:
                        neg_score += 2
                    elif word in ['bad', 'poor', 'rubbish', 'useless', 'disappoint']:
                        neg_score += 1.5
                    else:
                        neg_score += 1
            
            # Check for Bengali negative words
            for word in bengali_negative:
                if word in text:
                    neg_score += 2  # Bengali words get higher weight
            
            # Check for negative phrases (English + Bengali/Banglish)
            negative_phrases = [
                # English
                'very bad', 'so bad', 'not good', 'not working', 'does not work',
                'never again', 'stay away', 'waste of', 'not recommend', 'worst ever',
                # Bengali/Banglish
                'khub baje', 'onek baje', 'khub faltu', 'onek faltu', 'faltu app',
                'baje service', 'joghonno service', 'baler app', 'khub problem',
                'onek problem', 'time waste', 'waste of time', 'baje experience',
                'khub baje experience', 'not recommend', 'never order', 'ar order korbo na'
            ]
            for phrase in negative_phrases:
                if phrase in text_lower:
                    neg_score += 2
            
            # Check for emojis in original text (before cleaning)
            for emoji in positive_emojis:
                if emoji in original_text:
                    pos_score += 1.5
            
            for emoji in negative_emojis:
                if emoji in original_text:
                    neg_score += 1.5
            
            # Check for intensifiers that modify sentiment
            intensifiers = ['very', 'really', 'extremely', 'absolutely', 'totally', 'completely', 'super']
            for intensifier in intensifiers:
                if intensifier in text_lower:
                    # Check if it's modifying a positive or negative word nearby
                    words = text_lower.split()
                    for i, word in enumerate(words):
                        if intensifier in word or (i > 0 and words[i-1] == intensifier):
                            # Check surrounding words
                            context = ' '.join(words[max(0, i-2):min(len(words), i+3)])
                            if any(pw in context for pw in ['good', 'great', 'nice', 'excellent', 'amazing']):
                                pos_score += 0.5
                            elif any(nw in context for nw in ['bad', 'poor', 'terrible', 'horrible']):
                                neg_score += 0.5
            
            # Decision logic with balanced threshold
            # For short texts, require stronger evidence
            min_threshold = 2.0 if is_short_text else 1.5
            
            # Also check if the difference is significant enough
            score_diff = abs(pos_score - neg_score)
            min_diff = 0.8 if is_short_text else 0.5
            
            # Special handling for texts with strong positive words
            strong_positive_words = ['enjoy', 'delicious', 'love', 'amazing', 'excellent', 'wonderful', 
                                     'fantastic', 'outstanding', 'perfect', 'brilliant', 'superb']
            has_strong_positive = any(word in text_lower for word in strong_positive_words)
            
            # Special handling for positive phrases
            positive_phrase_patterns = ['time saver', 'timesaver', 'without stress', 'stress free', 
                                       'highly recommend', 'strongly recommend', 'great experience']
            has_positive_phrase = any(phrase in text_lower for phrase in positive_phrase_patterns)
            
            # For very short texts (1-2 words) with clear positive words, be more lenient
            very_short_positive_words = ['good', 'great', 'nice', 'wonderful', 'excellent', 'amazing', 
                                         'awesome', 'best', 'love', 'perfect', 'fantastic']
            is_very_short = word_count <= 2
            has_very_short_positive = is_very_short and any(word in text_lower.split() for word in very_short_positive_words)
            
            if has_very_short_positive:
                # For very short texts with clear positive words, classify as positive
                if pos_score >= 1.0 and pos_score > neg_score:
                    return 1  # Positive
            
            if (has_strong_positive or has_positive_phrase) and pos_score >= 1.5:
                # If we have strong positive indicators, be more lenient
                min_threshold = max(1.0, min_threshold - 0.5)
                min_diff = max(0.3, min_diff - 0.3)
            
            if pos_score > neg_score and pos_score >= min_threshold and score_diff >= min_diff:
                return 1  # Positive
            elif neg_score > pos_score and neg_score >= min_threshold and score_diff >= min_diff:
                return 0  # Negative
            else:
                return 2  # Neutral
        
        # Apply to original content before cleaning for better emoji detection
        # But also check cleaned text for word matching
        df['sentiment'] = df.apply(
            lambda row: detect_sentiment(row.get('content', '') if 'content' in row else row.get('cleaned_text', '')),
            axis=1
        )
        return df
    
    def train_model(self, df):
        """Train the sentiment detection model"""
        X = df['cleaned_text']
        y = df['sentiment']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Increased features and improved n-gram range for better context capture
        self.vectorizer = TfidfVectorizer(
            max_features=10000,  # Increased from 5000 to capture more features
            ngram_range=(1, 3),  # Increased to 3-grams to capture more context
            min_df=2,  # Ignore terms that appear in less than 2 documents
            max_df=0.95,  # Ignore terms that appear in more than 95% of documents
            sublinear_tf=True  # Apply sublinear tf scaling
        )
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        # Use ensemble of models for better accuracy and robustness
        lr = LogisticRegression(
            random_state=42, 
            max_iter=2000,
            class_weight='balanced',
            C=1.0,
            solver='lbfgs',
            multi_class='multinomial'
        )
        
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        nb = MultinomialNB(alpha=0.1)
        
        # Use Voting Classifier for ensemble
        self.model = VotingClassifier(
            estimators=[('lr', lr), ('rf', rf), ('nb', nb)],
            voting='soft',  # Use probability-based voting
            weights=[2, 2, 1]  # Give more weight to LR and RF
        )
        self.model.fit(X_train_tfidf, y_train)
        
        # Evaluate
        train_pred = self.model.predict(X_train_tfidf)
        test_pred = self.model.predict(X_test_tfidf)
        
        print("=== Model Performance ===")
        print(f"Training Accuracy: {accuracy_score(y_train, train_pred):.3f}")
        print(f"Testing Accuracy: {accuracy_score(y_test, test_pred):.3f}")
        print("\n=== Classification Report ===")
        print(classification_report(y_test, test_pred, target_names=['Negative', 'Positive', 'Neutral']))
        
        return X_test, y_test, test_pred
    
    def predict_sentiment(self, text):
        """Predict sentiment for new text"""
        if self.model is None:
            raise Exception("Model not trained yet!")
        
        cleaned_text = self.clean_text(text)
        text_tfidf = self.vectorizer.transform([cleaned_text])
        prediction = self.model.predict(text_tfidf)[0]
        probability = self.model.predict_proba(text_tfidf)[0]
        
        sentiment_map = {0: 'Negative', 1: 'Positive', 2: 'Neutral'}
        
        return {
            'text': text,
            'sentiment': sentiment_map[prediction],
            'confidence': max(probability),
            'probabilities': {
                'negative': probability[0],
                'positive': probability[1],
                'neutral': probability[2]
            }
        }
    
    def save_model(self, filepath):
        """Save the trained model"""
        model_data = {'model': self.model, 'vectorizer': self.vectorizer}
        joblib.dump(model_data, filepath)
        print(f"✓ Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.vectorizer = model_data['vectorizer']
        print(f"✓ Model loaded from {filepath}")
    
    def plot_sentiment_distribution(self, df):
        """Plot and return sentiment distribution"""
        sentiment_counts = df['sentiment'].value_counts().sort_index()
        
        plt.figure(figsize=(10, 6))
        sentiment_labels = ['Negative', 'Positive', 'Neutral']
        colors = ['#ff4444', '#44ff44', '#4444ff']
        
        bars = plt.bar(sentiment_labels, [sentiment_counts.get(i, 0) for i in range(3)], 
                      color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        plt.title('Sentiment Distribution', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Sentiment', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('sentiment_distribution.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: sentiment_distribution.png")
        plt.close()
        
        return sentiment_counts.to_dict()
    
    def plot_feature_importance(self, top_n=15):
        """Plot top N most important features"""
        if self.model is None or self.vectorizer is None:
            raise Exception("Model not trained yet!")
        
        # Get feature names
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Handle different model types
        if hasattr(self.model, 'coef_'):
            # Direct model (LogisticRegression, etc.)
            coefficients = self.model.coef_
            avg_importance = np.mean(np.abs(coefficients), axis=0)
        elif hasattr(self.model, 'named_estimators_'):
            # VotingClassifier ensemble
            importances_list = []
            # Try to get feature importance from each estimator
            for name, estimator in self.model.named_estimators_.items():
                if hasattr(estimator, 'coef_'):
                    # LogisticRegression
                    coef = estimator.coef_
                    importances_list.append(np.mean(np.abs(coef), axis=0))
                elif hasattr(estimator, 'feature_importances_'):
                    # RandomForest
                    importances_list.append(estimator.feature_importances_)
                # NaiveBayes doesn't have feature importance, skip it
            
            if importances_list:
                # Average importance across all estimators
                avg_importance = np.mean(importances_list, axis=0)
            else:
                # Fallback: use LogisticRegression if available
                if 'lr' in self.model.named_estimators_:
                    lr = self.model.named_estimators_['lr']
                    if hasattr(lr, 'coef_'):
                        coefficients = lr.coef_
                        avg_importance = np.mean(np.abs(coefficients), axis=0)
                    else:
                        raise Exception("Cannot extract feature importance from ensemble model")
                else:
                    raise Exception("Cannot extract feature importance from ensemble model")
        elif hasattr(self.model, 'feature_importances_'):
            # RandomForest or similar
            avg_importance = self.model.feature_importances_
        else:
            raise Exception("Model type not supported for feature importance extraction")
        
        # Get top N features
        top_indices = np.argsort(avg_importance)[-top_n:][::-1]
        top_features = [feature_names[i] for i in top_indices]
        top_importance = [avg_importance[i] for i in top_indices]
        
        plt.figure(figsize=(10, 8))
        bars = plt.barh(range(len(top_features)), top_importance, color='steelblue', alpha=0.7)
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Average Absolute Coefficient', fontsize=12)
        plt.title(f'Top {top_n} Most Important Features', fontsize=16, fontweight='bold', pad=20)
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, top_importance)):
            plt.text(val, i, f' {val:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: feature_importance.png")
        plt.close()
    
    def plot_confidence_distribution(self, test_texts, y_test, y_pred):
        """Plot confidence distribution for predictions"""
        if self.model is None:
            raise Exception("Model not trained yet!")
        
        # Get confidence scores
        confidences = []
        for text in test_texts:
            cleaned = self.clean_text(text)
            text_tfidf = self.vectorizer.transform([cleaned])
            proba = self.model.predict_proba(text_tfidf)[0]
            confidences.append(max(proba))
        
        confidences = np.array(confidences)
        
        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(confidences, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Confidence Score', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Confidence Distribution', fontsize=14, fontweight='bold')
        axes[0].grid(alpha=0.3, linestyle='--')
        axes[0].axvline(confidences.mean(), color='red', linestyle='--', 
                        label=f'Mean: {confidences.mean():.3f}')
        axes[0].legend()
        
        # Confidence by correctness
        correct = (y_test == y_pred)
        correct_conf = confidences[correct]
        incorrect_conf = confidences[~correct]
        
        axes[1].hist([correct_conf, incorrect_conf], bins=20, alpha=0.7, 
                    color=['green', 'red'], label=['Correct', 'Incorrect'], 
                    edgecolor='black')
        axes[1].set_xlabel('Confidence Score', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title('Confidence by Prediction Correctness', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig('confidence_distribution.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: confidence_distribution.png")
        plt.close()
    
    def plot_model_comparison(self, X_train_tfidf, X_test_tfidf, y_train, y_test):
        """Compare different models"""
        from sklearn.linear_model import LogisticRegression
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.svm import SVC
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, f1_score
        
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
            'Naive Bayes': MultinomialNB(),
            'SVM': SVC(random_state=42, class_weight='balanced', probability=True),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        }
        
        results = []
        for name, model in models.items():
            model.fit(X_train_tfidf, y_train)
            train_pred = model.predict(X_train_tfidf)
            test_pred = model.predict(X_test_tfidf)
            
            train_acc = accuracy_score(y_train, train_pred)
            test_acc = accuracy_score(y_test, test_pred)
            test_f1 = f1_score(y_test, test_pred, average='weighted')
            
            results.append({
                'Model': name,
                'Train Accuracy': train_acc,
                'Test Accuracy': test_acc,
                'Test F1': test_f1
            })
        
        results_df = pd.DataFrame(results)
        
        # Plot comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        x = range(len(results_df))
        width = 0.35
        
        axes[0].bar([i - width/2 for i in x], results_df['Train Accuracy'], 
                   width, label='Train', alpha=0.7, color='steelblue')
        axes[0].bar([i + width/2 for i in x], results_df['Test Accuracy'], 
                   width, label='Test', alpha=0.7, color='coral')
        axes[0].set_xlabel('Model', fontsize=12)
        axes[0].set_ylabel('Accuracy', fontsize=12)
        axes[0].set_title('Model Comparison: Accuracy', fontsize=14, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(results_df['Model'], rotation=45, ha='right')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3, linestyle='--')
        axes[0].set_ylim([0, 1])
        
        axes[1].bar(results_df['Model'], results_df['Test F1'], 
                   alpha=0.7, color='green', edgecolor='black')
        axes[1].set_xlabel('Model', fontsize=12)
        axes[1].set_ylabel('F1 Score (Weighted)', fontsize=12)
        axes[1].set_title('Model Comparison: F1 Score', fontsize=14, fontweight='bold')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(axis='y', alpha=0.3, linestyle='--')
        axes[1].set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: model_comparison.png")
        plt.close()
        
        print("\n=== Model Comparison Results ===")
        print(results_df.to_string(index=False))