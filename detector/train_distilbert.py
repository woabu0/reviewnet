# train_distilbert.py - Train sentiment model using DistilBERT
from sentiment_detector_distilbert import SentimentDetectorDistilBERT

def main():
    print("🚀 Starting DistilBERT Sentiment Detector Training...")
    
    # Initialize detector
    detector = SentimentDetectorDistilBERT()
    
    # Load data from all CSV files
    print("\n📁 Loading data files...")
    df = detector.load_data()
    print(f"✓ Total reviews loaded: {len(df)}")
    
    # Add sentiment labels
    print("\n🎯 Adding sentiment labels...")
    df = detector.add_sentiment_labels(df)
    
    # Show distribution
    sentiment_counts = df['sentiment'].value_counts().sort_index()
    print(f"\n📊 Sentiment Distribution:")
    print(f"   Positive (1): {sentiment_counts.get(1, 0)}")
    print(f"   Negative (0): {sentiment_counts.get(0, 0)}")
    print(f"   Neutral (2):  {sentiment_counts.get(2, 0)}")
    
    # Train the model
    print("\n🤖 Training DistilBERT model...")
    X_test, y_test, y_pred = detector.train_model(df)
    
    # Save the model
    detector.save_model('distilbert_sentiment_model')
    print("\n✅ Training completed! Model saved as 'distilbert_sentiment_model'")

if __name__ == "__main__":
    main()

