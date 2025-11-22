# train_model.py
from sentiment_detector import SentimentDetector
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def main():
    print("🚀 Starting Sentiment Detector Training...")
    
    # Initialize detector
    detector = SentimentDetector()
    
    # Load data from all CSV files
    print("\n📁 Loading data files...")
    df = detector.load_data()
    print(f"✓ Total reviews loaded: {len(df)}")
    
    # Add sentiment labels
    print("\n🎯 Adding sentiment labels...")
    df = detector.add_sentiment_labels(df)
    
    # Show sentiment distribution with plot
    print("\n📊 Plotting sentiment distribution...")
    sentiment_counts = detector.plot_sentiment_distribution(df)
    
    print(f"\n📈 Sentiment Counts:")
    print(f"   Positive: {sentiment_counts.get(1, 0)}")
    print(f"   Negative: {sentiment_counts.get(0, 0)}") 
    print(f"   Neutral:  {sentiment_counts.get(2, 0)}")
    
    # Train the model
    print("\n🤖 Training the model...")
    X_test, y_test, y_pred = detector.train_model(df)
    
    # Additional visualizations
    print("\n📊 Generating model visualizations...")
    
    # Feature importance
    print("1. Plotting feature importance...")
    detector.plot_feature_importance(top_n=15)
    
    # Confidence distribution
    print("2. Plotting confidence distribution...")
    test_texts = X_test.tolist()  # Convert back to list of texts
    detector.plot_confidence_distribution(test_texts, y_test, y_pred)
    
    # Model comparison (optional - takes longer)
    print("3. Comparing different models...")
    X = df['cleaned_text']
    y = df['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    detector.plot_model_comparison(X_train_tfidf, X_test_tfidf, y_train, y_test)
    
    # Save the model
    detector.save_model('trained_sentiment_model.pkl')
    print("\n✅ Training completed! Model saved as 'trained_sentiment_model.pkl'")

if __name__ == "__main__":
    main()