# analyze_results.py
from sentiment_detector import SentimentDetector
import pandas as pd

def analyze_sample_predictions(detector, test_texts, actual_sentiments, num_samples=10):
    """Analyze and display sample predictions"""
    print("\n" + "="*60)
    print("📋 SAMPLE PREDICTION ANALYSIS")
    print("="*60)
    
    correct = 0
    for i, (text, actual) in enumerate(zip(test_texts[:num_samples], actual_sentiments[:num_samples])):
        result = detector.predict_sentiment(text)
        actual_label = ['Negative', 'Positive', 'Neutral'][actual]
        
        status = "✓ CORRECT" if result['sentiment'] == actual_label else "✗ WRONG"
        if status == "✓ CORRECT":
            correct += 1
        
        print(f"\n{i+1}. {status}")
        print(f"   Text: {text[:100]}...")
        print(f"   Actual: {actual_label} → Predicted: {result['sentiment']}")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"   Probabilities: Neg({result['probabilities']['negative']:.3f}) "
              f"Pos({result['probabilities']['positive']:.3f}) "
              f"Neu({result['probabilities']['neutral']:.3f})")
    
    accuracy = correct / num_samples
    print(f"\n📈 Sample Accuracy: {accuracy:.1%} ({correct}/{num_samples} correct)")

def main():
    print("🔍 Analyzing Model Results...")
    
    # Load trained model
    detector = SentimentDetector()
    detector.load_model('trained_sentiment_model.pkl')
    
    # Load some test data
    df = detector.load_data()
    df = detector.add_sentiment_labels(df)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_text'], df['sentiment'], test_size=0.2, random_state=42
    )
    
    # Analyze sample predictions
    test_texts = X_test.tolist()
    analyze_sample_predictions(detector, test_texts, y_test.tolist(), num_samples=15)
    
    # Show feature importance
    print("\n" + "="*60)
    print("🔤 TOP PREDICTIVE WORDS")
    print("="*60)
    detector.plot_feature_importance(top_n=15)

if __name__ == "__main__":
    main()