# use_detector.py
from sentiment_detector import SentimentDetector

def main():
    print("🎯 Using Trained Sentiment Detector...")
    
    # Initialize detector
    detector = SentimentDetector()
    
    # Load the trained model
    detector.load_model('trained_sentiment_model.pkl')
    
    # Test with some example reviews
    test_reviews = [
        "This app is amazing! Fast delivery and great food quality.",
        "Worst experience ever. Never ordering again from this app.",
        "The food was okay but delivery was a bit late.",
        "They scammed me! Took my money and never delivered the food.",
        "Good service and reasonable prices. Will order again.",
        "App keeps crashing and customer service doesn't respond.",
        "Excellent food and very fast delivery. Highly recommended!",
        "Order was delayed by 2 hours and food was cold."
    ]
    
    print("\n🧪 Testing the detector:\n")
    for i, review in enumerate(test_reviews, 1):
        result = detector.predict_sentiment(review)
        print(f"{i}. {result['text']}")
        print(f"   → Sentiment: {result['sentiment']} (Confidence: {result['confidence']:.2f})")
        print()

if __name__ == "__main__":
    main()