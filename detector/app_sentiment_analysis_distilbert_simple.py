# app_sentiment_analysis_distilbert_simple.py - Analyze sentiment by app using DistilBERT (simplified)
from sentiment_detector_distilbert_simple import SentimentDetectorDistilBERTSimple
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analyze_sentiment_by_app():
    """Analyze and visualize sentiment distribution for each app using DistilBERT"""
    print("📊 Analyzing Sentiment by App (DistilBERT - Simplified)...\n")
    
    try:
        detector = SentimentDetectorDistilBERTSimple()
    except ImportError as e:
        print(f"❌ Error: {e}")
        print("\nPlease install required packages:")
        print("pip install transformers torch")
        return None
    
    # Map file names to app names
    app_mapping = {
        'khabarkoi.csv': 'Khabar Koi',
        'khaodao.csv': 'Khaodao',
        'coopers.csv': 'Coopers',
        'pizzahut.csv': 'Pizza Hut',
        'proyojon.csv': 'Proyojon',
        'sultansdine.csv': 'Sultans Dine'
    }
    
    # Load data for each app separately and predict
    app_data = {}
    
    for file_name, app_name in app_mapping.items():
        try:
            df = pd.read_csv(f'data/{file_name}')
            df['cleaned_text'] = df['content'].apply(detector.clean_text)
            df = df[df['cleaned_text'].str.len() > 0]
            
            # Predict sentiment for each review (process in smaller batches)
            predictions = []
            print(f"Processing {app_name} ({len(df)} reviews)...")
            
            batch_size = 10
            texts = df['content'].tolist()
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                try:
                    # Process batch
                    for text in batch:
                        try:
                            result = detector.predict_sentiment(text)
                            sentiment_label = result['sentiment']
                            # Map to numeric
                            if sentiment_label == 'Negative':
                                predictions.append(0)
                            elif sentiment_label == 'Positive':
                                predictions.append(1)
                            else:
                                predictions.append(2)
                        except Exception as e:
                            predictions.append(2)  # Default to neutral on error
                except Exception as e:
                    # If batch fails, add neutrals
                    predictions.extend([2] * len(batch))
                
                # Progress indicator
                if (i + batch_size) % 50 == 0 or (i + batch_size) >= len(texts):
                    print(f"  Processed {min(i + batch_size, len(texts))}/{len(df)} reviews...")
            
            df['sentiment'] = predictions
            
            # Count sentiments
            sentiment_counts = pd.Series(predictions).value_counts().sort_index()
            app_data[app_name] = {
                'negative': sentiment_counts.get(0, 0),
                'positive': sentiment_counts.get(1, 0),
                'neutral': sentiment_counts.get(2, 0),
                'total': len(df)
            }
            print(f"✓ {app_name}: {app_data[app_name]['total']} reviews")
            print(f"  Negative: {app_data[app_name]['negative']}, "
                  f"Neutral: {app_data[app_name]['neutral']}, "
                  f"Positive: {app_data[app_name]['positive']}\n")
        except Exception as e:
            print(f"✗ Error loading {file_name}: {e}")
    
    # Create visualization with grouped bars
    apps = list(app_data.keys())
    negative_counts = [app_data[app]['negative'] for app in apps]
    positive_counts = [app_data[app]['positive'] for app in apps]
    neutral_counts = [app_data[app]['neutral'] for app in apps]
    
    # Set up the plot
    x = np.arange(len(apps))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create grouped bars
    bars1 = ax.bar(x - width, negative_counts, width, label='Negative', 
                   color='#e74c3c', alpha=0.9, edgecolor='white', linewidth=1)
    bars2 = ax.bar(x, neutral_counts, width, label='Neutral', 
                   color='#95a5a6', alpha=0.9, edgecolor='white', linewidth=1)
    bars3 = ax.bar(x + width, positive_counts, width, label='Positive', 
                   color='#2ecc71', alpha=0.9, edgecolor='white', linewidth=1)
    
    # Add value labels
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom', 
                       fontsize=9, fontweight='bold')
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)
    
    # Customize the plot
    ax.set_xlabel('App', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Reviews', fontsize=14, fontweight='bold')
    ax.set_title('Sentiment Distribution by App (DistilBERT)', fontsize=18, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(apps, rotation=0, ha='center', fontsize=11, fontweight='bold')
    ax.legend(loc='upper left', fontsize=12, framealpha=0.95)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')
    ax.set_ylim(bottom=0)
    
    max_value = max(negative_counts + neutral_counts + positive_counts)
    ax.set_ylim(top=max_value * 1.1)
    
    plt.tight_layout()
    plt.savefig('app_sentiment_distribution_distilbert.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Graph saved as 'app_sentiment_distribution_distilbert.png'")
    plt.close()
    
    # Print summary table
    print("\n" + "="*80)
    print("📈 SENTIMENT SUMMARY BY APP (DistilBERT)")
    print("="*80)
    print(f"\n{'App':<20} {'Negative':<12} {'Neutral':<12} {'Positive':<12} {'Total':<10} {'% Positive':<12}")
    print("-"*80)
    
    for app in apps:
        data = app_data[app]
        pct_positive = (data['positive'] / data['total'] * 100) if data['total'] > 0 else 0
        print(f"{app:<20} {data['negative']:<12} {data['neutral']:<12} "
              f"{data['positive']:<12} {data['total']:<10} {pct_positive:.1f}%")
    
    return app_data

def main():
    app_data = analyze_sentiment_by_app()
    if app_data:
        print("\n✅ Analysis complete! Check the generated PNG file.")
    else:
        print("\n❌ Analysis failed. Please check the error messages above.")

if __name__ == "__main__":
    main()

