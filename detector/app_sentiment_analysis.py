# app_sentiment_analysis.py - Analyze sentiment distribution by app
from sentiment_detector import SentimentDetector
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analyze_sentiment_by_app():
    """Analyze and visualize sentiment distribution for each app"""
    print("📊 Analyzing Sentiment by App...\n")
    
    detector = SentimentDetector()
    
    # Map file names to app names
    app_mapping = {
        'khabarkoi.csv': 'Khabar Koi',
        'khaodao.csv': 'Khaodao',
        'coopers.csv': 'Coopers',
        'pizzahut.csv': 'Pizza Hut',
        'proyojon.csv': 'Proyojon',
        'sultansdine.csv': 'Sultans Dine'
    }
    
    # Load data for each app separately
    app_data = {}
    
    for file_name, app_name in app_mapping.items():
        try:
            df = pd.read_csv(f'data/{file_name}')
            df['cleaned_text'] = df['content'].apply(detector.clean_text)
            df = df[df['cleaned_text'].str.len() > 0]  # Remove empty texts
            df = detector.add_sentiment_labels(df)
            
            # Count sentiments
            sentiment_counts = df['sentiment'].value_counts().sort_index()
            app_data[app_name] = {
                'negative': sentiment_counts.get(0, 0),
                'positive': sentiment_counts.get(1, 0),
                'neutral': sentiment_counts.get(2, 0),
                'total': len(df)
            }
            print(f"✓ {app_name}: {app_data[app_name]['total']} reviews")
        except Exception as e:
            print(f"✗ Error loading {file_name}: {e}")
    
    # Create visualization with grouped bars (like the example)
    apps = list(app_data.keys())
    negative_counts = [app_data[app]['negative'] for app in apps]
    positive_counts = [app_data[app]['positive'] for app in apps]
    neutral_counts = [app_data[app]['neutral'] for app in apps]
    
    # Set up the plot - bars close together within groups, groups well separated
    x = np.arange(len(apps))
    width = 0.25  # Width of each bar
    # Position bars: Negative (left), Neutral (center), Positive (right)
    # Bars are close together (width apart), groups are separated by 1.0
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create grouped bars - bars are width apart, creating tight groups
    bars1 = ax.bar(x - width, negative_counts, width, label='Negative', 
                   color='#e74c3c', alpha=0.9, edgecolor='white', linewidth=1)
    bars2 = ax.bar(x, neutral_counts, width, label='Neutral', 
                   color='#95a5a6', alpha=0.9, edgecolor='white', linewidth=1)
    bars3 = ax.bar(x + width, positive_counts, width, label='Positive', 
                   color='#2ecc71', alpha=0.9, edgecolor='white', linewidth=1)
    
    # Add value labels on top of bars
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
    ax.set_title('Sentiment Distribution by App', fontsize=18, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(apps, rotation=0, ha='center', fontsize=11, fontweight='bold')
    ax.legend(loc='upper left', fontsize=12, framealpha=0.95)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')
    
    # Set y-axis to start from 0
    ax.set_ylim(bottom=0)
    
    # Adjust y-axis to show all data nicely
    max_value = max(negative_counts + neutral_counts + positive_counts)
    ax.set_ylim(top=max_value * 1.1)
    
    plt.tight_layout()
    plt.savefig('app_sentiment_distribution.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Graph saved as 'app_sentiment_distribution.png'")
    plt.close()
    
    # Print summary table
    print("\n" + "="*80)
    print("📈 SENTIMENT SUMMARY BY APP")
    print("="*80)
    print(f"\n{'App':<20} {'Negative':<12} {'Neutral':<12} {'Positive':<12} {'Total':<10} {'% Positive':<12}")
    print("-"*80)
    
    for app in apps:
        data = app_data[app]
        pct_positive = (data['positive'] / data['total'] * 100) if data['total'] > 0 else 0
        print(f"{app:<20} {data['negative']:<12} {data['neutral']:<12} "
              f"{data['positive']:<12} {data['total']:<10} {pct_positive:.1f}%")
    
    # Create percentage stacked bar chart
    create_percentage_chart(apps, app_data)
    
    return app_data

def create_percentage_chart(apps, app_data):
    """Create a percentage stacked bar chart"""
    # Calculate percentages
    negative_pct = [app_data[app]['negative'] / app_data[app]['total'] * 100 
                    if app_data[app]['total'] > 0 else 0 for app in apps]
    neutral_pct = [app_data[app]['neutral'] / app_data[app]['total'] * 100 
                   if app_data[app]['total'] > 0 else 0 for app in apps]
    positive_pct = [app_data[app]['positive'] / app_data[app]['total'] * 100 
                    if app_data[app]['total'] > 0 else 0 for app in apps]
    
    x = np.arange(len(apps))
    width = 0.6
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create stacked bars
    bars1 = ax.bar(x, negative_pct, width, label='Negative', 
                   color='#ff4444', alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax.bar(x, neutral_pct, width, bottom=negative_pct, label='Neutral', 
                   color='#888888', alpha=0.8, edgecolor='black', linewidth=1)
    bars3 = ax.bar(x, positive_pct, width, 
                   bottom=[negative_pct[i] + neutral_pct[i] for i in range(len(apps))], 
                   label='Positive', color='#44ff44', alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add percentage labels
    for i, app in enumerate(apps):
        # Negative label
        if negative_pct[i] > 5:
            ax.text(i, negative_pct[i]/2, f'{negative_pct[i]:.1f}%', 
                   ha='center', va='center', fontsize=9, fontweight='bold', color='white')
        # Neutral label
        if neutral_pct[i] > 5:
            ax.text(i, negative_pct[i] + neutral_pct[i]/2, f'{neutral_pct[i]:.1f}%', 
                   ha='center', va='center', fontsize=9, fontweight='bold', color='white')
        # Positive label
        if positive_pct[i] > 5:
            ax.text(i, negative_pct[i] + neutral_pct[i] + positive_pct[i]/2, f'{positive_pct[i]:.1f}%', 
                   ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    
    ax.set_xlabel('App', fontsize=14, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=14, fontweight='bold')
    ax.set_title('Sentiment Distribution by App (Percentage)', fontsize=18, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(apps, rotation=45, ha='right', fontsize=11)
    ax.legend(loc='upper left', fontsize=12, framealpha=0.9)
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig('app_sentiment_percentage.png', dpi=300, bbox_inches='tight')
    print(f"✅ Percentage chart saved as 'app_sentiment_percentage.png'")
    plt.close()

def main():
    app_data = analyze_sentiment_by_app()
    print("\n✅ Analysis complete! Check the generated PNG files.")

if __name__ == "__main__":
    main()

