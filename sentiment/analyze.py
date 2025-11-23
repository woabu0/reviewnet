# analyze.py - Analyze sentiment using trained DistilBERT model
import pandas as pd
import os
import sys
from detector import SentimentDetector

def main():
    print("=== 📱 App Review Sentiment Analyzer (Trained DistilBERT) ===")
    
    # Check if model exists
    if not os.path.exists('./distilbert_sentiment_model_final'):
        print("\n❌ Trained model not found!")
        print("Please train the model first by running:")
        print("  python train.py")
        return
    
    try:
        detector = SentimentDetector()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return
    
    # Get input file path
    input_path = input("\nEnter path to your CSV file (or press Enter to analyze all files in data/): ").strip()
    
    if not input_path:
        print("\n📁 Loading all CSV files from data folder...")
        try:
            df = detector.load_data()
            print(f"✓ Total reviews loaded: {len(df)}")
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return
    else:
        if os.path.isdir(input_path):
            print(f"\n📁 Detected directory. Loading all CSV files from {input_path}...")
            try:
                data_dir = input_path
                file_names = [
                    'khabarkoi.csv', 'khaodao.csv', 'coopers.csv',
                    'pizzahut.csv', 'proyojon.csv', 'sultansdine.csv'
                ]
                
                all_data = []
                for file_name in file_names:
                    file_path = os.path.join(data_dir, file_name)
                    if os.path.exists(file_path):
                        df_temp = pd.read_csv(file_path)
                        print(f"✓ Loaded {file_name}: {len(df_temp)} reviews")
                        all_data.append(df_temp)
                
                if all_data:
                    df = pd.concat(all_data, ignore_index=True)
                    df['cleaned_text'] = df['content'].apply(detector.clean_text)
                    df = df[df['cleaned_text'].str.len() > 0]
                    print(f"✓ Total reviews loaded: {len(df)}")
                else:
                    print("❌ No CSV files found in the directory.")
                    return
            except Exception as e:
                print(f"❌ Error loading data: {e}")
                return
        elif os.path.isfile(input_path):
            df = pd.read_csv(input_path)
            print(f"\n✓ Loaded {len(df)} reviews from {input_path}")
        else:
            print("❌ File or directory not found.")
            return
    
    # Get text column
    print("\nColumns found:", list(df.columns))
    text_col = input("\nEnter the column name containing review text (default: 'content'): ").strip()
    
    if not text_col:
        text_col = 'content'
    
    if text_col not in df.columns:
        print(f"❌ Column '{text_col}' not found in CSV.")
        return
    
    print(f"\n🔍 Analyzing sentiment using trained DistilBERT model...")
    print("This may take a few minutes. Please wait ⏳\n")
    
    # Analyze sentiment
    result_df = detector.analyze_dataframe(df, text_column=text_col)
    
    # Create final dataframe
    final_df = pd.DataFrame({
        'content': result_df['content'],
        'sentiment': result_df['sentiment'],
        'polarity': result_df['polarity'],
        'confidence': result_df['confidence']
    })
    
    # Save results
    output_csv = "app_review_sentiment_results.csv"
    output_xlsx = "app_review_sentiment_results.xlsx"
    
    final_df.to_csv(output_csv, index=False)
    final_df.to_excel(output_xlsx, index=False)
    
    print(f"\n✅ Analysis complete!")
    print(f"📄 Saved CSV: {output_csv}")
    print(f"📊 Saved Excel: {output_xlsx}")
    print("\nColumns in output file: [content, sentiment, polarity, confidence]")
    
    # Show summary
    print("\n" + "="*60)
    print("📈 SENTIMENT SUMMARY")
    print("="*60)
    sentiment_counts = final_df['sentiment'].value_counts()
    print(f"Positive: {sentiment_counts.get('Positive', 0)}")
    print(f"Negative: {sentiment_counts.get('Negative', 0)}")
    print(f"Neutral:  {sentiment_counts.get('Neutral', 0)}")
    print("="*60)


if __name__ == "__main__":
    main()

