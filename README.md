# ReviewNet - Intelligent App Review Analysis

Welcome to the **ReviewNet** project! This README provides an overview of the project, setup instructions, and other relevant details.

## Table of Contents

- [Visit](#visit)
- [About](#about)
- [Features](#features)
- [Installation](#installation)
- [Structure](#structure)
- [Contributors](#contributors)
- [Contributing](#contributing)
- [License](#license)

## Visit

- [Repository](https://github.com/woabu0/reviewnet)

## About

**ReviewNet** is a comprehensive data analysis framework designed to extract deep insights from mobile application reviews. By leveraging state-of-the-art Natural Language Processing (NLP) models—including discriminative transformers (BERT, RoBERTa) and novel tabular deep learning approaches (TabPFN)—it provides robust analysis of user sentiment, emotional undertones, and content toxicity.

## Features

- Google Play Scraper
- Advanced Sentiment Analysis
- Emotion Detection
- Toxicity Detection
- Word Cloud Generation
- Interactive Visualizations

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/woabu0/reviewnet.git
   ```

2. Navigate to the project directory:

   ```bash
   cd reviewnet
   ```

3. Set up the Environment (Python):

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
   pip install -r requirements.txt
   ```

## Structure

```
reviewnet/
├── dataset/                    # Raw and processed review datasets
│   ├── coopers.csv
│   ├── foodi.csv
│   ├── kfc.csv
│   ├── khabarkoi.csv
│   ├── khaodao.csv
│   ├── munchies.csv
│   ├── pizzahut.csv
│   ├── proyojon.csv
│   └── sultansdine.csv
├── emotion/                    # Emotion analysis scripts and results
│   ├── emotion_analysis/       # Analysis results & graphs
│   ├── theme_emotion_analysis/ # Theme-based analysis results & graphs
│   ├── emotion_analysis.py     # Core emotion detection logic
│   └── theme_emotion_analysis.py # Theme-specific emotion analysis
├── negative_dataset/           # Datasets containing negative reviews
│   ├── coopers.csv
│   ├── foodi.csv
│   ├── kfc.csv
│   ├── khabarkoi.csv
│   ├── khaodao.csv
│   ├── munchies.csv
│   ├── pizzahut.csv
│   ├── proyojon.csv
│   └── sultansdine.csv
├── scraper/                    # Google Play Store review scraper
│   ├── main.py                 # Scraper entry point
│   └── filter.py               # Data filtering utilities
├── sentiment/                  # Sentiment analysis models and training
│   ├── outputs/                # Confusion matrices, plots, and results
│   ├── bert_train.py           # BERT/RoBERTa training script
│   ├── ml_train.py             # Classical ML models (SVM, RF)
│   ├── negative_bert_train.py  # Negative sentiment focused BERT
│   ├── negative_ml_train.py    # Negative sentiment focused ML models
│   ├── negative_tabpfn_train.py # Negative sentiment with TabPFN
│   ├── sentiment_analysis.py   # Sentiment inference engine
│   ├── tabpfn_train.py         # TabPFN training script
│   └── theme_sentiment_analysis.py # Theme-based sentiment analysis
├── theme/                      # Theme categorization data
│   └── theme.csv
├── toxicity/                   # Toxicity detection modules
│   ├── outputs/                # Toxicity analysis graphs
│   ├── theme_toxicity_analysis.py # Theme-based toxicity analysis
│   └── toxicity_analysis.py    # Core toxicity detection
├── wordclouds/                 # Word cloud generation scripts
│   ├── outputs/                # Generated word cloud images
│   ├── sentiment_wordclouds.py # Sentiment-based word clouds
│   └── theme_wordclouds.py     # Theme-based word clouds
├── dataset.csv                 # Combined main dataset
├── negative_dataset.csv        # Combined negative review dataset
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```

## Contributors

<p align="center">
  <a href="https://github.com/woabu0/reviewnet/graphs/contributors">
    <img src="https://contrib.rocks/image?repo=woabu0/reviewnet" alt="Contributors" />
  </a>
</p>

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch:

   ```bash
   git checkout -b feature-name
   ```

3. Commit your changes:

   ```bash
   git commit -m "Add feature-name"
   ```

4. Push to the branch:

   ```bash
   git push origin feature-name
   ```

5. Open a pull request.

## License

This project is licensed under the [MIT License](LICENSE).