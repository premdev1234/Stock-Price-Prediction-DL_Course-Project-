# DL\_Course-Project: Stock Price Prediction using Sentiment-Aware Deep Learning

This project is a course submission focused on applying Natural Language Processing (NLP) and Deep Learning techniques to predict stock price movements based on sentiment analysis from financial news headlines.

## 📌 Objective

To build a deep learning model that incorporates sentiment signals from textual data (e.g., financial news, tweets) and numerical stock features to forecast stock price trends with improved predictive power.

## 🔍 Motivation

Financial markets are heavily influenced by public sentiment, especially in the short term. Traditional quantitative models often neglect this textual component. This project bridges that gap by integrating sentiment analysis with deep learning to enhance stock movement forecasting.

## 🧠 Techniques Used

* **Text Preprocessing**: Tokenization, stopword removal, and padding
* **Sentiment Analysis**: Using labeled news headlines as input
* **Deep Learning Models**:

  * Bidirectional LSTM
  * CNN-LSTM hybrid
  * Word2Vec and GloVe embeddings
* **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score
* **Data Handling**: Manual labeling, preprocessing with NLTK, scikit-learn

## 📁 Directory Structure

```
DL_Course-Project/
│
├── data/               # Contains labeled financial news data
│   ├── FinancialData.csv
│   └── preprocessed/   # Cleaned and vectorized text data
│
├── models/             # DL models used in experiments
│   ├── lstm_model.py
│   ├── cnn_lstm_model.py
│   └── utils.py        # Shared utilities for tokenization, embedding etc.
│
├── notebooks/          # Jupyter notebooks for EDA and training logs
│   └── Sentiment_Analysis_Stock.ipynb
│
├── results/            # Accuracy graphs, confusion matrix, etc.
│
├── requirements.txt    # Python dependencies
├── train.py            # Main training pipeline
└── README.md           # You're reading it!
```

## 📊 Dataset

A custom dataset comprising financial news headlines with manually assigned sentiment labels (`positive`, `negative`, `neutral`). Some examples were augmented with synthetic variations to improve class balance.

## 🚀 How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/premdev1234/DL_Course-Project.git
   cd DL_Course-Project
   ```

2. Install requirements:

   ```bash
   pip install -r requirements.txt
   ```

3. Train the model:

   ```bash
   python train.py --model lstm --epochs 20
   ```

4. Visualize results:
   Open the Jupyter notebook in the `notebooks/` directory to explore results interactively.

## 🧪 Results

| Model      | Accuracy | F1-Score |
| ---------- | -------- | -------- |
| BiLSTM     | 72.3%    | 0.70     |
| CNN+LSTM   | 74.1%    | 0.73     |
| GloVe+LSTM | 76.5%    | 0.75     |

The CNN+LSTM model with pre-trained GloVe embeddings delivered the best performance by capturing both local and temporal features of the input sequences.

## 📌 Limitations & Future Work

* **Dataset Size**: Limited labeled data; would improve with larger corpora.
* **Domain Knowledge**: Sentiment labeling could benefit from context-aware transformers.
* **Model Generalization**: Overfitting remains a challenge; future versions may use BERT/FinBERT for better transfer learning.

## 🧑‍💻 Contributors

* **[Premdev1234](https://github.com/premdev1234)** – Model Development, NLP, Evaluation

---

> **Note**: This project was part of an academic deep learning course and is not intended for live trading or financial advice.
