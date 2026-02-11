###############################################################
# FULL PYTORCH PIPELINE – CNN-LSTM-HAN WITH FINBERT SENTIMENT
# TensorFlow COMPLETELY REMOVED
###############################################################

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

###############################################################
# 1. DATA LOADING
###############################################################

data1 = pd.read_csv("F:\downloads\DL_Course-Project-main\DL_Course-Project-main\data\cnbc_headlines.csv")
data2 = pd.read_csv("F:\downloads\DL_Course-Project-main\DL_Course-Project-main\data\reuters_headlines.csv")

news_data = pd.concat([data1, data2]).dropna()

news_data['Date'] = pd.to_datetime(
    news_data['Time'],
    format='%I:%M %p ET %a, %d %B %Y',
    errors='coerce'
)

news_data = news_data.sort_values('Date')
news_data['DateOnly'] = news_data['Date'].dt.date

###############################################################
# 2. FINBERT SENTIMENT EXTRACTION (PYTORCH)
###############################################################

tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
finbert = AutoModel.from_pretrained("yiyanghkust/finbert-tone").to(device)

def batch_finbert_sentiment(texts, batch_size=32):
    scores = []

    finbert.eval()

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]

            inputs = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=256
            ).to(device)

            outputs = finbert(**inputs)

            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            batch_scores = cls_embeddings.mean(dim=1).cpu().numpy()

            scores.extend(batch_scores)

    return np.array(scores)


news_data['FinBERT_Sentiment'] = batch_finbert_sentiment(
    news_data['Headlines'].tolist()
)

daily_sentiment = news_data.groupby('DateOnly')['FinBERT_Sentiment'].mean().reset_index()
daily_sentiment.rename(columns={'DateOnly': 'Date'}, inplace=True)
daily_sentiment['Date'] = pd.to_datetime(daily_sentiment['Date'])

###############################################################
# 3. STOCK DATA
###############################################################

symbol = 'SPY'

# stock_data = yf.download(symbol, start="2017-12-01", end="2020-07-19")
import pandas_datareader.data as web

stock_data = web.DataReader(
    "SPY",
    "stooq",
    start="2017-12-01",
    end="2020-07-19"
).reset_index()

stock_data['Date'] = pd.to_datetime(stock_data['Date'])

stock_data.columns = [''.join(col).strip() if isinstance(col, tuple) else col for col in stock_data.columns]

combined_data = pd.merge(stock_data, daily_sentiment, on='Date', how='left')
combined_data['FinBERT_Sentiment'].fillna(method='ffill', inplace=True)

combined_data.to_csv(
    r"F:\downloads\DL_Course-Project-main\DL_Course-Project-main\data\combined_data.csv",
    index=False,
    encoding="utf-8"
)

###############################################################
# 4. TECHNICAL INDICATORS
###############################################################

combined_data['Returns'] = combined_data['Close'].pct_change()
combined_data['MA5'] = combined_data['Close'].rolling(5).mean()
combined_data['MA10'] = combined_data['Close'].rolling(10).mean()
combined_data['Volatility'] = combined_data['Returns'].rolling(5).std()

combined_data['RSI'] = 100 - (
    100 / (1 +
    combined_data['Returns'].rolling(14).mean() /
    combined_data['Returns'].rolling(14).std())
)

combined_data['OBV'] = (np.sign(combined_data['Returns']) * combined_data['Volume']).cumsum()

combined_data.ffill(inplace=True)
combined_data.dropna(inplace=True)

###############################################################
# 5. SEQUENCE CREATION
###############################################################

features = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'Returns', 'MA5', 'MA10', 'Volatility',
    'RSI', 'OBV', 'FinBERT_Sentiment'
]

def create_sequences(data, seq_len=10):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[features].iloc[i:i+seq_len].values)
        y.append(data['Close'].iloc[i+seq_len])
    return np.array(X), np.array(y)

X, y = create_sequences(combined_data)

split = int(0.8 * len(X))

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

###############################################################
# 6. SCALING
###############################################################

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

y_scaler = MinMaxScaler()
y_train = y_scaler.fit_transform(y_train.reshape(-1,1)).flatten()
y_test = y_scaler.transform(y_test.reshape(-1,1)).flatten()

###############################################################
# 7. PYTORCH DATASET
###############################################################

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(TimeSeriesDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(TimeSeriesDataset(X_test, y_test), batch_size=32)

###############################################################
# 8. MODEL – CNN + BiLSTM + ATTENTION
###############################################################

class HybridModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.conv = nn.Conv1d(input_dim, 32, kernel_size=3)
        self.lstm = nn.LSTM(32, 64, batch_first=True, bidirectional=True)

        self.attn = nn.Linear(128, 1)

        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = torch.relu(self.conv(x))
        x = x.permute(0, 2, 1)

        x, _ = self.lstm(x)

        attn_weights = torch.softmax(self.attn(x), dim=1)
        x = (x * attn_weights).sum(dim=1)

        return self.fc(x).squeeze()

model = HybridModel(X_train.shape[2]).to(device)

criterion = nn.HuberLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

###############################################################
# 9. TRAINING LOOP
###############################################################

for epoch in range(30):
    model.train()

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)

        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} completed with loss = {loss}")

###############################################################
# 10. EVALUATION
###############################################################

model.eval()
preds = []

with torch.no_grad():
    for xb, _ in test_loader:
        xb = xb.to(device)
        preds.extend(model(xb).cpu().numpy())

preds = y_scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
y_true = y_scaler.inverse_transform(y_test.reshape(-1,1)).flatten()

mse = mean_squared_error(y_true, preds)
rmse = np.sqrt(mse)

directional_accuracy = np.mean(
    np.sign(np.diff(y_true)) ==
    np.sign(np.diff(preds))
) * 100

print("MSE:", mse)
print("RMSE:", rmse)
print("Directional Accuracy:", directional_accuracy)

###############################################################
# 11. ARIMA BASELINE
###############################################################

arima = ARIMA(y_train, order=(2,1,2)).fit()
arima_preds = arima.forecast(steps=len(y_test))

arima_preds = y_scaler.inverse_transform(arima_preds.reshape(-1,1)).flatten()

print("ARIMA RMSE:",
      np.sqrt(mean_squared_error(y_true, arima_preds)))

###############################################################
# 12. ONNX EXPORT
###############################################################

dummy = torch.randn(1, X_train.shape[1], X_train.shape[2]).to(device)

torch.onnx.export(
    model,
    dummy,
    r"F:\downloads\DL_Course-Project-main\DL_Course-Project-main\saved_model\pytorch_cnn_lstm.onnx",
    input_names=["input"],
    output_names=["output"]
)

print("ONNX Exported")
