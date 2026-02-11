import numpy as np
import pandas as pd

def calculate_technical_indicators(df):

    df['Returns'] = df['Close'].pct_change()

    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA10'] = df['Close'].rolling(10).mean()

    df['Volatility'] = df['Returns'].rolling(5).std()

    df['RSI'] = 100 - (
        100 / (1 +
        df['Returns'].rolling(14).mean() /
        df['Returns'].rolling(14).std())
    )

    df['OBV'] = (np.sign(df['Returns']) * df['Volume']).cumsum()

    # Correct Bollinger Bands using Close price
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['STD_20'] = df['Close'].rolling(window=20).std()

    df['Bollinger_Upper'] = df['SMA_20'] + (df['STD_20'] * 2)
    df['Bollinger_Lower'] = df['SMA_20'] - (df['STD_20'] * 2)

    df.ffill(inplace=True)
    df.dropna(inplace=True)

    return df
