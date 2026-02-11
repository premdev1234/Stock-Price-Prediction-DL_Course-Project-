import pandas as pd
import pandas_datareader.data as web

def load_news_data(cnbc_path, reuters_path):
    data1 = pd.read_csv(cnbc_path)
    data2 = pd.read_csv(reuters_path)

    news_data = pd.concat([data1, data2]).dropna()

    news_data['Date'] = pd.to_datetime(
        news_data['Time'],
        format='%I:%M %p ET %a, %d %B %Y',
        errors='coerce'
    )

    news_data = news_data.sort_values('Date')
    news_data['DateOnly'] = news_data['Date'].dt.date

    return news_data


def download_stock_data(symbol, start, end):
    stock_data = web.DataReader(
        symbol,
        "stooq",
        start=start,
        end=end
    ).reset_index()

    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    return stock_data
