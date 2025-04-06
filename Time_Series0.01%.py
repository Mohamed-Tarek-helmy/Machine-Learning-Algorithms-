import os
import sys
import warnings
import logging
from contextlib import redirect_stdout, redirect_stderr
# Suppress installation output
with open(os.devnull, "w") as f:
    with redirect_stdout(f), redirect_stderr(f):
        os.system("pip install --upgrade ta")
        os.system("pip install --upgrade cmdstanpy")
## Suppres unwanted outputs :


# Suppress warnings
warnings.filterwarnings("ignore")

# Suppress logging from cmdstanpy and prophet
logging.getLogger("prophet").setLevel(logging.CRITICAL)
logging.getLogger("cmdstanpy").setLevel(logging.CRITICAL)
"""-----------------------------------------------------------------------"""


import pandas as pd
import numpy as np
import yfinance as yf
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
import ta
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import RANSACRegressor, LinearRegression

# Fetch raw stock data without preprocessing
def get_raw_data(ticker, start_date, end_date):
    """Fetch raw stock data without any transformations."""
    stock = yf.download(ticker, start=start_date, end=end_date, progress=False)
    stock.columns = [col[0] for col in stock.columns]
    return stock[['Open', 'High', 'Low', 'Close', 'Volume']]

# Technical indicators with lookback window handling
def add_technical_indicators(df, lookback_window=20):
    """Add technical indicators with proper lookback handling."""
    df = df.copy()
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    bb = ta.volatility.BollingerBands(df['Close'])
    df['BB_high'] = bb.bollinger_hband()
    df['BB_low'] = bb.bollinger_lband()
    df['Volatility'] = df['Close'].rolling(window=20).std()
    return df.iloc[lookback_window-1:]  # Remove invalid initial values

# Lag features with training context
def add_lag_features(df, train_context=None, lags=[1, 5, 10]):
    """Add lag features using training data context to prevent lookahead."""
    df = df.copy()
    for lag in lags:
        if train_context is not None:
            # Combine last 'lag' values from training with current df
            combined = pd.concat([train_context.tail(lag), df])
            df[f'Close_lag_{lag}'] = combined['Close'].shift(lag).iloc[lag:]
            df[f'Volume_lag_{lag}'] = combined['Volume'].shift(lag).iloc[lag:]
        else:
            df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
            df[f'Volume_lag_{lag}'] = df['Volume'].shift(lag)
    return df

# Time features
def add_time_features(df):
    """Add time-based features."""
    df = df.copy()
    df['Day_of_Week'] = df.index.dayofweek
    df['Month'] = df.index.month
    df['Quarter'] = df.index.quarter
    df['Day_of_Year'] = df.index.dayofyear
    df['Week_of_Year'] = df.index.isocalendar().week.astype(int)
    df['Is_Month_End'] = df.index.is_month_end.astype(int)
    return df

# External factors (VIX)
def add_external_factors(df):
    """Add VIX data as external factor."""
    vix = yf.download('^VIX', start=df.index.min(), end=df.index.max(), progress=False)
    vix.columns= [col[0] for col in vix.columns]
    df = df.join(vix['Close'].rename('VIX'), how='left')
    return df
# Preprocessing pipeline for training data
def preprocess_train(raw_df):
    """Full preprocessing pipeline for training data."""
    df = raw_df.copy()

    # Technical indicators
    df = add_technical_indicators(df)

    # Lag features
    df = add_lag_features(df)

    # Time features
    df = add_time_features(df)

    # External factors
    df = add_external_factors(df)

    # Price dynamics
    df['Price_Diff'] = df['Close'].diff()
    df['Returns'] = df['Close'].pct_change()

    # Handle missing values
    df = df.interpolate(method='linear').fillna(method='ffill').dropna()

    return df

# Preprocessing pipeline for test data
def preprocess_test(raw_df, train_df):
    """Preprocessing for test data using training context."""
    df = raw_df.copy()

    # Remove 'VIX' from training lookback to avoid duplication
    lookback = train_df.tail(19).drop(columns=['VIX'], errors='ignore')  # Key fix
    combined = pd.concat([lookback, df])
    df = add_technical_indicators(combined).loc[df.index]

    # Lag features with training context
    df = add_lag_features(df, train_context=train_df)

    # Time features
    df = add_time_features(df)

    # External factors (now safe to add VIX)
    df = add_external_factors(df)

    # Price dynamics with training context
    df['Price_Diff'] = df['Close'].diff()
    df['Price_Diff'].iloc[0] = df['Close'].iloc[0] - train_df['Close'].iloc[-1]
    df['Returns'] = df['Close'].pct_change()
    df['Returns'].iloc[0] = (df['Close'].iloc[0] - train_df['Close'].iloc[-1])/train_df['Close'].iloc[-1]

    # Handle missing values
    df = df.interpolate(method='linear').fillna(method='ffill').dropna()

    return df

# Main execution
if __name__ == "__main__":
    ticker = 'GC=F'
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)

    # Get and split raw data
    raw_data = get_raw_data(ticker, start_date, end_date)
    split_date = raw_data.index[int(len(raw_data)*0.93)]
    train_raw = raw_data.loc[:split_date]
    test_raw = raw_data.loc[split_date + timedelta(days=1):]

    # Preprocess datasets
    train_processed = preprocess_train(train_raw)
    test_processed = preprocess_test(test_raw, train_processed)

    # Prepare Prophet format
    def prepare_prophet_format(df, scaler=None):
        prophet_df = df.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
        continuous_cols = ['Open', 'High', 'Low', 'Volume', 'SMA_20', 'EMA_12', 'RSI',
                          'MACD', 'MACD_signal', 'BB_high', 'BB_low', 'Volatility',
                          'Price_Diff', 'Returns'] + \
                         [f'Close_lag_{lag}' for lag in [1,5,10]] + \
                         [f'Volume_lag_{lag}' for lag in [1,5,10]]

        if scaler is None:
            scaler = MinMaxScaler()
            prophet_df[continuous_cols] = scaler.fit_transform(prophet_df[continuous_cols])
            return prophet_df, scaler
        else:
            prophet_df[continuous_cols] = scaler.transform(prophet_df[continuous_cols])
            return prophet_df, scaler

    # Scale data
    train_prophet, scaler = prepare_prophet_format(train_processed)
    test_prophet, _ = prepare_prophet_format(test_processed, scaler)

    # Build and validate model
    model = Prophet(yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=False,
                    seasonality_mode='multiplicative')

    regressors = ['Open', 'High', 'Low', 'Volume', 'SMA_20', 'EMA_12', 'RSI',
                  'MACD', 'MACD_signal', 'BB_high', 'BB_low', 'Volatility',
                  'Price_Diff', 'Returns'] + [f'Close_lag_{lag}' for lag in [1,5,10]]

    for reg in regressors:
        model.add_regressor(reg)

    model.fit(train_prophet)

    # Make predictions
    future = test_prophet.drop('y', axis=1)
    # Make predictions
    forecast = model.predict(future)

    # Evaluate
    y_true = test_prophet['y'].values
    y_pred = forecast['yhat'].values

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred)/y_true)) * 100

    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.3f}%")
    # forecast.head()
