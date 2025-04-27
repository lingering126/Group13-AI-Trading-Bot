import numpy as np
import pandas as pd
from datetime import datetime

DATA_PATH = "data/BTC-Daily.csv"

def import_data(path):
    """ 
    Imports Bitcoin price data
    Returns: a 2d numpy array with unix timestamp and opening prices as the rows
    """
    df = pd.read_csv(path)
    arr = df[["unix", "close"]].to_numpy().T # Transpose so timestamp and price are rows
    arr = arr[:, arr[0,:].argsort()]         # Sort in ascending order of the first row (timestamp)
    return arr

def training_testing_split(data, unix_cutoff):
    training = data[:, data[0] < unix_cutoff]
    testing = data[:, data[0] >= unix_cutoff]
    return training, testing

def pad(price, days):
    """ Pads the price array as done in the task sheet """
    padding = -np.flip(price[1:days])
    return np.append(padding, price)

def WMA(price, days, kernel):
    price = pad(price, days)
    wma = np.correlate(price, kernel, mode="valid")
    print(kernel, sum(kernel))
    return wma

def SMA(price, days):
    """Simple moving average"""
    K = 1/days * np.ones(days)
    return WMA(price, days, K)

def LMA(price, days):
    """Linear moving average"""
    K = 2 * np.linspace(0, 1/days, days)
    return WMA(price, days, K)

def EMA(price, days, alpha):
    """Exponential moving average"""
    K = alpha * np.power(1 - alpha, np.linspace(0, days, days)[::-1])
    K/= sum(K)
    return WMA(price, days, K)

def combined_WMA(price, sma_weight, sma_length, lma_weight, lma_length, ema_weight, ema_length, ema_alpha):
    sma = SMA(price, sma_length)
    lma = LMA(price, lma_length)
    ema = EMA(price, ema_length, ema_alpha)
    return (sma_weight * sma + lma_weight * lma + ema_weight * ema) / (sma_weight + lma_weight + ema_weight)

def generate_ma_crossover_signals(short_ma, long_ma):
    """
    Generate moving average crossover signals
    
    This function creates trading signals based on crossover of short and long-term
    moving averages. 
    
    Args:
        short_ma: array containing the more reactive weighted moving averages
        long_ma: aeeay containing the less reactive weighted moving averages
        
    Returns:
        Tuple containing:
        - numpy array of signals: +1 (buy), -1 (sell), 0 (hold)
        - short moving average values
        - long moving average values
    """
    
    # Ensure arrays have the same length
    # This is important because different MA types/windows can produce arrays of different lengths
    if len(short_ma) != len(long_ma):
        print("WARNING: Moving averages are of different lengths")
        min_len = min(len(short_ma), len(long_ma))
        short_ma = short_ma[-min_len:]
        long_ma = long_ma[-min_len:]
    
    # Generate signals based on crossovers
    # Signal convention: +1 (buy), -1 (sell), 0 (hold)
    # This matches the expected format in the backtesting module
    crossover_signals = np.zeros(len(short_ma), dtype=int)
    diff = short_ma - long_ma
    
    for i in range(1, len(short_ma)):
        
        # Buy signal: short MA crosses above long MA
        if diff[i] > 0 and diff[i-1] <= 0:
            crossover_signals[i] = 1
        # Sell signal: short MA crosses below long MA
        elif diff[i] < 0 and diff[i-1] >= 0:
            crossover_signals[i] = -1
    
    return crossover_signals

def evaluate(weights, prices, start_amount=1000, fee_rate=0.03):
    """
    Evaluation function with all backtesting rules implemented inside.
    
    Parameters:
    - weights: np.ndarray of strategy parameters (length = 14 for 2 WMA with 7 weights each)
    - prices: np.ndarray of historical prices
    - start_amount: starting USD capital
    - fee_rate: transaction fee (default 3%)

    Returns:
    - Final portfolio value in USD
    """

    # Placeholder for WMA computation (just simulate)
    short_weights = weights[:7]
    long_weights = weights[7:]

    # Normalize weights
    short_weights /= short_weights.sum()
    long_weights /= long_weights.sum()

    # Calculate weighted moving averages
    def wma(prices, weights):
        w = len(weights)
        return np.convolve(prices, weights[::-1], mode='valid')

    short_wma = wma(prices, short_weights)
    long_wma = wma(prices, long_weights)

    # Align all arrays to the same length
    min_len = min(len(short_wma), len(long_wma))
    short_wma = short_wma[-min_len:]
    long_wma = long_wma[-min_len:]
    prices = prices[-min_len:]

    # Simulate trading logic according to evaluation spec
    cash = start_amount
    btc = 0.0
    prev_diff = 0

    for i in range(min_len):
        diff = short_wma[i] - long_wma[i]
        price = prices[i]

        if diff > 0 and prev_diff <= 0 and cash > 0:
            btc = (cash * (1 - fee_rate)) / price
            cash = 0.0
        elif diff < 0 and prev_diff >= 0 and btc > 0:
            cash = btc * price * (1 - fee_rate)
            btc = 0.0

        prev_diff = diff

    # Final sell if BTC remaining
    if btc > 0:
        cash = btc * prices[-1] * (1 - fee_rate)

    return cash

data = import_data(DATA_PATH)
training_data, testing_data = training_testing_split(data, datetime(2020, 1, 1).timestamp())

wma1 = combined_WMA(training_data[1], 0.8, 100, 0.1, 100, 0.1, 100, 0.5)
wma2 = combined_WMA(training_data[1], 0.8, 10, 0.1, 10, 0.1, 10, 0.5)

buy_signal = generate_ma_crossover_signals(wma2, wma1)

from matplotlib import pyplot as plt
plt.plot(training_data[0], training_data[1], label="y")
plt.plot(training_data[0], wma1, label="WMA1", linestyle="--")
plt.plot(training_data[0], wma2, label="WMA2", linestyle="--")
plt.plot(training_data[0], 10000*buy_signal, label="buy signal")
plt.legend()
plt.title("Training Data and EMA")
plt.xlabel("Timestamp")
plt.ylabel("Price")
plt.show()

