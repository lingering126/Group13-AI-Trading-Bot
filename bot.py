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

def generate_ma_crossover_signals(self, short_window, long_window, ma_type="SMA"):
    """
    Generate moving average crossover signals
    
    This function creates trading signals based on crossover of short and long-term
    moving averages. 
    
    Args:
        short_window: size of short-term moving average window
        long_window: size of long-term moving average window
        ma_type: type of moving average to use ('SMA', 'EMA', 'LMA')
        
    Returns:
        Tuple containing:
        - numpy array of signals: +1 (buy), -1 (sell), 0 (hold)
        - short moving average values
        - long moving average values
    """
    # Calculate moving averages based on type
    if ma_type == "SMA":
        short_ma = SMA(self.prices, short_window)
        long_ma = SMA(self.prices, long_window)
    elif ma_type == "EMA":
        short_ma = EMA(self.prices, short_window, 2/(short_window+1))
        long_ma = EMA(self.prices, long_window, 2/(long_window+1))
    elif ma_type == "LMA":
        short_ma = LMA(self.prices, short_window)
        long_ma = LMA(self.prices, long_window)
    else:
        raise ValueError(f"Unsupported MA type: {ma_type}")
    
    # Ensure arrays have the same length
    # This is important because different MA types/windows can produce arrays of different lengths
    min_len = min(len(short_ma), len(long_ma))
    short_ma = short_ma[-min_len:]
    long_ma = long_ma[-min_len:]
    
    # Generate signals based on crossovers
    # Signal convention: +1 (buy), -1 (sell), 0 (hold)
    # This matches the expected format in the backtesting module
    signals = np.zeros(min_len, dtype=int)
    prev_diff = short_ma[0] - long_ma[0]
    
    for i in range(1, min_len):
        curr_diff = short_ma[i] - long_ma[i]
        
        # Buy signal: short MA crosses above long MA
        if curr_diff > 0 and prev_diff <= 0:
            signals[i] = 1
        # Sell signal: short MA crosses below long MA
        elif curr_diff < 0 and prev_diff >= 0:
            signals[i] = -1
            
        prev_diff = curr_diff
    
    return signals, short_ma, long_ma

data = import_data(DATA_PATH)
training_data, testing_data = training_testing_split(data, datetime(2020, 1, 1).timestamp())

wma = EMA(training_data[1], 100, 2/101)

from matplotlib import pyplot as plt
plt.plot(training_data[0], training_data[1], label="y")
plt.plot(training_data[0], wma, label="WMA", linestyle="--")
plt.legend()
plt.title("Training Data and EMA")
plt.xlabel("Index")
plt.ylabel("Price")
plt.show()

