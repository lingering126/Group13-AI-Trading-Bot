#LATEST ADDITION
import numpy as np
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
from PSO import ParticleSwarmOptimizer

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
    sma = SMA(price, int(sma_length))
    lma = LMA(price, int(lma_length))
    ema = EMA(price, int(ema_length), ema_alpha)
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

def evaluate(parameters, prices, start_amount=1000, fee_rate=0.03, verbose=False, plot=False):
    """
    Evaluation function with all backtesting rules implemented inside.
    
    Parameters:
    - parameters: np.ndarray of strategy parameters (length = 14 for 2 WMA with 7 weights each)
    - prices: np.ndarray of historical prices
    - start_amount: starting USD capital
    - fee_rate: transaction fee (default 3%)

    Returns:
    - Final portfolio value in USD
    """

    # Split parameters between both wma
    short_params = parameters[:7]
    long_params = parameters[7:]

    # Calculate moving averages
    short_wma = combined_WMA(prices, *short_params)
    long_wma = combined_WMA(prices, *long_params)

    # Generate buy signal
    buy_signal = generate_ma_crossover_signals(short_wma, long_wma)

    # Simulate trading logic according to evaluation spec
    cash = start_amount
    btc = 0.0

    for i, signal in enumerate(buy_signal):
        current_price = prices[i]
        if signal == 1 and cash > 0: # Buy signal
            btc = cash / current_price * (1 - fee_rate)
            if verbose: print(f"Buying {btc:.2f}BTC with {cash:.2f}USD at time {i}")
            cash = 0.0
        elif signal == -1 and btc > 0:
            cash = btc * current_price * (1 - fee_rate)
            if verbose: print(f"Selling {btc:.2f}BTC into {cash:.2f}USD at time {i}")
            btc = 0.0

    # Final sell if BTC remaining
    if btc > 0:
        cash = btc * prices[-1] * (1 - fee_rate)

    if plot: plot_ma_signals(prices, short_wma, long_wma, buy_signal)

    return cash

def plot_ma_signals(prices, short_wma, long_wma, buy_signal, save_fig=False, fig_path=None):
    """
    Plot moving average signals
    
    Args:
        prices: Array of price data
        short_wma: Length of short moving average window
        long_wma: Length of long moving average window
        save_fig: Whether to save the figure
        fig_path: Path to save the figure
    """
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot price and moving averages
    plt.plot(prices, label='Price', alpha=0.5, linewidth=1)
    plt.plot(short_wma, label=f'Short WMA', linewidth=1.5)
    plt.plot(long_wma, label=f'Long WMA', linewidth=1.5)
    
    # Plot buy signals
    buy_indices = np.where(buy_signal == 1)[0]
    sell_indices = np.where(buy_signal == -1)[0]
    
    plt.plot(buy_indices, prices[buy_indices], '^', markersize=10, color='g', label='Buy Signal')
    plt.plot(sell_indices, prices[sell_indices], 'v', markersize=10, color='r', label='Sell Signal')
    
    plt.title('Moving Average Crossover Strategy')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    
    if save_fig and fig_path:
        import os
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)
        plt.savefig(fig_path)
        plt.close()
    else:
        plt.show()

PARAM_NAMES = ["Short SMA weight", "Short SMA length", "Short LMA weight", "Short LMA length", "Short EMA weight", "Short EMA length", "Short EMA alpha",
                "Long SMA weight", "Long SMA length", "Long LMA weight", "Long LMA length", "Long EMA weight", "Long EMA length", "Long EMA alpha"]

def print_parameters(parameters):
    for name, parameter in zip(PARAM_NAMES, parameters):
        if "length" in name:
            print(f"{name}: {int(parameter)}")
        else:
            print(f"{name}: {round(parameter, 6)}")

data = import_data(DATA_PATH)
training_data, testing_data = training_testing_split(data, datetime(2020, 1, 1).timestamp())

prices = training_data[1]

#EUSHA
# Defining the bounds for the parameters (14 total)
bounds = [
    [0, 1],    # short_wma_weight_1
    [1, 500], # short_wma_length_1
    [0, 1],    # short_wma_weight_2
    [1, 500], # short_wma_length_2
    [0, 1],    # short_wma_weight_3
    [1, 500], # short_wma_length_3
    [0.01, 1],    # short_wma_alpha for EMA
    [0, 1],    # long_wma_weight_1
    [1, 500], # long_wma_length_1
    [0, 1],    # long_wma_weight_2
    [1, 500], # long_wma_length_2
    [0, 1],    # long_wma_weight_3
    [1, 500], # long_wma_length_3
    [0.01, 1]     # long_wma_alpha for EMA (second one)
]

# Wrapping the evaluate function to accept only weights
def fitness_function(weights):
    return evaluate(weights, prices)

# Initialize and run PSO
pso = ParticleSwarmOptimizer(
    fitness_function=fitness_function,
    bounds=bounds,
    num_particles=100,
    max_iter=100
)

best_params, best_score = pso.optimize()

print(best_params, best_score)
print_parameters(best_params)
evaluate(best_params, prices=training_data[1], plot=True)
