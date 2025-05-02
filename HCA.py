import numpy as np
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt

DATA_PATH = "BTC-Daily.csv"


def import_data(path):
    """
    Import Bitcoin price data
    Returns: 2D numpy array with unix timestamp and closing prices as rows
    """
    df = pd.read_csv(path)
    arr = df[["unix", "close"]].to_numpy().T
    arr = arr[:, arr[0, :].argsort()]
    return arr


def training_testing_split(data, unix_cutoff):
    training = data[:, data[0] < unix_cutoff]
    testing = data[:, data[0] >= unix_cutoff]
    return training, testing


def pad(price, days):
    """ Pad the price array """
    padding = -np.flip(price[1:days])
    return np.append(padding, price)


def WMA(price, days, kernel):
    price = pad(price, days)
    wma = np.correlate(price, kernel, mode="valid")
    return wma


def SMA(price, days):
    """Simple Moving Average"""
    K = 1 / days * np.ones(days)
    return WMA(price, days, K)


def LMA(price, days):
    """Linear Moving Average"""
    K = 2 * np.linspace(0, 1 / days, days)
    return WMA(price, days, K)


def EMA(price, days, alpha):
    """Exponential Moving Average"""
    K = alpha * np.power(1 - alpha, np.linspace(0, days, days)[::-1])
    K_sum = np.sum(K)
    if K_sum > 0:  # Prevent division by zero
        K /= K_sum
    return WMA(price, days, K)


def combined_WMA(price, sma_weight, sma_length, lma_weight, lma_length, ema_weight, ema_length, ema_alpha):
    """Combined Weighted Moving Average"""
    # Ensure all length parameters are integers
    sma_length = int(round(sma_length))
    lma_length = int(round(lma_length))
    ema_length = int(round(ema_length))

    # Calculate individual moving averages
    sma = SMA(price, sma_length)
    lma = LMA(price, lma_length)
    ema = EMA(price, ema_length, ema_alpha)

    # Calculate total weight
    total_weight = sma_weight + lma_weight + ema_weight
    if total_weight > 0:  # Prevent division by zero
        return (sma_weight * sma + lma_weight * lma + ema_weight * ema) / total_weight
    else:
        return (sma + lma + ema) / 3  # Fallback to simple average if all weights are zero


def generate_ma_crossover_signals(short_ma, long_ma):
    """Generate moving average crossover signals"""
    if len(short_ma) != len(long_ma):
        print("Warning: Moving averages have different lengths")
        min_len = min(len(short_ma), len(long_ma))
        short_ma = short_ma[-min_len:]
        long_ma = long_ma[-min_len:]

    crossover_signals = np.zeros(len(short_ma), dtype=int)
    diff = short_ma - long_ma

    for i in range(1, len(short_ma)):
        if diff[i] > 0 and diff[i - 1] <= 0:
            crossover_signals[i] = 1
        elif diff[i] < 0 and diff[i - 1] >= 0:
            crossover_signals[i] = -1

    return crossover_signals


def evaluate(parameters, prices, start_amount=1000, fee_rate=0.03, verbose=False, plot=False):
    """Evaluate trading strategy performance"""
    # Ensure all length parameters are integers
    integer_params = [1, 3, 5, 8, 10, 12]
    for i in integer_params:
        parameters[i] = int(round(parameters[i]))

    # Split parameters into short-term and long-term
    short_params = parameters[:7]
    long_params = parameters[7:]

    # Calculate short-term and long-term moving averages
    short_wma = combined_WMA(prices,
                             short_params[0],  # sma_weight
                             short_params[1],  # sma_length
                             short_params[2],  # lma_weight
                             short_params[3],  # lma_length
                             short_params[4],  # ema_weight
                             short_params[5],  # ema_length
                             short_params[6])  # ema_alpha

    long_wma = combined_WMA(prices,
                            long_params[0],  # sma_weight
                            long_params[1],  # sma_length
                            long_params[2],  # lma_weight
                            long_params[3],  # lma_length
                            long_params[4],  # ema_weight
                            long_params[5],  # ema_length
                            long_params[6])  # ema_alpha

    buy_signal = generate_ma_crossover_signals(short_wma, long_wma)

    cash = start_amount
    btc = 0.0

    for i, signal in enumerate(buy_signal):
        current_price = prices[i]
        if signal == 1 and cash > 0:
            btc = cash / current_price * (1 - fee_rate)
            if verbose: print(f"Buy {btc:.2f}BTC with {cash:.2f}USD at time {i}")
            cash = 0.0
        elif signal == -1 and btc > 0:
            cash = btc * current_price * (1 - fee_rate)
            if verbose: print(f"Sell {btc:.2f}BTC for {cash:.2f}USD at time {i}")
            btc = 0.0

    if btc > 0:
        cash = btc * prices[-1] * (1 - fee_rate)

    if plot: plot_ma_signals(prices, short_wma, long_wma, buy_signal)

    return cash


def plot_ma_signals(prices, short_wma, long_wma, buy_signal, save_fig=False, fig_path=None):
    """Plot moving average signals"""
    plt.figure(figsize=(12, 8))

    plt.plot(prices, label='Price', alpha=0.5, linewidth=1)
    plt.plot(short_wma, label=f'Short WMA', linewidth=1.5)
    plt.plot(long_wma, label=f'Long WMA', linewidth=1.5)

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


class HCA:
    def __init__(self, objective_func, bounds, step_size=0.1, n_iterations=1000):
        """
        Hill Climbing Algorithm implementation

        Parameters:
        - objective_func: Objective function to optimize
        - bounds: Parameter bounds [(min, max), ...]
        - step_size: Size of step for parameter updates
        - n_iterations: Number of iterations
        """
        self.objective_func = objective_func
        self.bounds = bounds
        self.step_size = step_size
        self.n_iterations = n_iterations
        self.dim = len(bounds)

        # Initialize current position
        self.current_position = np.random.uniform(
            [b[0] for b in bounds],
            [b[1] for b in bounds]
        )

        # Initialize current fitness
        self.current_fitness = self.objective_func(self.current_position)

    def generate_neighbor(self, position):
        """Generate a neighboring solution"""
        neighbor = position.copy()

        # Randomly select a dimension to modify
        dim = np.random.randint(self.dim)

        # Generate a random step
        step = np.random.uniform(-self.step_size, self.step_size)

        # Update the selected dimension
        neighbor[dim] += step

        # Ensure the new value is within bounds
        neighbor[dim] = np.clip(neighbor[dim], self.bounds[dim][0], self.bounds[dim][1])

        return neighbor

    def optimize(self):
        """Execute optimization"""
        best_position = self.current_position.copy()
        best_fitness = self.current_fitness

        for iteration in range(self.n_iterations):
            # Generate a neighbor
            neighbor = self.generate_neighbor(self.current_position)

            # Evaluate the neighbor
            neighbor_fitness = self.objective_func(neighbor)

            # If the neighbor is better, move to it
            if neighbor_fitness > self.current_fitness:
                self.current_position = neighbor
                self.current_fitness = neighbor_fitness

                # Update best solution if necessary
                if neighbor_fitness > best_fitness:
                    best_position = neighbor.copy()
                    best_fitness = neighbor_fitness

            print(
                f"Iteration {iteration + 1}/{self.n_iterations}, Current fitness: {self.current_fitness:.2f}, Best fitness: {best_fitness:.2f}")

        return best_position, best_fitness


# Main program
if __name__ == "__main__":
    # Import data
    data = import_data(DATA_PATH)
    training_data, testing_data = training_testing_split(data, datetime(2020, 1, 1).timestamp())

    # Define parameter bounds
    bounds = [
        (0, 1), (1, 100), (0, 1),  # SMA parameters
        (1, 100), (0, 1), (1, 100),  # LMA parameters
        (0, 1), (0, 1), (1, 100),  # EMA parameters
        (0, 1), (1, 100), (0, 1),  # SMA parameters
        (1, 100), (0, 1), (1, 100),  # LMA parameters
        (0, 1)  # EMA parameters
    ]


    # Create objective function
    def objective_function(params):
        return evaluate(params, training_data[1])


    # Create HCA optimizer
    hca = HCA(
        objective_func=objective_function,
        bounds=bounds,
        step_size=0.1,
        n_iterations=1000
    )

    # Execute optimization
    best_params, best_fitness = hca.optimize()

    print("\nOptimization Results:")
    print(f"Best parameters: {best_params}")
    print(f"Best fitness: {best_fitness:.2f}")

    # Evaluate best parameters on test set
    test_fitness = evaluate(best_params, testing_data[1], plot=True)
    print(f"Test set fitness: {test_fitness:.2f}")