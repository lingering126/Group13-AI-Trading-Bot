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


class ABC:
    def __init__(self, objective_func, bounds, n_bees=50, n_iterations=100, limit=50):
        """
        Artificial Bee Colony Algorithm implementation

        Parameters:
        - objective_func: Objective function to optimize
        - bounds: Parameter bounds [(min, max), ...]
        - n_bees: Number of bees
        - n_iterations: Number of iterations
        - limit: Maximum number of trials
        """
        self.objective_func = objective_func
        self.bounds = bounds
        self.n_bees = n_bees
        self.n_iterations = n_iterations
        self.limit = limit
        self.dim = len(bounds)

        # Initialize bee positions
        self.positions = np.random.uniform(
            [b[0] for b in bounds],
            [b[1] for b in bounds],
            (n_bees, self.dim)
        )

        # Initialize fitness
        self.fitness = np.zeros(n_bees)
        self.trials = np.zeros(n_bees)
        self.best_solution = None
        self.best_fitness = float('-inf')

    def evaluate_fitness(self, position):
        """Evaluate fitness of a position"""
        return self.objective_func(position)

    def employed_bee_phase(self):
        """Employed bee phase"""
        for i in range(self.n_bees):
            # Randomly select a dimension
            j = np.random.randint(self.dim)
            # Randomly select a different bee
            k = np.random.randint(self.n_bees)
            while k == i:
                k = np.random.randint(self.n_bees)

            # Generate new solution
            new_position = self.positions[i].copy()
            phi = np.random.uniform(-1, 1)
            new_position[j] = self.positions[i][j] + phi * (self.positions[i][j] - self.positions[k][j])

            # Ensure new solution is within bounds
            new_position[j] = np.clip(new_position[j], self.bounds[j][0], self.bounds[j][1])

            # Evaluate new solution
            new_fitness = self.evaluate_fitness(new_position)

            # Update if new solution is better
            if new_fitness > self.fitness[i]:
                self.positions[i] = new_position
                self.fitness[i] = new_fitness
                self.trials[i] = 0
            else:
                self.trials[i] += 1

    def onlooker_bee_phase(self):
        """Onlooker bee phase"""
        # Calculate selection probabilities
        fitness_sum = np.sum(self.fitness)
        if fitness_sum == 0:
            probabilities = np.ones(self.n_bees) / self.n_bees
        else:
            probabilities = self.fitness / fitness_sum

        for _ in range(self.n_bees):
            # Roulette wheel selection
            i = np.random.choice(self.n_bees, p=probabilities)

            # Randomly select a dimension
            j = np.random.randint(self.dim)
            # Randomly select a different bee
            k = np.random.randint(self.n_bees)
            while k == i:
                k = np.random.randint(self.n_bees)

            # Generate new solution
            new_position = self.positions[i].copy()
            phi = np.random.uniform(-1, 1)
            new_position[j] = self.positions[i][j] + phi * (self.positions[i][j] - self.positions[k][j])

            # Ensure new solution is within bounds
            new_position[j] = np.clip(new_position[j], self.bounds[j][0], self.bounds[j][1])

            # Evaluate new solution
            new_fitness = self.evaluate_fitness(new_position)

            # Update if new solution is better
            if new_fitness > self.fitness[i]:
                self.positions[i] = new_position
                self.fitness[i] = new_fitness
                self.trials[i] = 0
            else:
                self.trials[i] += 1

    def scout_bee_phase(self):
        """Scout bee phase"""
        for i in range(self.n_bees):
            if self.trials[i] >= self.limit:
                # Reinitialize the position of this bee
                self.positions[i] = np.random.uniform(
                    [b[0] for b in self.bounds],
                    [b[1] for b in self.bounds],
                    self.dim
                )
                self.fitness[i] = self.evaluate_fitness(self.positions[i])
                self.trials[i] = 0

    def optimize(self):
        """Execute optimization"""
        # Initialize fitness
        for i in range(self.n_bees):
            self.fitness[i] = self.evaluate_fitness(self.positions[i])

        # Record best solution
        best_idx = np.argmax(self.fitness)
        self.best_solution = self.positions[best_idx].copy()
        self.best_fitness = self.fitness[best_idx]

        # Iterative optimization
        for iteration in range(self.n_iterations):
            self.employed_bee_phase()
            self.onlooker_bee_phase()
            self.scout_bee_phase()

            # Update best solution
            current_best_idx = np.argmax(self.fitness)
            if self.fitness[current_best_idx] > self.best_fitness:
                self.best_solution = self.positions[current_best_idx].copy()
                self.best_fitness = self.fitness[current_best_idx]

            print(f"Iteration {iteration + 1}/{self.n_iterations}, Best fitness: {self.best_fitness:.2f}")

        return self.best_solution, self.best_fitness


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


    # Create ABC optimizer
    abc = ABC(
        objective_func=objective_function,
        bounds=bounds,
        n_bees=50,
        n_iterations=100,
        limit=50
    )

    # Execute optimization
    best_params, best_fitness = abc.optimize()

    print("\nOptimization Results:")
    print(f"Best parameters: {best_params}")
    print(f"Best fitness: {best_fitness:.2f}")

    # Evaluate best parameters on test set
    test_fitness = evaluate(best_params, testing_data[1], plot=True)
    print(f"Test set fitness: {test_fitness:.2f}")