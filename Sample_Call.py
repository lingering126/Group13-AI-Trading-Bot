import pandas as pd

# Load the daily data CSV file
df = pd.read_csv('BTC-Daily.csv')  

# print(df.head()) -- was used to check for the column names

# Converting 'date' column to datetime format
df['date'] = pd.to_datetime(df['date'])

# Sorting the data by date
df.sort_values('date', inplace=True)

# Resetting the index
df.reset_index(drop=True, inplace=True)

# Extract the 'Close' prices as a NumPy array
close_prices = df['close'].values


def sma(prices, window):
    return np.convolve(prices, np.ones(window)/window, mode='valid')

# Real trading fitness function
def trading_bot_fitness(params):
    short_window = int(params[0])
    long_window = int(params[1])

    if short_window >= long_window or long_window >= len(close_prices) - 1:
        return -1e9  # Penalize invalid configurations

    short_sma = sma(close_prices, short_window)
    long_sma = sma(close_prices, long_window)

    min_len = min(len(short_sma), len(long_sma))
    short_sma = short_sma[-min_len:]
    long_sma = long_sma[-min_len:]
    prices = close_prices[-min_len:]

    usd = 1000
    btc = 0
    prev_diff = 0

    for i in range(min_len):
        diff = short_sma[i] - long_sma[i]
        price = prices[i]

        if diff > 0 and prev_diff <= 0 and usd > 0:
            btc = (usd * 0.97) / price
            usd = 0
        elif diff < 0 and prev_diff >= 0 and btc > 0:
            usd = (btc * price) * 0.97
            btc = 0

        prev_diff = diff

    if btc > 0:
        usd = (btc * prices[-1]) * 0.97

    return usd

# Define bounds for short and long SMA windows
bounds = [
    [5, 30],    # short_window
    [31, 90],   # long_window
]

# Initialize and run PSO
pso = ParticleSwarmOptimizer(
    fitness_function=trading_bot_fitness,
    bounds=bounds,
    num_particles=30,
    max_iter=50
)

best_params, best_score = pso.optimize()

print("📈 Best Parameters Found:", best_params)
print("💰 Final Portfolio Value:", best_score)
