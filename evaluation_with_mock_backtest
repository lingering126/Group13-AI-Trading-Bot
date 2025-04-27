import numpy as np

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
