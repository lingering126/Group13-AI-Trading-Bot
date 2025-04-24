import numpy as np
import pandas as pd
from datetime import datetime

# Import necessary functions from bot.py
# This allows the signal generator to use the existing implementation of these functions
# from the bot.py file if they are available, maintaining compatibility with the existing codebase
try:
    from bot import import_data, training_testing_split, SMA, EMA, LMA, WMA, pad
except ImportError:
    # Define these functions in case they're not available from bot.py
    # This fallback ensures the signal generator can work standalone if needed
    # while still using the same function signatures as in bot.py
    def pad(price, days):
        """ 
        Pads the price array as done in the task sheet 
        Used for calculating moving averages with proper lookback periods
        """
        padding = np.flip(price[1:days])
        return np.append(padding, price)

    def WMA(price, days, kernel):
        """
        Calculates Weighted Moving Average using the provided kernel weights
        
        Args:
            price: Array of price data
            days: Length of the moving average window
            kernel: Array of weights for the moving average
            
        Returns:
            Array of weighted moving average values
        """
        price = pad(price, days)
        wma = np.correlate(price, kernel, mode="valid")
        return wma

    def SMA(price, days):
        """
        Simple moving average - equal weights for all days in the window
        
        Args:
            price: Array of price data
            days: Length of the moving average window
            
        Returns:
            Array of simple moving average values
        """
        K = 1/days * np.ones(days)  # Equal weights for all days
        return WMA(price, days, K)

    def EMA(price, days, alpha):
        """
        Exponential moving average - more weight to recent prices
        
        Args:
            price: Array of price data
            days: Length of the moving average window
            alpha: Smoothing factor (typically 2/(days+1))
            
        Returns:
            Array of exponential moving average values
        """
        K = alpha * np.power(1 - alpha, np.linspace(0, days, days)[::-1])
        K/= sum(K)  # Normalize weights to sum to 1
        return WMA(price, days, K)

    def LMA(price, days):
        """
        Linear moving average - linearly increasing weights
        
        Args:
            price: Array of price data
            days: Length of the moving average window
            
        Returns:
            Array of linear moving average values
        """
        K = 2 * np.linspace(0, 1/days, days)  # Linearly increasing weights
        return WMA(price, days, K)

class SignalGenerator:
    def __init__(self, prices=None, data_path=None):
        """
        Initialize the signal generator with price data
        
        This class takes either direct price data or a path to a CSV file containing price data
        and provides methods to generate various types of trading signals.
        
        Args:
            prices: numpy array of price data (if already available)
            data_path: path to CSV file with price data (alternative to prices)
            
        Note: 
            The SignalGenerator is designed to work with both preprocessed price arrays
            and raw CSV data files that follow the project's expected format.
        """
        if prices is not None:
            self.prices = prices
        elif data_path is not None:
            # Import data from CSV
            # This follows the same format expected by other parts of the project
            df = pd.read_csv(data_path)
            self.prices = df['close'].values  # Using 'close' column like in bot.py
        else:
            raise ValueError("Either prices or data_path must be provided")
    
    def generate_ma_crossover_signals(self, short_window, long_window, ma_type="SMA"):
        """
        Generate moving average crossover signals
        
        This function creates trading signals based on crossovers of short and long-term
        moving averages. It's designed to be compatible with the moving average functions
        from bot.py.
        
        Args:
            short_window: size of short-term moving average window
            long_window: size of long-term moving average window
            ma_type: type of moving average to use ('SMA', 'EMA', 'LMA')
            
        Returns:
            Tuple containing:
            - numpy array of signals: +1 (buy), -1 (sell), 0 (hold)
            - short moving average values
            - long moving average values
            
        Integration points:
        - Uses the same MA functions as bot.py
        - Supports all MA types implemented in the project
        - Returns signals in the format expected by the backtesting system
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
    
    def generate_rsi_signals(self, window=14, overbought=70, oversold=30):
        """
        Generate signals based on Relative Strength Index (RSI)
        
        RSI is a momentum oscillator that measures the speed and change of price movements.
        Traditional interpretation: RSI > 70 = overbought, RSI < 30 = oversold.
        
        Args:
            window: RSI calculation window
            overbought: Threshold for overbought condition (sell signal)
            oversold: Threshold for oversold condition (buy signal)
            
        Returns:
            Tuple containing:
            - numpy array of signals: +1 (buy), -1 (sell), 0 (hold)
            - RSI values
            
        Integration points:
        - Uses the standard RSI formula that is industry-standard
        - Signal output format matches existing project expectations
        """
        # Calculate RSI
        rsi = self._calculate_rsi(window)
        
        # Generate signals
        signals = np.zeros(len(rsi), dtype=int)
        
        for i in range(1, len(rsi)):
            # Buy signal: RSI crosses above oversold threshold (typically 30)
            # This indicates a potential reversal from oversold conditions
            if rsi[i] > oversold and rsi[i-1] <= oversold:
                signals[i] = 1
            # Sell signal: RSI crosses below overbought threshold (typically 70)
            # This indicates a potential reversal from overbought conditions
            elif rsi[i] < overbought and rsi[i-1] >= overbought:
                signals[i] = -1
        
        return signals, rsi
    
    def generate_macd_signals(self, fast_period=12, slow_period=26, signal_period=9):
        """
        Generate signals based on MACD (Moving Average Convergence Divergence)
        
        MACD is calculated by subtracting the long-term EMA from the short-term EMA.
        The Signal line is an EMA of the MACD line.
        
        Args:
            fast_period: Fast EMA period (typically 12)
            slow_period: Slow EMA period (typically 26)
            signal_period: Signal line period (typically 9)
            
        Returns:
            Tuple containing:
            - numpy array of signals: +1 (buy), -1 (sell), 0 (hold)
            - MACD line values
            - Signal line values
            - Histogram values (MACD - Signal)
            
        Integration points:
        - Uses the same EMA calculation method as in other parts of the project
        - Follows standard MACD calculation methodology
        """
        # Calculate MACD components
        fast_ema = EMA(self.prices, fast_period, 2/(fast_period+1))
        slow_ema = EMA(self.prices, slow_period, 2/(slow_period+1))
        
        # Ensure arrays have the same length
        min_len = min(len(fast_ema), len(slow_ema))
        fast_ema = fast_ema[-min_len:]
        slow_ema = slow_ema[-min_len:]
        
        # Calculate MACD line (fast EMA - slow EMA)
        macd_line = fast_ema - slow_ema
        
        # Calculate signal line (EMA of MACD line)
        signal_line = np.zeros_like(macd_line)
        signal_line[:signal_period] = np.mean(macd_line[:signal_period])
        alpha = 2 / (signal_period + 1)
        
        for i in range(signal_period, len(macd_line)):
            signal_line[i] = alpha * macd_line[i] + (1 - alpha) * signal_line[i-1]
        
        # Calculate histogram (MACD line - signal line)
        # Used for visualization and additional analysis
        histogram = macd_line - signal_line
        
        # Generate signals
        signals = np.zeros(len(macd_line), dtype=int)
        
        for i in range(1, len(macd_line)):
            # Buy signal: MACD line crosses above signal line
            if macd_line[i] > signal_line[i] and macd_line[i-1] <= signal_line[i-1]:
                signals[i] = 1
            # Sell signal: MACD line crosses below signal line
            elif macd_line[i] < signal_line[i] and macd_line[i-1] >= signal_line[i-1]:
                signals[i] = -1
        
        return signals, macd_line, signal_line, histogram
    
    def generate_bollinger_band_signals(self, window=20, num_std=2):
        """
        Generate signals based on Bollinger Bands
        
        Bollinger Bands consist of:
        - Middle band: n-period moving average
        - Upper band: Middle band + (K * n-period standard deviation)
        - Lower band: Middle band - (K * n-period standard deviation)
        
        Args:
            window: Window for moving average (typically 20)
            num_std: Number of standard deviations for bands (typically 2)
            
        Returns:
            Tuple containing:
            - numpy array of signals: +1 (buy), -1 (sell), 0 (hold)
            - Middle band values (SMA)
            - Upper band values
            - Lower band values
            
        Integration points:
        - Uses the same SMA calculation as the rest of the project
        - Signal generation follows traditional Bollinger Band interpretation
        """
        # Calculate moving average (middle band)
        ma = SMA(self.prices, window)
        
        # Calculate standard deviation
        std = np.zeros(len(ma))
        
        # Calculate rolling standard deviation
        for i in range(len(ma)):
            idx = max(0, i - window + 1)
            std[i] = np.std(self.prices[idx:i+1])
        
        # Calculate Bollinger Bands
        upper_band = ma + (std * num_std)
        lower_band = ma - (std * num_std)
        
        # Align prices with bands (needed because MA calculation reduces array length)
        prices_aligned = self.prices[-len(ma):]
        
        # Generate signals
        signals = np.zeros(len(ma), dtype=int)
        
        for i in range(1, len(ma)):
            # Buy signal: Price crosses below lower band and then back above it
            # This indicates a potential bullish reversal after a price drop
            if (prices_aligned[i-1] <= lower_band[i-1] and 
                prices_aligned[i] > lower_band[i]):
                signals[i] = 1
            # Sell signal: Price crosses above upper band and then back below it
            # This indicates a potential bearish reversal after a price rise
            elif (prices_aligned[i-1] >= upper_band[i-1] and 
                  prices_aligned[i] < upper_band[i]):
                signals[i] = -1
        
        return signals, ma, upper_band, lower_band
    
    def generate_combined_signals(self, strategies, weights=None):
        """
        Combine multiple signal strategies with optional weighting
        
        This allows for a more robust trading approach that doesn't rely on just one indicator.
        
        Args:
            strategies: List of (signals, name) tuples from different strategies
            weights: Dictionary of weights for each strategy (default: equal weights)
            
        Returns:
            Tuple containing:
            - numpy array of signals: +1 (buy), -1 (sell), 0 (hold)
            - Combined signal strength array
            
        Integration points:
        - Can combine any number of strategies with flexible weighting
        - Provides a more robust signal generation approach
        - Returns signals in the same format as individual strategies
        """
        if not strategies:
            raise ValueError("No strategies provided")
        
        # Determine the minimum length of all signals
        # This ensures all signal arrays are aligned properly
        min_len = min(len(signals) for signals, _ in strategies)
        
        # Initialize combined signal strength
        combined = np.zeros(min_len)
        
        # Set default weights if not provided (equal weighting)
        if weights is None:
            weights = {name: 1.0 for _, name in strategies}
        
        # Combine signals with weights
        for signals, name in strategies:
            if name in weights:
                combined += weights[name] * signals[-min_len:]
        
        # Calculate threshold for signals
        # The threshold is half the total weight, which creates a balanced approach
        total_weight = sum(weights.values())
        threshold = total_weight * 0.5
        
        # Generate signals based on combined strength
        signals = np.zeros(min_len, dtype=int)
        prev_combined = combined[0]
        
        for i in range(1, min_len):
            # Buy signal: Combined strength crosses above threshold
            if combined[i] > threshold and prev_combined <= threshold:
                signals[i] = 1
            # Sell signal: Combined strength crosses below -threshold
            elif combined[i] < -threshold and prev_combined >= -threshold:
                signals[i] = -1
            
            prev_combined = combined[i]
        
        return signals, combined
    
    def _calculate_rsi(self, window=14):
        """
        Calculate Relative Strength Index
        
        RSI = 100 - (100 / (1 + RS))
        RS = Average Gain / Average Loss
        
        Args:
            window: Period for RSI calculation (typically 14)
            
        Returns:
            Array of RSI values
            
        Note:
            This is a private helper method used by generate_rsi_signals
            Implements the standard RSI calculation formula
        """
        # Calculate price changes
        deltas = np.diff(self.prices)
        
        # Separate gains and losses
        gains = np.maximum(deltas, 0)
        losses = -np.minimum(deltas, 0)
        
        # Initialize arrays
        avg_gain = np.zeros_like(deltas)
        avg_loss = np.zeros_like(deltas)
        
        # Calculate first average gain and loss (simple average)
        avg_gain[window-1] = np.mean(gains[:window])
        avg_loss[window-1] = np.mean(losses[:window])
        
        # Calculate subsequent values using the smoothing formula
        # This follows the standard Wilder's RSI formula
        for i in range(window, len(deltas)):
            avg_gain[i] = (avg_gain[i-1] * (window-1) + gains[i]) / window
            avg_loss[i] = (avg_loss[i-1] * (window-1) + losses[i]) / window
        
        # Calculate RS (Relative Strength)
        rs = np.zeros_like(avg_gain)
        non_zero_losses = avg_loss != 0
        rs[non_zero_losses] = avg_gain[non_zero_losses] / avg_loss[non_zero_losses]
        
        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

# Example usage
if __name__ == "__main__":
    # Load data from CSV
    data_path = "data/BTC-Daily.csv"
    
    try:
        # Try to use import_data from bot.py
        data = import_data(data_path)
        prices = data[1]  # Assuming prices are in the second row
    except (NameError, ImportError):
        # Fallback to pandas
        try:
            df = pd.read_csv(data_path)
            prices = df['close'].values
        except:
            print("Error: Could not load data. Please provide a valid data path.")
            exit(1)
    
    # Create signal generator
    sg = SignalGenerator(prices=prices)
    
    # Generate signals using different strategies
    ma_signals, short_ma, long_ma = sg.generate_ma_crossover_signals(10, 50, "SMA")
    rsi_signals, rsi = sg.generate_rsi_signals(14, 70, 30)
    macd_signals, macd_line, signal_line, histogram = sg.generate_macd_signals(12, 26, 9)
    bb_signals, ma, upper_band, lower_band = sg.generate_bollinger_band_signals(20, 2)
    
    # Combine signals
    strategies = [
        (ma_signals, "MA Crossover"),
        (rsi_signals, "RSI"),
        (macd_signals, "MACD"),
        (bb_signals, "Bollinger Bands")
    ]
    
    weights = {
        "MA Crossover": 1.0,
        "RSI": 1.0,
        "MACD": 1.0,
        "Bollinger Bands": 1.0
    }
    
    combined_signals, combined_strength = sg.generate_combined_signals(strategies, weights)
    
    # Count signals
    ma_buy_count = np.sum(ma_signals == 1)
    ma_sell_count = np.sum(ma_signals == -1)
    
    rsi_buy_count = np.sum(rsi_signals == 1)
    rsi_sell_count = np.sum(rsi_signals == -1)
    
    macd_buy_count = np.sum(macd_signals == 1)
    macd_sell_count = np.sum(macd_signals == -1)
    
    bb_buy_count = np.sum(bb_signals == 1)
    bb_sell_count = np.sum(bb_signals == -1)
    
    combined_buy_count = np.sum(combined_signals == 1)
    combined_sell_count = np.sum(combined_signals == -1)
    
    print(f"MA Crossover: {ma_buy_count} buy signals, {ma_sell_count} sell signals")
    print(f"RSI: {rsi_buy_count} buy signals, {rsi_sell_count} sell signals")
    print(f"MACD: {macd_buy_count} buy signals, {macd_sell_count} sell signals")
    print(f"Bollinger Bands: {bb_buy_count} buy signals, {bb_sell_count} sell signals")
    print(f"Combined: {combined_buy_count} buy signals, {combined_sell_count} sell signals") 