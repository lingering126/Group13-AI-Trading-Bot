import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class SignalGenerator:
    """
    Signal Generator for trading strategies
    
    This class generates trading signals based on technical analysis of price data.
    """
    
    def __init__(self, prices=None):
        """Initialize the signal generator with optional price data"""
        self.prices = prices
    
    def generate_ma_signals(self, prices, short_window=10, long_window=50):
        """
        Generate signals based on Moving Average crossover strategy
        
        Args:
            prices: Array of price data
            short_window: Length of short moving average window
            long_window: Length of long moving average window
            
        Returns:
            Tuple of (signals array, indicators dict)
        """
        # Check for sufficient data
        if len(prices) < long_window:
            raise ValueError(f"Not enough data points. Need at least {long_window} data points.")
        
        # Calculate moving averages
        short_ma = self._calculate_ma(prices, short_window)
        long_ma = self._calculate_ma(prices, long_window)
        
        # Generate signals
        signals = np.zeros(len(prices))
        
        # Ensure arrays are the same length by padding with zeros
        pad_length = long_window - 1
        short_ma_padded = np.pad(short_ma, (pad_length, 0), 'constant')
        long_ma_padded = np.pad(long_ma, (pad_length, 0), 'constant')
        
        # Only generate signals after both MAs are available
        for i in range(long_window, len(prices)):
            # Buy signal: short MA crosses above long MA
            if short_ma_padded[i] > long_ma_padded[i] and short_ma_padded[i-1] <= long_ma_padded[i-1]:
                signals[i] = 1
            # Sell signal: short MA crosses below long MA
            elif short_ma_padded[i] < long_ma_padded[i] and short_ma_padded[i-1] >= long_ma_padded[i-1]:
                signals[i] = -1
        
        # Store indicators
        indicators = {
            'short_ma': short_ma_padded,
            'long_ma': long_ma_padded
        }
        
        return signals, indicators
    
    def generate_ma_crossover_signals(self, short_window, long_window, ma_type="SMA"):
        """
        Generate moving average crossover signals
        
        This method exists for backward compatibility with existing code.
        
        Args:
            short_window: size of short-term moving average window
            long_window: size of long-term moving average window
            ma_type: type of moving average to use (not used in this implementation, kept for compatibility)
            
        Returns:
            Tuple containing:
            - numpy array of signals: +1 (buy), -1 (sell), 0 (hold)
            - short moving average values
            - long moving average values
        """
        if self.prices is None:
            raise ValueError("Prices must be set either in constructor or by calling generate_ma_signals directly")
        
        # Use the standard generate_ma_signals method
        signals, indicators = self.generate_ma_signals(self.prices, short_window, long_window)
        
        return signals, indicators['short_ma'], indicators['long_ma']
    
    def _calculate_ma(self, data, window):
        """
        Calculate moving average
        
        Args:
            data: Array of data
            window: Window size
            
        Returns:
            Array of moving averages
        """
        return np.array(pd.Series(data).rolling(window=window).mean().dropna())
    
    def optimize_ma_parameters(self, prices, short_window_range=(5, 20), long_window_range=(21, 100),
                                step=5, metric='return'):
        """
        Optimize moving average parameters
        
        Args:
            prices: Array of price data
            short_window_range: Tuple of (min, max) for short window
            long_window_range: Tuple of (min, max) for long window
            step: Step size for parameter search
            metric: Performance metric to optimize ('return', 'sharpe', 'drawdown')
            
        Returns:
            Dict with optimal parameters
        """
        from backtest import Backtester
        
        best_metric_value = -np.inf if metric != 'drawdown' else np.inf
        best_params = None
        results = []
        
        for short_window in range(short_window_range[0], short_window_range[1] + 1, step):
            for long_window in range(long_window_range[0], long_window_range[1] + 1, step):
                # Skip invalid combinations
                if short_window >= long_window:
                    continue
                
                # Generate signals with current parameters
                signals, _ = self.generate_ma_signals(prices, short_window, long_window)
                
                # Backtest the strategy
                backtester = Backtester()
                result = backtester.run_backtest(signals, prices)
                
                # Record result
                results.append({
                    'short_window': short_window,
                    'long_window': long_window,
                    'return': result['return'],
                    'sharpe_ratio': result['sharpe_ratio'],
                    'max_drawdown': result['max_drawdown']
                })
                
                # Update best parameters based on metric
                metric_value = result[metric.replace('return', 'return').replace('sharpe', 'sharpe_ratio').replace('drawdown', 'max_drawdown')]
                
                if metric == 'drawdown':
                    # For drawdown, lower is better
                    if metric_value < best_metric_value:
                        best_metric_value = metric_value
                        best_params = {'short_window': short_window, 'long_window': long_window}
                else:
                    # For return and Sharpe ratio, higher is better
                    if metric_value > best_metric_value:
                        best_metric_value = metric_value
                        best_params = {'short_window': short_window, 'long_window': long_window}
        
        # Convert results to DataFrame for analysis
        results_df = pd.DataFrame(results)
        
        return {
            'best_params': best_params,
            'best_metric_value': best_metric_value,
            'results_df': results_df
        }
    
    def plot_ma_signals(self, prices, short_window=10, long_window=50, save_fig=False, fig_path=None):
        """
        Plot moving average signals
        
        Args:
            prices: Array of price data
            short_window: Length of short moving average window
            long_window: Length of long moving average window
            save_fig: Whether to save the figure
            fig_path: Path to save the figure
        """
        signals, indicators = self.generate_ma_signals(prices, short_window, long_window)
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot price and moving averages
        plt.plot(prices, label='Price', alpha=0.5, linewidth=1)
        plt.plot(indicators['short_ma'], label=f'Short MA ({short_window})', linewidth=1.5)
        plt.plot(indicators['long_ma'], label=f'Long MA ({long_window})', linewidth=1.5)
        
        # Plot buy signals
        buy_indices = np.where(signals == 1)[0]
        sell_indices = np.where(signals == -1)[0]
        
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
    
    def plot_optimization_results(self, results_df, metric='return', save_fig=False, fig_path=None):
        """
        Plot parameter optimization results
        
        Args:
            results_df: DataFrame with optimization results
            metric: Performance metric to visualize
            save_fig: Whether to save the figure
            fig_path: Path to save the figure
        """
        # Pivot data for heatmap
        if 'short_window' in results_df.columns and 'long_window' in results_df.columns:
            pivot = results_df.pivot_table(
                index='short_window', 
                columns='long_window', 
                values=metric.replace('return', 'return').replace('sharpe', 'sharpe_ratio').replace('drawdown', 'max_drawdown')
            )
            
            # Create figure
            plt.figure(figsize=(10, 8))
            
            # Plot heatmap
            if metric == 'drawdown':
                # For drawdown, lower is better
                plt.imshow(pivot, cmap='coolwarm_r', aspect='auto')
            else:
                # For return and Sharpe ratio, higher is better
                plt.imshow(pivot, cmap='coolwarm', aspect='auto')
            
            plt.colorbar(label=metric.capitalize())
            plt.title(f'Moving Average Parameter Optimization ({metric.capitalize()})')
            plt.xlabel('Long Window')
            plt.ylabel('Short Window')
            
            # Set x and y ticks
            plt.xticks(range(len(pivot.columns)), pivot.columns)
            plt.yticks(range(len(pivot.index)), pivot.index)
            
            if save_fig and fig_path:
                import os
                os.makedirs(os.path.dirname(fig_path), exist_ok=True)
                plt.savefig(fig_path)
                plt.close()
            else:
                plt.show()
        else:
            print("Required columns not found in results DataFrame")

# Example usage
if __name__ == "__main__":
    # Generate sample price data
    np.random.seed(42)
    price = np.random.randn(200).cumsum() + 100
    
    # Create signal generator
    sg = SignalGenerator()
    
    # Generate MA signals
    signals, _ = sg.generate_ma_signals(price)
    
    # Plot signals
    sg.plot_ma_signals(price)
    
    # Optimize parameters
    optimization_results = sg.optimize_ma_parameters(price)
    
    # Print optimal parameters
    print("Optimal parameters:")
    print(optimization_results['best_params'])
    
    # Plot optimization results
    sg.plot_optimization_results(optimization_results['results_df']) 