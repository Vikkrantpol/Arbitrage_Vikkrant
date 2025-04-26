import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime
import matplotlib.pyplot as plt
from pandas.api.types import is_datetime64_any_dtype

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/backtest.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create necessary directories
os.makedirs('logs', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# Configuration
class Config:
    # Profit threshold (0.005%)
    MIN_PROFIT_PCT = 0.00005  # Configurable by user (0.005%)
    
    # Exchange fees (maker/taker)
    TRADING_FEES = {
        'binance': {'maker': 0.0002, 'taker': 0.0004},  # 0.02%/0.04%
        'delta': {'maker': 0.0001, 'taker': 0.0003}     # 0.01%/0.03%
    }
    
    # Slippage settings
    SLIPPAGE_PCT = 0.0002  # 0.02%
    
    # Trade parameters
    TRADE_AMOUNT = 0.05  # BTC per trade
    MAX_TRADE_AMOUNT = 0.05  # BTC maximum per trade
    MIN_TRADE_AMOUNT = 0.005  # BTC minimum per trade
    
    # Capital management
    INITIAL_CAPITAL = 15000  # USD
    RISK_PER_TRADE = 0.01  # 1% of capital per trade
    
    # Date range (set to None to use all available data)
    START_DATE = None  
    END_DATE = None    
    
    # Data columns mapping
    COLUMN_MAP = {
        'binance': {
            'timestamp': 'timestamp',
            'bid_price': 'bid_price_binance',
            'ask_price': 'ask_price_binance',
            'bid_volume': 'bid_volume_binance',
            'ask_volume': 'ask_volume_binance'
        },
        'delta': {
            'timestamp': 'timestamp',
            'bid_price': 'bid_price_delta',
            'ask_price': 'ask_price_delta',
            'bid_volume': 'bid_volume_delta',
            'ask_volume': 'ask_volume_delta'
        }
    }

def calculate_arbitrage(row, capital, config):
    """Calculate arbitrage opportunity between exchanges with enhanced logic."""
    try:
        # Extract prices and volumes
        binance_bid = row[config.COLUMN_MAP['binance']['bid_price']]
        binance_ask = row[config.COLUMN_MAP['binance']['ask_price']]
        delta_bid = row[config.COLUMN_MAP['delta']['bid_price']]
        delta_ask = row[config.COLUMN_MAP['delta']['ask_price']]
        
        binance_ask_vol = row[config.COLUMN_MAP['binance']['ask_volume']]
        delta_bid_vol = row[config.COLUMN_MAP['delta']['bid_volume']]
        delta_ask_vol = row[config.COLUMN_MAP['delta']['ask_volume']]
        binance_bid_vol = row[config.COLUMN_MAP['binance']['bid_volume']]
        
        # Dynamic trade amount based on capital and risk management
        max_trade_usd = capital * config.RISK_PER_TRADE
        trade_amount = min(
            config.TRADE_AMOUNT,
            max_trade_usd / binance_ask if binance_ask > 0 else config.TRADE_AMOUNT,
            max_trade_usd / delta_ask if delta_ask > 0 else config.TRADE_AMOUNT
        )
        trade_amount = max(min(trade_amount, config.MAX_TRADE_AMOUNT), config.MIN_TRADE_AMOUNT)
        
        # Initialize strategies list
        strategies = []
        
        # Strategy 1: Buy on Binance (taker), Sell on Delta (maker)
        trade_amount_buy_binance = min(trade_amount, binance_ask_vol, delta_bid_vol)
        if trade_amount_buy_binance > 0:
            buy_cost_binance = binance_ask * trade_amount_buy_binance * (1 + config.TRADING_FEES['binance']['taker'])
            sell_revenue_delta = delta_bid * trade_amount_buy_binance * (1 - config.TRADING_FEES['delta']['maker'])
            profit_buy_binance = sell_revenue_delta - buy_cost_binance
            profit_buy_binance *= (1 - config.SLIPPAGE_PCT)  # Apply slippage
            profit_pct_buy_binance = (profit_buy_binance / buy_cost_binance) * 100 if buy_cost_binance > 0 else 0
            logger.info(f"Strategy 1: profit_pct={profit_pct_buy_binance:.4f}%, MIN_PROFIT_PCT={config.MIN_PROFIT_PCT*100:.4f}%")
            if profit_pct_buy_binance >= config.MIN_PROFIT_PCT * 100 and profit_buy_binance > 0:
                strategies.append(('buy_binance_sell_delta', profit_buy_binance, profit_pct_buy_binance, trade_amount_buy_binance))
        
        # Strategy 2: Buy on Delta (taker), Sell on Binance (maker)
        trade_amount_buy_delta = min(trade_amount, delta_ask_vol, binance_bid_vol)
        if trade_amount_buy_delta > 0:
            buy_cost_delta = delta_ask * trade_amount_buy_delta * (1 + config.TRADING_FEES['delta']['taker'])
            sell_revenue_binance = binance_bid * trade_amount_buy_delta * (1 - config.TRADING_FEES['binance']['maker'])
            profit_buy_delta = sell_revenue_binance - buy_cost_delta
            profit_buy_delta *= (1 - config.SLIPPAGE_PCT)  # Apply slippage
            profit_pct_buy_delta = (profit_buy_delta / buy_cost_delta) * 100 if buy_cost_delta > 0 else 0
            logger.info(f"Strategy 2: profit_pct={profit_pct_buy_delta:.4f}%, MIN_PROFIT_PCT={config.MIN_PROFIT_PCT*100:.4f}%")
            if profit_pct_buy_delta >= config.MIN_PROFIT_PCT * 100 and profit_buy_delta > 0:
                strategies.append(('buy_delta_sell_binance', profit_buy_delta, profit_pct_buy_delta, trade_amount_buy_delta))
        
        # Return if no valid strategies
        if not strategies:
            return None, 0, 0, 0
        
        # Select strategy with highest profit percentage
        best_strategy = max(strategies, key=lambda x: x[2])
        return best_strategy
        
    except KeyError as e:
        logger.error(f"Missing column in row: {e}")
        return None, 0, 0, 0
    except Exception as e:
        logger.error(f"Error calculating arbitrage: {e}")
        return None, 0, 0, 0

def load_and_preprocess_data(binance_file, delta_file, config):
    """Load and preprocess exchange data with improved datetime handling."""
    try:
        # Load data with error handling
        logger.info(f"Loading Binance data from {binance_file}")
        binance_df = pd.read_csv(
            binance_file, 
            parse_dates=[config.COLUMN_MAP['binance']['timestamp']],
            dtype={
                config.COLUMN_MAP['binance']['bid_price']: float,
                config.COLUMN_MAP['binance']['ask_price']: float,
                config.COLUMN_MAP['binance']['bid_volume']: float,
                config.COLUMN_MAP['binance']['ask_volume']: float
            }
        )
        
        logger.info(f"Loading Delta data from {delta_file}")
        delta_df = pd.read_csv(
            delta_file, 
            parse_dates=[config.COLUMN_MAP['delta']['timestamp']],
            dtype={
                config.COLUMN_MAP['delta']['bid_price']: float,
                config.COLUMN_MAP['delta']['ask_price']: float,
                config.COLUMN_MAP['delta']['bid_volume']: float,
                config.COLUMN_MAP['delta']['ask_volume']: float
            }
        )
        
        # Process each dataframe
        for df, exchange in [(binance_df, 'binance'), (delta_df, 'delta')]:
            ts_col = config.COLUMN_MAP[exchange]['timestamp']
            logger.info(f"Processing {exchange} data with timestamp column {ts_col}")
            
            # Convert to datetime if not already
            if not is_datetime64_any_dtype(df[ts_col]):
                df[ts_col] = pd.to_datetime(df[ts_col])
            
            # Make timezone-aware if not already
            if not hasattr(df[ts_col].dtype, 'tz'):
                df[ts_col] = df[ts_col].dt.tz_localize('UTC')
            
            logger.info(f"{exchange} timestamp range: {df[ts_col].min()} to {df[ts_col].max()}")
        
        # Only filter by date range if dates are specified
        if config.START_DATE is not None:
            start_date = pd.to_datetime(config.START_DATE).tz_localize('UTC')
            binance_df = binance_df[binance_df[config.COLUMN_MAP['binance']['timestamp']] >= start_date]
            delta_df = delta_df[delta_df[config.COLUMN_MAP['delta']['timestamp']] >= start_date]
        
        if config.END_DATE is not None:
            end_date = pd.to_datetime(config.END_DATE).tz_localize('UTC')
            binance_df = binance_df[binance_df[config.COLUMN_MAP['binance']['timestamp']] <= end_date]
            delta_df = delta_df[delta_df[config.COLUMN_MAP['delta']['timestamp']] <= end_date]
        
        if binance_df.empty or delta_df.empty:
            logger.error("No data remaining after date filtering")
            logger.error(f"Binance data count: {len(binance_df)}")
            logger.error(f"Delta data count: {len(delta_df)}")
            return None
        
        # Merge data with time alignment
        logger.info("Merging datasets...")
        df = pd.merge_asof(
            binance_df.sort_values(config.COLUMN_MAP['binance']['timestamp']),
            delta_df.sort_values(config.COLUMN_MAP['delta']['timestamp']),
            left_on=config.COLUMN_MAP['binance']['timestamp'],
            right_on=config.COLUMN_MAP['delta']['timestamp'],
            suffixes=('_binance', '_delta'),
            tolerance=pd.Timedelta('30s'),
            direction='nearest'
        )
        
        if df.empty:
            logger.error("No data remaining after merge")
            return None
        
        # Clean data
        initial_count = len(df)
        df = df.dropna()
        df = df[
            (df[config.COLUMN_MAP['binance']['bid_price']] > 0) &
            (df[config.COLUMN_MAP['binance']['ask_price']] > 0) &
            (df[config.COLUMN_MAP['delta']['bid_price']] > 0) &
            (df[config.COLUMN_MAP['delta']['ask_price']] > 0)
        ]
        
        logger.info(f"Data cleaning removed {initial_count - len(df)} rows")
        logger.info(f"Final dataset contains {len(df)} rows")
        
        # Calculate price differences
        df['price_diff'] = df[config.COLUMN_MAP['binance']['ask_price']] - df[config.COLUMN_MAP['delta']['bid_price']]
        df['price_diff_pct'] = df['price_diff'] / df[config.COLUMN_MAP['binance']['ask_price']] * 100
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}", exc_info=True)
        return None

def run_backtest(df, config):
    """Run the backtest simulation."""
    if df is None or df.empty:
        logger.error("No valid data to backtest")
        return None, None
    
    # Initialize backtest variables
    capital = config.INITIAL_CAPITAL
    trades = []
    profits = []
    peak_capital = config.INITIAL_CAPITAL
    max_drawdown = 0
    trade_count = 0
    winning_trades = 0
    
    # Simulate trades
    for _, row in df.iterrows():
        strategy, profit, profit_pct, trade_amount = calculate_arbitrage(row, capital, config)
        
        if strategy:
            # Update capital and metrics
            capital += profit
            peak_capital = max(peak_capital, capital)
            current_drawdown = ((peak_capital - capital) / peak_capital) * 100 if peak_capital > 0 else 0
            max_drawdown = max(max_drawdown, current_drawdown)
            
            # Track trade performance
            trade_count += 1
            if profit > 0:
                winning_trades += 1
            
            trades.append({
                'timestamp': row[config.COLUMN_MAP['binance']['timestamp']],
                'strategy': strategy,
                'profit_usd': profit,
                'profit_pct': profit_pct,
                'trade_amount_btc': trade_amount,
                'capital': capital,
                'drawdown': current_drawdown,
                'binance_ask': row[config.COLUMN_MAP['binance']['ask_price']],
                'delta_bid': row[config.COLUMN_MAP['delta']['bid_price']],
                'price_diff_pct': row['price_diff_pct']
            })
            
            logger.info(
                f"Trade {trade_count}: {strategy} at {row[config.COLUMN_MAP['binance']['timestamp']]}, "
                f"Amount: {trade_amount:.4f} BTC, Profit: {profit:.2f} USD ({profit_pct:.4f}%), "
                f"Capital: {capital:.2f}, Drawdown: {current_drawdown:.2f}%"
            )
            
            profits.append(profit)
    
    return trades, profits

def calculate_metrics(trades, profits, initial_capital):
    """Calculate performance metrics."""
    if not trades:
        return {}
    
    total_trades = len(trades)
    total_profit = sum(profits)
    avg_profit = total_profit / total_trades if total_trades > 0 else 0
    avg_profit_pct = np.mean([t['profit_pct'] for t in trades]) if trades else 0
    win_rate = (sum(p > 0 for p in profits) / total_trades) * 100 if total_trades > 0 else 0
    max_drawdown = max([t['drawdown'] for t in trades]) if trades else 0
    
    # Calculate Sharpe ratio (simplified)
    returns = [p / initial_capital for p in profits]
    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
    
    # Calculate profit factor
    gross_profit = sum(p for p in profits if p > 0)
    gross_loss = abs(sum(p for p in profits if p < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Calculate CAGR
    if trades:
        start_date = trades[0]['timestamp']
        end_date = trades[-1]['timestamp']
        days = (end_date - start_date).days
        cagr = ((initial_capital + total_profit) / initial_capital) ** (365/days) - 1 if days > 0 else 0
    else:
        cagr = 0
    
    return {
        'total_trades': total_trades,
        'total_profit': total_profit,
        'avg_profit': avg_profit,
        'avg_profit_pct': avg_profit_pct,
        'win_rate': win_rate,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'profit_factor': profit_factor,
        'cagr': cagr * 100,  # as percentage
        'final_capital': initial_capital + total_profit,
        'return_pct': (total_profit / initial_capital) * 100
    }

def save_results(trades, metrics, config):
    """Save backtest results and generate plots."""
    # Save trades to CSV
    if trades:
        trades_df = pd.DataFrame(trades)
        trades_df.to_csv('results/backtest_trades.csv', index=False)
        logger.info("Trades saved to results/backtest_trades.csv")
        
        # Save metrics to JSON
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv('results/backtest_metrics.csv', index=False)
        logger.info("Metrics saved to results/backtest_metrics.csv")
        
        # Generate plots
        plt.figure(figsize=(12, 6))
        
        # Capital growth plot
        plt.subplot(1, 2, 1)
        plt.plot(trades_df['timestamp'], trades_df['capital'])
        plt.title('Capital Growth')
        plt.xlabel('Date')
        plt.ylabel('Capital (USD)')
        plt.grid(True)
        
        # Drawdown plot
        plt.subplot(1, 2, 2)
        plt.plot(trades_df['timestamp'], trades_df['drawdown'])
        plt.title('Drawdown')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('plots/backtest_results.png')
        plt.close()
        
        # Price difference histogram
        plt.figure(figsize=(8, 5))
        plt.hist(trades_df['price_diff_pct'], bins=50)
        plt.title('Arbitrage Price Differences')
        plt.xlabel('Price Difference (%)')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig('plots/price_differences.png')
        plt.close()
        
        logger.info("Plots saved to plots directory")

def main():
    """Main execution function."""
    logger.info("Starting arbitrage backtest")
    config = Config()
    
    # Load and preprocess data
    binance_file = 'data/binance_btcusd.csv'
    delta_file = 'data/delta_btcusd.csv'
    
    df = load_and_preprocess_data(binance_file, delta_file, config)
    if df is None or df.empty:
        logger.error("Failed to load or preprocess data")
        return
    
    # Run backtest
    trades, profits = run_backtest(df, config)
    
    # Calculate metrics
    metrics = calculate_metrics(trades, profits, config.INITIAL_CAPITAL)
    
    # Display and save results
    logger.info("\nBacktest Results:")
    for key, value in metrics.items():
        logger.info(f"{key.replace('_', ' ').title()}: {value:.2f}{'%' if 'pct' in key or 'rate' in key or 'return' in key or 'cagr' in key else ''}")
    
    save_results(trades, metrics, config)
    logger.info("Backtest completed")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error in backtest: {e}", exc_info=True)
