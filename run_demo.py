#!/usr/bin/env python3
"""
HFT Simulator - Complete Demo Script

This script demonstrates the full HFT simulation pipeline:
1. Data fetching from Binance
2. Feature engineering
3. ML model training
4. Strategy backtesting
5. Performance analysis
6. Visualization

Run with: python run_demo.py
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def create_directories():
    """Create required directories if they don't exist."""
    dirs = ['data/raw', 'data/processed', 'data/cache', 'models/saved', 'results']
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("✓ Created required directories")


def fetch_data():
    """Fetch market data from Binance."""
    from data.fetcher import BinanceDataFetcher
    from data.cache import DataCache
    
    print("\n" + "="*60)
    print("STEP 1: FETCHING MARKET DATA")
    print("="*60)
    
    # Initialize fetcher
    fetcher = BinanceDataFetcher(symbol="BTCUSDT")
    
    # Fetch 3 days of 1-minute data (4320 candles)
    print("Fetching BTC-USD data from Binance...")
    df = fetcher.fetch_klines(interval='1m', limit=4320)
    
    if df.empty:
        print("WARNING: Could not fetch data from API. Using synthetic data.")
        df = generate_synthetic_data()
    
    print(f"✓ Fetched {len(df)} candles")
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")
    print(f"  Price range: ${df['low'].min():,.2f} - ${df['high'].max():,.2f}")
    
    # Cache the data
    cache = DataCache(cache_dir='data/cache')
    cache.save(df, 'BTCUSDT', '1m')
    print("✓ Data cached to data/cache/")
    
    return df


def generate_synthetic_data():
    """Generate synthetic price data for testing."""
    print("Generating synthetic data...")
    np.random.seed(42)
    
    n_points = 4320
    dates = pd.date_range(end=pd.Timestamp.now(), periods=n_points, freq='1min')
    
    # Generate random walk with drift
    returns = np.random.normal(0.00001, 0.001, n_points)
    price = 50000 * np.exp(np.cumsum(returns))
    
    # Generate OHLCV
    high = price * (1 + np.abs(np.random.normal(0, 0.002, n_points)))
    low = price * (1 - np.abs(np.random.normal(0, 0.002, n_points)))
    open_price = low + (high - low) * np.random.random(n_points)
    volume = np.abs(np.random.normal(100, 30, n_points))
    
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': price,
        'volume': volume
    }, index=dates)
    
    return df


def preprocess_data(df):
    """Preprocess data and add technical indicators."""
    from data.preprocessor import DataPreprocessor
    
    print("\n" + "="*60)
    print("STEP 2: DATA PREPROCESSING")
    print("="*60)
    
    preprocessor = DataPreprocessor()
    
    print("Adding technical indicators...")
    df_processed = preprocessor.add_technical_indicators(df)
    
    print(f"✓ Added {len(df_processed.columns) - 5} technical indicators")
    print(f"  Total features: {len(df_processed.columns)}")
    print(f"  Rows after cleaning: {len(df_processed)}")
    
    return df_processed


def engineer_features(df):
    """Create ML features from processed data."""
    from ml.features import FeatureEngineer
    
    print("\n" + "="*60)
    print("STEP 3: FEATURE ENGINEERING")
    print("="*60)
    
    engineer = FeatureEngineer()
    
    print("Engineering ML features...")
    df_features = engineer.create_all_features(df)
    
    print(f"✓ Created {len(df_features.columns)} total features")
    print(f"  Feature categories: price, volatility, volume, candle, orderflow, time")
    
    return df_features, engineer


def train_model(df_features, engineer):
    """Train ML model for price prediction."""
    from ml.models import PriceLSTM
    
    print("\n" + "="*60)
    print("STEP 4: ML MODEL TRAINING")
    print("="*60)
    
    # Prepare training data
    print("Preparing training sequences...")
    X, y, feature_names = engineer.prepare_training_data(
        df_features,
        target_col='close',
        lookback=30,
        prediction_horizon=1
    )
    
    if len(X) < 100:
        print("WARNING: Not enough data for model training. Skipping ML.")
        return None
    
    # Create and train model
    print("\nTraining LSTM model...")
    model = PriceLSTM(
        input_size=X.shape[2],
        hidden_size=32,
        num_layers=2
    )
    
    history = model.train_model(
        X, y,
        epochs=20,
        batch_size=32,
        validation_split=0.2
    )
    
    print(f"\n✓ Training complete!")
    print(f"  Final training loss: {history['train_loss'][-1]:.4f}")
    print(f"  Final validation loss: {history['val_loss'][-1]:.4f}")
    print(f"  Final validation accuracy: {history['val_accuracy'][-1]:.4f}")
    
    return model


def run_backtest(df_features):
    """Run trading strategy backtest."""
    from strategies.momentum_strategy import MomentumStrategy
    from strategies.portfolio import Portfolio
    from strategies.risk_manager import RiskManager, RiskLimits
    from backtest.backtester import Backtester, BacktestConfig
    
    print("\n" + "="*60)
    print("STEP 5: BACKTESTING")
    print("="*60)
    
    # Initialize strategy
    strategy = MomentumStrategy(
        ml_threshold=0.55,
        momentum_threshold=0.0005,
        volume_threshold=1.2
    )
    
    # Configure backtest
    config = BacktestConfig(
        initial_capital=100000,
        fee_rate=0.001,
        position_size_pct=0.3,
        use_risk_manager=True
    )
    
    # Run backtest
    backtester = Backtester(strategy, config)
    
    print("Running backtest...")
    results = backtester.run(df_features, symbol='BTC')
    
    return results, backtester


def analyze_performance(results):
    """Calculate and display performance metrics."""
    from analytics.metrics import PerformanceMetrics
    
    print("\n" + "="*60)
    print("STEP 6: PERFORMANCE ANALYSIS")
    print("="*60)
    
    summary = results['summary']
    
    print("\n📊 Portfolio Performance:")
    print(f"  Initial Capital:    ${summary['initial_capital']:>12,.2f}")
    print(f"  Final Value:        ${summary['current_value']:>12,.2f}")
    print(f"  Total Return:       {summary['total_return_pct']:>12.2f}%")
    print(f"  Total PnL:          ${summary['total_pnl']:>12,.2f}")
    print(f"  Realized PnL:       ${summary['realized_pnl']:>12,.2f}")
    print(f"  Unrealized PnL:     ${summary['unrealized_pnl']:>12,.2f}")
    
    print("\n📈 Trading Activity:")
    print(f"  Total Signals:      {results['num_signals']:>12}")
    print(f"  Trades Executed:    {results['trades_executed']:>12}")
    print(f"  Total Fees:         ${summary['total_fees']:>12,.2f}")
    
    signal_stats = results['signal_stats']
    print("\n📊 Signal Distribution:")
    print(f"  BUY signals:        {signal_stats['buy_signals']:>12} ({signal_stats['buy_pct']:.1f}%)")
    print(f"  SELL signals:       {signal_stats['sell_signals']:>12} ({signal_stats['sell_pct']:.1f}%)")
    print(f"  HOLD signals:       {signal_stats['hold_signals']:>12} ({signal_stats['hold_pct']:.1f}%)")
    
    # Calculate additional metrics if we have trades
    if not results['trades'].empty:
        trades_df = results['trades']
        winning = trades_df[trades_df['pnl'] > 0]
        losing = trades_df[trades_df['pnl'] < 0]
        
        print("\n📊 Trade Statistics:")
        print(f"  Winning Trades:     {len(winning):>12}")
        print(f"  Losing Trades:      {len(losing):>12}")
        if len(trades_df) > 0:
            win_rate = len(winning) / len(trades_df) * 100
            print(f"  Win Rate:           {win_rate:>11.1f}%")
        if len(winning) > 0:
            print(f"  Avg Win:            ${winning['pnl'].mean():>12,.2f}")
        if len(losing) > 0:
            print(f"  Avg Loss:           ${losing['pnl'].mean():>12,.2f}")
    
    return results


def create_visualizations(results):
    """Generate performance visualizations."""
    from analytics.visualizations import TradingVisualizer
    
    print("\n" + "="*60)
    print("STEP 7: VISUALIZATIONS")
    print("="*60)
    
    viz = TradingVisualizer()
    
    equity_curve = results['equity_curve']
    trades = results['trades']
    
    if not equity_curve.empty:
        # Equity curve
        viz.plot_equity_curve(equity_curve, 
                             title="HFT Strategy - Equity Curve", 
                             save_path='results/equity_curve.png')
        
        # Drawdown
        viz.plot_drawdown(equity_curve, 
                         title="HFT Strategy - Drawdown",
                         save_path='results/drawdown.png')
        
        # PnL distribution
        if not trades.empty:
            viz.plot_pnl_distribution(trades,
                                     title="HFT Strategy - PnL Distribution",
                                     save_path='results/pnl_distribution.png')
        
        # Dashboard
        metrics = {
            'total_return_pct': results['summary']['total_return_pct'],
            'sharpe_ratio': 0.0,  # Would need more data to calculate
            'max_drawdown_pct': 0.0,
            'total_trades': results['trades_executed'],
            'win_rate': results['signal_stats']['buy_pct'],
            'profit_factor': 0.0
        }
        
        viz.plot_summary_dashboard(equity_curve, trades, metrics,
                                   save_path='results/dashboard.png')
        
        print("✓ Visualizations saved to results/")
        print("  - equity_curve.png")
        print("  - drawdown.png")
        print("  - pnl_distribution.png")
        print("  - dashboard.png")
    else:
        print("⚠ No equity curve data to visualize")


def export_results(results):
    """Export results to CSV files."""
    print("\n" + "="*60)
    print("STEP 8: EXPORTING RESULTS")
    print("="*60)
    
    # Export equity curve
    if not results['equity_curve'].empty:
        results['equity_curve'].to_csv('results/equity_curve.csv')
        print("✓ Equity curve saved to results/equity_curve.csv")
    
    # Export trades
    if not results['trades'].empty:
        results['trades'].to_csv('results/trades.csv')
        print("✓ Trades saved to results/trades.csv")
    
    # Export summary
    summary_df = pd.DataFrame([results['summary']])
    summary_df.to_csv('results/summary.csv', index=False)
    print("✓ Summary saved to results/summary.csv")


def main():
    """Run complete HFT simulation demo."""
    print("\n" + "="*60)
    print("       HFT SIMULATOR - COMPLETE DEMO")
    print("="*60)
    print("This demo runs a complete trading simulation pipeline.")
    print("="*60)
    
    try:
        # Setup
        create_directories()
        
        # Step 1: Fetch data
        df = fetch_data()
        
        # Step 2: Preprocess
        df_processed = preprocess_data(df)
        
        # Step 3: Engineer features
        df_features, engineer = engineer_features(df_processed)
        
        # Step 4: Train ML model (optional)
        model = train_model(df_features, engineer)
        
        # Step 5: Run backtest
        results, backtester = run_backtest(df_features)
        
        # Step 6: Analyze performance
        analyze_performance(results)
        
        # Step 7: Create visualizations
        create_visualizations(results)
        
        # Step 8: Export results
        export_results(results)
        
        print("\n" + "="*60)
        print("       ✅ DEMO COMPLETE!")
        print("="*60)
        print("\nResults have been saved to the 'results/' directory.")
        print("Check out the visualizations and CSV exports.")
        print("\nNext steps:")
        print("  - Open examples/example_backtest.ipynb for interactive exploration")
        print("  - Modify strategy parameters in src/strategies/")
        print("  - Experiment with different ML models in src/ml/")
        print("="*60 + "\n")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
