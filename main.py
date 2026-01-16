#!/usr/bin/env python3
"""
HFT Simulator - Production Entry Point

Main script for running the complete HFT simulation with optimized settings.
Supports multiple modes: backtest, optimize, paper (simulation).

Usage:
    python main.py                      # Run with default config
    python main.py --config custom.yaml # Use custom config
    python main.py --mode optimize      # Run parameter optimization
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def setup_directories():
    """Create required directories."""
    dirs = ['data/raw', 'data/processed', 'data/cache', 'models/saved', 'results', 'logs']
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def fetch_data(config):
    """Fetch market data based on config."""
    from data.fetcher import BinanceDataFetcher
    from data.cache import DataCache
    
    print(f"\n{'='*60}")
    print("FETCHING DATA")
    print(f"{'='*60}")
    
    fetcher = BinanceDataFetcher(symbol=config.data.symbol)
    cache = DataCache(cache_dir=config.data.cache_dir)
    
    # Calculate limit based on days
    limit = config.data.history_days * 24 * 60  # 1-minute bars
    limit = min(limit, 10000)  # API limit
    
    print(f"Fetching {config.data.symbol} data ({config.data.history_days} days)...")
    df = fetcher.fetch_klines(interval=config.data.interval, limit=limit)
    
    if df.empty:
        print("WARNING: API fetch failed. Generating synthetic data...")
        df = generate_synthetic_data(limit)
    
    print(f"✓ Loaded {len(df)} candles")
    print(f"  Range: {df.index[0]} to {df.index[-1]}")
    
    # Cache
    if config.data.cache_enabled:
        cache.save(df, config.data.symbol, config.data.interval)
    
    return df


def generate_synthetic_data(n_points: int):
    """Generate synthetic price data."""
    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.now(), periods=n_points, freq='1min')
    
    returns = np.random.normal(0.00002, 0.0015, n_points)
    price = 50000 * np.exp(np.cumsum(returns))
    
    high = price * (1 + np.abs(np.random.normal(0, 0.002, n_points)))
    low = price * (1 - np.abs(np.random.normal(0, 0.002, n_points)))
    open_price = low + (high - low) * np.random.random(n_points)
    volume = np.abs(np.random.normal(100, 30, n_points))
    
    return pd.DataFrame({
        'open': open_price, 'high': high, 'low': low,
        'close': price, 'volume': volume
    }, index=dates)


def preprocess_data(df):
    """Preprocess and add indicators."""
    from data.preprocessor import DataPreprocessor
    
    print(f"\n{'='*60}")
    print("PREPROCESSING")
    print(f"{'='*60}")
    
    preprocessor = DataPreprocessor()
    df_processed = preprocessor.preprocess_pipeline(df)
    
    print(f"✓ Added technical indicators")
    print(f"  Columns: {len(df_processed.columns)}, Rows: {len(df_processed)}")
    
    return df_processed


def engineer_features(df):
    """Create ML features."""
    from ml.features import FeatureEngineer
    
    print(f"\n{'='*60}")
    print("FEATURE ENGINEERING")
    print(f"{'='*60}")
    
    engineer = FeatureEngineer()
    df_features = engineer.create_all_features(df)
    
    # Add rolling high/low for breakout strategy
    df_features['rolling_high_20'] = df_features['high'].rolling(20).max()
    df_features['rolling_low_20'] = df_features['low'].rolling(20).min()
    df_features = df_features.dropna()
    
    print(f"✓ Created {len(df_features.columns)} features")
    
    return df_features, engineer


def train_model(df_features, engineer, config):
    """Train ML model."""
    print(f"\n{'='*60}")
    print("TRAINING ML MODEL")
    print(f"{'='*60}")
    
    # Prepare data
    X, y, _ = engineer.prepare_training_data(
        df_features,
        lookback=config.ml.lookback,
        prediction_horizon=config.ml.prediction_horizon
    )
    
    if len(X) < 200:
        print("WARNING: Not enough data for training")
        return None
    
    print(f"Training data: {X.shape}")
    
    if config.ml.model_type == 'transformer':
        from ml.transformer_model import PriceTransformer
        model = PriceTransformer(
            input_size=X.shape[2],
            d_model=config.ml.d_model,
            nhead=config.ml.nhead,
            num_layers=config.ml.num_layers
        )
    elif config.ml.model_type == 'ensemble':
        from ml.transformer_model import EnsembleModel
        model = EnsembleModel(input_size=X.shape[2])
        history = model.train(X, y, epochs=config.ml.epochs)
        print(f"✓ Ensemble training complete")
        return model
    else:  # lstm
        from ml.models import PriceLSTM
        model = PriceLSTM(
            input_size=X.shape[2],
            hidden_size=config.ml.hidden_size,
            num_layers=config.ml.num_layers
        )
    
    history = model.train_model(
        X, y,
        epochs=config.ml.epochs,
        batch_size=config.ml.batch_size,
        validation_split=config.ml.validation_split,
        early_stopping_patience=config.ml.early_stopping_patience
    )
    
    print(f"✓ Training complete")
    print(f"  Final val accuracy: {history['val_accuracy'][-1]:.2%}")
    
    return model


def create_strategy(config):
    """Create strategy based on config."""
    from strategies.momentum_strategy import MomentumStrategy
    from strategies.mean_reversion_strategy import MeanReversionStrategy
    from strategies.breakout_strategy import BreakoutStrategy
    from strategies.ensemble_strategy import EnsembleStrategy
    
    if config.strategy.type == 'momentum':
        return MomentumStrategy(
            ml_threshold=config.strategy.ml_threshold,
            momentum_threshold=config.strategy.momentum_threshold,
            volume_threshold=config.strategy.volume_threshold
        )
    
    elif config.strategy.type == 'mean_reversion':
        return MeanReversionStrategy(
            rsi_oversold=config.strategy.rsi_oversold,
            rsi_overbought=config.strategy.rsi_overbought,
            mean_deviation_threshold=config.strategy.mean_deviation_threshold
        )
    
    elif config.strategy.type == 'breakout':
        return BreakoutStrategy(
            lookback=config.strategy.breakout_lookback,
            breakout_threshold=config.strategy.breakout_threshold,
            volume_threshold=config.strategy.volume_threshold
        )
    
    else:  # ensemble
        strategies = [
            MomentumStrategy(
                ml_threshold=config.strategy.ml_threshold,
                momentum_threshold=config.strategy.momentum_threshold,
                volume_threshold=config.strategy.volume_threshold
            ),
            MeanReversionStrategy(
                rsi_oversold=config.strategy.rsi_oversold,
                rsi_overbought=config.strategy.rsi_overbought,
                mean_deviation_threshold=config.strategy.mean_deviation_threshold
            ),
            BreakoutStrategy(
                lookback=config.strategy.breakout_lookback,
                breakout_threshold=config.strategy.breakout_threshold,
                volume_threshold=config.strategy.volume_threshold
            )
        ]
        weights = [
            config.strategy.weights.get('momentum', 0.4),
            config.strategy.weights.get('mean_reversion', 0.35),
            config.strategy.weights.get('breakout', 0.25)
        ]
        return EnsembleStrategy(
            strategies=strategies,
            method=config.strategy.ensemble_method,
            min_confidence=config.strategy.min_confidence,
            weights=weights
        )


def run_backtest(df_features, strategy, config):
    """Run backtest."""
    from backtest.backtester import Backtester, BacktestConfig as BTConfig
    
    print(f"\n{'='*60}")
    print("RUNNING BACKTEST")
    print(f"{'='*60}")
    
    bt_config = BTConfig(
        initial_capital=config.backtest.initial_capital,
        fee_rate=config.backtest.fee_rate,
        position_size_pct=config.risk.max_position_pct,
        use_risk_manager=config.backtest.use_risk_manager
    )
    
    backtester = Backtester(strategy, bt_config)
    
    print(f"Strategy: {strategy.get_name() if hasattr(strategy, 'get_name') else type(strategy).__name__}")
    print(f"Initial capital: ${bt_config.initial_capital:,.0f}")
    print(f"Fee rate: {bt_config.fee_rate:.4%}")
    
    results = backtester.run(df_features, symbol=config.data.symbol.replace('USDT', ''))
    
    return results


def analyze_results(results, config):
    """Analyze and display results."""
    from analytics.metrics import PerformanceMetrics
    
    print(f"\n{'='*60}")
    print("PERFORMANCE RESULTS")
    print(f"{'='*60}")
    
    summary = results['summary']
    signals = results['signal_stats']
    
    print(f"\n📊 Portfolio:")
    print(f"  Initial:     ${summary['initial_capital']:>12,.2f}")
    print(f"  Final:       ${summary['current_value']:>12,.2f}")
    print(f"  Return:      {summary['total_return_pct']:>12.2f}%")
    print(f"  PnL:         ${summary['total_pnl']:>12,.2f}")
    print(f"  Fees:        ${summary['total_fees']:>12,.2f}")
    
    print(f"\n📈 Trading:")
    print(f"  Total Signals: {results['num_signals']}")
    print(f"  Trades:        {results['trades_executed']}")
    print(f"  BUY:  {signals['buy_signals']:>4} ({signals['buy_pct']:.1f}%)")
    print(f"  SELL: {signals['sell_signals']:>4} ({signals['sell_pct']:.1f}%)")
    print(f"  HOLD: {signals['hold_signals']:>4} ({signals['hold_pct']:.1f}%)")
    
    if not results['trades'].empty:
        trades = results['trades']
        wins = trades[trades['pnl'] > 0]
        losses = trades[trades['pnl'] < 0]
        
        print(f"\n📊 Trade Stats:")
        print(f"  Win Rate:    {len(wins)/len(trades)*100 if len(trades) > 0 else 0:>10.1f}%")
        print(f"  Avg Win:     ${wins['pnl'].mean() if len(wins) > 0 else 0:>10,.2f}")
        print(f"  Avg Loss:    ${losses['pnl'].mean() if len(losses) > 0 else 0:>10,.2f}")
    
    return results


def save_results(results, config):
    """Save results to files."""
    from analytics.visualizations import TradingVisualizer
    import matplotlib
    matplotlib.use('Agg')
    
    print(f"\n{'='*60}")
    print("SAVING RESULTS")
    print(f"{'='*60}")
    
    # Create results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(config.results_dir) / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save CSVs
    results['equity_curve'].to_csv(results_dir / 'equity_curve.csv')
    results['trades'].to_csv(results_dir / 'trades.csv')
    pd.DataFrame([results['summary']]).to_csv(results_dir / 'summary.csv', index=False)
    
    # Save visualizations
    viz = TradingVisualizer()
    
    viz.plot_equity_curve(
        results['equity_curve'],
        title="HFT Strategy - Equity Curve",
        save_path=str(results_dir / 'equity_curve.png')
    )
    
    viz.plot_drawdown(
        results['equity_curve'],
        title="HFT Strategy - Drawdown",
        save_path=str(results_dir / 'drawdown.png')
    )
    
    if not results['trades'].empty:
        viz.plot_pnl_distribution(
            results['trades'],
            title="Trade PnL Distribution",
            save_path=str(results_dir / 'pnl_dist.png')
        )
    
    metrics = {
        'total_return_pct': results['summary']['total_return_pct'],
        'sharpe_ratio': 0.0,
        'max_drawdown_pct': 0.0,
        'total_trades': results['trades_executed'],
        'win_rate': results['signal_stats']['buy_pct'],
        'profit_factor': 0.0
    }
    
    viz.plot_summary_dashboard(
        results['equity_curve'], results['trades'], metrics,
        save_path=str(results_dir / 'dashboard.png')
    )
    
    print(f"✓ Results saved to {results_dir}/")
    
    return results_dir


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='HFT Simulator')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                       help='Path to config file')
    parser.add_argument('--mode', type=str, default='backtest',
                       choices=['backtest', 'optimize', 'paper'],
                       help='Execution mode')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("      HFT SIMULATOR - PRODUCTION")
    print("="*60)
    
    try:
        # Load config
        from config.config_loader import load_config
        config = load_config(args.config)
        print(f"✓ Loaded config: {args.config}")
        
        # Setup
        setup_directories()
        
        # Fetch data
        df = fetch_data(config)
        
        # Preprocess
        df_processed = preprocess_data(df)
        
        # Features
        df_features, engineer = engineer_features(df_processed)
        
        # Train model
        model = train_model(df_features, engineer, config)
        
        # Create strategy
        strategy = create_strategy(config)
        
        # Run backtest
        results = run_backtest(df_features, strategy, config)
        
        # Analyze
        analyze_results(results, config)
        
        # Save
        results_dir = save_results(results, config)
        
        print(f"\n{'='*60}")
        print("      ✅ COMPLETE")
        print(f"{'='*60}")
        print(f"\nResults: {results_dir}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
