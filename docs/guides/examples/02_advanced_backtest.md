# Advanced Backtest Example

A comprehensive example with ML training, custom settings, and full analysis.

## Full Code

```python
#!/usr/bin/env python3
"""
Advanced Backtest Example

Complete workflow with:
- ML model training
- Custom strategy parameters
- Risk management
- Full metrics analysis
- Visualization
"""

import sys
import os
sys.path.insert(0, 'src')

import warnings
warnings.filterwarnings('ignore')

# Create output directories
os.makedirs('results', exist_ok=True)
os.makedirs('models/saved', exist_ok=True)

# =============================================================================
# 1. DATA PIPELINE
# =============================================================================

print("=" * 60)
print("STEP 1: DATA PIPELINE")
print("=" * 60)

from data.fetcher import BinanceDataFetcher
from data.cache import DataCache
from data.preprocessor import DataPreprocessor

# Initialize
fetcher = BinanceDataFetcher(symbol="BTCUSDT")
cache = DataCache(cache_dir="data/cache")
preprocessor = DataPreprocessor()

# Fetch or load from cache
if cache.exists("BTCUSDT", "1m"):
    print("Loading from cache...")
    df = cache.load("BTCUSDT", "1m")
else:
    print("Fetching from Binance API...")
    df = fetcher.fetch_historical_data(interval="1m", days=7)
    cache.save(df, "BTCUSDT", "1m")

print(f"Data loaded: {len(df)} candles")
print(f"Date range: {df.index[0]} to {df.index[-1]}")

# Preprocess
df_processed = preprocessor.preprocess_pipeline(df)
print(f"After preprocessing: {len(df_processed)} rows")

# =============================================================================
# 2. FEATURE ENGINEERING
# =============================================================================

print("\n" + "=" * 60)
print("STEP 2: FEATURE ENGINEERING")
print("=" * 60)

from ml.features import FeatureEngineer

engineer = FeatureEngineer()
df_features = engineer.create_all_features(df_processed)
print(f"Features created: {len(df_features.columns)} columns")

# =============================================================================
# 3. ML MODEL TRAINING
# =============================================================================

print("\n" + "=" * 60)
print("STEP 3: ML MODEL TRAINING")
print("=" * 60)

from ml.models import PriceLSTM

# Prepare training data
X, y, feature_names = engineer.prepare_training_data(
    df_features,
    target_col='close',
    lookback=30,
    prediction_horizon=1
)

print(f"Training data: X={X.shape}, y={y.shape}")
print(f"Class balance: {y.mean():.1%} up, {1-y.mean():.1%} down")

# Train model
model = PriceLSTM(
    input_size=X.shape[2],
    hidden_size=32,
    num_layers=2,
    dropout=0.2
)

history = model.train_model(
    X, y,
    epochs=15,
    batch_size=32,
    validation_split=0.2,
    early_stopping_patience=5
)

print(f"\nTraining complete!")
print(f"Final validation accuracy: {history['val_accuracy'][-1]:.1%}")

# =============================================================================
# 4. STRATEGY CONFIGURATION
# =============================================================================

print("\n" + "=" * 60)
print("STEP 4: STRATEGY CONFIGURATION")
print("=" * 60)

from strategies.momentum_strategy import MomentumStrategy
from strategies.risk_manager import RiskLimits

# Strategy with tuned parameters
strategy = MomentumStrategy(
    ml_threshold=0.55,
    momentum_threshold=0.0005,
    volume_threshold=1.2,
    confidence_scaling=True
)

# Risk limits
risk_limits = RiskLimits(
    max_position_pct=0.3,
    max_drawdown_pct=0.15,
    stop_loss_pct=0.02,
    max_daily_loss_pct=0.05
)

print("Strategy: MomentumStrategy")
print(f"  ML threshold: {strategy.ml_threshold}")
print(f"  Momentum threshold: {strategy.momentum_threshold}")
print(f"  Volume threshold: {strategy.volume_threshold}")
print("\nRisk Limits:")
print(f"  Max position: {risk_limits.max_position_pct:.0%}")
print(f"  Max drawdown: {risk_limits.max_drawdown_pct:.0%}")
print(f"  Stop loss: {risk_limits.stop_loss_pct:.0%}")

# =============================================================================
# 5. BACKTEST EXECUTION
# =============================================================================

print("\n" + "=" * 60)
print("STEP 5: BACKTEST EXECUTION")
print("=" * 60)

from backtest.backtester import Backtester, BacktestConfig

config = BacktestConfig(
    initial_capital=100000.0,
    fee_rate=0.001,
    position_size_pct=0.3,
    use_risk_manager=True,
    use_ml_model=False  # Using signals from strategy
)

print("Running backtest...")
backtester = Backtester(strategy, config)
results = backtester.run(df_features, symbol='BTC')
print("Backtest complete!")

# =============================================================================
# 6. PERFORMANCE ANALYSIS
# =============================================================================

print("\n" + "=" * 60)
print("STEP 6: PERFORMANCE ANALYSIS")
print("=" * 60)

from analytics.metrics import PerformanceMetrics

# Calculate comprehensive metrics
metrics = PerformanceMetrics.calculate_all_metrics(
    results['equity_curve'],
    results['trades'],
    config.initial_capital
)

# Display results
summary = results['summary']
signal_stats = results['signal_stats']

print("\n📊 Portfolio Performance:")
print(f"  Initial Capital:    ${summary['initial_capital']:>12,.2f}")
print(f"  Final Value:        ${summary['current_value']:>12,.2f}")
print(f"  Total Return:       {summary['total_return_pct']:>12.2f}%")
print(f"  Realized PnL:       ${summary['realized_pnl']:>12,.2f}")
print(f"  Unrealized PnL:     ${summary['unrealized_pnl']:>12,.2f}")
print(f"  Total Fees:         ${summary['total_fees']:>12,.2f}")

print("\n📈 Risk-Adjusted Metrics:")
print(f"  Sharpe Ratio:       {metrics.get('sharpe_ratio', 0):>12.2f}")
print(f"  Sortino Ratio:      {metrics.get('sortino_ratio', 0):>12.2f}")
print(f"  Calmar Ratio:       {metrics.get('calmar_ratio', 0):>12.2f}")

print("\n⚠️ Risk Metrics:")
print(f"  Max Drawdown:       {metrics.get('max_drawdown_pct', 0):>11.2f}%")
print(f"  Volatility:         {metrics.get('annual_volatility_pct', 0):>11.2f}%")

print("\n📊 Trading Activity:")
print(f"  Total Signals:      {results['num_signals']:>12}")
print(f"  Trades Executed:    {results['trades_executed']:>12}")
print(f"  BUY signals:        {signal_stats['buy_signals']:>12} ({signal_stats['buy_pct']:.1f}%)")
print(f"  SELL signals:       {signal_stats['sell_signals']:>12} ({signal_stats['sell_pct']:.1f}%)")
print(f"  HOLD signals:       {signal_stats['hold_signals']:>12} ({signal_stats['hold_pct']:.1f}%)")

if metrics.get('total_trades', 0) > 0:
    print("\n📊 Trade Statistics:")
    print(f"  Win Rate:           {metrics.get('win_rate', 0):>11.1f}%")
    print(f"  Profit Factor:      {metrics.get('profit_factor', 0):>12.2f}")
    print(f"  Avg Win:            ${metrics.get('avg_win', 0):>12,.2f}")
    print(f"  Avg Loss:           ${metrics.get('avg_loss', 0):>12,.2f}")

# =============================================================================
# 7. VISUALIZATION
# =============================================================================

print("\n" + "=" * 60)
print("STEP 7: VISUALIZATION")
print("=" * 60)

from analytics.visualizations import TradingVisualizer
import matplotlib
matplotlib.use('Agg')  # Headless mode

viz = TradingVisualizer()

# Create visualizations
viz.plot_equity_curve(
    results['equity_curve'],
    title='Advanced Backtest - Equity Curve',
    save_path='results/advanced_equity.png'
)
print("Saved: results/advanced_equity.png")

viz.plot_drawdown(
    results['equity_curve'],
    title='Advanced Backtest - Drawdown',
    save_path='results/advanced_drawdown.png'
)
print("Saved: results/advanced_drawdown.png")

if not results['trades'].empty:
    viz.plot_pnl_distribution(
        results['trades'],
        title='Advanced Backtest - Trade PnL',
        save_path='results/advanced_pnl.png'
    )
    print("Saved: results/advanced_pnl.png")

viz.plot_summary_dashboard(
    results['equity_curve'],
    results['trades'],
    metrics,
    save_path='results/advanced_dashboard.png'
)
print("Saved: results/advanced_dashboard.png")

# =============================================================================
# 8. EXPORT RESULTS
# =============================================================================

print("\n" + "=" * 60)
print("STEP 8: EXPORT RESULTS")
print("=" * 60)

import pandas as pd

# Export equity curve
results['equity_curve'].to_csv('results/advanced_equity.csv')
print("Exported: results/advanced_equity.csv")

# Export trades
if not results['trades'].empty:
    results['trades'].to_csv('results/advanced_trades.csv')
    print("Exported: results/advanced_trades.csv")

# Export metrics
pd.DataFrame([metrics]).to_csv('results/advanced_metrics.csv', index=False)
print("Exported: results/advanced_metrics.csv")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("✅ ADVANCED BACKTEST COMPLETE")
print("=" * 60)
print(f"\nFinal Return: {summary['total_return_pct']:.2f}%")
print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
print(f"\nAll results saved to results/")
```

## Running the Example

```bash
cd hft-sim
python docs/guides/examples/02_advanced_backtest.py
```

## Key Differences from Basic Example

| Feature | Basic | Advanced |
|---------|-------|----------|
| ML Model | No | Yes (LSTM) |
| Risk Management | Default | Custom limits |
| Caching | No | Yes |
| Metrics | Basic | Comprehensive |
| Visualization | No | Full dashboard |
| Export | No | CSV + PNG |

## Customization Points

1. **Data Period**: Change `days=7` for more/less data
2. **ML Architecture**: Adjust `hidden_size`, `num_layers`
3. **Strategy Parameters**: Tune thresholds
4. **Risk Limits**: Adjust position and drawdown limits

## Next Steps

- See [Custom Indicators](03_custom_indicators.md)
- Read [Parameter Optimization](../03_optimization.md)
