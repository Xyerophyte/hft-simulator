# Complete Workflow

This guide walks through the entire HFT simulation process from start to finish.

## Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     COMPLETE WORKFLOW                       │
├─────────────────────────────────────────────────────────────┤
│  1. Setup        → Install, configure                       │
│  2. Data         → Fetch, cache, preprocess                 │
│  3. Features     → Engineer ML features                     │
│  4. Model        → Train LSTM predictor                     │
│  5. Strategy     → Configure trading rules                  │
│  6. Backtest     → Run simulation                           │
│  7. Analyze      → Calculate metrics                        │
│  8. Visualize    → Create charts                            │
│  9. Iterate      → Improve and repeat                       │
└─────────────────────────────────────────────────────────────┘
```

---

## Step 1: Setup

### Project Structure

```bash
hft-sim/
├── src/           # Source code
├── data/          # Data storage
├── results/       # Output files
├── models/        # Saved models
└── run_demo.py    # Quick start script
```

### Quick Test

```bash
# Run the complete demo
python run_demo.py
```

---

## Step 2: Fetch Data

### Get Market Data

```python
from data.fetcher import BinanceDataFetcher
from data.cache import DataCache

# Initialize
fetcher = BinanceDataFetcher(symbol="BTCUSDT")
cache = DataCache(cache_dir="data/cache")

# Fetch 3 days of 1-minute data
print("Fetching data...")
df = fetcher.fetch_klines(interval='1m', limit=4320)

# Cache for later use
cache.save(df, 'BTCUSDT', '1m')

print(f"Downloaded {len(df)} candles")
print(f"Date range: {df.index[0]} to {df.index[-1]}")
print(f"Columns: {list(df.columns)}")
```

**Output:**
```
Fetching data...
Downloaded 4320 candles
Date range: 2024-01-28 10:00:00 to 2024-01-31 10:00:00
Columns: ['open', 'high', 'low', 'close', 'volume']
```

---

## Step 3: Preprocess Data

### Add Technical Indicators

```python
from data.preprocessor import DataPreprocessor

preprocessor = DataPreprocessor()

print("Adding technical indicators...")
df_processed = preprocessor.preprocess_pipeline(df)

print(f"Original columns: 5")
print(f"After preprocessing: {len(df_processed.columns)}")
print(f"Rows after cleaning: {len(df_processed)}")

# Preview
print("\nSample indicators:")
print(df_processed[['close', 'sma_20', 'rsi_14', 'atr_14']].tail())
```

**Output:**
```
Adding technical indicators...
Original columns: 5
After preprocessing: 34
Rows after cleaning: 4270

Sample indicators:
                     close    sma_20   rsi_14   atr_14
2024-01-31 09:56  42150.5  42089.25    52.34   125.67
2024-01-31 09:57  42175.0  42095.50    54.12   124.89
...
```

---

## Step 4: Engineer Features

### Create ML Features

```python
from ml.features import FeatureEngineer

engineer = FeatureEngineer()

print("Engineering features...")
df_features = engineer.create_all_features(df_processed)

print(f"Total features: {len(df_features.columns)}")
print(f"Dataset size: {len(df_features)} rows")

# Feature categories
print("\nFeature categories:")
print("- Price features: returns, momentum, log_returns")
print("- Volatility: rolling_std, atr, bollinger")
print("- Volume: vwap, volume_ratio, obv")
print("- Candle: body_size, shadows, doji")
print("- Order flow: vpin proxies")
print("- Time: hour, day_of_week")
```

### Prepare Training Data

```python
# Create sequences for LSTM
X, y, feature_names = engineer.prepare_training_data(
    df_features,
    lookback=30,           # Use 30 minutes of history
    prediction_horizon=1   # Predict 1 minute ahead
)

print(f"\nTraining data shape:")
print(f"  X: {X.shape} (samples, timesteps, features)")
print(f"  y: {y.shape} (samples,)")
print(f"  Class balance: {y.mean():.1%} up, {1-y.mean():.1%} down")
```

---

## Step 5: Train Model

### Create and Train LSTM

```python
from ml.models import PriceLSTM

print("Creating LSTM model...")
model = PriceLSTM(
    input_size=X.shape[2],  # Number of features
    hidden_size=32,         # LSTM hidden units
    num_layers=2            # LSTM layers
)

print(f"Model architecture:")
print(f"  Input: {X.shape[2]} features")
print(f"  Hidden: 32 units × 2 layers")
print(f"  Output: Binary (up/down)")

print("\nTraining...")
history = model.train_model(
    X, y,
    epochs=15,
    batch_size=32,
    validation_split=0.2
)

print(f"\nTraining complete!")
print(f"Final validation accuracy: {history['val_accuracy'][-1]:.1%}")
```

**Output:**
```
Training on cpu with 2500 samples...
Epoch 5/15 - Loss: 0.6923, Val Loss: 0.6918, Val Acc: 0.5134
Epoch 10/15 - Loss: 0.6875, Val Loss: 0.6890, Val Acc: 0.5287
Epoch 15/15 - Loss: 0.6812, Val Loss: 0.6878, Val Acc: 0.5356

Training complete!
Final validation accuracy: 53.6%
```

---

## Step 6: Setup Strategy

### Configure Strategy

```python
from strategies.momentum_strategy import MomentumStrategy
from strategies.risk_manager import RiskLimits
from backtest.backtester import BacktestConfig

# Create strategy
strategy = MomentumStrategy(
    ml_threshold=0.55,        # ML confidence for signals
    momentum_threshold=0.0005, # 0.05% momentum
    volume_threshold=1.2       # Volume confirmation
)

# Risk limits
limits = RiskLimits(
    max_position_pct=0.3,     # 30% max position
    stop_loss_pct=0.02,       # 2% stop loss
    max_drawdown_pct=0.15     # 15% max drawdown
)

# Backtest configuration
config = BacktestConfig(
    initial_capital=100000,
    fee_rate=0.001,
    position_size_pct=0.3,
    use_risk_manager=True
)

print("Strategy configured:")
print(f"  ML threshold: {strategy.ml_threshold}")
print(f"  Max position: {config.position_size_pct:.0%}")
print(f"  Fee rate: {config.fee_rate:.2%}")
```

---

## Step 7: Run Backtest

### Execute Simulation

```python
from backtest.backtester import Backtester

print("Running backtest...")
backtester = Backtester(strategy, config)
results = backtester.run(df_features, symbol='BTC')

print("\n" + "="*50)
print("BACKTEST RESULTS")
print("="*50)

summary = results['summary']
print(f"\nPortfolio:")
print(f"  Initial Capital: ${summary['initial_capital']:,.2f}")
print(f"  Final Value: ${summary['current_value']:,.2f}")
print(f"  Total Return: {summary['total_return_pct']:.2f}%")
print(f"  Total PnL: ${summary['total_pnl']:,.2f}")

print(f"\nTrading:")
print(f"  Total Signals: {results['num_signals']}")
print(f"  Trades Executed: {results['trades_executed']}")
print(f"  Total Fees: ${summary['total_fees']:,.2f}")

signals = results['signal_stats']
print(f"\nSignal Distribution:")
print(f"  BUY: {signals['buy_signals']} ({signals['buy_pct']:.1f}%)")
print(f"  SELL: {signals['sell_signals']} ({signals['sell_pct']:.1f}%)")
print(f"  HOLD: {signals['hold_signals']} ({signals['hold_pct']:.1f}%)")
```

---

## Step 8: Analyze Performance

### Calculate Metrics

```python
from analytics.metrics import PerformanceMetrics

# Calculate all metrics
metrics = PerformanceMetrics.calculate_all_metrics(
    results['equity_curve'],
    results['trades'],
    config.initial_capital
)

print("\n" + "="*50)
print("PERFORMANCE METRICS")
print("="*50)

print(f"\nRisk-Adjusted:")
print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"  Sortino Ratio: {metrics['sortino_ratio']:.2f}")
print(f"  Calmar Ratio: {metrics['calmar_ratio']:.2f}")

print(f"\nRisk:")
print(f"  Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
print(f"  Volatility: {metrics['annual_volatility_pct']:.1f}%")

print(f"\nTrade Quality:")
print(f"  Win Rate: {metrics['win_rate']:.1f}%")
print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
print(f"  Avg Win: ${metrics['avg_win']:.2f}")
print(f"  Avg Loss: ${metrics['avg_loss']:.2f}")
```

---

## Step 9: Visualize Results

### Create Charts

```python
from analytics.visualizations import TradingVisualizer
import matplotlib.pyplot as plt

viz = TradingVisualizer()

# 1. Equity Curve
viz.plot_equity_curve(
    results['equity_curve'],
    title='Portfolio Equity Over Time',
    save_path='results/equity_curve.png'
)
print("Saved: equity_curve.png")

# 2. Drawdown
viz.plot_drawdown(
    results['equity_curve'],
    title='Drawdown Analysis',
    save_path='results/drawdown.png'
)
print("Saved: drawdown.png")

# 3. Trade Distribution
viz.plot_pnl_distribution(
    results['trades'],
    title='Trade PnL Distribution',
    save_path='results/pnl_distribution.png'
)
print("Saved: pnl_distribution.png")

# 4. Dashboard
viz.plot_summary_dashboard(
    results['equity_curve'],
    results['trades'],
    metrics,
    save_path='results/dashboard.png'
)
print("Saved: dashboard.png")
```

---

## Step 10: Export Results

### Save Data

```python
import pandas as pd
import os

os.makedirs('results', exist_ok=True)

# Save equity curve
results['equity_curve'].to_csv('results/equity_curve.csv')
print("Saved: results/equity_curve.csv")

# Save trades
results['trades'].to_csv('results/trades.csv')
print("Saved: results/trades.csv")

# Save metrics
metrics_df = pd.DataFrame([metrics])
metrics_df.to_csv('results/metrics.csv', index=False)
print("Saved: results/metrics.csv")

# Save summary
summary_df = pd.DataFrame([summary])
summary_df.to_csv('results/summary.csv', index=False)
print("Saved: results/summary.csv")

print("\n✅ All results exported to results/")
```

---

## Complete Script

All of the above in one runnable script:

```python
#!/usr/bin/env python3
"""Complete HFT Simulation Workflow"""

import sys
sys.path.insert(0, 'src')

# Imports
from data.fetcher import BinanceDataFetcher
from data.preprocessor import DataPreprocessor
from ml.features import FeatureEngineer
from ml.models import PriceLSTM
from strategies.momentum_strategy import MomentumStrategy
from backtest.backtester import Backtester, BacktestConfig
from analytics.metrics import PerformanceMetrics
from analytics.visualizations import TradingVisualizer
import os

# Create directories
os.makedirs('results', exist_ok=True)

# 1. Fetch data
print("Step 1: Fetching data...")
fetcher = BinanceDataFetcher()
df = fetcher.fetch_klines(limit=3000)

# 2. Preprocess
print("Step 2: Preprocessing...")
preprocessor = DataPreprocessor()
df_proc = preprocessor.preprocess_pipeline(df)

# 3. Features
print("Step 3: Engineering features...")
engineer = FeatureEngineer()
df_features = engineer.create_all_features(df_proc)

# 4. Train model (optional)
print("Step 4: Training model...")
X, y, _ = engineer.prepare_training_data(df_features, lookback=30)
model = PriceLSTM(input_size=X.shape[2])
model.train_model(X, y, epochs=10)

# 5. Backtest
print("Step 5: Running backtest...")
strategy = MomentumStrategy()
config = BacktestConfig(initial_capital=100000)
backtester = Backtester(strategy, config)
results = backtester.run(df_features)

# 6. Analyze
print("Step 6: Analyzing...")
metrics = PerformanceMetrics.calculate_all_metrics(
    results['equity_curve'], results['trades'], 100000
)

# 7. Visualize
print("Step 7: Creating visualizations...")
viz = TradingVisualizer()
viz.plot_summary_dashboard(
    results['equity_curve'], results['trades'], metrics,
    save_path='results/dashboard.png'
)

# 8. Report
print("\n" + "="*50)
print("FINAL RESULTS")
print("="*50)
print(f"Return: {results['summary']['total_return_pct']:.2f}%")
print(f"Sharpe: {metrics['sharpe_ratio']:.2f}")
print(f"Max DD: {metrics['max_drawdown_pct']:.1f}%")
print(f"Win Rate: {metrics['win_rate']:.1f}%")
print("="*50)
print("✅ Complete! Check results/ for outputs.")
```

---

## What's Next?

After completing your first backtest:

1. **Experiment with parameters**
   - Try different ML thresholds
   - Adjust position sizes
   - Change lookback periods

2. **Improve the model**
   - More training data
   - Different architectures
   - Feature selection

3. **Add strategies**
   - Mean reversion
   - Market making
   - Multiple assets

4. **Walk-forward test**
   - Avoid overfitting
   - More realistic results

---

**Congratulations!** You've completed the full HFT simulation workflow. 🎉
