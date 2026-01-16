# Basic Backtest Example

A simple, complete example of running a backtest.

## Full Code

```python
#!/usr/bin/env python3
"""
Basic Backtest Example

This script demonstrates a simple backtest using default settings.
"""

import sys
sys.path.insert(0, 'src')

from data.fetcher import BinanceDataFetcher
from data.preprocessor import DataPreprocessor
from ml.features import FeatureEngineer
from strategies.momentum_strategy import MomentumStrategy
from backtest.backtester import Backtester, BacktestConfig

# 1. Fetch Data
print("Fetching data...")
fetcher = BinanceDataFetcher(symbol="BTCUSDT")
df = fetcher.fetch_klines(interval='1m', limit=2000)
print(f"Downloaded {len(df)} candles")

# 2. Preprocess
print("\nPreprocessing...")
preprocessor = DataPreprocessor()
df_processed = preprocessor.preprocess_pipeline(df)
print(f"After preprocessing: {len(df_processed)} rows")

# 3. Create Features
print("\nCreating features...")
engineer = FeatureEngineer()
df_features = engineer.create_all_features(df_processed)
print(f"Features created: {len(df_features.columns)} columns")

# 4. Setup Strategy
strategy = MomentumStrategy(
    ml_threshold=0.55,
    momentum_threshold=0.0005,
    volume_threshold=1.2
)

# 5. Configure Backtest
config = BacktestConfig(
    initial_capital=100000.0,
    fee_rate=0.001,
    position_size_pct=0.3
)

# 6. Run Backtest
print("\nRunning backtest...")
backtester = Backtester(strategy, config)
results = backtester.run(df_features, symbol='BTC')

# 7. Display Results
print("\n" + "="*50)
print("RESULTS")
print("="*50)

summary = results['summary']
print(f"Initial Capital: ${summary['initial_capital']:,.2f}")
print(f"Final Value:     ${summary['current_value']:,.2f}")
print(f"Total Return:    {summary['total_return_pct']:.2f}%")
print(f"Total PnL:       ${summary['total_pnl']:,.2f}")
print(f"Total Fees:      ${summary['total_fees']:,.2f}")
print(f"Trades:          {results['trades_executed']}")

print("\n" + "="*50)
print("Done!")
```

## Running the Example

```bash
cd hft-sim
python docs/guides/examples/01_basic_backtest.py
```

## Expected Output

```
Fetching data...
Downloaded 2000 candles

Preprocessing...
Starting preprocessing pipeline...
After cleaning: 2000 rows
Added 29 total columns
After removing NaN: 1950 rows
After preprocessing: 1950 rows

Creating features...
Creating features...
Created 90 total columns
Features created: 95 columns

Running backtest...

==================================================
RESULTS
==================================================
Initial Capital: $100,000.00
Final Value:     $99,234.56
Total Return:    -0.77%
Total PnL:       $-123.45
Total Fees:      $567.89
Trades:          45

==================================================
Done!
```

## Key Points

1. **Data Source**: Uses Binance public API (no auth needed)
2. **Default Settings**: Uses sensible defaults for all parameters
3. **Minimal Code**: Complete backtest in ~40 lines
4. **Ready to Extend**: Easy to modify parameters and strategy

## Next Steps

- Try different `ml_threshold` values (0.52, 0.55, 0.60)
- Adjust `initial_capital` 
- Use more data: `limit=5000`
- See [Advanced Backtest](02_advanced_backtest.md) for more options
