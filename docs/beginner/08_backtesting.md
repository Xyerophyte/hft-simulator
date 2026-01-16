# Backtesting Guide

This guide explains how to test trading strategies on historical data.

## What is Backtesting?

**Backtesting** = Testing your strategy on old data to see if it would have worked.

### The Idea

1. Get historical data (e.g., Bitcoin prices from last month)
2. Pretend you're trading in the past
3. Make decisions using only data available at that time
4. See if you would have made money

---

## Why Backtest?

### Before Risking Real Money

```
Without backtesting:
"I think this strategy will work!" → Risk $10,000 → Lose $3,000 → "Oops"

With backtesting:
"Let me test this first..." → Simulate on 6 months of data → 
    "Hmm, lost 5% on average" → Improve strategy → Test again → 
    "Now it's profitable!" → More confident to try with real money
```

### What Backtesting Tells You

- **Return**: Would you have made money?
- **Risk**: What was the worst drawdown?
- **Consistency**: Did it work in different market conditions?
- **Costs**: How much in fees?
- **Frequency**: How many trades?

---

## Event-Driven Backtesting

Our backtester works **bar by bar** through history:

```
Time 10:00 → Get data → Generate signal → Execute trade (if any)
Time 10:01 → Get data → Generate signal → Execute trade (if any)
Time 10:02 → Get data → Generate signal → Execute trade (if any)
... repeat for entire history ...
```

### Why Event-Driven?

**Realistic**: Mimics how real trading works
- You only know data up to NOW
- You can't peek at the future
- Each decision is made in sequence

---

## Running a Backtest

### Basic Example

```python
from backtest.backtester import Backtester, BacktestConfig
from strategies.momentum_strategy import MomentumStrategy

# 1. Setup strategy
strategy = MomentumStrategy(
    ml_threshold=0.55,
    momentum_threshold=0.0005
)

# 2. Configure backtest
config = BacktestConfig(
    initial_capital=100000,
    fee_rate=0.001,
    position_size_pct=0.3
)

# 3. Create backtester
backtester = Backtester(strategy, config)

# 4. Run!
results = backtester.run(df_features, symbol='BTC')

# 5. See results
print(f"Final Value: ${results['summary']['current_value']:,.2f}")
print(f"Return: {results['summary']['total_return_pct']:.2f}%")
print(f"Trades: {results['trades_executed']}")
```

---

## Understanding Results

### Summary Statistics

```python
summary = results['summary']

print(f"Initial Capital: ${summary['initial_capital']:,.2f}")
print(f"Final Value: ${summary['current_value']:,.2f}")
print(f"Total PnL: ${summary['total_pnl']:,.2f}")
print(f"Total Return: {summary['total_return_pct']:.2f}%")
print(f"Total Fees: ${summary['total_fees']:,.2f}")
```

### Signal Statistics

```python
signals = results['signal_stats']

print(f"Total Signals: {results['num_signals']}")
print(f"BUY signals: {signals['buy_signals']} ({signals['buy_pct']:.1f}%)")
print(f"SELL signals: {signals['sell_signals']} ({signals['sell_pct']:.1f}%)")
print(f"HOLD signals: {signals['hold_signals']} ({signals['hold_pct']:.1f}%)")
```

### Trade History

```python
trades = results['trades']

print(trades.head())
#    trade_id symbol side  quantity    price       timestamp    fee      pnl
# 0         1    BTC  BUY      0.30  50000.0  2024-01-01 10:00  15.0      0.0
# 1         2    BTC SELL      0.30  50250.0  2024-01-01 11:00  15.08    45.0
# 2         3    BTC  BUY      0.25  50100.0  2024-01-01 13:00  12.53     0.0
```

### Equity Curve

```python
equity = results['equity_curve']

print(equity.tail())
#                        equity     cash  positions_value
# 2024-01-31 23:58:00  102345.67  52345.67        50000.00
# 2024-01-31 23:59:00  102123.45  52345.67        49777.78
# 2024-01-31 23:60:00  102456.78  102456.78           0.00
```

---

## Visualizing Results

```python
from analytics.visualizations import TradingVisualizer
import matplotlib.pyplot as plt

viz = TradingVisualizer()

# Equity curve
viz.plot_equity_curve(results['equity_curve'])
plt.show()

# Drawdown
viz.plot_drawdown(results['equity_curve'])
plt.show()

# Trade PnL distribution
viz.plot_pnl_distribution(results['trades'])
plt.show()

# Full dashboard
viz.plot_summary_dashboard(
    results['equity_curve'],
    results['trades'],
    metrics,
    save_path='results/dashboard.png'
)
```

---

## Important Considerations

### 1. Transaction Costs

**Always include fees!**

```python
# Without fees (unrealistic)
config = BacktestConfig(fee_rate=0.0)
# Result: +15% return

# With realistic fees
config = BacktestConfig(fee_rate=0.001)
# Result: +8% return

# Fees ate 7% of profits!
```

### 2. Slippage

**Real execution isn't instant**

```python
# In simulation
# Order: Buy at $50,000
# Execution: Buy at $50,000 ✓

# In real life
# Order: Buy at $50,000
# Execution: Buy at $50,025 (price moved while order processed)
```

Our simulator models slippage:
```python
engine = MatchingEngine(slippage_pct=0.0005)  # 0.05% slippage
```

### 3. Look-Ahead Bias

**Only use data available at each moment**

❌ **Wrong (cheating):**
```python
# At 10:00, using data from 10:05
future_price = df.loc['10:05', 'close']  # Cheating!
```

✅ **Correct:**
```python
# At 10:00, only use data up to 10:00
current_data = df.loc[:'10:00']
```

Our backtester does this correctly automatically.

### 4. Survivorship Bias

**We only have data for assets that still exist**

- Failed companies aren't in the data
- Delisted coins aren't in the data
- This can inflate historical returns

---

## Walk-Forward Testing

**Best practice**: Don't train and test on same data!

### Rolling Window Approach

```
Month 1-3: Train model
Month 4: Test model → Record results

Month 2-4: Train model  
Month 5: Test model → Record results

Month 3-5: Train model
Month 6: Test model → Record results

... combine all test results for realistic estimate ...
```

### Implementation

```python
# Walk-forward backtest
results_list = []
train_window = 1000  # bars
test_window = 200    # bars

for i in range(0, len(df) - train_window - test_window, test_window):
    # Split data
    train_data = df.iloc[i:i + train_window]
    test_data = df.iloc[i + train_window:i + train_window + test_window]
    
    # Train model on training data
    # ...
    
    # Test on unseen data
    backtester = Backtester(strategy, config)
    results = backtester.run(test_data)
    
    results_list.append(results['summary']['total_return_pct'])

print(f"Average return per window: {np.mean(results_list):.2f}%")
```

---

## Key Metrics to Watch

### Good Signs

| Metric | Good Value |
|--------|------------|
| Sharpe Ratio | > 1.0 |
| Win Rate | > 50% |
| Profit Factor | > 1.5 |
| Max Drawdown | < 15% |

### Warning Signs

| Sign | What It Means |
|------|---------------|
| Sharpe > 3.0 | Too good, probably wrong |
| Win Rate > 80% | Suspiciously high |
| No losing months | Unrealistic |
| Huge returns | Check for bugs |

---

## Common Mistakes

### 1. Over-Optimization

```
Tested 1000 parameter combinations
Found one that made 200% return!
→ Probably just luck, won't work in future
```

**Fix**: Use walk-forward testing, limit parameter search

### 2. Too Little Data

```
Tested on 1 week of data
Strategy worked great!
→ Might just be lucky week
```

**Fix**: Test on at least 6 months, multiple market conditions

### 3. Ignoring Costs

```
Strategy makes 0.1% per trade
Fees are 0.1% per trade
→ Net profit: 0%
```

**Fix**: Always include realistic fees and slippage

---

## Quick Reference

### Running a Backtest

```python
# 1. Get data
from data.fetcher import BinanceDataFetcher
fetcher = BinanceDataFetcher()
df = fetcher.fetch_historical_data(days=30)

# 2. Create features
from ml.features import FeatureEngineer
engineer = FeatureEngineer()
df_features = engineer.create_all_features(df)

# 3. Setup and run
from backtest.backtester import Backtester, BacktestConfig
from strategies.momentum_strategy import MomentumStrategy

strategy = MomentumStrategy()
config = BacktestConfig(initial_capital=100000)
backtester = Backtester(strategy, config)
results = backtester.run(df_features)

# 4. Analyze
print(f"Return: {results['summary']['total_return_pct']:.2f}%")
```

---

## Next Steps

- [Metrics Explained](09_metrics.md) - Understanding performance numbers
- [Complete Workflow](10_complete_workflow.md) - Putting it all together
