# Walk-Forward Analysis Guide

This guide explains walk-forward testing to avoid overfitting.

## The Problem: Overfitting

### What is Overfitting?

**Overfitting** = Your strategy memorizes the past instead of learning real patterns.

```
Backtest result: +50% return!  🎉
Live trading: -15% return  😢

What happened? The strategy was overfit.
```

### Simple Analogy

Imagine a student preparing for an exam:

- **Good learning**: Understands concepts → Can solve new problems
- **Overfitting**: Memorizes answer key → Fails when questions change

---

## Walk-Forward Testing

### The Idea

Don't test your strategy on data it was trained on!

```
Traditional (wrong):
┌──────────────────────────────────────────┐
│        Train + Test on ALL data          │
└──────────────────────────────────────────┘
Problem: Model sees test data during training

Walk-Forward (correct):
┌─────────────────────┬───────┐
│     Train           │ Test  │  ← Test on unseen data
└─────────────────────┴───────┘
         Roll forward ──────────▶
┌─────────────────────┬───────┐
│         Train       │ Test  │  ← New test period
└─────────────────────┴───────┘
```

---

## How It Works

### Step by Step

1. **Split data** into train and test windows
2. **Train model** on training window
3. **Test** on test window (never seen)
4. **Record** performance
5. **Move forward** in time
6. **Repeat**
7. **Combine** all test results

### Visual Example

```
Data: January to June

Round 1:
├── Train: Jan-Feb ──┤── Test: March ──┤
                      └── Record results

Round 2:
    ├── Train: Feb-Mar ──┤── Test: April ──┤
                          └── Record results

Round 3:
        ├── Train: Mar-Apr ──┤── Test: May ──┤
                              └── Record results

Round 4:
            ├── Train: Apr-May ──┤── Test: June ──┤
                                  └── Record results

Final: Combine all test results (March + April + May + June)
```

---

## Implementation

### Basic Walk-Forward

```python
import numpy as np
import pandas as pd
from ml.features import FeatureEngineer
from ml.models import PriceLSTM
from strategies.momentum_strategy import MomentumStrategy
from backtest.backtester import Backtester, BacktestConfig

def walk_forward_backtest(
    df: pd.DataFrame,
    train_window: int = 2000,   # Training samples
    test_window: int = 500,      # Test samples
    step_size: int = 500         # How far to move each iteration
):
    """
    Perform walk-forward analysis.
    
    Args:
        df: Feature DataFrame
        train_window: Number of bars for training
        test_window: Number of bars for testing
        step_size: How many bars to move forward each round
        
    Returns:
        Combined results from all test windows
    """
    results_list = []
    
    # Calculate number of rounds
    n_rounds = (len(df) - train_window - test_window) // step_size + 1
    print(f"Running {n_rounds} walk-forward rounds...")
    
    for i in range(n_rounds):
        start = i * step_size
        train_end = start + train_window
        test_end = train_end + test_window
        
        if test_end > len(df):
            break
            
        # Split data
        train_data = df.iloc[start:train_end]
        test_data = df.iloc[train_end:test_end]
        
        print(f"\nRound {i+1}/{n_rounds}:")
        print(f"  Train: rows {start} to {train_end}")
        print(f"  Test: rows {train_end} to {test_end}")
        
        # Train model on training data
        engineer = FeatureEngineer()
        X_train, y_train, _ = engineer.prepare_training_data(
            train_data, lookback=30
        )
        
        model = PriceLSTM(input_size=X_train.shape[2])
        model.train_model(X_train, y_train, epochs=10, validation_split=0.2)
        
        # Test on unseen data
        strategy = MomentumStrategy()
        config = BacktestConfig(initial_capital=100000)
        backtester = Backtester(strategy, config)
        
        test_results = backtester.run(test_data, symbol='BTC')
        
        # Record results
        results_list.append({
            'round': i + 1,
            'train_start': start,
            'test_start': train_end,
            'return_pct': test_results['summary']['total_return_pct'],
            'trades': test_results['trades_executed'],
            'final_value': test_results['summary']['current_value']
        })
        
        print(f"  Return: {test_results['summary']['total_return_pct']:.2f}%")
    
    return pd.DataFrame(results_list)
```

### Running Walk-Forward

```python
# Load and prepare data
from data.fetcher import BinanceDataFetcher
from data.preprocessor import DataPreprocessor
from ml.features import FeatureEngineer

fetcher = BinanceDataFetcher()
df = fetcher.fetch_historical_data(days=30)

preprocessor = DataPreprocessor()
df_proc = preprocessor.preprocess_pipeline(df)

engineer = FeatureEngineer()
df_features = engineer.create_all_features(df_proc)

# Run walk-forward analysis
wf_results = walk_forward_backtest(
    df_features,
    train_window=5000,
    test_window=1000,
    step_size=1000
)

# Analyze results
print("\n" + "="*50)
print("WALK-FORWARD RESULTS")
print("="*50)
print(wf_results)

print(f"\nAggregate Statistics:")
print(f"  Average Return: {wf_results['return_pct'].mean():.2f}%")
print(f"  Std Dev Return: {wf_results['return_pct'].std():.2f}%")
print(f"  Min Return: {wf_results['return_pct'].min():.2f}%")
print(f"  Max Return: {wf_results['return_pct'].max():.2f}%")
print(f"  Win Rate: {(wf_results['return_pct'] > 0).mean()*100:.1f}%")
```

---

## Interpreting Results

### Good Results

```
Walk-Forward Results:
  Round 1: +2.3%
  Round 2: +1.8%
  Round 3: -0.5%
  Round 4: +3.1%
  Round 5: +1.2%

Average: +1.58%
Win Rate: 80%

✅ Consistent positive returns across periods
✅ Most periods are profitable
✅ No huge outliers
```

### Bad Results (Overfitting)

```
Traditional Backtest: +45%  ← Looks great!

Walk-Forward Results:
  Round 1: +8.5%   ← First period still in-sample
  Round 2: -2.3%
  Round 3: -4.1%
  Round 4: -1.8%
  Round 5: -3.2%

Average: -0.58%
Win Rate: 20%

❌ Only first period is profitable
❌ Strategy doesn't generalize
❌ Traditional backtest was misleading
```

---

## Window Size Considerations

### Training Window

| Size | Pros | Cons |
|------|------|------|
| **Small** (500) | Fast, adapts quickly | May not learn enough |
| **Medium** (2000) | Balanced | Good default |
| **Large** (5000) | Learns more patterns | Slow to adapt, old data |

### Test Window

| Size | Pros | Cons |
|------|------|------|
| **Small** (100) | More test periods | Noisy results |
| **Medium** (500) | Balanced | Good default |
| **Large** (1000) | More reliable per period | Fewer test periods |

### Rule of Thumb

```python
# Good starting point
train_window = 3 * test_window

# Example
train_window = 3000  # 3000 minutes = ~2 days
test_window = 1000   # 1000 minutes = ~17 hours
```

---

## Advanced: Expanding Window

Instead of rolling window, use **expanding window**:

```
Round 1:
├── Train: Jan ─────┤── Test: Feb ──┤

Round 2:
├── Train: Jan-Feb ────────┤── Test: Mar ──┤

Round 3:
├── Train: Jan-Feb-Mar ────────────┤── Test: Apr ──┤

Training window grows over time (uses all historical data)
```

```python
def expanding_window_backtest(df, initial_train=2000, test_window=500):
    """Walk-forward with expanding training window."""
    results = []
    
    for test_start in range(initial_train, len(df) - test_window, test_window):
        train_data = df.iloc[:test_start]  # All data up to test
        test_data = df.iloc[test_start:test_start + test_window]
        
        # Train and test...
        
    return results
```

---

## Validation Metrics

### Key Questions

1. **Is average return positive?**
   - Should be significantly > 0

2. **What's the win rate?**
   - What % of test periods are profitable?
   - Should be > 50%

3. **Is volatility reasonable?**
   - Big variance means unreliable
   
4. **Any trend in performance?**
   - Getting worse over time? Strategy may be dated

### Statistical Test

```python
from scipy import stats

# Is average return significantly different from 0?
t_stat, p_value = stats.ttest_1samp(wf_results['return_pct'], 0)

print(f"t-statistic: {t_stat:.2f}")
print(f"p-value: {p_value:.4f}")

if p_value < 0.05:
    print("✅ Returns are statistically significant")
else:
    print("❌ Returns may be due to chance")
```

---

## Common Mistakes

### 1. Leaking Future Data

```python
# WRONG - using all data to compute features
df['sma'] = df['close'].rolling(20).mean()  # Uses future data!

# CORRECT - compute within each window
for window in walk_forward_windows:
    window['sma'] = window['close'].rolling(20).mean()
```

### 2. Too Much Optimization

```python
# WRONG
for params in 1000_parameter_combinations:
    # Pick best on training data
    # This over-optimizes to training!

# BETTER
# Use fixed or few parameters
# Or use nested cross-validation
```

### 3. Ignoring Transaction Costs

```python
# WRONG
config = BacktestConfig(fee_rate=0.0)  # No fees

# CORRECT
config = BacktestConfig(fee_rate=0.001)  # Realistic fees
```

---

## Summary

| Approach | Use When | Risk |
|----------|----------|------|
| **Traditional backtest** | Quick sanity check | High overfitting risk |
| **Train/test split** | Moderate validation | Some overfitting |
| **Walk-forward** | Serious validation | Most realistic |
| **Expanding window** | Maximum data use | Good for production |

---

## Key Takeaways

1. **Never test on training data** - it's cheating!
2. **Walk-forward** simulates real trading
3. **Consistent results** across periods = robust strategy
4. **Variable results** = strategy may be overfit
5. **Average performance** is what matters, not best period

---

## Next Steps

- [Parameter Optimization](03_optimization.md) - Finding best parameters
- [Custom Strategies](04_custom_strategies.md) - Building your own
