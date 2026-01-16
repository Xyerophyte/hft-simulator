# Parameter Optimization Guide

This guide explains how to find the best parameters for your strategy.

## Why Optimize?

Different parameter values give different results:

```
Parameter Set A: Sharpe = 0.8
Parameter Set B: Sharpe = 1.4  ← Better!
Parameter Set C: Sharpe = 1.1
```

Optimization helps find the best settings systematically.

---

## The Dangers

### Over-Optimization

```
Tested 10,000 combinations
Found one with +200% return!

Reality: Just found the lucky combination that fit noise
Future result: -20%
```

### How to Avoid

1. **Limit search space** - Don't try every possible value
2. **Use walk-forward validation** - Test on unseen data
3. **Be suspicious** of extreme results
4. **Prefer robust regions** over single points

---

## Grid Search

### The Simplest Approach

Try every combination of parameter values.

```python
from itertools import product
import pandas as pd

def grid_search_backtest(df_features, param_grid):
    """
    Test all parameter combinations.
    
    Args:
        df_features: Feature DataFrame
        param_grid: Dict of parameter values to try
        
    Returns:
        DataFrame with results for each combination
    """
    from strategies.momentum_strategy import MomentumStrategy
    from backtest.backtester import Backtester, BacktestConfig
    from analytics.metrics import PerformanceMetrics
    
    results = []
    
    # Generate all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(product(*values))
    
    print(f"Testing {len(combinations)} parameter combinations...")
    
    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        
        # Create strategy with these parameters
        strategy = MomentumStrategy(
            ml_threshold=params.get('ml_threshold', 0.55),
            momentum_threshold=params.get('momentum_threshold', 0.0005),
            volume_threshold=params.get('volume_threshold', 1.2)
        )
        
        config = BacktestConfig(
            initial_capital=100000,
            position_size_pct=params.get('position_size', 0.3)
        )
        
        # Run backtest
        backtester = Backtester(strategy, config)
        bt_results = backtester.run(df_features)
        
        # Calculate metrics
        metrics = PerformanceMetrics.calculate_all_metrics(
            bt_results['equity_curve'],
            bt_results['trades'],
            100000
        )
        
        # Record results
        result = {
            **params,
            'sharpe': metrics['sharpe_ratio'],
            'return_pct': metrics['total_return_pct'],
            'max_dd': metrics['max_drawdown_pct'],
            'trades': bt_results['trades_executed']
        }
        results.append(result)
        
        if (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{len(combinations)}")
    
    return pd.DataFrame(results)


# Example usage
param_grid = {
    'ml_threshold': [0.52, 0.55, 0.58, 0.60],
    'momentum_threshold': [0.0003, 0.0005, 0.0007],
    'volume_threshold': [1.0, 1.2, 1.5],
    'position_size': [0.2, 0.3]
}

results_df = grid_search_backtest(df_features, param_grid)

# Find best by Sharpe ratio
best = results_df.loc[results_df['sharpe'].idxmax()]
print("\nBest parameters:")
print(best)
```

---

## Random Search

### Why Random?

- Grid search tests systematically but may miss good regions
- Random search explores more of the space
- Often finds good solutions faster

```python
import numpy as np

def random_search_backtest(df_features, param_ranges, n_trials=50):
    """
    Random search for best parameters.
    
    Args:
        df_features: Feature DataFrame
        param_ranges: Dict with (min, max) for each parameter
        n_trials: Number of random trials
    """
    results = []
    
    for trial in range(n_trials):
        # Generate random parameters
        params = {}
        for name, (min_val, max_val) in param_ranges.items():
            if isinstance(min_val, float):
                params[name] = np.random.uniform(min_val, max_val)
            else:
                params[name] = np.random.randint(min_val, max_val + 1)
        
        # Run backtest with these parameters
        # ... (similar to grid search)
        
        results.append({**params, 'sharpe': sharpe, 'return': return_pct})
        
        print(f"Trial {trial + 1}: Sharpe = {sharpe:.2f}")
    
    return pd.DataFrame(results)


# Example
param_ranges = {
    'ml_threshold': (0.50, 0.65),
    'momentum_threshold': (0.0001, 0.001),
    'volume_threshold': (1.0, 2.0),
    'position_size': (0.1, 0.4)
}

results = random_search_backtest(df_features, param_ranges, n_trials=100)
```

---

## Robustness Analysis

### Don't Just Pick the Best Point

Look at the **neighborhood** around the best parameters.

```python
def analyze_robustness(results_df, param_name, best_value, tolerance=0.1):
    """
    Check if performance is stable around best parameter.
    """
    nearby = results_df[
        (results_df[param_name] >= best_value * (1 - tolerance)) &
        (results_df[param_name] <= best_value * (1 + tolerance))
    ]
    
    print(f"Results near {param_name} = {best_value}:")
    print(f"  Count: {len(nearby)}")
    print(f"  Avg Sharpe: {nearby['sharpe'].mean():.2f}")
    print(f"  Std Sharpe: {nearby['sharpe'].std():.2f}")
    print(f"  Min Sharpe: {nearby['sharpe'].min():.2f}")
    
    if nearby['sharpe'].std() < 0.1:
        print("  ✅ Robust region")
    else:
        print("  ⚠️ Unstable region, be cautious")


# Find robust parameters
best_ml_threshold = 0.57
analyze_robustness(results_df, 'ml_threshold', best_ml_threshold)
```

### Visual Robustness Check

```python
import matplotlib.pyplot as plt

def plot_parameter_surface(results_df, param1, param2, metric='sharpe'):
    """Plot performance surface for two parameters."""
    pivot = results_df.pivot_table(
        values=metric, 
        index=param1, 
        columns=param2
    )
    
    plt.figure(figsize=(10, 6))
    plt.imshow(pivot, cmap='RdYlGn', aspect='auto')
    plt.colorbar(label=metric)
    plt.xlabel(param2)
    plt.ylabel(param1)
    plt.title(f'{metric} by {param1} and {param2}')
    plt.show()
    
    # Look for smooth gradients (robust) vs spiky peaks (overfit)


plot_parameter_surface(results_df, 'ml_threshold', 'momentum_threshold')
```

---

## Multi-Objective Optimization

### Don't Just Optimize Sharpe

Consider multiple objectives:

```python
def score_strategy(sharpe, max_dd, trades):
    """
    Multi-objective scoring function.
    
    Balances:
    - High Sharpe (good returns per risk)
    - Low drawdown (risk control)
    - Enough trades (statistical significance)
    """
    score = 0
    
    # Sharpe contribution (0-40 points)
    score += min(sharpe * 20, 40)
    
    # Drawdown penalty (0-30 points)
    if max_dd < 5:
        score += 30
    elif max_dd < 10:
        score += 20
    elif max_dd < 15:
        score += 10
    
    # Trade count (0-30 points)
    if trades >= 50:
        score += 30
    elif trades >= 20:
        score += 20
    elif trades >= 10:
        score += 10
    
    return score


# Score all results
results_df['score'] = results_df.apply(
    lambda r: score_strategy(r['sharpe'], r['max_dd'], r['trades']),
    axis=1
)

# Find best balanced parameters
best_balanced = results_df.loc[results_df['score'].idxmax()]
print("Best balanced parameters:")
print(best_balanced)
```

---

## Walk-Forward Optimization

### The Gold Standard

Optimize on training data, validate on test data.

```python
def walk_forward_optimization(df, param_grid, train_size=0.7):
    """
    Optimize on training, validate on test.
    """
    # Split data chronologically
    split_idx = int(len(df) * train_size)
    train_data = df.iloc[:split_idx]
    test_data = df.iloc[split_idx:]
    
    print(f"Training on {len(train_data)} samples")
    print(f"Testing on {len(test_data)} samples")
    
    # Grid search on training data
    print("\nOptimizing on training data...")
    train_results = grid_search_backtest(train_data, param_grid)
    
    # Find best on training
    best_train = train_results.loc[train_results['sharpe'].idxmax()]
    print(f"\nBest on training: Sharpe = {best_train['sharpe']:.2f}")
    
    # Validate on test data
    print("\nValidating on test data...")
    best_params = {
        'ml_threshold': best_train['ml_threshold'],
        'momentum_threshold': best_train['momentum_threshold'],
        'volume_threshold': best_train['volume_threshold']
    }
    
    strategy = MomentumStrategy(**best_params)
    config = BacktestConfig()
    backtester = Backtester(strategy, config)
    test_results = backtester.run(test_data)
    
    test_metrics = PerformanceMetrics.calculate_all_metrics(
        test_results['equity_curve'],
        test_results['trades'],
        100000
    )
    
    print(f"\nTest results with optimized parameters:")
    print(f"  Sharpe: {test_metrics['sharpe_ratio']:.2f}")
    print(f"  Return: {test_metrics['total_return_pct']:.2f}%")
    
    # Compare train vs test
    degradation = best_train['sharpe'] - test_metrics['sharpe_ratio']
    print(f"\nPerformance degradation: {degradation:.2f}")
    
    if degradation < 0.3:
        print("✅ Good generalization")
    elif degradation < 0.5:
        print("⚠️ Some overfitting")
    else:
        print("❌ Significant overfitting, reconsider strategy")
    
    return best_params, test_metrics
```

---

## Best Practices

### 1. Keep Parameter Ranges Sensible

```python
# Good - based on financial intuition
param_grid = {
    'ml_threshold': [0.52, 0.55, 0.58],  # Reasonable confidence levels
    'stop_loss': [0.01, 0.02, 0.03],      # 1-3% reasonable stops
}

# Bad - too many random values
param_grid = {
    'ml_threshold': np.linspace(0.1, 0.9, 50),  # Many unreasonable values
}
```

### 2. Use Validation Set

```python
# Split data three ways
train = df[:6000]   # 60% - optimize on this
val = df[6000:8000] # 20% - tune parameters
test = df[8000:]    # 20% - final evaluation (touch once!)
```

### 3. Limit Search Space

```python
# Too many combinations
4 × 5 × 5 × 4 × 3 = 1200 combinations  # Bad

# Reasonable number
3 × 3 × 3 = 27 combinations  # Good for initial search
```

### 4. Document Everything

```python
# Keep a record
optimization_log = {
    'date': '2024-01-15',
    'data_period': 'Jan 2023 - Dec 2023',
    'best_params': {'ml_threshold': 0.55, ...},
    'train_sharpe': 1.4,
    'test_sharpe': 1.1,
    'notes': 'Volume threshold seems less important'
}
```

---

## Summary

| Method | Pros | Cons |
|--------|------|------|
| **Grid Search** | Exhaustive, simple | Slow for many parameters |
| **Random Search** | Faster, explores more | May miss optimal |
| **Walk-Forward** | Most realistic | Complex, time-consuming |

---

## Key Takeaways

1. **Optimize carefully** - Easy to overfit!
2. **Use validation data** - Don't test on training
3. **Check robustness** - Best point may be lucky
4. **Multiple objectives** - Balance return and risk
5. **Walk-forward** - Gold standard validation

---

## Next Steps

- [Custom Strategies](04_custom_strategies.md) - Build your own
- [Troubleshooting](05_troubleshooting.md) - Common issues
