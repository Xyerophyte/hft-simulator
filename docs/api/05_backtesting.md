# Backtesting API Reference

This document covers the backtesting and analytics components.

---

## Backtester

**Module:** `src/backtest/backtester.py`

Event-driven backtesting engine.

### Constructor

```python
from backtest.backtester import Backtester, BacktestConfig

config = BacktestConfig(
    initial_capital=100000.0,
    fee_rate=0.001,
    position_size_pct=0.3,
    use_risk_manager=True,
    use_ml_model=False
)

backtester = Backtester(strategy, config)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `strategy` | MomentumStrategy | Trading strategy instance |
| `config` | BacktestConfig | Configuration settings |

### Methods

#### `run()`

Run backtest on historical data.

```python
results = backtester.run(
    df: pd.DataFrame,
    symbol: str = 'BTC'
) -> Dict
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `df` | DataFrame | Historical data with features |
| `symbol` | str | Trading symbol |

**Returns:**
```python
{
    'summary': Dict,           # Portfolio summary
    'equity_curve': DataFrame, # Equity over time
    'trades': DataFrame,       # Trade history
    'returns': Series,         # Return series
    'num_signals': int,        # Total signals generated
    'trades_executed': int,    # Trades actually executed
    'signal_stats': Dict       # Signal distribution stats
}
```

**Example:**
```python
backtester = Backtester(strategy, config)
results = backtester.run(df_features, symbol='BTC')

print(f"Total Return: {results['summary']['total_return_pct']:.2f}%")
print(f"Trades Executed: {results['trades_executed']}")
```

#### `load_ml_model()`

Load a trained ML model for predictions.

```python
backtester.load_ml_model(
    model_path: str,
    input_size: int,
    device: str = 'cpu'
)
```

---

## BacktestConfig

**Module:** `src/backtest/backtester.py`

```python
@dataclass
class BacktestConfig:
    initial_capital: float = 100000.0
    fee_rate: float = 0.001
    position_size_pct: float = 0.3
    use_risk_manager: bool = True
    use_ml_model: bool = False
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `initial_capital` | float | 100000.0 | Starting capital |
| `fee_rate` | float | 0.001 | Trading fee (0.1%) |
| `position_size_pct` | float | 0.3 | Max position as % of capital |
| `use_risk_manager` | bool | True | Enable risk checks |
| `use_ml_model` | bool | False | Use ML predictions |

---

## PerformanceMetrics

**Module:** `src/analytics/metrics.py`

Calculates performance metrics.

### Methods

All methods are static - no need to instantiate.

#### `calculate_returns()`

Calculate returns from equity curve.

```python
returns = PerformanceMetrics.calculate_returns(
    equity_curve: pd.Series
) -> pd.Series
```

#### `calculate_sharpe_ratio()`

Calculate annualized Sharpe ratio.

```python
sharpe = PerformanceMetrics.calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 525600  # Minutes
) -> float
```

**Formula:** `sqrt(periods) * (mean(excess_returns) / std(excess_returns))`

#### `calculate_sortino_ratio()`

Calculate Sortino ratio (uses downside deviation).

```python
sortino = PerformanceMetrics.calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 525600
) -> float
```

#### `calculate_max_drawdown()`

Calculate maximum drawdown and duration.

```python
dd_metrics = PerformanceMetrics.calculate_max_drawdown(
    equity_curve: pd.Series
) -> Dict
```

**Returns:**
```python
{
    'max_drawdown': float,         # Maximum dollar drawdown
    'max_drawdown_pct': float,     # Maximum percentage drawdown
    'drawdown_duration': int,       # Duration in periods
    'current_drawdown_pct': float  # Current drawdown
}
```

#### `calculate_win_rate()`

Calculate win rate and trade statistics.

```python
trade_stats = PerformanceMetrics.calculate_win_rate(
    trades: pd.DataFrame
) -> Dict
```

**Returns:**
```python
{
    'total_trades': int,
    'winning_trades': int,
    'losing_trades': int,
    'win_rate': float,        # Percentage
    'avg_win': float,
    'avg_loss': float,
    'largest_win': float,
    'largest_loss': float,
    'profit_factor': float    # Gross profit / Gross loss
}
```

#### `calculate_calmar_ratio()`

Calculate Calmar ratio (return / drawdown).

```python
calmar = PerformanceMetrics.calculate_calmar_ratio(
    returns: pd.Series,
    equity_curve: pd.Series,
    periods_per_year: int = 525600
) -> float
```

#### `calculate_all_metrics()`

Calculate all metrics at once.

```python
all_metrics = PerformanceMetrics.calculate_all_metrics(
    equity_curve: pd.DataFrame,
    trades: pd.DataFrame,
    initial_capital: float,
    risk_free_rate: float = 0.02
) -> Dict
```

**Returns:**
```python
{
    'total_return_pct': float,
    'cagr_pct': float,
    'sharpe_ratio': float,
    'sortino_ratio': float,
    'calmar_ratio': float,
    'annual_volatility_pct': float,
    'max_drawdown': float,
    'max_drawdown_pct': float,
    'drawdown_duration': int,
    'current_drawdown_pct': float,
    'total_trades': int,
    'winning_trades': int,
    'losing_trades': int,
    'win_rate': float,
    'avg_win': float,
    'avg_loss': float,
    'largest_win': float,
    'largest_loss': float,
    'profit_factor': float
}
```

---

## TradingVisualizer

**Module:** `src/analytics/visualizations.py`

Creates performance visualizations.

### Constructor

```python
from analytics.visualizations import TradingVisualizer

viz = TradingVisualizer(style: str = 'seaborn-v0_8-darkgrid')
```

### Methods

#### `plot_equity_curve()`

Plot equity over time.

```python
viz.plot_equity_curve(
    equity_curve: pd.DataFrame,
    title: str = "Equity Curve",
    save_path: Optional[str] = None
)
```

#### `plot_drawdown()`

Plot drawdown chart.

```python
viz.plot_drawdown(
    equity_curve: pd.DataFrame,
    title: str = "Drawdown",
    save_path: Optional[str] = None
)
```

#### `plot_pnl_distribution()`

Plot trade PnL distribution.

```python
viz.plot_pnl_distribution(
    trades: pd.DataFrame,
    title: str = "PnL Distribution",
    save_path: Optional[str] = None
)
```

#### `plot_returns_distribution()`

Plot returns distribution.

```python
viz.plot_returns_distribution(
    returns: pd.Series,
    title: str = "Returns Distribution",
    save_path: Optional[str] = None
)
```

#### `plot_summary_dashboard()`

Create multi-panel dashboard.

```python
viz.plot_summary_dashboard(
    equity_curve: pd.DataFrame,
    trades: pd.DataFrame,
    metrics: Dict,
    save_path: Optional[str] = None
)
```

---

## Usage Examples

### Complete Backtest Analysis

```python
from backtest.backtester import Backtester, BacktestConfig
from strategies.momentum_strategy import MomentumStrategy
from analytics.metrics import PerformanceMetrics
from analytics.visualizations import TradingVisualizer

# Setup
strategy = MomentumStrategy()
config = BacktestConfig(initial_capital=100000)
backtester = Backtester(strategy, config)

# Run backtest
results = backtester.run(df_features, symbol='BTC')

# Calculate metrics
metrics = PerformanceMetrics.calculate_all_metrics(
    results['equity_curve'],
    results['trades'],
    config.initial_capital
)

# Display results
print("=== Performance Summary ===")
print(f"Total Return: {metrics['total_return_pct']:.2f}%")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
print(f"Win Rate: {metrics['win_rate']:.1f}%")
print(f"Profit Factor: {metrics['profit_factor']:.2f}")

# Create visualizations
viz = TradingVisualizer()
viz.plot_summary_dashboard(
    results['equity_curve'],
    results['trades'],
    metrics,
    save_path='results/dashboard.png'
)
```

### Parameter Sweep

```python
import itertools

param_grid = {
    'ml_threshold': [0.50, 0.55, 0.60],
    'momentum_threshold': [0.0003, 0.0005, 0.0007],
    'position_size_pct': [0.2, 0.3, 0.4]
}

results_list = []

for params in itertools.product(*param_grid.values()):
    ml_t, mom_t, pos_size = params
    
    strategy = MomentumStrategy(
        ml_threshold=ml_t,
        momentum_threshold=mom_t
    )
    config = BacktestConfig(position_size_pct=pos_size)
    
    backtester = Backtester(strategy, config)
    results = backtester.run(df_features)
    
    metrics = PerformanceMetrics.calculate_all_metrics(
        results['equity_curve'],
        results['trades'],
        config.initial_capital
    )
    
    results_list.append({
        'ml_threshold': ml_t,
        'momentum_threshold': mom_t,
        'position_size': pos_size,
        'sharpe': metrics['sharpe_ratio'],
        'return': metrics['total_return_pct'],
        'max_dd': metrics['max_drawdown_pct']
    })

# Find best parameters
results_df = pd.DataFrame(results_list)
best = results_df.loc[results_df['sharpe'].idxmax()]
print(f"Best parameters: {best.to_dict()}")
```

---

## Next Steps

- [Getting Started Guide](../guides/01_getting_started.md)
- [Walk-Forward Analysis](../guides/02_walk_forward.md)
