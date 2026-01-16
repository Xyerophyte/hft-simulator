# Trading Strategies API Reference

This document covers the trading strategy components.

---

## MomentumStrategy

**Module:** `src/strategies/momentum_strategy.py`

Generates trading signals using ML predictions and momentum.

### Constructor

```python
from strategies.momentum_strategy import MomentumStrategy

strategy = MomentumStrategy(
    ml_threshold: float = 0.55,
    momentum_threshold: float = 0.0005,
    volume_threshold: float = 1.2,
    confidence_scaling: bool = True
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ml_threshold` | float | 0.55 | ML probability threshold for BUY |
| `momentum_threshold` | float | 0.0005 | Minimum momentum for signal (0.05%) |
| `volume_threshold` | float | 1.2 | Volume ratio for confirmation |
| `confidence_scaling` | bool | True | Scale position by confidence |

### Methods

#### `generate_signal()`

Generate signal from a single data row.

```python
signal = strategy.generate_signal(
    row: pd.Series,
    ml_prediction: Optional[float] = None
) -> TradeSignal
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `row` | Series | DataFrame row with features |
| `ml_prediction` | float | ML model probability (0-1) |

**Returns:** TradeSignal object

#### `generate_signals_batch()`

Generate signals for entire DataFrame.

```python
signals = strategy.generate_signals_batch(
    df: pd.DataFrame,
    ml_predictions: Optional[np.ndarray] = None
) -> List[TradeSignal]
```

**Returns:** List of TradeSignal objects

#### `get_position_size()`

Calculate position size based on signal.

```python
size = strategy.get_position_size(
    signal: TradeSignal,
    account_balance: float,
    current_position: float = 0.0,
    max_position_pct: float = 0.5
) -> float
```

**Returns:** Position size (positive=buy, negative=sell)

#### `get_signal_statistics()`

Get statistics about generated signals.

```python
stats = strategy.get_signal_statistics() -> Dict
```

**Returns:**
```python
{
    'total_signals': int,
    'buy_signals': int,
    'sell_signals': int,
    'hold_signals': int,
    'avg_confidence': float,
    'buy_pct': float,
    'sell_pct': float,
    'hold_pct': float
}
```

---

## Signal & TradeSignal

**Module:** `src/strategies/momentum_strategy.py`

### Signal Enum

```python
class Signal(Enum):
    BUY = 1
    SELL = -1
    HOLD = 0
```

### TradeSignal Dataclass

```python
@dataclass
class TradeSignal:
    timestamp: pd.Timestamp   # When signal was generated
    signal: Signal            # BUY, SELL, or HOLD
    confidence: float         # 0.0 to 1.0
    price: float             # Current price
    reason: str              # Human-readable reason
    ml_probability: Optional[float]  # ML prediction if available
```

---

## Portfolio

**Module:** `src/strategies/portfolio.py`

Tracks positions, trades, and calculates PnL.

### Constructor

```python
from strategies.portfolio import Portfolio

portfolio = Portfolio(
    initial_capital: float = 100000.0,
    fee_rate: float = 0.001
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `initial_capital` | float | 100000.0 | Starting capital in USD |
| `fee_rate` | float | 0.001 | Trading fee per trade (0.1%) |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `cash` | float | Current cash balance |
| `positions` | Dict[str, Position] | Current positions by symbol |
| `trades` | List[Trade] | Trade history |
| `realized_pnl` | float | Total realized profit/loss |
| `total_fees` | float | Total fees paid |

### Methods

#### `execute_trade()`

Execute a trade and update positions.

```python
trade = portfolio.execute_trade(
    symbol: str,
    quantity: float,     # Positive=buy, negative=sell
    price: float,
    timestamp: pd.Timestamp
) -> Trade
```

**Returns:** Trade object with execution details

**Example:**
```python
# Buy 0.5 BTC at $50,000
trade = portfolio.execute_trade('BTC', 0.5, 50000.0, pd.Timestamp.now())
print(f"Bought {trade.quantity} {trade.symbol} @ ${trade.price:,.2f}")
print(f"Fee: ${trade.fee:.2f}")
print(f"Remaining cash: ${portfolio.cash:,.2f}")
```

#### `update_prices()`

Update current prices for positions.

```python
portfolio.update_prices(
    prices: Dict[str, float],
    timestamp: pd.Timestamp
)
```

This also records a portfolio state snapshot.

#### `get_position()`

Get position for a symbol.

```python
position = portfolio.get_position(symbol: str) -> Optional[Position]
```

#### `get_total_pnl()`

Get total PnL (realized + unrealized).

```python
total_pnl = portfolio.get_total_pnl() -> float
```

#### `get_total_return()`

Get total return as percentage.

```python
return_pct = portfolio.get_total_return() -> float
```

#### `get_equity_curve()`

Get equity history as DataFrame.

```python
equity_df = portfolio.get_equity_curve() -> pd.DataFrame
```

**Returns:** DataFrame with columns: `equity`, `cash`, `positions_value`, `unrealized_pnl`, `realized_pnl`

#### `get_trade_history()`

Get trade history as DataFrame.

```python
trades_df = portfolio.get_trade_history() -> pd.DataFrame
```

#### `get_summary()`

Get portfolio summary statistics.

```python
summary = portfolio.get_summary() -> Dict
```

**Returns:**
```python
{
    'initial_capital': float,
    'current_value': float,
    'cash': float,
    'positions_value': float,
    'realized_pnl': float,
    'unrealized_pnl': float,
    'total_pnl': float,
    'total_return_pct': float,
    'total_fees': float,
    'num_trades': int,
    'num_positions': int
}
```

---

## Position & Trade

**Module:** `src/strategies/portfolio.py`

### Position Dataclass

```python
@dataclass
class Position:
    symbol: str
    quantity: float        # Positive=long, negative=short
    entry_price: float
    entry_time: pd.Timestamp
    current_price: float = 0.0
```

**Properties:**
- `market_value`: quantity × current_price
- `cost_basis`: |quantity| × entry_price
- `unrealized_pnl`: Current profit/loss
- `unrealized_pnl_pct`: Percentage profit/loss

### Trade Dataclass

```python
@dataclass
class Trade:
    trade_id: int
    symbol: str
    side: str             # 'BUY' or 'SELL'
    quantity: float
    price: float
    timestamp: pd.Timestamp
    fee: float = 0.0
    pnl: float = 0.0
```

**Properties:**
- `notional`: quantity × price
- `total_cost`: notional + fee

---

## RiskManager

**Module:** `src/strategies/risk_manager.py`

Enforces risk limits and controls.

### Constructor

```python
from strategies.risk_manager import RiskManager, RiskLimits

limits = RiskLimits(
    max_position_pct=0.5,
    max_total_exposure=1.0,
    max_drawdown_pct=0.15,
    max_daily_loss_pct=0.05,
    stop_loss_pct=0.02,
    volatility_limit=0.05
)

risk_mgr = RiskManager(limits)
```

### Methods

#### `check_position_limit()`

Check if position size is within limits.

```python
within_limit = risk_mgr.check_position_limit(
    position_value: float,
    total_equity: float
) -> bool
```

#### `check_drawdown()`

Check if drawdown exceeds limit.

```python
within_limit = risk_mgr.check_drawdown(
    current_equity: float,
    initial_equity: float
) -> bool
```

#### `check_daily_loss()`

Check if daily loss exceeds limit.

```python
within_limit = risk_mgr.check_daily_loss(
    current_equity: float,
    daily_start_equity: Optional[float] = None
) -> bool
```

#### `check_stop_loss()`

Check if stop loss is triggered.

```python
triggered = risk_mgr.check_stop_loss(
    entry_price: float,
    current_price: float,
    is_long: bool
) -> bool
```

**Returns:** True if stop loss is triggered

#### `calculate_position_size()`

Calculate risk-adjusted position size.

```python
size = risk_mgr.calculate_position_size(
    signal_size: float,
    current_equity: float,
    current_price: float,
    volatility: Optional[float] = None
) -> float
```

#### `calculate_var()`

Calculate Value at Risk.

```python
var = risk_mgr.calculate_var(
    returns: pd.Series,
    confidence: float = 0.95,
    window: int = 252
) -> float
```

#### `get_risk_metrics()`

Get current risk metrics.

```python
metrics = risk_mgr.get_risk_metrics(
    current_equity: float,
    initial_equity: float,
    positions_value: float,
    returns: Optional[pd.Series] = None
) -> RiskMetrics
```

---

## RiskMetrics & RiskLimits

### RiskLimits Dataclass

```python
@dataclass
class RiskLimits:
    max_position_pct: float = 0.5
    max_total_exposure: float = 1.0
    max_drawdown_pct: float = 0.15
    max_daily_loss_pct: float = 0.05
    stop_loss_pct: float = 0.02
    volatility_limit: float = 0.05
    min_sharpe_ratio: float = 0.5
```

### RiskMetrics Dataclass

```python
@dataclass
class RiskMetrics:
    total_exposure_pct: float
    max_position_pct: float
    current_drawdown_pct: float
    daily_pnl_pct: float
    portfolio_volatility: float
    var_95: float
    num_violations: int
    violations: List[RiskViolation]
```

---

## Usage Examples

### Complete Trading Flow

```python
from strategies.momentum_strategy import MomentumStrategy
from strategies.portfolio import Portfolio
from strategies.risk_manager import RiskManager, RiskLimits

# Initialize
strategy = MomentumStrategy(ml_threshold=0.55)
portfolio = Portfolio(initial_capital=100000)
risk_mgr = RiskManager(RiskLimits(max_position_pct=0.3))

# Generate signal
signal = strategy.generate_signal(data_row, ml_prediction=0.65)

if signal.signal != Signal.HOLD:
    # Check risk
    current_equity = portfolio.cash + sum(p.market_value for p in portfolio.positions.values())
    
    if risk_mgr.check_position_limit(signal.price * 0.1, current_equity):
        # Execute trade
        trade = portfolio.execute_trade('BTC', 0.1, signal.price, pd.Timestamp.now())
        print(f"Executed: {trade.side} {trade.quantity} @ ${trade.price:,.2f}")
```

---

## Next Steps

- [Backtesting API](05_backtesting.md)
- [Getting Started Guide](../guides/01_getting_started.md)
