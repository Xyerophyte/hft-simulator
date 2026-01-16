# Portfolio & Risk Management

This guide explains how positions and risk are managed.

## What is a Portfolio?

A **portfolio** tracks everything you own and owe:
- **Cash**: Money available to trade
- **Positions**: Assets you currently hold
- **Value**: Total worth of everything
- **PnL**: Profit or loss

---

## The Portfolio Class

```python
from strategies.portfolio import Portfolio

# Start with $100,000
portfolio = Portfolio(initial_capital=100000.0, fee_rate=0.001)
```

### Properties

| Property | Description |
|----------|-------------|
| `cash` | Available cash |
| `positions` | Current holdings |
| `equity` | Total portfolio value |
| `realized_pnl` | PnL from closed trades |
| `unrealized_pnl` | Paper PnL from open positions |

---

## Executing Trades

### Buying

```python
# Buy 0.5 BTC at $50,000
trade = portfolio.execute_trade('BTC', 0.5, 50000.0, timestamp)

# What happens:
# - Cash: $100,000 → $74,975 (paid $25,000 + $25 fee)
# - Position: 0 → 0.5 BTC
# - Total value: $100,000 → $99,975 (fee loss)
```

### Selling

```python
# Later, price is $55,000
# Sell all 0.5 BTC
trade = portfolio.execute_trade('BTC', -0.5, 55000.0, timestamp)

# What happens:
# - Received: $27,500 (0.5 × $55,000)
# - Fee: $27.50
# - Cash: $74,975 → $102,447.50
# - Position: 0.5 → 0 BTC
# - Realized PnL: +$2,500 - fees
```

---

## Understanding PnL

### Realized vs Unrealized

**Realized PnL**: Profit/loss from completed trades
```
Bought 1 BTC at $50,000
Sold 1 BTC at $55,000
Realized PnL: +$5,000 (minus fees)
```

**Unrealized PnL**: Paper profit/loss from open positions
```
Bought 1 BTC at $50,000
Current price: $52,000
Position is still open
Unrealized PnL: +$2,000 (not real until you sell)
```

### Total PnL

```
Total PnL = Realized PnL + Unrealized PnL - Total Fees
```

---

## Equity Curve

The **equity curve** shows your portfolio value over time.

```
Time    | Cash      | Position Value | Total Equity
--------|-----------|----------------|-------------
10:00   | $100,000  | $0             | $100,000
10:15   | $75,000   | $25,000        | $100,000
10:30   | $75,000   | $26,000        | $101,000  ← Price went up
10:45   | $75,000   | $24,500        | $99,500   ← Price went down
11:00   | $101,500  | $0             | $101,500  ← Sold for profit
```

```python
# Get equity history
equity_df = portfolio.get_equity_curve()

# Plot it
equity_df['equity'].plot(title='Equity Curve')
```

---

## What is Risk Management?

**Risk Management** protects you from large losses.

### Why It Matters

Without risk management:
```
Trade 1: -$5,000
Trade 2: -$8,000
Trade 3: -$15,000
Trade 4: -$25,000

After 4 bad trades: Lost 53% of account!
```

With risk management:
```
Trade 1: -$2,000 (hit stop loss)
Trade 2: -$2,000 (hit stop loss)
Trade 3: -$2,000 (hit stop loss)
Trade 4: Blocked! (daily loss limit reached)

After 4 bad trades: Lost 6% of account
```

---

## Risk Limits

### Position Limits

**Maximum position size** prevents over-concentration.

```python
from strategies.risk_manager import RiskManager, RiskLimits

limits = RiskLimits(
    max_position_pct=0.3  # Max 30% of equity in one position
)

# With $100,000 equity
# Max position = $30,000
```

### Stop Loss

**Stop loss** automatically sells when loss exceeds threshold.

```python
limits = RiskLimits(
    stop_loss_pct=0.02  # 2% stop loss
)

# Bought at $50,000
# Stop triggers at: $49,000 (2% loss)
# Maximum loss per trade: 2%
```

### Maximum Drawdown

**Drawdown** is how far you are below your peak.

```python
limits = RiskLimits(
    max_drawdown_pct=0.15  # Max 15% drawdown allowed
)

# Peak equity: $110,000
# Current equity: $93,500
# Drawdown: ($110,000 - $93,500) / $110,000 = 15%
# → STOP TRADING (limit reached)
```

### Daily Loss Limit

**Daily limit** caps losses per day.

```python
limits = RiskLimits(
    max_daily_loss_pct=0.05  # Max 5% daily loss
)

# Start of day: $100,000
# Current: $95,000
# Daily loss: 5%
# → No more trading today
```

---

## Risk Checks in Practice

```python
from strategies.risk_manager import RiskManager, RiskLimits

# Create risk manager
limits = RiskLimits(
    max_position_pct=0.3,
    stop_loss_pct=0.02,
    max_drawdown_pct=0.15
)
risk_mgr = RiskManager(limits)

# Before each trade, check:
current_equity = 100000
position_size = 25000

# 1. Check position limit
if not risk_mgr.check_position_limit(position_size, current_equity):
    print("BLOCKED: Position too large")
    
# 2. Check drawdown
if not risk_mgr.check_drawdown(current_equity, initial_equity=110000):
    print("BLOCKED: Drawdown too high")
    
# 3. Check stop loss on existing positions
if risk_mgr.check_stop_loss(entry_price=50000, current_price=48500, is_long=True):
    print("STOP LOSS: Close position now")
```

---

## Position Sizing with Risk

The risk manager adjusts position sizes:

```python
# Strategy wants to buy $30,000 worth
desired_size = 30000

# Risk manager adjusts
adjusted_size = risk_mgr.calculate_position_size(
    signal_size=desired_size,
    current_equity=100000,
    current_price=50000,
    volatility=0.03  # 3% daily volatility
)

# If volatility is high, size is reduced
# adjusted_size might be $20,000
```

---

## Tracking Positions

```python
# After several trades
print(portfolio.get_summary())

# Output:
{
    'initial_capital': 100000.0,
    'current_value': 102500.0,
    'cash': 52500.0,
    'positions_value': 50000.0,
    'realized_pnl': 1500.0,
    'unrealized_pnl': 1000.0,
    'total_pnl': 2500.0,
    'total_return_pct': 2.5,
    'total_fees': 225.0,
    'num_trades': 15,
    'num_positions': 1
}
```

---

## Common Risk Scenarios

### Scenario 1: Position Too Large

```
Want to buy: $50,000 of BTC
Equity: $100,000
Max position: 30%

$50,000 > $30,000 limit
→ Trade blocked or reduced to $30,000
```

### Scenario 2: Stop Loss Triggered

```
Bought BTC at: $50,000
Stop loss: 2%
Current price: $48,900 (2.2% loss)

Stop triggered!
→ Automatic sell at $48,900
→ Limited loss to ~2%
```

### Scenario 3: Drawdown Limit

```
Peak equity: $120,000
Current equity: $100,000
Drawdown: 16.7%
Max drawdown: 15%

Limit exceeded!
→ No new trades allowed
→ Consider reducing positions
```

---

## Best Practices

### 1. Start Small
```python
limits = RiskLimits(
    max_position_pct=0.1,  # Only 10% per position
    stop_loss_pct=0.01     # Tight 1% stops
)
```

### 2. Track Everything
```python
# After each trade
print(f"Position: {portfolio.get_position('BTC')}")
print(f"Unrealized PnL: ${portfolio.unrealized_pnl:,.2f}")
print(f"Total PnL: ${portfolio.get_total_pnl():,.2f}")
```

### 3. Review Daily
```python
# End of day review
summary = portfolio.get_summary()
print(f"Today's PnL: ${summary['total_pnl']:,.2f}")
print(f"Current drawdown: {risk_mgr.current_drawdown*100:.1f}%")
```

---

## Key Takeaways

1. **Portfolio** tracks all your assets and trades
2. **Realized PnL** = from closed trades
3. **Unrealized PnL** = from open positions (paper)
4. **Risk limits** prevent catastrophic losses
5. **Stop loss** limits loss per trade
6. **Max drawdown** stops trading after big losses
7. **Position sizing** adjusts for risk

---

## Next Steps

- [Backtesting](08_backtesting.md) - Testing with risk management
- [Metrics Explained](09_metrics.md) - Measuring performance
