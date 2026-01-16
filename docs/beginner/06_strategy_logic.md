# Strategy Logic

This guide explains how trading decisions are made in our simulator.

## What is a Trading Strategy?

A **strategy** is a set of rules that decides:
1. When to **BUY**
2. When to **SELL**
3. How **MUCH** to buy/sell

---

## Our Momentum Strategy

We use a **Momentum Strategy** that combines three signals:

### 1. ML Signal
What does our AI model predict?

```
ML Probability > 0.55 → BUY signal
ML Probability < 0.45 → SELL signal
ML Probability 0.45-0.55 → No signal (uncertain)
```

### 2. Momentum Signal
Is price moving strongly in one direction?

```
Price increased > 0.05% in last 10 bars → BUY signal
Price decreased > 0.05% in last 10 bars → SELL signal
Price roughly flat → No signal
```

### 3. Volume Confirmation
Are lots of people trading? (confirms the move is real)

```
Current volume > 1.2x average volume → Confirmed
Current volume < 1.2x average volume → Not confirmed
```

---

## How Signals Combine

The strategy looks for **agreement** between signals:

### Strong BUY Signal
```
ML says BUY (>0.55)  ✓
Momentum is up       ✓
Volume is high       ✓

Result: BUY with HIGH confidence
```

### Medium BUY Signal
```
ML says BUY (>0.55)  ✓
Momentum is flat     ✗
Volume is high       ✓

Result: BUY with MEDIUM confidence
```

### Weak BUY Signal
```
ML says BUY (>0.55)  ✓
Momentum is flat     ✗
Volume is low        ✗

Result: No trade (not confident enough)
```

### Decision Table

| ML Signal | Momentum | Volume | Decision | Confidence |
|-----------|----------|--------|----------|------------|
| BUY | Up | High | BUY | High |
| BUY | Up | Low | BUY | Medium |
| BUY | Flat | High | BUY | Medium |
| BUY | Flat | Low | HOLD | - |
| SELL | Down | High | SELL | High |
| SELL | Down | Low | SELL | Medium |
| HOLD | Any | Any | HOLD | - |

---

## Confidence Scoring

Each signal gets a **confidence score** from 0.0 to 1.0.

### How Confidence is Calculated

```python
# ML confidence
if ml_prediction > 0.55:
    ml_confidence = ml_prediction  # e.g., 0.65
    
# Momentum confidence
momentum_confidence = min(abs(momentum) / threshold, 1.0)
# e.g., momentum = 0.001, threshold = 0.0005
# confidence = 0.001 / 0.0005 = 2.0 → capped at 1.0

# Combined confidence
if ml_signal == momentum_signal:
    confidence = (ml_confidence + momentum_confidence) / 2
    if volume_confirmed:
        confidence *= 1.2  # Boost
    else:
        confidence *= 0.8  # Reduce
```

**Example:**
```
ML confidence: 0.65
Momentum confidence: 0.80
Volume confirmed: Yes

Combined = (0.65 + 0.80) / 2 = 0.725
With volume boost = 0.725 × 1.2 = 0.87

Final confidence: 0.87 (very confident)
```

---

## Position Sizing

How much to buy/sell based on confidence?

### Basic Formula

```
Position Value = Account Balance × Max Position % × Confidence
```

**Example:**
```
Account Balance: $100,000
Max Position %: 30%
Confidence: 0.80

Position Value = $100,000 × 0.30 × 0.80 = $24,000
```

### With Risk Limits

The risk manager may reduce position size:

```
Strategy wants: $24,000 position
Risk limit: $20,000 max position

Final position: $20,000 (limited by risk)
```

---

## The Trading Flow

```
┌──────────────────────────────────────────────────────────┐
│                    New Market Data                       │
└─────────────────────────┬────────────────────────────────┘
                          ▼
┌──────────────────────────────────────────────────────────┐
│               Calculate ML Prediction                    │
│               (What does AI think?)                      │
└─────────────────────────┬────────────────────────────────┘
                          ▼
┌──────────────────────────────────────────────────────────┐
│               Calculate Momentum                         │
│               (Is price trending?)                       │
└─────────────────────────┬────────────────────────────────┘
                          ▼
┌──────────────────────────────────────────────────────────┐
│               Check Volume                               │
│               (Is the move confirmed?)                   │
└─────────────────────────┬────────────────────────────────┘
                          ▼
┌──────────────────────────────────────────────────────────┐
│               Generate Signal                            │
│               (BUY, SELL, or HOLD)                       │
└─────────────────────────┬────────────────────────────────┘
                          ▼
             ┌────────────┴────────────┐
             ▼                         ▼
        ┌─────────┐              ┌──────────┐
        │  HOLD   │              │ BUY/SELL │
        └────┬────┘              └────┬─────┘
             │                        ▼
             │               ┌──────────────────┐
             │               │ Check Risk Limits │
             │               └────────┬─────────┘
             │                        ▼
             │               ┌──────────────────┐
             │               │ Calculate Size    │
             │               └────────┬─────────┘
             │                        ▼
             │               ┌──────────────────┐
             │               │ Execute Trade    │
             │               └────────┬─────────┘
             │                        │
             └────────────┬───────────┘
                          ▼
┌──────────────────────────────────────────────────────────┐
│               Update Portfolio                           │
│               (Track positions and PnL)                  │
└──────────────────────────────────────────────────────────┘
```

---

## Code Example

```python
from strategies.momentum_strategy import MomentumStrategy, Signal

# Create strategy
strategy = MomentumStrategy(
    ml_threshold=0.55,
    momentum_threshold=0.0005,
    volume_threshold=1.2
)

# Generate signal for one data point
signal = strategy.generate_signal(
    row=data.iloc[-1],  # Latest data
    ml_prediction=0.65   # ML model output
)

# Check the signal
print(f"Signal: {signal.signal.name}")
print(f"Confidence: {signal.confidence:.2f}")
print(f"Reason: {signal.reason}")

# Example output:
# Signal: BUY
# Confidence: 0.87
# Reason: ML + Momentum + Volume (BUY)
```

---

## Signal Statistics

After running a backtest, see how signals were distributed:

```python
stats = strategy.get_signal_statistics()

print(f"Total signals: {stats['total_signals']}")
print(f"BUY signals: {stats['buy_signals']} ({stats['buy_pct']:.1f}%)")
print(f"SELL signals: {stats['sell_signals']} ({stats['sell_pct']:.1f}%)")
print(f"HOLD signals: {stats['hold_signals']} ({stats['hold_pct']:.1f}%)")
print(f"Average confidence: {stats['avg_confidence']:.2f}")
```

**Typical output:**
```
Total signals: 1000
BUY signals: 105 (10.5%)
SELL signals: 102 (10.2%)
HOLD signals: 793 (79.3%)
Average confidence: 0.62
```

**Observation**: The strategy is conservative - it only trades when confident (about 20% of the time).

---

## Tuning the Strategy

### More Trades (Aggressive)

```python
strategy = MomentumStrategy(
    ml_threshold=0.52,       # Lower threshold
    momentum_threshold=0.0003, # More sensitive
    volume_threshold=1.0      # No volume filter
)
```

### Fewer Trades (Conservative)

```python
strategy = MomentumStrategy(
    ml_threshold=0.60,        # Higher threshold
    momentum_threshold=0.001,  # Less sensitive
    volume_threshold=1.5       # Stricter volume filter
)
```

---

## Key Takeaways

1. **Strategy** combines ML, momentum, and volume signals
2. **Confidence** determines position size
3. **Risk limits** cap maximum positions
4. **Conservative** approach - trade only when confident
5. **Tunable** parameters let you adjust aggressiveness

---

## Next Steps

- [Portfolio & Risk](07_portfolio_risk.md) - Managing positions and risk
- [Backtesting](08_backtesting.md) - Testing strategies on history
