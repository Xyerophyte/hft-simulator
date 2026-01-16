# Metrics Explained

This guide explains performance metrics and what they mean.

## Why Metrics Matter

**Metrics** = Numbers that tell you if your strategy is good

Without metrics:
- "I made money!" (But was it luck?)
- "I had some losses" (How bad?)

With metrics:
- "My Sharpe is 1.5" (Solid risk-adjusted return)
- "My max drawdown was 8%" (Reasonable risk)
- "My win rate is 55%" (Slightly better than chance)

---

## Return Metrics

### Total Return

**How much you made (or lost) overall.**

```
Total Return = (Final Value - Initial Value) / Initial Value × 100%

Example:
Start: $100,000
End: $112,000
Return = ($112,000 - $100,000) / $100,000 = 12%
```

### Annualized Return (CAGR)

**What would be your annual return if you traded for a year?**

```
CAGR = (Final/Initial)^(365/days) - 1

Example:
12% in 30 days
CAGR = (1.12)^(365/30) - 1 = 286%

(This shows why short-term results can be misleading!)
```

---

## Risk-Adjusted Metrics

These are the **most important** metrics because they account for risk.

### Sharpe Ratio

**Return per unit of risk.** Higher is better.

```
Sharpe = (Strategy Return - Risk Free Rate) / Volatility

Example:
Strategy return: 15% annually
Risk-free rate: 2%
Volatility: 10%

Sharpe = (15% - 2%) / 10% = 1.3
```

**Interpretation:**

| Sharpe | Rating |
|--------|--------|
| < 0 | Losing money |
| 0 - 0.5 | Poor |
| 0.5 - 1.0 | Acceptable |
| 1.0 - 2.0 | Good |
| 2.0 - 3.0 | Excellent |
| > 3.0 | Suspicious (check for errors) |

### Sortino Ratio

**Like Sharpe, but only penalizes downside volatility.**

Why? Some volatility is GOOD (when prices go up!).

```
Sortino = (Return - Risk Free Rate) / Downside Deviation

Downside Deviation = Standard deviation of NEGATIVE returns only
```

**Key insight**: Sortino > Sharpe usually means your gains are bigger than losses.

### Calmar Ratio

**Return divided by worst drawdown.**

```
Calmar = Annualized Return / Max Drawdown

Example:
Annualized return: 25%
Max drawdown: 10%
Calmar = 25% / 10% = 2.5
```

Higher is better. Calmar > 1 means returns exceed worst drawdown.

---

## Risk Metrics

### Volatility

**How much returns jump around.**

```
Daily Volatility = Standard Deviation of Daily Returns
Annual Volatility = Daily × sqrt(252)  # 252 trading days

Example:
Daily volatility: 2%
Annual volatility: 2% × 15.87 = 31.7%
```

**Interpretation:**
- 10% annual vol = Low volatility (stable)
- 20% annual vol = Moderate
- 40%+ annual vol = High volatility (risky)

### Maximum Drawdown

**Biggest peak-to-trough loss.**

```
Peak equity: $120,000
Lowest point after: $95,000
Max Drawdown = ($120,000 - $95,000) / $120,000 = 20.8%
```

**Why it matters:**
- Drawdown measures your worst experience
- Can you handle a 20% drop mentally?
- Bigger drawdowns are harder to recover from:
  - 10% loss needs 11% gain to recover
  - 20% loss needs 25% gain to recover
  - 50% loss needs 100% gain to recover!

### Value at Risk (VaR)

**Maximum expected loss with 95% confidence.**

```
95% VaR = -2.3% means:
"95% of days, you won't lose more than 2.3%"
"5% of days, you might lose MORE than 2.3%"
```

---

## Trading Metrics

### Win Rate

**Percentage of profitable trades.**

```
Win Rate = Winning Trades / Total Trades × 100%

Example:
60 winning trades
100 total trades
Win Rate = 60%
```

**Important**: High win rate doesn't mean profitable!

```
80% win rate, avg win = $10
20% loss rate, avg loss = $50

Expected: 0.8 × $10 - 0.2 × $50 = $8 - $10 = -$2 per trade (losing!)
```

### Profit Factor

**Gross profits divided by gross losses.**

```
Profit Factor = Sum of Wins / Sum of Losses

Example:
Total wins: $15,000
Total losses: $10,000
Profit Factor = 1.5

Interpretation: You make $1.50 for every $1 lost
```

| Profit Factor | Rating |
|---------------|--------|
| < 1.0 | Losing money |
| 1.0 - 1.25 | Marginal |
| 1.25 - 1.75 | Good |
| 1.75 - 2.5 | Very good |
| > 2.5 | Excellent (check for errors) |

### Average Win / Average Loss

```
Average Win = Sum of Winning Trades / Number of Winners
Average Loss = Sum of Losing Trades / Number of Losers

Example:
Avg Win: $150
Avg Loss: $100
Ratio: 1.5:1 (wins are 50% bigger than losses)
```

### Expected Value per Trade

```
Expected Value = (Win Rate × Avg Win) - (Loss Rate × Avg Loss)

Example:
Win Rate: 55%
Avg Win: $100
Loss Rate: 45%
Avg Loss: $80

Expected = 0.55 × $100 - 0.45 × $80 = $55 - $36 = $19 per trade
```

---

## Visualizing Metrics

### Equity Curve

![Equity Curve Description](Shows portfolio value over time)

- Upward slope = Making money
- Flat = Breaking even
- Downward slope = Losing money
- Dips = Drawdowns

### Drawdown Chart

![Drawdown Description](Shows how far below peak you are)

- Always 0% or negative
- Deeper valleys = Bigger losses
- Time in drawdown matters too

### PnL Distribution

![PnL Distribution](Histogram of trade outcomes)

- Should be roughly centered above 0
- Long right tail = Big occasional wins
- Long left tail = Big occasional losses (bad!)

---

## Reading a Performance Report

```
=== PERFORMANCE SUMMARY ===

Returns:
  Total Return:        12.5%
  Annualized Return:   45.3%
  
Risk-Adjusted:
  Sharpe Ratio:        1.42
  Sortino Ratio:       1.89
  Calmar Ratio:        2.08
  
Risk:
  Max Drawdown:        6.0%
  Annual Volatility:   31.8%
  95% VaR:             -2.1%
  
Trading:
  Total Trades:        145
  Win Rate:            54.5%
  Profit Factor:       1.67
  Average Win:         $127.34
  Average Loss:        $89.12
```

**Analysis:**
- ✅ Sharpe 1.42 = Good risk-adjusted return
- ✅ Sortino > Sharpe = More upside than downside
- ✅ Max DD 6.0% = Excellent risk control
- ✅ Win Rate 54.5% = Slightly better than chance
- ✅ Profit Factor 1.67 = Winners bigger than losers
- ✅ Avg Win > Avg Loss = Good trade quality

---

## Red Flags

### Things That Are TOO Good

| Metric | Suspicious If |
|--------|---------------|
| Sharpe | > 3.0 |
| Win Rate | > 80% |
| Max Drawdown | 0% (no losses ever) |
| Profit Factor | > 4.0 |

### These Usually Mean

1. **Bug in code** - Check your logic
2. **Look-ahead bias** - Using future data
3. **Overfitting** - Only works on test data
4. **Data error** - Wrong prices

---

## What Good Looks Like

### For HFT Strategies

| Metric | Realistic Range |
|--------|-----------------|
| Sharpe | 0.8 - 2.0 |
| Win Rate | 48% - 58% |
| Profit Factor | 1.2 - 2.0 |
| Max Drawdown | 5% - 20% |
| Avg Trade | 0.02% - 0.2% |

---

## Quick Calculation

```python
from analytics.metrics import PerformanceMetrics

# Calculate all metrics at once
metrics = PerformanceMetrics.calculate_all_metrics(
    equity_curve,
    trades,
    initial_capital=100000
)

# Print key metrics
print(f"Sharpe: {metrics['sharpe_ratio']:.2f}")
print(f"Sortino: {metrics['sortino_ratio']:.2f}")
print(f"Max DD: {metrics['max_drawdown_pct']:.1f}%")
print(f"Win Rate: {metrics['win_rate']:.1f}%")
print(f"Profit Factor: {metrics['profit_factor']:.2f}")
```

---

## Key Takeaways

1. **Sharpe Ratio** = Most important single metric (aim for > 1.0)
2. **Max Drawdown** = Your worst day, be ready for it
3. **Win Rate alone** doesn't tell the whole story
4. **Profit Factor** > 1.0 to make money
5. **Be suspicious** of metrics that look too good
6. **Risk-adjusted returns** matter more than raw returns

---

## Next Steps

- [Complete Workflow](10_complete_workflow.md) - Full end-to-end example
- [API Reference](../api/05_backtesting.md) - Detailed metrics API
