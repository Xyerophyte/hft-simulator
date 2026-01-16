# How It Works - System Overview

This guide provides a high-level overview of how the HFT Simulator works.

## The Big Picture

The simulator takes historical market data and simulates what would happen if you traded using a specific strategy.

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Market     │ ──▶ │   Strategy   │ ──▶ │   Results    │
│   Data       │     │   Logic      │     │              │
└──────────────┘     └──────────────┘     └──────────────┘

Input:               Processing:           Output:
- Price history      - Generate signals    - Performance
- Volume             - Execute trades      - Trade history
- Indicators         - Manage risk         - Charts
```

---

## The Main Components

### 1. Data Pipeline 📊

Gets and prepares market data.

```
Binance → Raw OHLCV → Preprocessing → Technical Indicators
```

**What it does:**
- Fetches Bitcoin prices from Binance (free API)
- Cleans missing/bad data
- Adds 29 technical indicators (SMA, RSI, etc.)

### 2. Feature Engineering 🔧

Creates patterns for the AI to learn.

```
Technical Indicators → Feature Engineering → 90+ Features
```

**What it does:**
- Takes indicator data
- Creates advanced features (momentum, volatility, etc.)
- Organizes data for the AI model

### 3. Machine Learning 🧠

AI that predicts price direction.

```
Historical Features → LSTM Neural Network → "Price going UP/DOWN"
```

**What it does:**
- Looks at 30 minutes of history
- Learns patterns from thousands of examples
- Predicts if price will go up or down
- ~54% accuracy (better than guessing!)

### 4. Strategy 🎯

Makes trading decisions.

```
ML Prediction + Momentum + Volume → BUY / SELL / HOLD
```

**What it does:**
- Combines multiple signals
- Decides when conditions are favorable
- Calculates confidence level

### 5. Risk Management ⚠️

Protects against big losses.

```
Trade Request → Risk Check → Approved / Blocked
```

**What it does:**
- Limits position sizes
- Enforces stop losses
- Prevents excessive drawdowns

### 6. Backtester 📈

Simulates trading on historical data.

```
Historical Data → Bar-by-Bar Simulation → Performance Report
```

**What it does:**
- Replays history one bar at a time
- Generates signals and executes trades
- Tracks portfolio value

### 7. Analytics 📊

Measures and visualizes performance.

```
Trade Results → Metrics Calculation → Charts & Reports
```

**What it does:**
- Calculates Sharpe ratio, win rate, etc.
- Creates equity curves and dashboards
- Exports results to files

---

## The Workflow

### Step-by-Step Process

```
1️⃣ FETCH DATA
   └── Download candles from Binance

2️⃣ PREPROCESS
   └── Clean data and add indicators

3️⃣ ENGINEER FEATURES
   └── Create 90+ ML features

4️⃣ (OPTIONAL) TRAIN MODEL
   └── Teach AI to predict prices

5️⃣ CONFIGURE STRATEGY
   └── Set trading rules and risk limits

6️⃣ RUN BACKTEST
   └── Simulate trading bar-by-bar

7️⃣ ANALYZE RESULTS
   └── Calculate performance metrics

8️⃣ VISUALIZE
   └── Create charts and reports
```

---

## How a Single Trade Happens

Let's follow one trade through the system:

```
Time: 10:15 AM

1. New bar arrives
   └── Price: $50,000, Volume: 100 BTC, RSI: 35

2. ML Model predicts
   └── Output: 0.72 (72% confident price goes UP)

3. Strategy checks signals
   ├── ML says: BUY (>0.55 threshold)
   ├── Momentum: +0.12% (positive, confirms)
   └── Volume: 1.5x average (high, confirms)

4. Signal generated
   └── BUY with 0.85 confidence

5. Risk check
   ├── Current drawdown: 3% (under 15% limit ✓)
   ├── Position would be 25% of equity (under 30% ✓)
   └── APPROVED

6. Trade executed
   ├── Buy 0.5 BTC at $50,000
   ├── Fee: $25 (0.1%)
   └── Cash: $100,000 → $74,975

7. Position recorded
   └── Position: 0.5 BTC, Entry: $50,000

8. Portfolio updated
   └── Equity: $99,975 (small loss from fee)
```

---

## Key Design Decisions

### Why Event-Driven?

We process one bar at a time, just like real trading.

**Benefits:**
- Realistic simulation
- No "cheating" by seeing future data
- Same logic could work for live trading

### Why Machine Learning?

ML can find subtle patterns humans miss.

**Our approach:**
- LSTM neural network (good for sequences)
- Binary classification (up or down)
- Combines with traditional indicators

### Why Risk Management?

Even good strategies have losing streaks.

**Protection:**
- Position limits prevent over-concentration
- Stop losses limit per-trade damage
- Drawdown limits protect against extended losses

---

## What Makes This Different

### Compared to Simple Backtests

| Feature | Simple | This Simulator |
|---------|--------|----------------|
| Fees | Often ignored | ✓ Included |
| Slippage | Ignored | ✓ Modeled |
| Position sizing | Fixed | ✓ Risk-based |
| ML integration | No | ✓ Yes |
| Risk management | Basic | ✓ Comprehensive |

### Compared to Professional Systems

| Feature | Professional | This Simulator |
|---------|--------------|----------------|
| Real-time trading | ✓ | ✗ (historical only) |
| Multiple assets | ✓ | ✗ (single asset) |
| Order book simulation | Simple | ✓ Basic |
| Research-ready | ✓ | ✓ Yes |

---

## The Technology Stack

```
Python 3.8+
├── pandas      - Data manipulation
├── numpy       - Numerical operations  
├── torch       - ML model (LSTM)
├── scikit-learn- Preprocessing
├── matplotlib  - Visualizations
└── requests    - API calls
```

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      run_demo.py                            │
│                  (Orchestrates everything)                  │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   src/data/     │ │   src/ml/       │ │  src/analytics/ │
│  - fetcher.py   │ │  - features.py  │ │  - metrics.py   │
│  - cache.py     │ │  - models.py    │ │  - viz.py       │
│  - preproc.py   │ └────────┬────────┘ └────────┬────────┘
└────────┬────────┘          │                   │
         │           ┌───────┴───────┐           │
         │           ▼               │           │
         │   ┌─────────────────┐     │           │
         └──▶│  src/backtest/  │◀────┘           │
             │  - backtester   │◀────────────────┘
             └────────┬────────┘
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│src/market/  │ │src/strategy/│ │  Output     │
│ - orderbook │ │ - momentum  │ │ - results/  │
│ - matching  │ │ - portfolio │ │ - models/   │
└─────────────┘ │ - risk_mgr  │ └─────────────┘
                └─────────────┘
```

---

## Quick Code Example

See the entire system in action:

```python
# The complete workflow in ~20 lines

from data.fetcher import BinanceDataFetcher
from data.preprocessor import DataPreprocessor
from ml.features import FeatureEngineer
from strategies.momentum_strategy import MomentumStrategy
from backtest.backtester import Backtester, BacktestConfig

# Get data
df = BinanceDataFetcher().fetch_klines(limit=2000)

# Process
df = DataPreprocessor().preprocess_pipeline(df)
df = FeatureEngineer().create_all_features(df)

# Backtest
strategy = MomentumStrategy()
backtester = Backtester(strategy, BacktestConfig())
results = backtester.run(df)

# Show results
print(f"Return: {results['summary']['total_return_pct']:.2f}%")
```

---

## Next Steps

- [Data Flow](03_data_flow.md) - Detailed data journey
- [Order Books](04_order_books.md) - Market microstructure
- [Getting Started](../guides/01_getting_started.md) - Run your first backtest