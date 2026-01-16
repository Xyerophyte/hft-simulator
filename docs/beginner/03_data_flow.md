# Understanding Data Flow

This guide traces how data moves through the HFT Simulator.

## Data Journey Overview

```
   ┌─────────────────────────────────────────────────────────┐
   │                  BINANCE EXCHANGE                       │
   │              (Live market data)                         │
   └──────────────────────┬──────────────────────────────────┘
                          │
                          ▼ REST API
   ┌──────────────────────────────────────────────────────────┐
   │                 RAW DATA (OHLCV)                         │
   │  [timestamp, open, high, low, close, volume]            │
   └──────────────────────┬──────────────────────────────────┘
                          │
                          ▼ Preprocessing
   ┌──────────────────────────────────────────────────────────┐
   │             CLEANED + INDICATORS                         │
   │  [... + SMA, EMA, RSI, ATR, Bollinger, etc.]            │
   └──────────────────────┬──────────────────────────────────┘
                          │
                          ▼ Feature Engineering
   ┌──────────────────────────────────────────────────────────┐
   │              90+ FEATURES                                │
   │  [... + momentum, volatility, volume ratios, etc.]      │
   └──────────────────────┬──────────────────────────────────┘
                          │
                          ▼ ML Training/Prediction
   ┌──────────────────────────────────────────────────────────┐
   │            SEQUENCES + PREDICTIONS                       │
   │  [30-bar sequences → UP/DOWN probability]               │
   └──────────────────────┬──────────────────────────────────┘
                          │
                          ▼ Strategy
   ┌──────────────────────────────────────────────────────────┐
   │             TRADING SIGNALS                              │
   │  [BUY/SELL/HOLD + confidence]                           │
   └──────────────────────┬──────────────────────────────────┘
                          │
                          ▼ Execution
   ┌──────────────────────────────────────────────────────────┐
   │             TRADES & PORTFOLIO                           │
   │  [positions, cash, equity, PnL]                         │
   └──────────────────────┬──────────────────────────────────┘
                          │
                          ▼ Analytics
   ┌──────────────────────────────────────────────────────────┐
   │             RESULTS & METRICS                            │
   │  [Sharpe ratio, drawdown, charts, CSV files]            │
   └──────────────────────────────────────────────────────────┘
```

---

## Stage 1: Data Fetching

### From Binance API

```python
from data.fetcher import BinanceDataFetcher

fetcher = BinanceDataFetcher(symbol="BTCUSDT")
df = fetcher.fetch_klines(interval='1m', limit=1000)
```

### What We Get

```
                            open      high       low     close    volume
timestamp                                                                
2024-01-15 10:00:00    42150.50  42175.00  42140.00  42165.00   25.1234
2024-01-15 10:01:00    42165.00  42180.00  42160.00  42170.50   18.5678
2024-01-15 10:02:00    42170.50  42190.00  42165.00  42185.00   32.8901
...                           ...       ...       ...       ...       ...
```

**Data points:**
- `open`: Price at bar start
- `high`: Highest price in bar
- `low`: Lowest price in bar
- `close`: Price at bar end
- `volume`: Total volume traded

---

## Stage 2: Preprocessing

### Cleaning and Indicators

```python
from data.preprocessor import DataPreprocessor

preprocessor = DataPreprocessor()
df_processed = preprocessor.preprocess_pipeline(df)
```

### What Gets Added

```
Original columns: 5
After preprocessing: 34
```

**New columns include:**

| Type | Examples |
|------|----------|
| Returns | `returns`, `log_returns` |
| Moving Averages | `sma_10`, `sma_20`, `ema_10`, `ema_20` |
| Volatility | `volatility_10`, `volatility_20`, `atr_14` |
| Momentum | `rsi_14`, `roc_10`, `momentum_10` |
| Volume | `volume_ma_10`, `volume_ma_20` |
| Bands | `bb_upper`, `bb_lower` |

---

## Stage 3: Feature Engineering

### Creating ML Features

```python
from ml.features import FeatureEngineer

engineer = FeatureEngineer()
df_features = engineer.create_all_features(df_processed)
```

### What Gets Added

```
After feature engineering: 95 columns
```

**Feature categories:**

| Category | Count | Examples |
|----------|-------|----------|
| Price | 15+ | `log_return_5`, `price_range_ratio` |
| Volatility | 12+ | `rolling_std_10`, `range_pct` |
| Volume | 12+ | `vwap`, `volume_ratio_10`, `obv` |
| Candlestick | 8 | `body_size`, `upper_shadow`, `is_doji` |
| Order Flow | 9+ | `vpin_proxy`, `buy_volume_pct` |
| Time | 4 | `hour`, `day_of_week`, `is_weekend` |

### Sample Feature Data

```
                     close  momentum_10  volume_ratio_10   rsi_14   vwap
timestamp                                                              
2024-01-15 10:30  42185.00       0.0012             1.25    52.34  42175.50
2024-01-15 10:31  42190.00       0.0015             1.30    54.12  42178.00
2024-01-15 10:32  42175.00       0.0008             1.15    48.90  42180.25
```

---

## Stage 4: ML Preparation

### Creating Sequences

```python
X, y, feature_names = engineer.prepare_training_data(
    df_features,
    lookback=30,           # 30 bars of history
    prediction_horizon=1   # Predict 1 bar ahead
)
```

### What This Creates

```
X shape: (samples, 30, 90)
   │        │   └── 90 features per bar
   │        └── 30 bars of lookback
   └── Number of samples

y shape: (samples,)
   └── 0 = price went DOWN, 1 = price went UP
```

### Visual Representation

```
Sample 1:
  ┌────────────────────────────────────────┐
  │  Bar 1   Bar 2   Bar 3  ...  Bar 30    │ X[0]
  │  [90]    [90]    [90]   ...  [90]      │
  └────────────────────────────────────────┘
                                    ↓
                                   y[0] = 1 (price went UP at Bar 31)

Sample 2:
  ┌────────────────────────────────────────┐
  │  Bar 2   Bar 3   Bar 4  ...  Bar 31    │ X[1]
  │  [90]    [90]    [90]   ...  [90]      │
  └────────────────────────────────────────┘
                                    ↓
                                   y[1] = 0 (price went DOWN at Bar 32)
```

---

## Stage 5: ML Prediction

### Training Flow

```python
model = PriceLSTM(input_size=90)
history = model.train_model(X, y, epochs=20)
```

```
Training data:  80% of samples
Validation:     20% of samples

Epoch 1:  Loss: 0.693  Val Acc: 50.5%
Epoch 10: Loss: 0.685  Val Acc: 53.2%
Epoch 20: Loss: 0.678  Val Acc: 54.1%
```

### Prediction Flow

```
Input:  [30 bars × 90 features]
   ↓
LSTM processes sequence
   ↓
Output: 0.72 (72% probability of UP)
```

---

## Stage 6: Signal Generation

### Strategy Processing

```python
strategy = MomentumStrategy(
    ml_threshold=0.55,
    momentum_threshold=0.0005
)

signal = strategy.generate_signal(current_bar, ml_prediction=0.72)
```

### Signal Logic

```
Inputs:
  - ML prediction: 0.72 (> 0.55 threshold ✓)
  - Momentum: +0.0012 (> 0.0005 ✓)
  - Volume ratio: 1.25 (> 1.2 ✓)

All conditions met!
   ↓
Output: BUY signal, confidence 0.85
```

---

## Stage 7: Trade Execution

### From Signal to Trade

```python
# Signal approved by risk manager
trade = portfolio.execute_trade(
    symbol='BTC',
    quantity=0.5,        # Positive = buy
    price=42185.00,
    timestamp=current_time
)
```

### Portfolio Update

```
Before:
  Cash: $100,000
  BTC: 0

Trade:
  Buy 0.5 BTC @ $42,185
  Cost: $21,092.50
  Fee: $21.09 (0.1%)

After:
  Cash: $78,886.41
  BTC: 0.5
  Equity: $99,978.91 (lost fee)
```

---

## Stage 8: Results Collection

### During Backtest

```python
results = backtester.run(df_features)
```

Each bar produces:
```python
{
    'timestamp': '2024-01-15 10:30',
    'equity': 99978.91,
    'cash': 78886.41,
    'positions_value': 21092.50,
    'signal': 'BUY',
    'trade_executed': True
}
```

### Final Output

```python
results = {
    'summary': {
        'initial_capital': 100000,
        'current_value': 101234.56,
        'total_return_pct': 1.23,
        'total_pnl': 1234.56,
        'total_fees': 567.89
    },
    'equity_curve': DataFrame,  # Equity over time
    'trades': DataFrame,         # All trades
    'returns': Series            # Return per bar
}
```

---

## Stage 9: Analytics

### Metrics Calculation

```python
from analytics.metrics import PerformanceMetrics

metrics = PerformanceMetrics.calculate_all_metrics(
    results['equity_curve'],
    results['trades'],
    initial_capital=100000
)
```

### Output Metrics

```python
{
    'total_return_pct': 1.23,
    'sharpe_ratio': 1.42,
    'sortino_ratio': 1.89,
    'max_drawdown_pct': 3.5,
    'win_rate': 55.2,
    'profit_factor': 1.67,
    ...
}
```

---

## Stage 10: Visualization

### Charts Created

```python
from analytics.visualizations import TradingVisualizer

viz = TradingVisualizer()
viz.plot_summary_dashboard(equity, trades, metrics, save_path='dashboard.png')
```

### Files Saved

```
results/
├── equity_curve.png      # Portfolio value over time
├── drawdown.png          # Drawdown chart
├── pnl_distribution.png  # Trade PnL histogram
├── dashboard.png         # Summary dashboard
├── equity_curve.csv      # Raw equity data
├── trades.csv            # Trade history
└── summary.csv           # Performance metrics
```

---

## Data Size at Each Stage

| Stage | Rows | Columns | Size (MB) |
|-------|------|---------|-----------|
| Raw API | 1000 | 5 | ~0.05 |
| Preprocessed | 950 | 34 | ~0.3 |
| Features | 900 | 95 | ~0.8 |
| ML Sequences | 870 | 30×90 | ~20 |
| Predictions | 870 | 1 | ~0.01 |
| Signals | 870 | 5 | ~0.04 |
| Trades | ~50 | 8 | ~0.01 |

---

## Key Takeaways

1. **Raw data** → Preprocessed → Features → Sequences → Predictions → Signals → Trades → Metrics

2. Data **grows** during feature engineering (5 → 95 columns)

3. Data **reduces** during ML (sequence compression)

4. **One bar flows through** the entire pipeline each moment

5. **Results accumulate** as trades execute

---

## Next Steps

- [Order Books](04_order_books.md) - Market microstructure
- [ML Basics](05_ml_basics.md) - How predictions work