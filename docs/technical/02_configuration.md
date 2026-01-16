# Configuration Guide

This guide covers all configuration options for the HFT Simulator.

## Quick Reference

| Component | File | Key Settings |
|-----------|------|--------------|
| Backtester | `BacktestConfig` | initial_capital, fee_rate, position_size |
| Risk Manager | `RiskLimits` | max_position, max_drawdown, stop_loss |
| Strategy | `MomentumStrategy` | ml_threshold, momentum_threshold |
| ML Model | `PriceLSTM` | hidden_size, num_layers, dropout |

---

## Backtest Configuration

The `BacktestConfig` dataclass controls backtesting behavior.

```python
from backtest.backtester import BacktestConfig

config = BacktestConfig(
    initial_capital=100000.0,  # Starting capital in USD
    fee_rate=0.001,            # 0.1% trading fee per trade
    position_size_pct=0.3,     # Max 30% of capital per position
    use_risk_manager=True,     # Enable risk checks
    use_ml_model=False         # Use ML predictions
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `initial_capital` | float | 100000.0 | Starting capital in USD |
| `fee_rate` | float | 0.001 | Trading fee as a decimal (0.001 = 0.1%) |
| `position_size_pct` | float | 0.3 | Maximum position size as fraction of capital |
| `use_risk_manager` | bool | True | Enable/disable risk management |
| `use_ml_model` | bool | False | Use ML predictions for signals |

---

## Risk Limits Configuration

The `RiskLimits` dataclass defines risk parameters.

```python
from strategies.risk_manager import RiskLimits

limits = RiskLimits(
    max_position_pct=0.5,      # Max 50% of capital per position
    max_total_exposure=1.0,    # Max 100% total exposure
    max_drawdown_pct=0.15,     # Stop if drawdown exceeds 15%
    max_daily_loss_pct=0.05,   # Max 5% daily loss
    stop_loss_pct=0.02,        # 2% stop loss per position
    volatility_limit=0.05,     # Max 5% volatility allowed
    min_sharpe_ratio=0.5       # Minimum required Sharpe
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_position_pct` | float | 0.5 | Max position size as fraction of equity |
| `max_total_exposure` | float | 1.0 | Max total exposure (1.0 = 100%) |
| `max_drawdown_pct` | float | 0.15 | Maximum allowed drawdown |
| `max_daily_loss_pct` | float | 0.05 | Maximum daily loss allowed |
| `stop_loss_pct` | float | 0.02 | Stop loss percentage per position |
| `volatility_limit` | float | 0.05 | Max volatility for position sizing |
| `min_sharpe_ratio` | float | 0.5 | Minimum Sharpe ratio target |

---

## Strategy Configuration

The `MomentumStrategy` class has several tunable parameters.

```python
from strategies.momentum_strategy import MomentumStrategy

strategy = MomentumStrategy(
    ml_threshold=0.55,           # ML probability threshold for signals
    momentum_threshold=0.0005,   # Momentum % threshold
    volume_threshold=1.2,        # Volume must be 1.2x average
    confidence_scaling=True      # Scale position by confidence
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ml_threshold` | float | 0.55 | ML probability above this = BUY signal |
| `momentum_threshold` | float | 0.0005 | Minimum momentum for signal (0.05%) |
| `volume_threshold` | float | 1.2 | Volume ratio for confirmation |
| `confidence_scaling` | bool | True | Scale position by signal confidence |

### Signal Logic

```
BUY Signal Conditions:
- ML probability > 0.55 OR momentum > 0.05%
- Volume ratio >= 1.2x average
- Both ML and momentum agree = higher confidence

SELL Signal Conditions:
- ML probability < 0.45 OR momentum < -0.05%
- Volume ratio >= 1.2x average
```

---

## ML Model Configuration

### PriceLSTM Model

```python
from ml.models import PriceLSTM

model = PriceLSTM(
    input_size=90,          # Number of input features
    hidden_size=64,         # LSTM hidden layer size
    num_layers=2,           # Number of LSTM layers
    dropout=0.2,            # Dropout rate
    bidirectional=False     # Use bidirectional LSTM
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_size` | int | - | Number of input features (required) |
| `hidden_size` | int | 64 | Size of LSTM hidden state |
| `num_layers` | int | 2 | Number of stacked LSTM layers |
| `dropout` | float | 0.2 | Dropout probability |
| `bidirectional` | bool | False | Use bidirectional LSTM |

### Training Configuration

```python
history = model.train_model(
    X, y,
    epochs=50,                      # Maximum training epochs
    batch_size=32,                  # Batch size
    validation_split=0.2,           # Validation data fraction
    learning_rate=0.001,            # Adam learning rate
    early_stopping_patience=10      # Early stopping patience
)
```

---

## Feature Engineering Configuration

### Window Sizes

```python
from ml.features import FeatureEngineer

engineer = FeatureEngineer()
df_features = engineer.create_all_features(
    df,
    windows=[5, 10, 20]  # Lookback windows for features
)
```

### Features Created

| Category | Count | Examples |
|----------|-------|----------|
| Price | 15+ | returns, log_returns, momentum |
| Volatility | 12+ | rolling_std, atr, bollinger |
| Volume | 12+ | volume_ma, vwap, volume_ratio |
| Candle | 8 | body_size, upper_shadow, doji |
| Order Flow | 9+ | vpin, buy_volume_pct |
| Time | 4 | hour, day_of_week, is_weekend |

---

## Data Pipeline Configuration

### Data Fetcher

```python
from data.fetcher import BinanceDataFetcher

fetcher = BinanceDataFetcher(symbol="BTCUSDT")

# Fetch with parameters
df = fetcher.fetch_klines(
    interval='1m',      # Candle interval: 1m, 5m, 15m, 1h, 1d
    limit=1000          # Number of candles
)
```

### Data Cache

```python
from data.cache import DataCache

cache = DataCache(cache_dir='data/cache')

# Save data
cache.save(df, symbol='BTCUSDT', interval='1m', file_format='parquet')

# Load data
df = cache.load(symbol='BTCUSDT', interval='1m', file_format='parquet')
```

### Preprocessor

```python
from data.preprocessor import DataPreprocessor

preprocessor = DataPreprocessor()
df_clean = preprocessor.preprocess_pipeline(
    df,
    add_features=True   # Add technical indicators
)
```

---

## Matching Engine Configuration

```python
from market.matching_engine import MatchingEngine

engine = MatchingEngine(
    maker_fee=0.0001,      # 0.01% maker fee
    taker_fee=0.0002,      # 0.02% taker fee
    slippage_pct=0.0005    # 0.05% slippage
)
```

---

## Portfolio Configuration

```python
from strategies.portfolio import Portfolio

portfolio = Portfolio(
    initial_capital=100000.0,  # Starting capital
    fee_rate=0.001             # Trading fee rate
)
```

---

## Environment Variables

For production use, configure via environment variables:

```bash
# Data settings
export HFT_DATA_DIR="./data"
export HFT_CACHE_DIR="./data/cache"

# API settings (if using authenticated endpoints)
export BINANCE_API_KEY="your-api-key"
export BINANCE_SECRET="your-secret"

# Model settings
export HFT_MODEL_DIR="./models/saved"

# Logging
export HFT_LOG_LEVEL="INFO"
```

---

## Configuration Files

### config.yaml (Example)

```yaml
backtest:
  initial_capital: 100000
  fee_rate: 0.001
  position_size_pct: 0.3

risk:
  max_position_pct: 0.5
  max_drawdown_pct: 0.15
  stop_loss_pct: 0.02

strategy:
  ml_threshold: 0.55
  momentum_threshold: 0.0005
  volume_threshold: 1.2

model:
  hidden_size: 64
  num_layers: 2
  dropout: 0.2
  epochs: 50
  batch_size: 32
```

---

## Best Practices

### 1. Start Conservative

```python
# Conservative settings for new strategies
config = BacktestConfig(
    position_size_pct=0.1,  # Only 10% of capital
)

limits = RiskLimits(
    max_drawdown_pct=0.05,  # Only 5% max drawdown
    stop_loss_pct=0.01,     # Tight 1% stop loss
)
```

### 2. Gradually Increase Risk

Only increase position sizes and risk limits after validating strategy performance across multiple time periods.

### 3. Version Control Configuration

Keep configuration in version control to track changes and reproduce results.

---

## Next Steps

- [Performance Guide](03_performance.md) - Optimization tips
- [API Reference](../api/01_data_pipeline.md) - Detailed API docs
