# System Architecture

This document describes the architecture and design of the HFT Simulator.

## Overview

The HFT Simulator follows a **modular, layered architecture** that separates concerns and allows for easy testing and extension.

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface                            │
│              (run_demo.py / Jupyter Notebook)                    │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                     Backtesting Layer                            │
│                      backtester.py                               │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────────────┐
│   Strategy    │   │   Portfolio   │   │   Risk Management     │
│  Layer        │   │   Layer       │   │   Layer               │
│  - Signals    │   │  - Positions  │   │  - Limits             │
│  - Decisions  │   │  - PnL        │   │  - Stop Loss          │
└───────────────┘   └───────────────┘   └───────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│                     ML Prediction Layer                          │
│              features.py / models.py                             │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Market Simulation Layer                       │
│              orderbook.py / matching_engine.py                   │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Data Pipeline Layer                         │
│            fetcher.py / cache.py / preprocessor.py               │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│                      External APIs                               │
│                    (Binance REST API)                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Data Pipeline (`src/data/`)

The data pipeline is responsible for acquiring, storing, and preprocessing market data.

| Component | File | Purpose |
|-----------|------|---------|
| **BinanceDataFetcher** | `fetcher.py` | Fetches OHLCV data from Binance API |
| **DataCache** | `cache.py` | Persists data to Parquet/CSV files |
| **DataPreprocessor** | `preprocessor.py` | Cleans data, adds technical indicators |

**Data Flow:**
```
Binance API → Fetcher → Cache → Preprocessor → Features
```

### 2. Market Simulation (`src/market/`)

Simulates realistic market conditions for order execution.

| Component | File | Purpose |
|-----------|------|---------|
| **OrderBook** | `orderbook.py` | Maintains bid/ask price levels |
| **MatchingEngine** | `matching_engine.py` | Executes orders with fees/slippage |

**Key Features:**
- Price-time priority matching
- Configurable maker/taker fees
- Slippage modeling
- Order book imbalance calculation

### 3. Machine Learning (`src/ml/`)

Provides predictive signals using deep learning.

| Component | File | Purpose |
|-----------|------|---------|
| **FeatureEngineer** | `features.py` | Creates 90+ features from price data |
| **PriceLSTM** | `models.py` | LSTM model for price direction |
| **ModelTrainer** | `models.py` | Handles training and evaluation |

**Feature Categories:**
- Price-based (returns, momentum)
- Volatility-based (ATR, Bollinger Bands)
- Volume-based (VWAP, volume ratio)
- Candlestick patterns
- Order flow proxies
- Time features (hour, day of week)

### 4. Trading Strategy (`src/strategies/`)

Implements trading logic and position management.

| Component | File | Purpose |
|-----------|------|---------|
| **MomentumStrategy** | `momentum_strategy.py` | Generates buy/sell signals |
| **Portfolio** | `portfolio.py` | Tracks positions and calculates PnL |
| **RiskManager** | `risk_manager.py` | Enforces risk limits |

**Signal Generation Flow:**
```
ML Prediction + Momentum + Volume → Signal → Risk Check → Trade
```

### 5. Backtesting (`src/backtest/`)

Runs historical simulations of trading strategies.

| Component | File | Purpose |
|-----------|------|---------|
| **Backtester** | `backtester.py` | Event-driven backtest engine |
| **BacktestConfig** | `backtester.py` | Configuration settings |

**Backtest Process:**
1. Load historical data
2. Generate ML predictions (optional)
3. Generate trading signals
4. Check risk limits
5. Execute trades
6. Update portfolio
7. Record metrics

### 6. Analytics (`src/analytics/`)

Calculates and visualizes performance.

| Component | File | Purpose |
|-----------|------|---------|
| **PerformanceMetrics** | `metrics.py` | Sharpe, Sortino, drawdown, etc. |
| **TradingVisualizer** | `visualizations.py` | Charts and dashboards |

---

## Design Patterns

### 1. Dependency Injection

Components are passed as dependencies rather than created internally:

```python
# Components are injected, not created
backtester = Backtester(strategy, config)
backtester.run(df)
```

### 2. Configuration Objects

Settings are encapsulated in dataclasses:

```python
@dataclass
class BacktestConfig:
    initial_capital: float = 100000.0
    fee_rate: float = 0.001
    position_size_pct: float = 0.3
```

### 3. Strategy Pattern

Trading strategies implement a common interface:

```python
class MomentumStrategy:
    def generate_signal(self, row, ml_prediction):
        # Returns TradeSignal
        pass
```

### 4. Observer Pattern (Implied)

Portfolio updates are triggered by events:

```python
# When trade executes, portfolio is notified
portfolio.execute_trade(symbol, quantity, price, timestamp)
```

---

## Directory Structure

```
hft-sim/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── fetcher.py       # API interaction
│   │   ├── cache.py         # Storage
│   │   └── preprocessor.py  # Data cleaning
│   ├── market/
│   │   ├── orderbook.py     # Order book
│   │   └── matching_engine.py # Execution
│   ├── ml/
│   │   ├── features.py      # Feature engineering
│   │   └── models.py        # LSTM models
│   ├── strategies/
│   │   ├── momentum_strategy.py
│   │   ├── portfolio.py
│   │   └── risk_manager.py
│   ├── backtest/
│   │   └── backtester.py
│   └── analytics/
│       ├── metrics.py
│       └── visualizations.py
├── tests/
├── docs/
├── examples/
├── data/
├── results/
└── models/
```

---

## Data Flow Diagram

```
                    ┌──────────────────┐
                    │   Binance API    │
                    └────────┬─────────┘
                             │ OHLCV data
                             ▼
                    ┌──────────────────┐
                    │  BinanceDataFetcher │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │    DataCache     │ ◄── Save/Load
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │ DataPreprocessor │ ── Add indicators
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │ FeatureEngineer  │ ── 90+ features
                    └────────┬─────────┘
                             │
           ┌─────────────────┼─────────────────┐
           │                 │                 │
           ▼                 ▼                 ▼
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │  PriceLSTM   │  │  OrderBook   │  │  Strategy    │
    │  (predict)   │  │  (simulate)  │  │  (signals)   │
    └──────┬───────┘  └──────────────┘  └──────┬───────┘
           │                                    │
           └────────────────┬───────────────────┘
                            ▼
                    ┌──────────────────┐
                    │   Backtester     │
                    └────────┬─────────┘
                             │
           ┌─────────────────┼─────────────────┐
           ▼                 ▼                 ▼
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │  Portfolio   │  │ RiskManager  │  │   Metrics    │
    └──────────────┘  └──────────────┘  └──────────────┘
                             │
                    ┌────────▼─────────┐
                    │  Visualizations  │
                    └──────────────────┘
```

---

## Key Interfaces

### TradeSignal
```python
@dataclass
class TradeSignal:
    timestamp: pd.Timestamp
    signal: Signal  # BUY, SELL, HOLD
    confidence: float
    price: float
    reason: str
    ml_probability: Optional[float]
```

### Order
```python
@dataclass
class Order:
    order_id: str
    side: OrderSide  # BUY, SELL
    order_type: OrderType  # MARKET, LIMIT
    price: Optional[float]
    quantity: float
    timestamp: float
```

### BacktestConfig
```python
@dataclass
class BacktestConfig:
    initial_capital: float = 100000.0
    fee_rate: float = 0.001
    position_size_pct: float = 0.3
    use_risk_manager: bool = True
    use_ml_model: bool = False
```

---

## Extension Points

The architecture is designed for easy extension:

1. **New Data Sources**: Create new fetcher classes
2. **New Strategies**: Implement strategy interface
3. **New ML Models**: Extend PriceLSTM or add new architectures
4. **New Metrics**: Add methods to PerformanceMetrics
5. **New Visualizations**: Add methods to TradingVisualizer

---

## Next Steps

- [Configuration Guide](02_configuration.md) - Customize settings
- [Performance Guide](03_performance.md) - Optimization tips
- [API Reference](../api/01_data_pipeline.md) - Detailed API docs
