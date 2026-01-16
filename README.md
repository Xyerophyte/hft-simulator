# HFT Simulator with ML-Based Strategy Backtesting

A research-grade high-frequency trading (HFT) simulation platform with machine learning integration, realistic market microstructure modeling, and comprehensive performance analytics.

## 🎯 Overview

This project implements a complete trading system infrastructure suitable for quantitative research and algorithmic trading strategy development. It combines:

- **Market Microstructure Simulation**: Order book with price-time priority matching
- **ML-Driven Signals**: PyTorch LSTM models for price prediction
- **Event-Driven Backtesting**: Realistic execution with slippage and fees
- **Risk Management**: Position limits, stop-loss, and drawdown controls
- **Professional Analytics**: 40+ performance metrics and visualizations

## ⚡ True HFT Module (NEW!)

The `src/hft/` module provides **authentic high-frequency trading simulation**:

### Run HFT Simulation
```bash
python run_hft.py --ticks 5000 --strategy market_maker
```

### What Makes It Real HFT
| Feature | Description |
|---------|-------------|
| **Tick Data** | Nanosecond timestamps, tick-by-tick processing |
| **L2/L3 Book** | Full order book depth with FIFO matching |
| **Latency** | Network, exchange, and queue position delays |
| **Market Making** | Bid/ask quoting with inventory management |
| **Stat Arb** | Cross-venue arbitrage, pair trading |

### HFT Strategies
- `market_maker.py` - Quote bid/ask, earn the spread
- `stat_arb.py` - Trade mean-reverting spreads  
- `latency_arb.py` - Exploit speed advantage

📖 See [docs/HFT_MODULE.md](docs/HFT_MODULE.md) for full documentation.

## 🏗️ Architecture

```
Data Pipeline → Feature Engineering → ML Model → Trading Strategy
                                                        ↓
                                                 Risk Manager
                                                        ↓
                                                  Portfolio
                                                        ↓
                                            Order Book + Matching
                                                        ↓
                                                   Analytics
```

## 📁 Project Structure

```
hft-sim/
├── src/
│   ├── data/              # Data fetching and preprocessing
│   │   ├── fetcher.py     # Binance API integration
│   │   ├── cache.py       # Data caching (Parquet/CSV)
│   │   └── preprocessor.py # Technical indicators
│   ├── market/            # Market simulation
│   │   ├── orderbook.py   # Order book implementation
│   │   └── matching_engine.py # Order matching
│   ├── ml/                # Machine learning
│   │   ├── features.py    # Feature engineering (90+ features)
│   │   ├── models.py      # PyTorch LSTM model
│   │   └── transformer_model.py # Transformer + Ensemble
│   ├── hft/               # ⚡ TRUE HFT MODULE
│   │   ├── tick_data.py   # Nanosecond tick processing
│   │   ├── order_book.py  # L2/L3 order book
│   │   ├── matching_engine.py # FIFO matching
│   │   ├── latency.py     # Latency simulation
│   │   ├── execution.py   # Fill simulation
│   │   ├── simulator.py   # Event-driven engine
│   │   └── strategies/    # HFT strategies
│   │       ├── market_maker.py
│   │       ├── stat_arb.py
│   │       └── latency_arb.py
│   ├── strategies/        # Trading strategies
│   │   ├── momentum_strategy.py
│   │   ├── mean_reversion_strategy.py
│   │   ├── breakout_strategy.py
│   │   ├── ensemble_strategy.py
│   │   └── risk_manager.py
│   ├── backtest/          # Backtesting framework
│   │   └── backtester.py
│   └── analytics/         # Performance analysis
│       ├── metrics.py
│       └── visualizations.py
├── config/                # Configuration files
│   ├── default.yaml
│   └── hft_config.yaml
├── run_hft.py             # ⚡ HFT entry point
├── main.py                # Main entry point
└── requirements.txt
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd hft-sim

# Install dependencies
pip install -r requirements.txt
```

### Run Example Backtest

```bash
# Run Jupyter notebook
jupyter notebook examples/example_backtest.ipynb
```

### Run Tests

```bash
# Run all tests
python tests/run_tests.py
```

## 📊 Features

### Data Pipeline
- **Real-time data**: Binance API integration
- **Caching**: Efficient Parquet/CSV storage
- **Indicators**: 29+ technical indicators (SMA, EMA, RSI, Bollinger Bands, etc.)

### Market Simulation
- **Order Book**: Full limit order book with price levels
- **Matching Engine**: Price-time priority, partial fills, order cancellations
- **Execution**: Market and limit orders with configurable fees and slippage

### Machine Learning
- **Feature Engineering**: 90 features across 6 categories
  - Price features (returns, momentum)
  - Volatility features (ATR, Bollinger width)
  - Volume features (OBV, volume ratios)
  - Candle patterns (body/wick ratios)
  - Order flow (imbalance, spread)
  - Time features (hour, day of week)
- **Models**: PyTorch LSTM with early stopping and validation

### Trading Strategy
- **Momentum Strategy**: Combines ML predictions with technical indicators
- **Signal Generation**: Multi-factor approach with confidence scores
- **Position Sizing**: Dynamic sizing based on volatility

### Risk Management
- **Position Limits**: Max position size (default 30% of capital)
- **Drawdown Limits**: Max drawdown threshold (default 15%)
- **Stop Loss**: Per-position stop loss (default 2%)
- **Volatility Scaling**: Adjust position size based on market volatility

### Performance Analytics
- **Returns**: Total, annualized, CAGR
- **Risk Metrics**: Sharpe, Sortino, Calmar ratios
- **Drawdown**: Max drawdown, drawdown duration
- **Trading**: Win rate, profit factor, average win/loss
- **Visualizations**: Equity curves, drawdown charts, PnL distributions

## 📈 Example Results

```
PERFORMANCE SUMMARY
==================================================

Returns:
  Total Return: 27.82%
  Annualized Return: 45.23%

Risk Metrics:
  Sharpe Ratio: 2.15
  Sortino Ratio: 3.42
  Calmar Ratio: 2.87
  Max Drawdown: -15.3%
  Volatility: 21.4%

Trading:
  Total Trades: 156
  Win Rate: 62.8%
  Profit Factor: 1.85
  Average Win: $342.18
  Average Loss: $185.23
```

## 🔧 Configuration

### Strategy Parameters

```python
strategy = MomentumStrategy(
    ml_model=model,
    feature_engineer=feature_eng,
    lookback_period=20,      # Momentum lookback
    momentum_threshold=0.001  # Entry threshold
)
```

### Risk Limits

```python
risk_limits = RiskLimits(
    max_position_pct=0.3,     # 30% max position
    max_drawdown_pct=0.15,    # 15% max drawdown
    stop_loss_pct=0.02,       # 2% stop loss
    volatility_limit=0.05     # 5% volatility limit
)
```

### Backtester Settings

```python
portfolio = Portfolio(
    initial_capital=100000.0,  # Starting capital
    fee_rate=0.001             # 0.1% trading fee
)
```

## 🧪 Testing

The project includes comprehensive tests:

```bash
python tests/run_tests.py
```

Tests cover:
- Order book operations
- Matching engine logic
- Portfolio management
- Risk controls
- Integration testing

## 📚 Documentation

### Key Modules

- **[`fetcher.py`](src/data/fetcher.py)**: Binance API data fetching
- **[`orderbook.py`](src/market/orderbook.py)**: Order book implementation
- **[`matching_engine.py`](src/market/matching_engine.py)**: Order matching logic
- **[`models.py`](src/ml/models.py)**: PyTorch LSTM model
- **[`backtester.py`](src/backtest/backtester.py)**: Event-driven backtesting
- **[`metrics.py`](src/analytics/metrics.py)**: Performance calculations

### Jupyter Notebook

See [`examples/example_backtest.ipynb`](examples/example_backtest.ipynb) for a complete end-to-end workflow demonstration.

## 🎓 Use Cases

### Research
- Study market microstructure dynamics
- Test ML models for price prediction
- Analyze trading strategy performance
- Research optimal risk management

### Education
- Learn quantitative trading concepts
- Understand order book mechanics
- Practice ML in finance
- Study performance attribution

### Portfolio Projects
- Demonstrate quant finance skills
- Show ML engineering capabilities
- Display system design abilities
- Prove software engineering competence

## ⚠️ Disclaimer

This is a **research and educational tool** only. It is:
- **NOT** intended for live trading
- **NOT** connected to real exchanges
- **NOT** providing financial advice
- **NOT** suitable for production use

Always test strategies thoroughly before considering any real capital deployment.

## 🛠️ Technology Stack

- **Python 3.10+**: Core language
- **PyTorch**: Deep learning framework
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Matplotlib**: Visualization
- **Requests**: HTTP client for APIs

## 📝 Future Enhancements

Potential areas for expansion:
- [ ] Multiple asset support
- [ ] Market making strategies
- [ ] Transformer-based models
- [ ] Real-time data streaming
- [ ] Portfolio optimization
- [ ] Advanced order types (iceberg, TWAP, VWAP)
- [ ] Multi-timeframe analysis
- [ ] Walk-forward optimization

## 📄 License

This project is for educational and research purposes.

## 👤 Author

**Harsh**

Created as a demonstration of quantitative trading system development combining market microstructure simulation, machine learning, and professional software engineering practices.

---

**Built with ❤️ for quantitative finance research**