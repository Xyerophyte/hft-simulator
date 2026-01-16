# 🚀 HFT Simulator

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

A **production-grade High-Frequency Trading simulator** with machine learning integration, realistic market microstructure modeling, and comprehensive performance analytics. Built for quantitative researchers, algorithmic traders, and fintech developers.

---

## 📋 Overview

This project implements a complete trading system infrastructure suitable for quantitative research and algorithmic trading strategy development. It combines **tick-by-tick market simulation** with nanosecond precision, **deep learning models** for price prediction, and **realistic execution simulation** including latency, slippage, and order book dynamics.

The HFT Simulator bridges the gap between academic research and production trading systems. Unlike simple backtesting frameworks that use candlestick data, this simulator operates at the tick level with full L2/L3 order book depth, FIFO price-time priority matching, and microsecond latency modeling—the same components used by real HFT firms.

Whether you're developing market-making algorithms, testing statistical arbitrage strategies, or researching ML-driven trading signals, this platform provides the infrastructure needed to simulate realistic market conditions and measure true strategy performance.

---

## 📑 Table of Contents

- [Features](#-features)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [API Documentation](#-api-documentation)
- [Development](#-development)
- [Testing](#-testing)
- [Deployment](#-deployment)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)
- [Authors and Acknowledgments](#-authors-and-acknowledgments)
- [Support](#-support)
- [Changelog](#-changelog)

---

## ✨ Features

### Core Trading Infrastructure
- **Tick-by-Tick Processing** — Nanosecond-precision timestamps for authentic HFT simulation
- **L2/L3 Order Book** — Full depth order book with bid/ask levels and queue position tracking
- **FIFO Matching Engine** — Price-time priority order matching with partial fills
- **Latency Modeling** — Network, exchange, and queue delays (configurable from µs to ms)

### Trading Strategies
- **Market Making** — Bid/ask quoting with inventory management and adverse selection detection
- **Statistical Arbitrage** — Pair trading with z-score signals and mean reversion
- **Latency Arbitrage** — Stale quote detection and speed-based trading
- **Momentum Strategy** — ML-enhanced trend following with risk controls

### Machine Learning
- **LSTM Networks** — PyTorch-based sequential models for price prediction
- **Transformer Models** — Attention-based architecture for time series
- **Ensemble Methods** — Combine multiple models for robust signals
- **90+ Engineered Features** — Technical indicators, microstructure features, and more

### Analytics & Backtesting
- **Event-Driven Backtester** — Realistic execution with fees and slippage
- **40+ Performance Metrics** — Sharpe, Sortino, max drawdown, win rate, etc.
- **Visualization Suite** — Equity curves, drawdown charts, trade analysis

---

## 📋 Prerequisites

| Requirement | Minimum Version | Recommended |
|-------------|-----------------|-------------|
| Python | 3.9+ | 3.11+ |
| pip | 21.0+ | Latest |
| OS | Windows 10 / Linux / macOS | Any |
| RAM | 4 GB | 8+ GB |
| Storage | 500 MB | 2+ GB (for data) |

### Required Python Packages
```
numpy>=1.21.0
pandas>=1.3.0
torch>=1.9.0
scikit-learn>=0.24.0
PyYAML>=5.4
matplotlib>=3.4.0
requests>=2.25.0
```

---

## 🔧 Installation

### Option 1: Standard Installation

```bash
# Clone the repository
git clone https://github.com/Xyerophyte/hft-simulator.git
cd hft-simulator

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Development Installation

```bash
# Clone with full history
git clone https://github.com/Xyerophyte/hft-simulator.git
cd hft-simulator

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Verify Installation

```bash
python -c "from src.hft import OrderBook, MatchingEngine; print('✅ Installation successful!')"
```

---

## ⚡ Quick Start

### Run HFT Simulation (30 seconds)

```bash
# Run market making simulation with 5000 ticks
python run_hft.py --ticks 5000 --strategy market_maker
```

### Run ML Backtest

```bash
# Run full backtest with ML model
python main.py
```

### Minimal Python Example

```python
from src.hft import OrderBook, Side, MatchingEngine
from src.hft.strategies import SimpleMarketMaker

# Create order book and matching engine
book = OrderBook(symbol="BTC")
engine = MatchingEngine(book)

# Initialize market maker
mm = SimpleMarketMaker(spread_bps=2.0, max_inventory=10.0)

# Make market at current price
result = mm.make_market(mid_price=50000.0)
print(f"Bid: ${result['bid']:.2f}, Ask: ${result['ask']:.2f}")
```

---

## 📖 Usage

### Command Line Interface

```bash
# HFT Simulation
python run_hft.py [OPTIONS]

Options:
  --ticks INT       Number of ticks to simulate (default: 1000)
  --strategy STR    Strategy: market_maker, stat_arb, latency_arb (default: market_maker)
  --price FLOAT     Initial price (default: 50000)
  --spread FLOAT    Initial spread in bps (default: 2.0)
  --volatility FLOAT Price volatility (default: 0.001)
  --seed INT        Random seed (default: 42)
```

### Python API Usage

#### Market Making
```python
from src.hft.strategies.market_maker import MarketMaker, MarketMakerConfig

config = MarketMakerConfig(
    default_spread_bps=3.0,
    max_position=5.0,
    inventory_skew_factor=0.15
)

mm = MarketMaker(config=config)
quote = mm.calculate_quote(mid_price=50000, volatility=0.001)
print(f"Quote: {quote.bid_price:.2f} / {quote.ask_price:.2f}")
```

#### Statistical Arbitrage
```python
from src.hft.strategies.stat_arb import StatisticalArbitrage

arb = StatisticalArbitrage()
arb.add_pair("BTC_ETH", "BTC", "ETH")

# Update with prices
signal = arb.update_prices("BTC_ETH", price_1=50000, price_2=3000)
if signal and signal.z_score > 2.0:
    print(f"Arbitrage signal: {signal.signal}")
```

#### Order Book Operations
```python
from src.hft.order_book import OrderBook, Order, Side

book = OrderBook(symbol="BTC")

# Add orders
book.add_order(Order(order_id=1, side=Side.BID, price=49990, size=10, timestamp_ns=0))
book.add_order(Order(order_id=2, side=Side.ASK, price=50010, size=10, timestamp_ns=0))

# Get market state
print(f"Mid: ${book.mid_price:.2f}")
print(f"Spread: {book.spread_bps:.1f} bps")
print(f"Depth: {book.get_depth(5)}")
```

---

## ⚙️ Configuration

### Configuration Files

| File | Purpose |
|------|---------|
| `config/default.yaml` | Main configuration |
| `config/hft_config.yaml` | HFT-specific parameters |

### Environment Variables

```bash
# Data Settings
export HFT_SYMBOL="BTCUSDT"
export HFT_INTERVAL="1m"
export HFT_HISTORY_DAYS=30

# Trading Settings
export HFT_INITIAL_CAPITAL=10000
export HFT_FEE_RATE=0.001
export HFT_STRATEGY="market_maker"

# ML Settings
export HFT_MODEL_TYPE="lstm"
export HFT_EPOCHS=50

# Output
export HFT_RESULTS_DIR="./results"
export HFT_LOG_LEVEL="INFO"
```

### Example Configuration (YAML)

```yaml
# config/hft_config.yaml
market_maker:
  spread_bps: 3.0
  base_size: 0.5
  max_position: 5.0
  inventory_skew_factor: 0.15

stat_arb:
  entry_z_threshold: 2.5
  exit_z_threshold: 0.3
  lookback_periods: 50

risk:
  max_loss_per_trade: 20.0
  daily_loss_limit: 500.0
  position_limit: 5.0
```

---

## 📚 API Documentation

### Core Modules

| Module | Description |
|--------|-------------|
| `src.hft.tick_data` | Tick processing with nanosecond timestamps |
| `src.hft.order_book` | L2/L3 order book implementation |
| `src.hft.matching_engine` | FIFO price-time priority matching |
| `src.hft.latency` | Network and exchange latency modeling |
| `src.hft.execution` | Fill simulation with slippage |

### Strategy Interface

```python
from abc import ABC, abstractmethod

class BaseStrategy(ABC):
    @abstractmethod
    def generate_signal(self, data) -> TradeSignal:
        """Generate trading signal from market data."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return strategy name."""
        pass
```

For detailed API documentation, see [docs/HFT_MODULE.md](docs/HFT_MODULE.md).

---

## 🛠️ Development

### Setup Development Environment

```bash
# Clone and install dev dependencies
git clone https://github.com/Xyerophyte/hft-simulator.git
cd hft-simulator
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Code Style

This project uses:
- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Run linter
flake8 src/ tests/

# Type check
mypy src/
```

### Project Structure

```
hft-simulator/
├── src/
│   ├── hft/               # ⚡ HFT Module
│   │   ├── tick_data.py
│   │   ├── order_book.py
│   │   ├── matching_engine.py
│   │   ├── latency.py
│   │   ├── execution.py
│   │   ├── simulator.py
│   │   └── strategies/
│   ├── ml/                # Machine Learning
│   ├── strategies/        # Trading Strategies
│   ├── backtest/          # Backtesting Framework
│   └── analytics/         # Performance Analytics
├── config/                # Configuration Files
├── docs/                  # Documentation
├── tests/                 # Test Suite
├── run_hft.py            # HFT Entry Point
├── main.py               # Main Entry Point
└── requirements.txt
```

---

## 🧪 Testing

### Run All Tests

```bash
# Run test suite
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html  # macOS
start htmlcov/index.html # Windows
```

### Run Specific Tests

```bash
# Test HFT module
python -m pytest tests/test_hft.py -v

# Test strategies
python -m pytest tests/test_strategies.py -v

# Test with markers
python -m pytest -m "not slow" tests/
```

---

## 🚀 Deployment

### Production Considerations

1. **Data Source**: Configure real-time data feeds (Binance, Polygon, etc.)
2. **Latency**: Deploy close to exchange servers for minimum latency
3. **Monitoring**: Set up alerts for PnL, position limits, and errors
4. **Logging**: Use structured logging with appropriate levels

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["python", "run_hft.py", "--ticks", "10000"]
```

```bash
docker build -t hft-simulator .
docker run -it hft-simulator
```

---

## 🔧 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'src'` | Run from project root or add to PYTHONPATH |
| `torch not found` | Install PyTorch: `pip install torch` |
| `sortedcontainers not found` | Install: `pip install sortedcontainers` |
| Slow simulation | Reduce tick count or use simpler strategy |
| Memory issues | Process data in chunks, reduce lookback periods |

### Debug Mode

```bash
# Run with verbose logging
export HFT_LOG_LEVEL=DEBUG
python run_hft.py --ticks 100
```

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code of Conduct

Please read our [Code of Conduct](CODE_OF_CONDUCT.md) before contributing.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors and Acknowledgments

**Author**: [Xyerophyte](https://github.com/Xyerophyte)

### Acknowledgments

- Inspired by real-world HFT system architectures
- Built with PyTorch, NumPy, and Pandas
- Market data from Binance API

---

## 💬 Support

- **Issues**: [GitHub Issues](https://github.com/Xyerophyte/hft-simulator/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Xyerophyte/hft-simulator/discussions)
- **Email**: Create an issue for support requests

---

## 📝 Changelog

### v1.0.0 (2026-01-16)
- ✨ Added true HFT module with tick-by-tick processing
- ✨ Implemented L2/L3 order book with FIFO matching
- ✨ Added market making, stat arb, and latency arb strategies
- ✨ Integrated latency modeling (µs precision)
- ✨ Added Transformer and Ensemble ML models
- 📚 Comprehensive documentation

See [CHANGELOG.md](CHANGELOG.md) for full history.

---

<p align="center">
  Made with ❤️ for quantitative trading research
</p>