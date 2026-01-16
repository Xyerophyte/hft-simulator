# HFT Simulator - Complete Technical Documentation

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Installation & Setup](#installation--setup)
3. [Configuration](#configuration)
4. [API Reference](#api-reference)
5. [Usage Examples](#usage-examples)
6. [Testing](#testing)
7. [Deployment](#deployment)
8. [Troubleshooting](#troubleshooting)
9. [Performance Considerations](#performance-considerations)

---

## Architecture Overview

### System Design

The HFT Simulator follows a modular, event-driven architecture designed for high-performance quantitative trading research:

```
┌─────────────────────────────────────────────────────────────┐
│                     HFT Simulator                            │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐     ┌──────────────┐     ┌─────────────┐ │
│  │ Data Pipeline│────▶│ ML Pipeline  │────▶│  Strategy   │ │
│  │              │     │              │     │   Engine    │ │
│  │  • Fetcher   │     │  • Features  │     │  • Momentum │ │
│  │  • Cache     │     │  • LSTM      │     │  • Signals  │ │
│  │  • Preproc   │     │  • Training  │     │             │ │
│  └──────────────┘     └──────────────┘     └─────────────┘ │
│         │                                          │         │
│         ▼                                          ▼         │
│  ┌──────────────┐     ┌──────────────┐     ┌─────────────┐ │
│  │   Market     │────▶│  Execution   │────▶│   Risk      │ │
│  │  Simulator   │     │   Engine     │     │ Management  │ │
│  │              │     │              │     │             │ │
│  │  • OrderBook │     │  • Matching  │     │  • Limits   │ │
│  │  • LOB       │     │  • Portfolio │     │  • VaR      │ │
│  └──────────────┘     └──────────────┘     └─────────────┘ │
│                                ▼                             │
│                       ┌──────────────┐                       │
│                       │  Backtesting │                       │
│                       │   & Analytics│                       │
│                       │              │                       │
│                       │  • Metrics   │                       │
│                       │  • Charts    │                       │
│                       └──────────────┘                       │
└─────────────────────────────────────────────────────────────┘
```

### Component Hierarchy

**Layer 1: Data Acquisition**
- `src/data/fetcher.py` - External API integration
- `src/data/cache.py` - Data persistence
- `src/data/preprocessor.py` - Data transformation

**Layer 2: Market Simulation**
- `src/market/orderbook.py` - Limit order book
- `src/market/matching_engine.py` - Order execution

**Layer 3: Intelligence**
- `src/ml/features.py` - Feature engineering
- `src/ml/models.py` - Predictive models

**Layer 4: Trading Logic**
- `src/strategies/momentum_strategy.py` - Signal generation
- `src/strategies/portfolio.py` - Position management
- `src/strategies/risk_manager.py` - Risk controls

**Layer 5: Evaluation**
- `src/backtest/backtester.py` - Event-driven simulation
- `src/analytics/metrics.py` - Performance measurement
- `src/analytics/visualizations.py` - Result visualization

---

## Installation & Setup

### System Requirements

- **Python**: 3.8 or higher
- **RAM**: Minimum 4GB, recommended 8GB+
- **Storage**: 500MB for dependencies, 1GB+ for data
- **OS**: Windows, macOS, or Linux

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/hft-sim.git
cd hft-sim
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Requirements Breakdown:**

```
# Core Data Processing
numpy>=1.24.0           # Numerical operations
pandas>=2.0.0           # Data structures
pyarrow>=12.0.0         # Fast serialization

# Machine Learning
torch>=2.0.0            # Deep learning framework
scikit-learn>=1.3.0     # ML utilities

# Technical Analysis
ta>=0.11.0              # Technical indicators

# Data Sources
requests>=2.31.0        # HTTP client
yfinance>=0.2.0         # Alternative data source

# Visualization
matplotlib>=3.7.0       # Plotting
seaborn>=0.12.0         # Statistical plots

# Testing
pytest>=7.4.0           # Test framework
```

### Step 4: Verify Installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import pandas; print(f'Pandas {pandas.__version__}')"
python tests/run_tests.py
```

Expected output:
```
PyTorch 2.1.0
Pandas 2.1.0
[SUCCESS] All tests passed!
```

---

## Configuration

### Directory Structure

```
hft-sim/
├── src/
│   ├── data/               # Data acquisition & processing
│   ├── market/             # Market simulation
│   ├── ml/                 # Machine learning
│   ├── strategies/         # Trading strategies
│   ├── backtest/           # Backtesting engine
│   └── analytics/          # Performance analytics
├── data/
│   └── cache/              # Cached market data
├── models/
│   └── saved/              # Trained ML models
├── plots/                  # Generated visualizations
├── tests/                  # Unit & integration tests
├── notebooks/              # Jupyter examples
├── requirements.txt        # Python dependencies
└── config.py               # Global configuration
```

### Configuration File (config.py)

```python
# Data Configuration
DATA_CONFIG = {
    'symbol': 'BTCUSDT',
    'interval': '1m',
    'lookback_days': 7,
    'cache_dir': 'data/cache'
}

# Model Configuration
MODEL_CONFIG = {
    'input_size': 90,
    'hidden_size': 128,
    'num_layers': 2,
    'dropout': 0.2,
    'learning_rate': 0.001,
    'batch_size': 64,
    'epochs': 50
}

# Strategy Configuration
STRATEGY_CONFIG = {
    'position_size': 0.1,           # 10% of capital per trade
    'confidence_threshold': 0.6,     # Minimum ML confidence
    'momentum_threshold': 0.02,      # 2% momentum trigger
    'volume_multiplier': 1.5         # Volume surge detection
}

# Risk Configuration
RISK_CONFIG = {
    'max_position_pct': 0.30,       # 30% max position size
    'stop_loss_pct': 0.02,          # 2% stop loss
    'max_drawdown_pct': 0.10,       # 10% max drawdown
    'var_confidence': 0.95           # 95% VaR confidence
}

# Backtest Configuration
BACKTEST_CONFIG = {
    'initial_capital': 100000,
    'commission': 0.001,             # 0.1% per trade
    'slippage': 0.0005               # 0.05% slippage
}
```

---

## API Reference

### Data Pipeline

#### DataFetcher

**Purpose**: Fetch historical market data from Binance API

```python
from src.data.fetcher import DataFetcher

fetcher = DataFetcher(symbol='BTCUSDT', interval='1m')
```

**Methods:**

**`fetch_historical(days: int) -> pd.DataFrame`**
- Fetches historical OHLCV data
- Parameters:
  - `days`: Number of days of historical data
- Returns: DataFrame with columns [timestamp, open, high, low, close, volume]
- Example:
```python
df = fetcher.fetch_historical(days=7)
print(f"Fetched {len(df)} candles")
```

**`fetch_realtime() -> dict`**
- Fetches current market price
- Returns: Dictionary with current price data
- Example:
```python
current = fetcher.fetch_realtime()
print(f"Current price: ${current['price']:.2f}")
```

---

#### DataCache

**Purpose**: Persist and retrieve market data efficiently

```python
from src.data.cache import DataCache

cache = DataCache(cache_dir='data/cache')
```

**Methods:**

**`save(data: pd.DataFrame, symbol: str, interval: str)`**
- Saves DataFrame to Parquet format with metadata
- Parameters:
  - `data`: Market data DataFrame
  - `symbol`: Trading symbol (e.g., 'BTCUSDT')
  - `interval`: Time interval (e.g., '1m')
- Example:
```python
cache.save(df, symbol='BTCUSDT', interval='1m')
```

**`load(symbol: str, interval: str) -> pd.DataFrame`**
- Loads cached data with validation
- Parameters:
  - `symbol`: Trading symbol
  - `interval`: Time interval
- Returns: DataFrame or None if cache miss
- Example:
```python
df = cache.load('BTCUSDT', '1m')
if df is not None:
    print(f"Loaded {len(df)} cached rows")
```

**`get_cache_info(symbol: str, interval: str) -> dict`**
- Returns cache metadata
- Returns: Dict with size, age, row count
- Example:
```python
info = cache.get_cache_info('BTCUSDT', '1m')
print(f"Cache size: {info['size_mb']:.2f} MB")
```

---

#### DataPreprocessor

**Purpose**: Add technical indicators and normalize data

```python
from src.data.preprocessor import DataPreprocessor

preprocessor = DataPreprocessor()
```

**Methods:**

**`add_technical_indicators(data: pd.DataFrame) -> pd.DataFrame`**
- Adds 29 technical indicators
- Indicators include:
  - Moving averages (SMA, EMA)
  - Momentum (RSI, MACD, Stochastic)
  - Volatility (ATR, Bollinger Bands)
  - Volume (OBV, MFI, VWAP)
- Example:
```python
df_with_indicators = preprocessor.add_technical_indicators(df)
print(f"Added indicators, now {len(df_with_indicators.columns)} columns")
```

**`normalize_data(data: pd.DataFrame) -> pd.DataFrame`**
- Z-score normalization for ML models
- Example:
```python
normalized = preprocessor.normalize_data(df)
```

---

### Market Simulation

#### OrderBook

**Purpose**: Simulate limit order book with price-time priority

```python
from src.market.orderbook import OrderBook

orderbook = OrderBook(symbol='BTCUSDT')
```

**Methods:**

**`add_order(side: str, price: float, quantity: float, order_id: str)`**
- Adds limit order to book
- Parameters:
  - `side`: 'bid' or 'ask'
  - `price`: Limit price
  - `quantity`: Order size
  - `order_id`: Unique identifier
- Example:
```python
orderbook.add_order('bid', 90000.0, 0.5, 'order_123')
```

**`cancel_order(order_id: str) -> bool`**
- Cancels existing order
- Returns: True if successful
- Example:
```python
cancelled = orderbook.cancel_order('order_123')
```

**`get_best_bid() -> float`**
- Returns highest bid price
- Example:
```python
best_bid = orderbook.get_best_bid()
```

**`get_best_ask() -> float`**
- Returns lowest ask price
- Example:
```python
best_ask = orderbook.get_best_ask()
```

**`get_mid_price() -> float`**
- Returns mid-market price
- Formula: (best_bid + best_ask) / 2
- Example:
```python
mid = orderbook.get_mid_price()
```

**`get_spread() -> float`**
- Returns bid-ask spread in basis points
- Formula: (ask - bid) / mid * 10000
- Example:
```python
spread_bps = orderbook.get_spread()
print(f"Spread: {spread_bps:.2f} bps")
```

**`get_depth(levels: int = 5) -> dict`**
- Returns market depth
- Parameters:
  - `levels`: Number of price levels
- Returns: Dict with bids and asks
- Example:
```python
depth = orderbook.get_depth(levels=5)
print(f"Total bid volume: {depth['total_bid_volume']}")
```

**`get_imbalance() -> float`**
- Returns order book imbalance
- Formula: (bid_volume - ask_volume) / (bid_volume + ask_volume)
- Range: -1 (all asks) to +1 (all bids)
- Example:
```python
imbalance = orderbook.get_imbalance()
```

---

#### MatchingEngine

**Purpose**: Execute orders with realistic fills, fees, and slippage

```python
from src.market.matching_engine import MatchingEngine

engine = MatchingEngine(orderbook, commission=0.001, slippage=0.0005)
```

**Methods:**

**`execute_market_order(side: str, quantity: float) -> dict`**
- Executes market order immediately
- Parameters:
  - `side`: 'buy' or 'sell'
  - `quantity`: Size to execute
- Returns: Dict with fill details
- Example:
```python
fill = engine.execute_market_order('buy', 0.1)
print(f"Filled {fill['quantity']} @ ${fill['price']:.2f}")
print(f"Fees: ${fill['commission']:.2f}")
```

**`execute_limit_order(side: str, price: float, quantity: float) -> dict`**
- Executes limit order if price available
- Parameters:
  - `side`: 'buy' or 'sell'
  - `price`: Limit price
  - `quantity`: Order size
- Returns: Dict with fill details or None
- Example:
```python
fill = engine.execute_limit_order('buy', 89000.0, 0.1)
if fill:
    print(f"Limit order filled")
else:
    print("Price not available, order pending")
```

**`get_execution_stats() -> dict`**
- Returns execution statistics
- Example:
```python
stats = engine.get_execution_stats()
print(f"Total fills: {stats['total_fills']}")
print(f"Avg fill price: ${stats['avg_fill_price']:.2f}")
```

---

### Machine Learning

#### FeatureEngineering

**Purpose**: Create ML features from market data

```python
from src.ml.features import FeatureEngineering

feature_eng = FeatureEngineering()
```

**Methods:**

**`create_features(data: pd.DataFrame) -> pd.DataFrame`**
- Creates 90 ML features across 6 categories
- Categories:
  1. Price Features (returns, log returns)
  2. Volume Features (volume changes, VWAP)
  3. Momentum Features (RSI, MACD, Stochastic)
  4. Volatility Features (ATR, Bollinger width)
  5. Order Book Features (spread, imbalance)
  6. Derived Features (interactions, polynomials)
- Example:
```python
features_df = feature_eng.create_features(df)
print(f"Created {len(features_df.columns)} features")
```

**`get_feature_importance(model, feature_names: list) -> pd.DataFrame`**
- Returns feature importance scores
- Example:
```python
importance = feature_eng.get_feature_importance(trained_model, feature_names)
print(importance.head(10))  # Top 10 features
```

---

#### LSTMPredictor

**Purpose**: PyTorch LSTM model for price direction prediction

```python
from src.ml.models import LSTMPredictor

model = LSTMPredictor(
    input_size=90,
    hidden_size=128,
    num_layers=2,
    dropout=0.2
)
```

**Methods:**

**`fit(X_train, y_train, X_val, y_val, epochs=50, batch_size=64, learning_rate=0.001)`**
- Trains model with early stopping
- Parameters:
  - `X_train`: Training features (numpy array)
  - `y_train`: Training labels
  - `X_val`: Validation features
  - `y_val`: Validation labels
  - `epochs`: Maximum training epochs
  - `batch_size`: Mini-batch size
  - `learning_rate`: Adam optimizer learning rate
- Example:
```python
history = model.fit(
    X_train, y_train,
    X_val, y_val,
    epochs=50,
    batch_size=64
)
print(f"Best validation accuracy: {max(history['val_accuracy']):.2%}")
```

**`predict(X: np.ndarray) -> np.ndarray`**
- Generates predictions
- Returns: Probabilities for each class
- Example:
```python
predictions = model.predict(X_test)
# predictions shape: (n_samples, 3) for [down, neutral, up]
```

**`predict_proba(X: np.ndarray) -> np.ndarray`**
- Returns class probabilities
- Example:
```python
probabilities = model.predict_proba(X_test)
confidence = probabilities.max(axis=1)
```

**`save_model(path: str)`**
- Saves model weights and architecture
- Example:
```python
model.save_model('models/saved/lstm_v1.pth')
```

**`load_model(path: str)`**
- Loads saved model
- Example:
```python
model.load_model('models/saved/lstm_v1.pth')
```

---

### Trading Strategies

#### MomentumStrategy

**Purpose**: Generate trading signals using ML + momentum + volume

```python
from src.strategies.momentum_strategy import MomentumStrategy

strategy = MomentumStrategy(
    model=trained_lstm,
    confidence_threshold=0.6,
    momentum_threshold=0.02,
    volume_multiplier=1.5
)
```

**Methods:**

**`generate_signal(data: pd.DataFrame, current_index: int) -> dict`**
- Generates trading signal for current bar
- Returns: Dict with signal, confidence, reasoning
- Signal types: 'BUY', 'SELL', 'HOLD'
- Example:
```python
signal = strategy.generate_signal(df, current_index=100)
print(f"Signal: {signal['action']}")
print(f"Confidence: {signal['confidence']:.2%}")
print(f"Reasoning: {signal['reasoning']}")
```

**Signal Logic:**
1. ML model predicts price direction
2. Check momentum (% change vs threshold)
3. Check volume surge (vs moving average)
4. Combine signals with weighted confidence
5. Apply confidence threshold filter

---

#### Portfolio

**Purpose**: Track positions, trades, and PnL

```python
from src.strategies.portfolio import Portfolio

portfolio = Portfolio(initial_capital=100000)
```

**Methods:**

**`execute_trade(symbol: str, side: str, quantity: float, price: float, commission: float)`**
- Records trade execution
- Parameters:
  - `symbol`: Trading symbol
  - `side`: 'buy' or 'sell'
  - `quantity`: Trade size
  - `price`: Execution price
  - `commission`: Transaction fees
- Example:
```python
portfolio.execute_trade(
    symbol='BTCUSDT',
    side='buy',
    quantity=0.5,
    price=90000.0,
    commission=45.0
)
```

**`get_position(symbol: str) -> dict`**
- Returns current position details
- Example:
```python
position = portfolio.get_position('BTCUSDT')
print(f"Size: {position['quantity']}")
print(f"Avg entry: ${position['avg_entry_price']:.2f}")
print(f"Unrealized PnL: ${position['unrealized_pnl']:.2f}")
```

**`update_market_price(symbol: str, current_price: float)`**
- Updates mark-to-market valuation
- Example:
```python
portfolio.update_market_price('BTCUSDT', 91000.0)
```

**`get_total_value() -> float`**
- Returns total portfolio value
- Formula: cash + sum(position_values)
- Example:
```python
total = portfolio.get_total_value()
print(f"Portfolio value: ${total:,.2f}")
```

**`get_pnl() -> dict`**
- Returns realized and unrealized PnL
- Example:
```python
pnl = portfolio.get_pnl()
print(f"Realized: ${pnl['realized']:.2f}")
print(f"Unrealized: ${pnl['unrealized']:.2f}")
print(f"Total: ${pnl['total']:.2f}")
```

**`get_trades() -> list`**
- Returns list of all executed trades
- Example:
```python
trades = portfolio.get_trades()
print(f"Total trades: {len(trades)}")
```

---

#### RiskManager

**Purpose**: Enforce risk limits and calculate risk metrics

```python
from src.strategies.risk_manager import RiskManager

risk_mgr = RiskManager(
    max_position_pct=0.30,
    stop_loss_pct=0.02,
    max_drawdown_pct=0.10
)
```

**Methods:**

**`check_position_limit(symbol: str, quantity: float, price: float, capital: float) -> dict`**
- Validates if trade within position limits
- Returns: Dict with allowed status and max allowed
- Example:
```python
check = risk_mgr.check_position_limit(
    symbol='BTCUSDT',
    quantity=0.5,
    price=90000.0,
    capital=100000.0
)
if not check['allowed']:
    print(f"Position too large! Max: {check['max_quantity']}")
```

**`check_stop_loss(entry_price: float, current_price: float, side: str) -> bool`**
- Checks if stop loss triggered
- Returns: True if stop loss hit
- Example:
```python
triggered = risk_mgr.check_stop_loss(
    entry_price=90000.0,
    current_price=88200.0,  # -2% = stop loss
    side='long'
)
if triggered:
    print("Stop loss triggered! Close position")
```

**`calculate_var(returns: np.ndarray, confidence: float = 0.95) -> float`**
- Calculates Value at Risk
- Parameters:
  - `returns`: Array of historical returns
  - `confidence`: Confidence level (0.95 = 95%)
- Returns: VaR as percentage
- Example:
```python
var_95 = risk_mgr.calculate_var(portfolio_returns, confidence=0.95)
print(f"95% VaR: {var_95:.2%}")
```

**`get_risk_metrics(portfolio: Portfolio) -> dict`**
- Returns comprehensive risk metrics
- Metrics include:
  - Position concentration
  - Leverage ratio
  - Current drawdown
  - VaR
- Example:
```python
metrics = risk_mgr.get_risk_metrics(portfolio)
print(f"Position concentration: {metrics['concentration']:.1%}")
print(f"Current drawdown: {metrics['drawdown']:.2%}")
```

---

### Backtesting

#### Backtester

**Purpose**: Event-driven backtesting with realistic execution

```python
from src.backtest.backtester import Backtester

backtester = Backtester(
    strategy=momentum_strategy,
    orderbook=orderbook,
    matching_engine=engine,
    portfolio=portfolio,
    risk_manager=risk_mgr,
    initial_capital=100000
)
```

**Methods:**

**`run(data: pd.DataFrame) -> dict`**
- Runs complete backtest
- Process:
  1. Iterate through historical data bar-by-bar
  2. Update order book state
  3. Generate trading signal
  4. Check risk limits
  5. Execute trades if allowed
  6. Update portfolio
  7. Record metrics
- Returns: Dict with results and statistics
- Example:
```python
results = backtester.run(historical_data)
print(f"Final capital: ${results['final_capital']:,.2f}")
print(f"Total return: {results['total_return']:.2%}")
print(f"Number of trades: {results['num_trades']}")
```

**`get_equity_curve() -> pd.Series`**
- Returns portfolio value over time
- Example:
```python
equity = backtester.get_equity_curve()
equity.plot(title='Equity Curve')
```

**`get_trade_log() -> pd.DataFrame`**
- Returns detailed trade history
- Columns: timestamp, symbol, side, quantity, price, pnl
- Example:
```python
trades = backtester.get_trade_log()
print(trades.head())
```

---

### Analytics

#### MetricsCalculator

**Purpose**: Calculate 40+ performance metrics

```python
from src.analytics.metrics import MetricsCalculator

calculator = MetricsCalculator()
```

**Methods:**

**`calculate_all_metrics(returns: pd.Series, trades: list, equity: pd.Series) -> dict`**
- Calculates comprehensive metrics
- Returns: Dict with 40+ metrics
- Example:
```python
metrics = calculator.calculate_all_metrics(
    returns=portfolio_returns,
    trades=trade_history,
    equity=equity_curve
)

# Access specific metrics
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
print(f"Win Rate: {metrics['win_rate']:.1%}")
```

**Metric Categories:**

1. **Return Metrics**
   - Total return
   - Annualized return
   - Monthly/daily returns
   - Cumulative return

2. **Risk-Adjusted Metrics**
   - Sharpe ratio (excess return / volatility)
   - Sortino ratio (excess return / downside deviation)
   - Calmar ratio (return / max drawdown)
   - Information ratio

3. **Risk Metrics**
   - Volatility (annualized)
   - Maximum drawdown
   - Value at Risk (95%, 99%)
   - Conditional VaR (CVaR)

4. **Trade Metrics**
   - Win rate (% profitable trades)
   - Profit factor (gross profit / gross loss)
   - Average win / average loss
   - Expectancy per trade

5. **Advanced Metrics**
   - Omega ratio
   - Skewness
   - Kurtosis
   - Beta (if benchmark provided)

**`calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float`**
- Sharpe Ratio = (Mean Return - Risk Free) / Std Dev
- Example:
```python
sharpe = calculator.calculate_sharpe_ratio(returns)
```

**`calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float`**
- Sortino Ratio = (Mean Return - Risk Free) / Downside Deviation
- Only considers downside volatility
- Example:
```python
sortino = calculator.calculate_sortino_ratio(returns)
```

**`calculate_max_drawdown(equity: pd.Series) -> dict`**
- Returns max drawdown and recovery info
- Example:
```python
dd = calculator.calculate_max_drawdown(equity)
print(f"Max drawdown: {dd['drawdown']:.2%}")
print(f"Recovery days: {dd['recovery_days']}")
```

---

#### Visualizations

**Purpose**: Generate publication-quality charts

```python
from src.analytics.visualizations import Visualizations

viz = Visualizations(output_dir='plots')
```

**Methods:**

**`plot_equity_curve(equity: pd.Series, trades: list = None, title: str = 'Equity Curve')`**
- Plots portfolio value over time with trade markers
- Example:
```python
viz.plot_equity_curve(
    equity=equity_curve,
    trades=trade_list,
    title='HFT Strategy Performance'
)
# Saves to plots/equity_curve.png
```

**`plot_drawdown(equity: pd.Series, title: str = 'Drawdown Chart')`**
- Plots underwater equity curve
- Shows drawdown periods
- Example:
```python
viz.plot_drawdown(equity_curve)
# Saves to plots/drawdown.png
```

**`plot_returns_distribution(returns: pd.Series, title: str = 'Returns Distribution')`**
- Histogram with normal curve overlay
- Example:
```python
viz.plot_returns_distribution(daily_returns)
# Saves to plots/returns_dist.png
```

**`plot_monthly_returns(returns: pd.Series, title: str = 'Monthly Returns Heatmap')`**
- Heatmap of monthly performance
- Example:
```python
viz.plot_monthly_returns(daily_returns)
# Saves to plots/monthly_returns.png
```

**`create_dashboard(equity: pd.Series, returns: pd.Series, trades: list, metrics: dict)`**
- Creates comprehensive 4-panel dashboard
- Panels: equity, drawdown, returns, metrics table
- Example:
```python
viz.create_dashboard(
    equity=equity_curve,
    returns=returns,
    trades=trades,
    metrics=performance_metrics
)
# Saves to plots/dashboard.png
```

---

## Usage Examples

### Example 1: Basic Backtest

```python
# 1. Fetch data
from src.data.fetcher import DataFetcher
from src.data.preprocessor import DataPreprocessor

fetcher = DataFetcher('BTCUSDT', '1m')
df = fetcher.fetch_historical(days=7)

preprocessor = DataPreprocessor()
df = preprocessor.add_technical_indicators(df)

# 2. Create features and train model
from src.ml.features import FeatureEngineering
from src.ml.models import LSTMPredictor

feature_eng = FeatureEngineering()
features_df = feature_eng.create_features(df)

# Prepare training data
X = features_df.values
y = (df['close'].pct_change().shift(-1) > 0).astype(int).values

# Train/validation split
split = int(0.8 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

model = LSTMPredictor(input_size=90, hidden_size=128)
model.fit(X_train, y_train, X_val, y_val, epochs=50)

# 3. Setup trading components
from src.market.orderbook import OrderBook
from src.market.matching_engine import MatchingEngine
from src.strategies.momentum_strategy import MomentumStrategy
from src.strategies.portfolio import Portfolio
from src.strategies.risk_manager import RiskManager

orderbook = OrderBook('BTCUSDT')
engine = MatchingEngine(orderbook, commission=0.001)
portfolio = Portfolio(initial_capital=100000)
risk_mgr = RiskManager()
strategy = MomentumStrategy(model)

# 4. Run backtest
from src.backtest.backtester import Backtester

backtester = Backtester(
    strategy=strategy,
    orderbook=orderbook,
    matching_engine=engine,
    portfolio=portfolio,
    risk_manager=risk_mgr
)

results = backtester.run(df)

# 5. Analyze results
from src.analytics.metrics import MetricsCalculator
from src.analytics.visualizations import Visualizations

calculator = MetricsCalculator()
metrics = calculator.calculate_all_metrics(
    returns=results['returns'],
    trades=results['trades'],
    equity=results['equity']
)

print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
print(f"Win Rate: {metrics['win_rate']:.1%}")

# 6. Visualize
viz = Visualizations()
viz.create_dashboard(
    equity=results['equity'],
    returns=results['returns'],
    trades=results['trades'],
    metrics=metrics
)
```

### Example 2: Walk-Forward Analysis

```python
# Walk-forward optimization to avoid overfitting
import pandas as pd
from datetime import datetime, timedelta

def walk_forward_backtest(data, train_window=30, test_window=7):
    """
    Rolling window backtest
    - Train model on N days
    - Test on next M days
    - Roll forward and repeat
    """
    results = []
    
    for i in range(0, len(data) - train_window - test_window, test_window):
        # Split data
        train_data = data.iloc[i:i+train_window]
        test_data = data.iloc[i+train_window:i+train_window+test_window]
        
        # Train model
        features = feature_eng.create_features(train_data)
        X_train = features.values
        y_train = (train_data['close'].pct_change().shift(-1) > 0).astype(int).values
        
        model = LSTMPredictor(input_size=90, hidden_size=128)
        model.fit(X_train, y_train, X_train, y_train, epochs=20)
        
        # Test model
        strategy = MomentumStrategy(model)
        backtester = Backtester(strategy, orderbook, engine, portfolio, risk_mgr)
        test_results = backtester.run(test_data)
        
        results.append({
            'period': test_data.index[0],
            'return': test_results['total_return'],
            'sharpe': test_results['sharpe_ratio'],
            'trades': len(test_results['trades'])
        })
    
    return pd.DataFrame(results)

# Run walk-forward
wf_results = walk_forward_backtest(df, train_window=30, test_window=7)
print(wf_results)
```

### Example 3: Parameter Optimization

```python
# Grid search for optimal strategy parameters
from itertools import product

def optimize_parameters(data, param_grid):
    """
    Test multiple parameter combinations
    """
    results = []
    
    for conf_thresh, mom_thresh, vol_mult in product(*param_grid.values()):
        strategy = MomentumStrategy(
            model=trained_model,
            confidence_threshold=conf_thresh,
            momentum_threshold=mom_thresh,
            volume_multiplier=vol_mult
        )
        
        backtester = Backtester(strategy, orderbook, engine, portfolio, risk_mgr)
        backtest_results = backtester.run(data)
        
        results.append({
            'conf_threshold': conf_thresh,
            'mom_threshold': mom_thresh,
            'vol_multiplier': vol_mult,
            'sharpe': backtest_results['sharpe_ratio'],
            'return': backtest_results['total_return'],
            'max_dd': backtest_results['max_drawdown']
        })
    
    return pd.DataFrame(results).sort_values('sharpe', ascending=False)

# Define parameter grid
param_grid = {
    'conf_threshold': [0.55, 0.60, 0.65],
    'mom_threshold': [0.01, 0.02, 0.03],
    'vol_multiplier': [1.3, 1.5, 1.7]
}

# Run optimization
opt_results = optimize_parameters(df, param_grid)
print("Top 5 parameter sets:")
print(opt_results.head())
```

---

## Testing

### Running Tests

```bash
# Run all tests
python tests/run_tests.py

# Run specific test file
pytest tests/test_orderbook.py -v

# Run with coverage
pytest --cov=src tests/
```

### Test Structure

```
tests/
├── run_tests.py              # Test runner
├── test_orderbook.py         # OrderBook tests
├── test_matching_engine.py   # Execution tests
├── test_portfolio.py         # Portfolio tests
├── test_risk_manager.py      # Risk tests
└── test_integration.py       # End-to-end tests
```

### Writing Custom Tests

```python
import pytest
from src.market.orderbook import OrderBook

def test_orderbook_add_order():
    """Test adding order to book"""
    ob = OrderBook('BTCUSDT')
    ob.add_order('bid', 90000.0, 0.5, 'order_1')
    
    assert ob.get_best_bid() == 90000.0
    assert ob.get_total_bid_volume() == 0.5

def test_orderbook_matching():
    """Test order matching logic"""
    ob = OrderBook('BTCUSDT')
    ob.add_order('bid', 90000.0, 0.5, 'bid_1')
    ob.add_order('ask', 90010.0, 0.3, 'ask_1')
    
    # Market buy should match with best ask
    fill = ob.match_market_order('buy', 0.3)
    assert fill['price'] == 90010.0
    assert fill['quantity'] == 0.3
```

---

## Deployment

### Local Deployment

**For development and testing:**

```bash
# Activate environment
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Run backtest
python examples/run_backtest.py

# Start Jupyter notebook
jupyter notebook notebooks/example_workflow.ipynb
```

### Docker Deployment

**Dockerfile:**

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "examples/run_backtest.py"]
```

**Build and run:**

```bash
docker build -t hft-simulator .
docker run -v $(pwd)/data:/app/data -v $(pwd)/plots:/app/plots hft-simulator
```

### Cloud Deployment (AWS)

```bash
# Package application
zip -r hft-sim.zip . -x "*.git*" -x "*venv*" -x "*__pycache__*"

# Upload to S3
aws s3 cp hft-sim.zip s3://your-bucket/

# Deploy to EC2 or Lambda for scheduled backtests
```

---

## Troubleshooting

### Common Issues

#### Issue 1: Import Errors

**Symptom:**
```
ModuleNotFoundError: No module named 'src'
```

**Solution:**
```bash
# Ensure you're in project root
cd hft-sim

# Add project to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # Linux/Mac
set PYTHONPATH=%PYTHONPATH%;%CD%          # Windows
```

#### Issue 2: Data Fetch Failures

**Symptom:**
```
requests.exceptions.ConnectionError: Failed to fetch data
```

**Solution:**
1. Check internet connection
2. Verify Binance API is accessible
3. Reduce data request size
4. Add retry logic with exponential backoff

```python
import time

def fetch_with_retry(fetcher, days, max_retries=3):
    for attempt in range(max_retries):
        try:
            return fetcher.fetch_historical(days)
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise
```

#### Issue 3: Out of Memory

**Symptom:**
```
MemoryError: Unable to allocate array
```

**Solution:**
1. Reduce data size (fewer days)
2. Process in chunks
3. Use data generators instead of loading all at once

```python
def process_in_chunks(data, chunk_size=1000):
    for i in range(0, len(data), chunk_size):
        chunk = data.iloc[i:i+chunk_size]
        yield process_chunk(chunk)
```

#### Issue 4: Slow Training

**Symptom:**
Model training takes too long

**Solution:**
1. Use GPU if available
2. Reduce model size (fewer layers/units)
3. Use smaller batch size
4. Enable early stopping

```python
# Check PyTorch CUDA availability
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Move model to GPU
model = LSTMPredictor(...).to(device)
```

#### Issue 5: Poor Model Performance

**Symptom:**
Low accuracy, negative Sharpe ratio

**Solution:**
1. Check data quality (missing values, outliers)
2. Increase training data
3. Add more features
4. Tune hyperparameters
5. Use walk-forward validation

```python
# Data quality check
print(df.isnull().sum())  # Check missing values
print(df.describe())       # Check for outliers

# Feature importance analysis
importance = feature_eng.get_feature_importance(model, feature_names)
print("Top features:", importance.head(10))
```

---

## Performance Considerations

### Optimization Techniques

#### 1. Vectorization

**Avoid:**
```python
# Slow: Loop over DataFrame
for i in range(len(df)):
    df.loc[i, 'sma'] = df['close'].iloc[i-20:i].mean()
```

**Use:**
```python
# Fast: Vectorized operation
df['sma'] = df['close'].rolling(20).mean()
```

#### 2. Efficient Data Structures

**Use NumPy arrays for numerical operations:**
```python
# Convert to numpy for faster computations
returns = df['close'].pct_change().values
sharpe = returns.mean() / returns.std() * np.sqrt(252)
```

#### 3. Caching

**Cache expensive computations:**
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def calculate_indicators(symbol, start_date, end_date):
    # Expensive computation
    return result
```

#### 4. Parallel Processing

**Use multiprocessing for backtests:**
```python
from multiprocessing import Pool

def run_backtest_for_params(params):
    # Run backtest with specific parameters
    return results

# Parallel parameter optimization
with Pool(processes=4) as pool:
    results = pool.map(run_backtest_for_params, param_list)
```

### Performance Benchmarks

**Target Performance:**
- Data fetch: < 5 seconds for 7 days
- Feature engineering: < 2 seconds for 10k bars
- Model training: < 60 seconds per epoch
- Backtest: > 1000 bars/second
- Order matching: < 1ms per order

**Monitoring:**
```python
import time

def benchmark_function(func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - start
    print(f"{func.__name__} took {elapsed:.3f} seconds")
    return result

# Usage
df = benchmark_function(fetcher.fetch_historical, days=7)
```

---

## Best Practices

### 1. Data Management
- Always cache fetched data
- Validate data quality before processing
- Use Parquet format for large datasets
- Version control your data pipeline

### 2. Model Development
- Use walk-forward validation
- Implement early stopping
- Save model checkpoints
- Log training metrics
- Version control model configurations

### 3. Risk Management
- Always enforce position limits
- Use stop losses
- Monitor drawdowns in real-time
- Calculate VaR regularly
- Implement circuit breakers

### 4. Backtesting
- Use event-driven architecture
- Model realistic execution (slippage, fees)
- Test multiple time periods
- Perform sensitivity analysis
- Document assumptions

### 5. Code Quality
- Write unit tests for all modules
- Use type hints
- Follow PEP 8 style guide
- Document all functions
- Use version control (git)

---

## Further Reading

### Books
- "Advances in Financial Machine Learning" by Marcos López de Prado
- "Algorithmic Trading" by Ernest P. Chan
- "Quantitative Trading" by Ernest P. Chan

### Papers
- "Market Microstructure in Practice" by Lehalle & Laruelle
- "The Profitability of Technical Trading Rules" by Brock et al.

### Online Resources
- QuantStart: https://www.quantstart.com
- Quantopian Lectures: https://www.quantopian.com/lectures
- ArXiv Quantitative Finance: https://arxiv.org/list/q-fin/recent

---

## Support & Contributing

### Getting Help
- GitHub Issues: Report bugs and request features
- Discussions: Ask questions and share ideas
- Documentation: Check this guide first

### Contributing
1. Fork the repository
2. Create feature branch
3. Write tests for new features
4. Submit pull request
5. Ensure all tests pass

### Code Standards
- Follow PEP 8
- Write docstrings for all functions
- Add type hints
- Include unit tests
- Update documentation

---

**Version:** 1.0.0
**Last Updated:** 2026-01-09
**License:** MIT