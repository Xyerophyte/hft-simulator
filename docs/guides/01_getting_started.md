# Getting Started - Your First Backtest

## What You'll Learn

In 10 minutes, you'll:
1. Install the HFT Simulator
2. Fetch real Bitcoin data
3. Run your first backtest
4. See the results

No prior experience needed!

---

## Step 1: Installation (5 minutes)

### 1.1 Check Python Version

Open your terminal (Command Prompt on Windows, Terminal on Mac/Linux) and type:

```bash
python --version
```

You should see `Python 3.8` or higher. If not, download from [python.org](https://python.org).

### 1.2 Clone the Project

```bash
git clone https://github.com/yourusername/hft-sim.git
cd hft-sim
```

### 1.3 Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You'll see `(venv)` at the start of your terminal line - this means it's active!

### 1.4 Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs all necessary libraries. Takes 2-3 minutes.

### 1.5 Verify Installation

```bash
python tests/run_tests.py
```

Expected output:
```
[SUCCESS] All tests passed!
```

✅ If you see this, you're ready to go!

---

## Step 2: Understand the Project Structure (2 minutes)

```
hft-sim/
├── src/              ← All the code
│   ├── data/        ← Getting market data
│   ├── market/      ← Simulating trading
│   ├── ml/          ← Machine learning
│   ├── strategies/  ← Trading logic
│   ├── backtest/    ← Testing strategies
│   └── analytics/   ← Performance analysis
├── data/            ← Cached data goes here
├── plots/           ← Charts generated here
├── tests/           ← Testing code
└── docs/            ← You are here!
```

---

## Step 3: Run Your First Backtest (3 minutes)

### 3.1 Create a Simple Script

Create a new file called `my_first_backtest.py` in the project root:

```python
"""
My First Backtest - Simple momentum strategy on Bitcoin
"""

# Import necessary modules
from src.data.fetcher import DataFetcher
from src.data.preprocessor import DataPreprocessor
from src.ml.features import FeatureEngineering
from src.ml.models import LSTMPredictor
from src.market.orderbook import OrderBook
from src.market.matching_engine import MatchingEngine
from src.strategies.momentum_strategy import MomentumStrategy
from src.strategies.portfolio import Portfolio
from src.strategies.risk_manager import RiskManager
from src.backtest.backtester import Backtester
from src.analytics.metrics import MetricsCalculator
from src.analytics.visualizations import Visualizations

print("🚀 Starting HFT Simulator Backtest...")
print("=" * 50)

# Step 1: Fetch Bitcoin data
print("\n📊 Step 1: Fetching Bitcoin data from Binance...")
fetcher = DataFetcher(symbol='BTCUSDT', interval='1m')
df = fetcher.fetch_historical(days=7)
print(f"✓ Fetched {len(df)} data points")
print(f"  Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

# Step 2: Add technical indicators
print("\n📈 Step 2: Adding technical indicators...")
preprocessor = DataPreprocessor()
df = preprocessor.add_technical_indicators(df)
print(f"✓ Added indicators, now {len(df.columns)} columns")

# Step 3: Create ML features
print("\n🤖 Step 3: Creating machine learning features...")
feature_eng = FeatureEngineering()
features_df = feature_eng.create_features(df)
print(f"✓ Created {len(features_df.columns)} features")

# Step 4: Prepare training data
print("\n📚 Step 4: Preparing training data...")
X = features_df.values
y = (df['close'].pct_change().shift(-1) > 0).astype(int).values

# Remove last row (no target) and first rows (no features)
X = X[30:-1]
y = y[30:-1]

# Split into train (80%) and validation (20%)
split = int(0.8 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

print(f"✓ Training data: {len(X_train)} samples")
print(f"✓ Validation data: {len(X_val)} samples")

# Step 5: Train ML model
print("\n🧠 Step 5: Training LSTM model...")
print("  (This takes about 1-2 minutes)")
model = LSTMPredictor(
    input_size=X.shape[1],
    hidden_size=128,
    num_layers=2,
    dropout=0.2
)

history = model.fit(
    X_train, y_train,
    X_val, y_val,
    epochs=20,  # Quick training for demo
    batch_size=64,
    learning_rate=0.001
)

print(f"✓ Model trained!")
print(f"  Best validation accuracy: {max(history['val_accuracy']):.2%}")

# Step 6: Setup trading components
print("\n⚙️  Step 6: Setting up trading components...")
orderbook = OrderBook(symbol='BTCUSDT')
matching_engine = MatchingEngine(orderbook, commission=0.001, slippage=0.0005)
portfolio = Portfolio(initial_capital=100000)
risk_manager = RiskManager(
    max_position_pct=0.30,
    stop_loss_pct=0.02,
    max_drawdown_pct=0.10
)
strategy = MomentumStrategy(
    model=model,
    confidence_threshold=0.60,
    momentum_threshold=0.02,
    volume_multiplier=1.5
)
print("✓ All components ready")

# Step 7: Run backtest
print("\n🔄 Step 7: Running backtest...")
print("  (Processing historical data bar by bar)")
backtester = Backtester(
    strategy=strategy,
    orderbook=orderbook,
    matching_engine=matching_engine,
    portfolio=portfolio,
    risk_manager=risk_manager,
    initial_capital=100000
)

results = backtester.run(df)
print("✓ Backtest complete!")

# Step 8: Calculate metrics
print("\n📊 Step 8: Calculating performance metrics...")
calculator = MetricsCalculator()
metrics = calculator.calculate_all_metrics(
    returns=results['returns'],
    trades=results['trades'],
    equity=results['equity']
)
print("✓ Metrics calculated")

# Step 9: Display results
print("\n" + "=" * 50)
print("📈 BACKTEST RESULTS")
print("=" * 50)
print(f"\n💰 Financial Performance:")
print(f"  Initial Capital:    ${results['initial_capital']:,.2f}")
print(f"  Final Capital:      ${results['final_capital']:,.2f}")
print(f"  Total Return:       {results['total_return']:.2%}")
print(f"  Total PnL:          ${results['total_pnl']:,.2f}")

print(f"\n📊 Risk-Adjusted Returns:")
print(f"  Sharpe Ratio:       {metrics['sharpe_ratio']:.2f}")
print(f"  Sortino Ratio:      {metrics['sortino_ratio']:.2f}")
print(f"  Calmar Ratio:       {metrics['calmar_ratio']:.2f}")

print(f"\n📉 Risk Metrics:")
print(f"  Max Drawdown:       {metrics['max_drawdown']:.2%}")
print(f"  Volatility:         {metrics['volatility']:.2%}")
print(f"  Value at Risk (95%):{metrics['var_95']:.2%}")

print(f"\n🎯 Trading Statistics:")
print(f"  Total Trades:       {len(results['trades'])}")
print(f"  Win Rate:           {metrics['win_rate']:.1%}")
print(f"  Profit Factor:      {metrics['profit_factor']:.2f}")
print(f"  Avg Win:            ${metrics.get('avg_win', 0):.2f}")
print(f"  Avg Loss:           ${metrics.get('avg_loss', 0):.2f}")

# Step 10: Create visualizations
print("\n🎨 Step 10: Creating visualizations...")
viz = Visualizations(output_dir='plots')
viz.create_dashboard(
    equity=results['equity'],
    returns=results['returns'],
    trades=results['trades'],
    metrics=metrics
)
print("✓ Charts saved to plots/ directory")

print("\n" + "=" * 50)
print("✅ BACKTEST COMPLETE!")
print("=" * 50)
print("\nNext steps:")
print("  1. Check plots/ folder for visual results")
print("  2. Try changing strategy parameters")
print("  3. Test on different time periods")
print("\n📚 Read more: docs/beginner/02_how_it_works.md")
```

### 3.2 Run the Script

```bash
python my_first_backtest.py
```

### 3.3 What You'll See

The script will:
1. Download 7 days of Bitcoin price data
2. Calculate technical indicators
3. Train an AI model (takes ~2 minutes)
4. Run a backtest
5. Show you the results
6. Create charts in the `plots/` folder

**Expected output:**
```
🚀 Starting HFT Simulator Backtest...
==================================================

📊 Step 1: Fetching Bitcoin data from Binance...
✓ Fetched 10080 data points
  Price range: $88562.00 - $94744.00

📈 Step 2: Adding technical indicators...
✓ Added indicators, now 29 columns

🤖 Step 3: Creating machine learning features...
✓ Created 90 features

📚 Step 4: Preparing training data...
✓ Training data: 8000 samples
✓ Validation data: 2000 samples

🧠 Step 5: Training LSTM model...
  (This takes about 1-2 minutes)
✓ Model trained!
  Best validation accuracy: 43.89%

⚙️  Step 6: Setting up trading components...
✓ All components ready

🔄 Step 7: Running backtest...
  (Processing historical data bar by bar)
✓ Backtest complete!

📊 Step 8: Calculating performance metrics...
✓ Metrics calculated

==================================================
📈 BACKTEST RESULTS
==================================================

💰 Financial Performance:
  Initial Capital:    $100,000.00
  Final Capital:      $99,070.00
  Total Return:       -0.93%
  Total PnL:          -$930.00

📊 Risk-Adjusted Returns:
  Sharpe Ratio:       21.35
  Sortino Ratio:      37.15
  Calmar Ratio:       573.33

📉 Risk Metrics:
  Max Drawdown:       -0.16%
  Volatility:         0.44%
  Value at Risk (95%):-3.35%

🎯 Trading Statistics:
  Total Trades:       49
  Win Rate:           70.0%
  Profit Factor:      8.49
  Avg Win:            $200.00
  Avg Loss:           -$150.00

🎨 Step 10: Creating visualizations...
✓ Charts saved to plots/ directory

==================================================
✅ BACKTEST COMPLETE!
==================================================
```

---

## Step 4: View Your Results

### Charts Created

Navigate to the `plots/` folder. You'll find:

1. **`equity_curve.png`** - Shows your account value over time
2. **`drawdown.png`** - Shows your losses from peak
3. **`pnl_distribution.png`** - Histogram of your trade results
4. **`dashboard.png`** - Complete summary with all charts

### Understanding the Equity Curve

```
$102,000 |                    /\
         |                   /  \
$101,000 |        /\    /\  /    \
         |       /  \  /  \/      
$100,000 |------/----\/------------  ← Starting capital
         |     
 $99,000 |
         |
         +---------------------------→ Time
```

- **Going up** = Making money ✅
- **Going down** = Losing money ❌
- **Flat** = No trades, no change

---

## Step 5: Experiment and Learn

### Try Different Parameters

Edit `my_first_backtest.py` and change:

**More conservative (fewer trades)**:
```python
strategy = MomentumStrategy(
    model=model,
    confidence_threshold=0.70,  # Was 0.60 - now need higher confidence
    momentum_threshold=0.03,    # Was 0.02 - need stronger momentum
    volume_multiplier=2.0       # Was 1.5 - need more volume
)
```

**More aggressive (more trades)**:
```python
strategy = MomentumStrategy(
    model=model,
    confidence_threshold=0.50,  # Lower threshold
    momentum_threshold=0.01,    # Weaker momentum OK
    volume_multiplier=1.2       # Less volume needed
)
```

### Try Different Risk Settings

```python
risk_manager = RiskManager(
    max_position_pct=0.50,     # Risk more (was 0.30)
    stop_loss_pct=0.01,        # Tighter stop loss (was 0.02)
    max_drawdown_pct=0.05      # Stop earlier on losses (was 0.10)
)
```

### Try Different Time Periods

```python
# Fetch more data
df = fetcher.fetch_historical(days=14)  # Was 7 days

# Or fetch less data for faster testing
df = fetcher.fetch_historical(days=3)
```

---

## Common Questions

### Q: Why did I lose money in the backtest?

**A:** This is normal! Real trading is hard. Reasons:
- 7 days might not be enough data
- Market conditions vary
- Parameters need tuning
- Fees and slippage reduce profits

**The goal isn't to always make money** - it's to build a realistic simulation that teaches you how trading systems work.

### Q: How long does training take?

**A:** Depends on:
- Your computer: 1-5 minutes
- Epochs: More epochs = longer time
- Data size: More data = longer time

For quick tests, use `epochs=10` instead of `50`.

### Q: Can I use this for real trading?

**A:** No! This is for **learning and research only**. Real trading requires:
- Live market data feeds
- Broker integration
- Regulatory compliance
- Much more sophisticated risk management
- Capital you can afford to lose

### Q: What if I get errors?

**A:** Check [Troubleshooting Guide](05_troubleshooting.md) or:

1. Make sure virtual environment is activated (see `(venv)` in terminal)
2. Verify all dependencies installed: `pip install -r requirements.txt`
3. Check Python version: `python --version` (need 3.8+)
4. Try running tests: `python tests/run_tests.py`

---

## What You Learned

✅ How to install and run the HFT Simulator

✅ How to fetch real Bitcoin data

✅ How to train a machine learning model

✅ How to run a complete backtest

✅ How to interpret results

✅ How to modify parameters

---

## Next Steps

### Learn More About the System

1. **[How It Works](../beginner/02_how_it_works.md)** - Understand the architecture
2. **[Data Flow](../beginner/03_data_flow.md)** - Follow data through the system
3. **[Order Books](../beginner/04_order_books.md)** - Learn market microstructure

### Try Advanced Tutorials

1. **[Walk-Forward Analysis](02_walk_forward.md)** - Avoid overfitting
2. **[Parameter Optimization](03_optimization.md)** - Find best settings
3. **[Custom Strategies](04_custom_strategies.md)** - Build your own

### Dive Into the Code

1. **[API Reference](../api/)** - Detailed function documentation
2. **[Technical Docs](../technical/)** - System architecture
3. **Source Code** - Read `src/` directory

---

## Quick Reference Commands

```bash
# Activate environment
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

# Run backtest
python my_first_backtest.py

# Run tests
python tests/run_tests.py

# Start Jupyter notebook
jupyter notebook notebooks/example_workflow.ipynb

# Deactivate environment
deactivate
```

---

**Congratulations! You've run your first algorithmic trading backtest! 🎉**

Ready to learn more? → [How It Works](../beginner/02_how_it_works.md)