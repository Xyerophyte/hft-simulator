# PRD Alignment - Basic Version Scope

This document maps the full PRD requirements to what we're building in the **basic version** (v1).

## ✅ Included in Basic Version

### Market Data Engine
- ✅ Load historical BTC-USD data from Binance API
- ✅ OHLCV candle data (1-minute resolution)
- ✅ Basic caching for offline use
- ❌ NOT: Full order book snapshots (simplified to bid/ask spread)
- ❌ NOT: Sub-second tick data

### Order Book Simulator
- ✅ Simplified order book (best bid/ask tracking)
- ✅ Market orders
- ✅ Basic limit orders
- ❌ NOT: Full LOB depth with multiple price levels
- ❌ NOT: Post-only orders
- ❌ NOT: Order cancellations

### Matching Engine
- ✅ Basic price-time priority
- ✅ Instant fills with slippage model
- ✅ Simple fee structure
- ❌ NOT: Partial fills
- ❌ NOT: Microsecond-level timing

### ML Prediction Module
- ✅ PyTorch LSTM model
- ✅ Binary classification (price direction)
- ✅ Basic features (returns, volatility, volume)
- ✅ Target: 1-minute ahead prediction
- ❌ NOT: Advanced features (order book imbalance requires full LOB)
- ❌ NOT: Ensemble models

### Strategy Engine
- ✅ Simple momentum strategy
- ✅ ML probability-based signals
- ✅ Basic entry/exit logic
- ❌ NOT: Market making
- ❌ NOT: Mean reversion strategies

### Risk Management
- ✅ Basic position limits
- ✅ Simple stop-loss
- ❌ NOT: Dynamic VaR calculation
- ❌ NOT: Volatility-based circuit breakers

### Backtesting Framework
- ✅ Replay historical data
- ✅ Single-period backtest
- ✅ Deterministic results with seed
- ❌ NOT: Multi-period validation
- ❌ NOT: Monte Carlo stress testing

### Analytics & Reporting
- ✅ Core metrics: Sharpe, max drawdown, win rate, total return
- ✅ Basic equity curve visualization
- ❌ NOT: 40+ metrics
- ❌ NOT: Advanced attribution analysis
- ❌ NOT: Interactive dashboards

## Performance Targets (Relaxed for v1)

| Metric | PRD Target | Basic Version Target |
|--------|------------|---------------------|
| Order throughput | 120k orders/sec | Not measured (focus on correctness) |
| Latency p99.9 | <450 μs | Not optimized initially |
| Sharpe Ratio | 2.5+ | 1.0+ (learning baseline) |
| Max Drawdown | <7% | <15% |
| ML Accuracy | >68% | >55% |

## What We're Deferring

These are explicitly **out of scope** for the basic version:

1. **Advanced Order Book Features**
   - Full depth visualization
   - Order cancellations
   - Complex order types

2. **High-Performance Optimization**
   - Microsecond timing
   - Parallel processing
   - C++ extensions

3. **Advanced ML**
   - Transformer models
   - Ensemble methods
   - Real-time training

4. **Complex Strategies**
   - Market making
   - Arbitrage
   - Multi-asset strategies

5. **Production Features**
   - Live trading integration
   - WebSocket data streams
   - Database persistence

## Success Criteria for Basic Version

The basic version is successful if it:

1. ✅ Fetches and stores real BTC-USD historical data
2. ✅ Simulates basic order execution (market orders)
3. ✅ Trains an ML model that beats random guessing (>50% accuracy)
4. ✅ Runs a complete backtest from start to finish
5. ✅ Generates core performance metrics (Sharpe, drawdown, win rate)
6. ✅ Produces an equity curve visualization
7. ✅ Has modular, testable code structure
8. ✅ Includes documentation and example notebook

## Philosophy: Build to Learn

This basic version prioritizes:

- **Completeness over perfection** - Working end-to-end system
- **Clarity over optimization** - Readable code, clear logic
- **Learning over production** - Understanding concepts deeply
- **Iteration over scope** - Ship v1, improve in v2

The goal is to have a **functional research platform** that demonstrates understanding of:
- Market microstructure basics
- ML integration in trading systems
- Quantitative performance evaluation
- Software engineering practices in finance

---

*Once the basic version is complete, we can incrementally add advanced features from the full PRD.*