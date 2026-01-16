# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2026-01-16

### Added
- **True HFT Module** with tick-by-tick processing and nanosecond timestamps
- L2/L3 order book simulation with FIFO price-time priority matching
- Latency modeling (network, exchange, queue delays)
- Market Making strategy with inventory management
- Statistical Arbitrage with z-score signals
- Latency Arbitrage with stale quote detection
- PyTorch ML models (LSTM, Transformer, Ensemble)
- 90+ engineered features for signal generation
- Event-driven backtesting framework
- 40+ performance metrics and visualizations
- Comprehensive documentation

### Components
- `src/hft/tick_data.py` - Tick processing
- `src/hft/order_book.py` - Order book simulation
- `src/hft/matching_engine.py` - Order matching
- `src/hft/latency.py` - Latency modeling
- `src/hft/execution.py` - Execution simulation
- `src/hft/simulator.py` - Event-driven engine
- `src/hft/strategies/` - HFT strategies
- `run_hft.py` - HFT entry point
