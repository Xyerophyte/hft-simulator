# HFT Module Documentation

## Overview

The `src/hft/` module contains true high-frequency trading components that simulate 
real HFT operations with microsecond-level precision.

## Quick Start

```python
# Run HFT simulation
python run_hft.py --ticks 5000

# Use specific strategy
python run_hft.py --strategy market_maker --ticks 10000
```

## Components

### 1. Tick Data (`tick_data.py`)

Process tick-by-tick market data with nanosecond timestamps.

```python
from hft.tick_data import Tick, TickStream

tick = Tick(
    timestamp_ns=1705123456789000000,
    price=50000.0,
    size=1.5,
    side='buy'
)
```

### 2. Order Book (`order_book.py`)

L2/L3 order book with full depth.

```python
from hft.order_book import OrderBook, Side

book = OrderBook()
book.add_order(Order(side=Side.BID, price=50000, size=10))

# Get market state
mid = book.mid_price
spread = book.spread_bps  # In basis points
depth = book.get_depth(levels=10)
```

### 3. Matching Engine (`matching_engine.py`)

FIFO price-time priority matching.

```python
from hft.matching_engine import MatchingEngine

engine = MatchingEngine(order_book)
result = engine.submit_order(Side.BID, size=1.0, price=50000)
```

### 4. Latency Model (`latency.py`)

Simulate realistic network/exchange latency.

```python
from hft.latency import LatencyModel, LatencyProfile

# HFT firm (co-located, ~10 microseconds)
latency = LatencyModel(LatencyProfile.HFT)

# Get round-trip latency
rtt = latency.get_total_order_latency()  # Returns nanoseconds
```

### 5. Market Making (`strategies/market_maker.py`)

Core HFT strategy: quote bid/ask to earn the spread.

```python
from hft.strategies.market_maker import SimpleMarketMaker

mm = SimpleMarketMaker(spread_bps=2.0, max_inventory=10.0)
result = mm.make_market(mid_price=50000)
# result = {'bid': 49995, 'ask': 50005, 'traded': False}
```

### 6. Statistical Arbitrage (`strategies/stat_arb.py`)

Trade on mean-reverting spreads between instruments.

```python
from hft.strategies.stat_arb import StatisticalArbitrage

arb = StatisticalArbitrage()
arb.add_pair("BTC_ETH", "BTC", "ETH")
signal = arb.update_prices("BTC_ETH", 50000, 3000)
```

### 7. Latency Arbitrage (`strategies/latency_arb.py`)

Exploit speed advantage to pick off stale quotes.

```python
from hft.strategies.latency_arb import LatencyArbitrage

la = LatencyArbitrage()
signals = la.scan_opportunities(fair_value=50000, current_ns=time.time_ns())
```

## Performance Metrics

The simulator tracks:

- **Throughput**: Ticks processed per second
- **Latency**: Processing time per tick (microseconds)
- **PnL**: Profit and loss from trading
- **Fill Rate**: Percentage of quotes that trade
- **Inventory**: Current position

## Configuration

Edit `config/hft_config.yaml`:

```yaml
market_maker:
  spread_bps: 3.0        # Wider = more profit, less fills
  max_position: 5.0      # Inventory limit
```

## Architecture

```
run_hft.py              # Entry point
└── src/hft/
    ├── __init__.py
    ├── tick_data.py    # Tick processing
    ├── order_book.py   # L2/L3 book
    ├── matching_engine.py
    ├── latency.py      # Latency simulation
    ├── execution.py    # Fill simulation
    ├── simulator.py    # Event loop
    └── strategies/
        ├── market_maker.py
        ├── stat_arb.py
        └── latency_arb.py
```
