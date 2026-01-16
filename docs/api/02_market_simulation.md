# Market Simulation API Reference

This document covers the market simulation components.

---

## OrderBook

**Module:** `src/market/orderbook.py`

Simulates a limit order book with price-time priority.

### Constructor

```python
from market.orderbook import OrderBook

orderbook = OrderBook()
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `bids` | Dict[float, List[Order]] | Buy orders by price level |
| `asks` | Dict[float, List[Order]] | Sell orders by price level |
| `orders` | Dict[str, Order] | All orders by ID |

### Methods

#### `add_order()`

Add an order to the book.

```python
order_id = orderbook.add_order(order: Order) -> str
```

**Example:**
```python
from market.orderbook import Order, OrderSide, OrderType

order = Order(
    order_id="bid1",
    side=OrderSide.BUY,
    order_type=OrderType.LIMIT,
    price=50000.0,
    quantity=0.1,
    timestamp=time.time()
)

orderbook.add_order(order)
```

#### `cancel_order()`

Cancel an existing order.

```python
success = orderbook.cancel_order(order_id: str) -> bool
```

**Returns:** True if order was found and cancelled

#### `get_best_bid()`

Get the highest bid price.

```python
price = orderbook.get_best_bid() -> Optional[float]
```

#### `get_best_ask()`

Get the lowest ask price.

```python
price = orderbook.get_best_ask() -> Optional[float]
```

#### `get_spread()`

Get the bid-ask spread.

```python
spread = orderbook.get_spread() -> Optional[float]
```

**Returns:** Ask price - Bid price

#### `get_spread_bps()`

Get spread in basis points.

```python
spread_bps = orderbook.get_spread_bps() -> Optional[float]
```

**Returns:** Spread as percentage of mid-price × 10000

#### `get_order_book_imbalance()`

Calculate order book imbalance.

```python
imbalance = orderbook.get_order_book_imbalance() -> Optional[float]
```

**Returns:** (bid_volume - ask_volume) / (bid_volume + ask_volume)

Range: -1.0 (all asks) to +1.0 (all bids)

#### `get_depth()`

Get order book depth at multiple levels.

```python
bid_depth, ask_depth = orderbook.get_depth(levels: int = 5) -> Tuple[List, List]
```

**Returns:** Lists of (price, total_quantity) tuples

---

## Order

**Module:** `src/market/orderbook.py`

Represents a single order.

```python
@dataclass
class Order:
    order_id: str
    side: OrderSide        # BUY or SELL
    order_type: OrderType  # MARKET or LIMIT
    price: Optional[float] # None for market orders
    quantity: float
    timestamp: float
    filled_quantity: float = 0.0
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `remaining_quantity` | float | quantity - filled_quantity |

### Methods

#### `fill()`

Fill (partially or completely) an order.

```python
filled = order.fill(quantity: float) -> float
```

**Returns:** Actual quantity filled (may be less if order is nearly full)

#### `is_filled()`

Check if order is completely filled.

```python
is_complete = order.is_filled() -> bool
```

---

## OrderSide & OrderType

Enumerations for order properties.

```python
from market.orderbook import OrderSide, OrderType

class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
```

---

## MatchingEngine

**Module:** `src/market/matching_engine.py`

Executes orders against the order book.

### Constructor

```python
from market.matching_engine import MatchingEngine

engine = MatchingEngine(
    maker_fee: float = 0.0001,  # 0.01%
    taker_fee: float = 0.0002,  # 0.02%
    slippage_pct: float = 0.0005  # 0.05%
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `maker_fee` | float | 0.0001 | Fee for providing liquidity |
| `taker_fee` | float | 0.0002 | Fee for taking liquidity |
| `slippage_pct` | float | 0.0005 | Simulated market impact |

### Methods

#### `match_market_order()`

Execute a market order.

```python
fills = engine.match_market_order(
    order: Order,
    order_book: OrderBook
) -> List[Fill]
```

**Returns:** List of Fill objects representing executed trades

**Example:**
```python
# Create market buy order
order = Order(
    order_id="m1",
    side=OrderSide.BUY,
    order_type=OrderType.MARKET,
    price=None,  # Market order - no price specified
    quantity=0.5,
    timestamp=time.time()
)

fills = engine.match_market_order(order, orderbook)

for fill in fills:
    print(f"Filled {fill.quantity} @ {fill.price}, fee: {fill.fee}")
```

#### `match_limit_order()`

Execute a limit order.

```python
fills = engine.match_limit_order(
    order: Order,
    order_book: OrderBook
) -> List[Fill]
```

If the limit order crosses the spread, it matches immediately.
Otherwise, it's added to the order book.

#### `get_trade_statistics()`

Get statistics about executed trades.

```python
stats = engine.get_trade_statistics() -> dict
```

**Returns:**
```python
{
    'total_trades': int,
    'total_volume': float,
    'total_fees': float,
    'avg_price': float,
    'buy_volume': float,
    'sell_volume': float
}
```

---

## Fill

**Module:** `src/market/matching_engine.py`

Represents a trade execution.

```python
@dataclass
class Fill:
    order_id: str
    side: OrderSide
    price: float
    quantity: float
    fee: float
    timestamp: float
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `value` | float | price × quantity |
| `total_cost` | float | value + fee |

---

## Usage Examples

### Setting Up Order Book

```python
from market.orderbook import OrderBook, Order, OrderSide, OrderType
from market.matching_engine import MatchingEngine
import time

# Create order book and engine
orderbook = OrderBook()
engine = MatchingEngine(maker_fee=0.001, taker_fee=0.002)

# Add some liquidity (limit orders)
orderbook.add_order(Order("bid1", OrderSide.BUY, OrderType.LIMIT, 49900.0, 1.0, time.time()))
orderbook.add_order(Order("bid2", OrderSide.BUY, OrderType.LIMIT, 49800.0, 2.0, time.time()))
orderbook.add_order(Order("ask1", OrderSide.SELL, OrderType.LIMIT, 50100.0, 1.0, time.time()))
orderbook.add_order(Order("ask2", OrderSide.SELL, OrderType.LIMIT, 50200.0, 2.0, time.time()))

# Check book state
print(f"Best Bid: ${orderbook.get_best_bid():,.2f}")
print(f"Best Ask: ${orderbook.get_best_ask():,.2f}")
print(f"Spread: ${orderbook.get_spread():,.2f}")
print(f"Imbalance: {orderbook.get_order_book_imbalance():.2f}")
```

### Executing Trades

```python
# Market buy order
buy_order = Order("m1", OrderSide.BUY, OrderType.MARKET, None, 0.5, time.time())
fills = engine.match_market_order(buy_order, orderbook)

print(f"\nMarket Buy Order Executed:")
total_cost = 0
for fill in fills:
    print(f"  {fill.quantity:.4f} BTC @ ${fill.price:,.2f} (fee: ${fill.fee:.2f})")
    total_cost += fill.total_cost
print(f"  Total Cost: ${total_cost:,.2f}")

# Check updated book
print(f"\nAfter trade - Best Ask: ${orderbook.get_best_ask():,.2f}")
```

### Limit Orders That Cross

```python
# Limit buy above best ask - will match immediately
crossing_order = Order("l1", OrderSide.BUY, OrderType.LIMIT, 50150.0, 0.3, time.time())
fills = engine.match_limit_order(crossing_order, orderbook)

if fills:
    print("Limit order matched immediately!")
else:
    print("Limit order added to book")
```

---

## Next Steps

- [Machine Learning API](03_machine_learning.md)
- [Trading Strategies API](04_strategies.md)
