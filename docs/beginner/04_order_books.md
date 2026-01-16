# Order Books Explained

This guide explains what an order book is and how it works.

## What is an Order Book?

An **order book** is a list of all buy and sell orders for a trading asset, organized by price.

Think of it like a physical auction:
- **Buyers** shout out prices they're willing to pay
- **Sellers** shout out prices they want to receive
- When a buyer and seller agree, a trade happens

---

## The Two Sides

### Bids (Buy Orders)

**Bids** are orders from people who want to **buy**.

```
BIDS (People wanting to buy)

Price     | Quantity | Who's Buying
----------|----------|-------------
$50,100   | 2.5 BTC  | Someone willing to pay this much
$50,000   | 5.0 BTC  | Someone willing to pay less
$49,900   | 3.2 BTC  | Someone willing to pay even less
```

The **best bid** is the highest price someone is willing to pay ($50,100 in this example).

### Asks (Sell Orders)

**Asks** are orders from people who want to **sell**.

```
ASKS (People wanting to sell)

Price     | Quantity | Who's Selling
----------|----------|---------------
$50,200   | 1.0 BTC  | Someone wanting at least this much
$50,300   | 4.0 BTC  | Someone wanting more
$50,400   | 2.0 BTC  | Someone wanting even more
```

The **best ask** is the lowest price someone is willing to sell ($50,200 in this example).

---

## The Spread

The **spread** is the difference between the best ask and best bid.

```
Best Ask:  $50,200
Best Bid:  $50,100
-----------------
Spread:    $100
```

### Why Does the Spread Matter?

The spread represents the **cost of trading immediately**:
- If you want to buy RIGHT NOW, you pay the ask ($50,200)
- If your position is worth the bid ($50,100)
- Immediate loss: $100 per Bitcoin (the spread)

**Tight spread** (small gap) = Easy to trade, low cost
**Wide spread** (big gap) = Expensive to trade, less liquidity

---

## How Orders Match

### Market Orders

A **market order** says "I want to trade NOW at the best available price."

**Example - Market Buy:**
```
You: "I want to buy 1 BTC at market price"

Order book before:
  Best Ask: $50,200 (1.0 BTC available)
  
Result:
  You buy 1.0 BTC at $50,200
  That ask order disappears from the book
```

### Limit Orders

A **limit order** says "I only want to trade at THIS price or better."

**Example - Limit Buy:**
```
You: "I want to buy 1 BTC, but only at $50,000 or less"

Order book:
  Best Ask: $50,200 (too expensive for you)
  
Result:
  Your order is added to the BIDS at $50,000
  You wait until someone is willing to sell at $50,000
```

---

## Price-Time Priority

When multiple orders are at the same price, who gets filled first?

**Answer: The one that arrived first** (time priority)

```
BIDS at $50,000:
  Order A: 1.0 BTC (arrived 10:00:00)  ← First in line
  Order B: 2.0 BTC (arrived 10:00:01)  ← Second
  Order C: 0.5 BTC (arrived 10:00:02)  ← Third

If someone sells 1.5 BTC at market:
  Order A gets filled completely (1.0 BTC)
  Order B gets partially filled (0.5 of their 2.0 BTC)
  Order C is still waiting
```

---

## Order Book Depth

**Depth** refers to how many orders exist at different price levels.

```
        BIDS                                    ASKS
        
Price     | Volume                Price     | Volume
$50,100   | ████████ 2.5         $50,200   | ██ 1.0
$50,000   | █████████████ 5.0    $50,300   | ████████████ 4.0
$49,900   | █████████ 3.2        $50,400   | ██████ 2.0
$49,800   | ████████████ 4.8     $50,500   | ████████████████ 6.5
$49,700   | ██████ 2.1           $50,600   | █████ 1.8
```

**Deep order book:** Many orders at many price levels → Good liquidity
**Shallow order book:** Few orders → Large orders will move the price

---

## Order Book Imbalance

**Imbalance** measures whether there are more buyers or sellers.

**Formula:**
```
Imbalance = (Bid Volume - Ask Volume) / (Bid Volume + Ask Volume)
```

**Interpretation:**
- Imbalance = +1.0: All bids, no asks (everyone wants to buy)
- Imbalance = 0.0: Equal bids and asks (balanced)
- Imbalance = -1.0: All asks, no bids (everyone wants to sell)

**Example:**
```
Bid Volume: 10 BTC
Ask Volume: 5 BTC

Imbalance = (10 - 5) / (10 + 5) = 5/15 = 0.33

Interpretation: Slightly more buying pressure
```

---

## Slippage

**Slippage** happens when your order moves the price.

**Example - Large Market Buy:**
```
You want to buy 5 BTC

Order book:
  $50,200 | 1.0 BTC
  $50,300 | 2.0 BTC
  $50,400 | 2.0 BTC

Your order fills:
  1.0 BTC @ $50,200 = $50,200
  2.0 BTC @ $50,300 = $100,600
  2.0 BTC @ $50,400 = $100,800
  --------------------------
  Total: 5.0 BTC for $251,600

Average price: $251,600 / 5 = $50,320

You wanted: $50,200 (best ask)
You got: $50,320
Slippage: $120 per BTC (0.24%)
```

---

## In Our Simulator

Our `OrderBook` class simulates all of this:

```python
from market.orderbook import OrderBook, Order, OrderSide, OrderType

# Create order book
book = OrderBook()

# Add buy orders (bids)
book.add_order(Order("b1", OrderSide.BUY, OrderType.LIMIT, 50100.0, 2.5, time.time()))
book.add_order(Order("b2", OrderSide.BUY, OrderType.LIMIT, 50000.0, 5.0, time.time()))

# Add sell orders (asks)
book.add_order(Order("a1", OrderSide.SELL, OrderType.LIMIT, 50200.0, 1.0, time.time()))
book.add_order(Order("a2", OrderSide.SELL, OrderType.LIMIT, 50300.0, 4.0, time.time()))

# Check the book
print(f"Best Bid: ${book.get_best_bid():,.2f}")
print(f"Best Ask: ${book.get_best_ask():,.2f}")
print(f"Spread: ${book.get_spread():,.2f}")
print(f"Imbalance: {book.get_order_book_imbalance():.2f}")
```

---

## Key Terms Summary

| Term | Definition |
|------|------------|
| **Bid** | Price someone will pay (buy order) |
| **Ask** | Price someone wants (sell order) |
| **Spread** | Difference between best ask and best bid |
| **Depth** | Volume of orders at various prices |
| **Imbalance** | Ratio of buy vs sell pressure |
| **Slippage** | Price movement caused by your order |
| **Market order** | Execute immediately at best price |
| **Limit order** | Only execute at specified price or better |

---

## Next Steps

- [Machine Learning Basics](05_ml_basics.md) - How AI predicts prices
- [Strategy Logic](06_strategy_logic.md) - Using order book in trading
