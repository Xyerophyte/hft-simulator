"""
Order book data structure for simulating market microstructure.
Simplified version tracking best bid/ask and basic depth.
"""

from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, List
import time


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"


@dataclass
class Order:
    """
    Represents a single order in the order book.
    """
    order_id: str
    side: OrderSide
    order_type: OrderType
    price: Optional[float]  # None for market orders
    quantity: float
    timestamp: float
    filled_quantity: float = 0.0
    
    @property
    def remaining_quantity(self) -> float:
        """Get unfilled quantity."""
        return self.quantity - self.filled_quantity
    
    @property
    def is_filled(self) -> bool:
        """Check if order is completely filled."""
        return self.filled_quantity >= self.quantity
    
    def fill(self, quantity: float) -> float:
        """
        Fill order with specified quantity.
        
        Args:
            quantity: Quantity to fill
            
        Returns:
            Actual filled quantity
        """
        fillable = min(quantity, self.remaining_quantity)
        self.filled_quantity += fillable
        return fillable


@dataclass
class PriceLevel:
    """
    Represents a single price level in the order book.
    """
    price: float
    orders: deque  # Queue of orders at this price
    total_quantity: float = 0.0
    
    def add_order(self, order: Order):
        """Add order to this price level."""
        self.orders.append(order)
        self.total_quantity += order.remaining_quantity
    
    def remove_order(self, order: Order):
        """Remove order from this price level."""
        if order in self.orders:
            self.orders.remove(order)
            self.total_quantity -= order.remaining_quantity


class OrderBook:
    """
    Simplified order book with price-time priority.
    Maintains best bid/ask and basic order matching.
    """
    
    def __init__(self):
        """Initialize empty order book."""
        self.bids: Dict[float, PriceLevel] = {}  # Buy orders (price -> PriceLevel)
        self.asks: Dict[float, PriceLevel] = {}  # Sell orders (price -> PriceLevel)
        self.orders: Dict[str, Order] = {}  # All orders by ID
        self.order_counter = 0
        
    def generate_order_id(self) -> str:
        """Generate unique order ID."""
        self.order_counter += 1
        return f"ORD_{self.order_counter}_{int(time.time() * 1000000)}"
    
    def add_order(self, order: Order) -> str:
        """
        Add order to the book.
        
        Args:
            order: Order to add
            
        Returns:
            Order ID
        """
        if order.order_type == OrderType.MARKET:
            # Market orders are matched immediately
            return order.order_id
            
        # Add limit order to appropriate side
        price_levels = self.bids if order.side == OrderSide.BUY else self.asks
        
        if order.price not in price_levels:
            price_levels[order.price] = PriceLevel(
                price=order.price,
                orders=deque()
            )
        
        price_levels[order.price].add_order(order)
        self.orders[order.order_id] = order
        
        return order.order_id
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: ID of order to cancel
            
        Returns:
            True if cancelled, False if not found
        """
        if order_id not in self.orders:
            return False
            
        order = self.orders[order_id]
        price_levels = self.bids if order.side == OrderSide.BUY else self.asks
        
        if order.price in price_levels:
            price_levels[order.price].remove_order(order)
            
            # Remove price level if empty
            if price_levels[order.price].total_quantity == 0:
                del price_levels[order.price]
        
        del self.orders[order_id]
        return True
    
    def get_best_bid(self) -> Optional[float]:
        """Get best (highest) bid price."""
        if not self.bids:
            return None
        return max(self.bids.keys())
    
    def get_best_ask(self) -> Optional[float]:
        """Get best (lowest) ask price."""
        if not self.asks:
            return None
        return min(self.asks.keys())
    
    def get_mid_price(self) -> Optional[float]:
        """Get mid-market price."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid is None or best_ask is None:
            return None
            
        return (best_bid + best_ask) / 2.0
    
    def get_spread(self) -> Optional[float]:
        """Get bid-ask spread."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid is None or best_ask is None:
            return None
            
        return best_ask - best_bid
    
    def get_spread_bps(self) -> Optional[float]:
        """Get bid-ask spread in basis points."""
        spread = self.get_spread()
        mid = self.get_mid_price()
        
        if spread is None or mid is None or mid == 0:
            return None
            
        return (spread / mid) * 10000  # Convert to bps
    
    def get_depth(self, side: OrderSide, levels: int = 5) -> List[tuple]:
        """
        Get order book depth for a side.
        
        Args:
            side: BUY or SELL
            levels: Number of price levels to return
            
        Returns:
            List of (price, quantity) tuples
        """
        price_levels = self.bids if side == OrderSide.BUY else self.asks
        
        if not price_levels:
            return []
        
        # Sort prices (descending for bids, ascending for asks)
        sorted_prices = sorted(
            price_levels.keys(),
            reverse=(side == OrderSide.BUY)
        )[:levels]
        
        return [(price, price_levels[price].total_quantity) 
                for price in sorted_prices]
    
    def get_volume_at_price(self, price: float, side: OrderSide) -> float:
        """
        Get total volume at a specific price.
        
        Args:
            price: Price level
            side: BUY or SELL
            
        Returns:
            Total quantity at price level
        """
        price_levels = self.bids if side == OrderSide.BUY else self.asks
        
        if price not in price_levels:
            return 0.0
            
        return price_levels[price].total_quantity
    
    def get_order_book_imbalance(self) -> Optional[float]:
        """
        Calculate order book imbalance.
        
        Returns:
            Imbalance ratio: (bid_volume - ask_volume) / (bid_volume + ask_volume)
            Positive = more bids, Negative = more asks
        """
        bid_volume = sum(level.total_quantity for level in self.bids.values())
        ask_volume = sum(level.total_quantity for level in self.asks.values())
        
        total_volume = bid_volume + ask_volume
        
        if total_volume == 0:
            return None
            
        return (bid_volume - ask_volume) / total_volume
    
    def clear(self):
        """Clear all orders from the book."""
        self.bids.clear()
        self.asks.clear()
        self.orders.clear()
    
    def __repr__(self) -> str:
        """String representation of order book."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        spread = self.get_spread()
        spread_str = f"{spread:.2f}" if spread is not None else "N/A"
        
        return (f"OrderBook(bid={best_bid}, ask={best_ask}, "
                f"spread={spread_str}, "
                f"orders={len(self.orders)})")
    
    def get_book_snapshot(self) -> dict:
        """
        Get current state of order book.
        
        Returns:
            Dictionary with book statistics
        """
        return {
            'best_bid': self.get_best_bid(),
            'best_ask': self.get_best_ask(),
            'mid_price': self.get_mid_price(),
            'spread': self.get_spread(),
            'spread_bps': self.get_spread_bps(),
            'imbalance': self.get_order_book_imbalance(),
            'total_orders': len(self.orders),
            'bid_levels': len(self.bids),
            'ask_levels': len(self.asks),
            'bid_depth': self.get_depth(OrderSide.BUY, 5),
            'ask_depth': self.get_depth(OrderSide.SELL, 5)
        }


# Example usage
if __name__ == "__main__":
    # Create order book
    book = OrderBook()
    
    # Add some limit orders
    bid1 = Order(
        order_id=book.generate_order_id(),
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=100.0,
        quantity=10.0,
        timestamp=time.time()
    )
    
    bid2 = Order(
        order_id=book.generate_order_id(),
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=99.5,
        quantity=20.0,
        timestamp=time.time()
    )
    
    ask1 = Order(
        order_id=book.generate_order_id(),
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        price=101.0,
        quantity=15.0,
        timestamp=time.time()
    )
    
    ask2 = Order(
        order_id=book.generate_order_id(),
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        price=101.5,
        quantity=25.0,
        timestamp=time.time()
    )
    
    # Add orders to book
    book.add_order(bid1)
    book.add_order(bid2)
    book.add_order(ask1)
    book.add_order(ask2)
    
    # Display book state
    print(book)
    print("\nBook snapshot:")
    snapshot = book.get_book_snapshot()
    for key, value in snapshot.items():
        print(f"  {key}: {value}")