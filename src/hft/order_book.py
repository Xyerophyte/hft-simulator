"""
L2/L3 Order Book Simulation for HFT.

Provides full order book depth with bid/ask levels,
order management, and market impact modeling.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
from collections import defaultdict
import heapq
from sortedcontainers import SortedDict
import time


class Side(Enum):
    """Order side."""
    BID = "bid"
    ASK = "ask"


class OrderType(Enum):
    """Order type."""
    MARKET = "market"
    LIMIT = "limit"
    CANCEL = "cancel"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    OPEN = "open"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """
    Represents a single order in the book.
    
    In real HFT, orders have nanosecond timestamps for
    price-time priority matching.
    """
    order_id: int
    side: Side
    price: float
    size: float
    timestamp_ns: int
    order_type: OrderType = OrderType.LIMIT
    status: OrderStatus = OrderStatus.PENDING
    filled_size: float = 0.0
    average_fill_price: float = 0.0
    
    @property
    def remaining_size(self) -> float:
        return self.size - self.filled_size
    
    @property
    def is_complete(self) -> bool:
        return self.status in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED)
    
    def fill(self, size: float, price: float) -> float:
        """Fill order and return amount filled."""
        fill_amount = min(size, self.remaining_size)
        
        if fill_amount > 0:
            # Update average fill price
            old_value = self.filled_size * self.average_fill_price
            new_value = fill_amount * price
            self.filled_size += fill_amount
            self.average_fill_price = (old_value + new_value) / self.filled_size
            
            # Update status
            if self.remaining_size <= 0:
                self.status = OrderStatus.FILLED
            else:
                self.status = OrderStatus.PARTIAL
        
        return fill_amount


@dataclass
class PriceLevel:
    """
    Single price level in the order book.
    
    Contains all orders at a specific price, ordered by time priority.
    """
    price: float
    orders: List[Order] = field(default_factory=list)
    
    @property
    def total_size(self) -> float:
        return sum(o.remaining_size for o in self.orders)
    
    @property
    def order_count(self) -> int:
        return len(self.orders)
    
    def add_order(self, order: Order) -> None:
        """Add order to level (FIFO queue)."""
        self.orders.append(order)
    
    def remove_order(self, order_id: int) -> Optional[Order]:
        """Remove order by ID."""
        for i, order in enumerate(self.orders):
            if order.order_id == order_id:
                return self.orders.pop(i)
        return None
    
    def match(self, size: float, price: float) -> Tuple[float, List[Tuple[Order, float]]]:
        """
        Match incoming order against this level.
        
        Returns:
            Tuple of (remaining_size, list of (order, fill_size))
        """
        fills = []
        remaining = size
        
        while remaining > 0 and self.orders:
            order = self.orders[0]
            fill_amount = order.fill(remaining, price)
            
            if fill_amount > 0:
                fills.append((order, fill_amount))
                remaining -= fill_amount
            
            if order.is_complete:
                self.orders.pop(0)
        
        return remaining, fills


class OrderBook:
    """
    Full L2/L3 Order Book simulation.
    
    Features:
    - Multiple price levels with depth
    - FIFO matching within levels
    - Queue position tracking
    - Market impact estimation
    """
    
    def __init__(self, symbol: str = "BTC"):
        self.symbol = symbol
        
        # Use sorted dicts for efficient price level access
        # Bids: highest first (negative keys for reverse sort)
        # Asks: lowest first
        self.bids: Dict[float, PriceLevel] = {}
        self.asks: Dict[float, PriceLevel] = {}
        
        # Order tracking
        self.orders: Dict[int, Order] = {}
        self._next_order_id = 1
        
        # Book stats
        self.last_trade_price = 0.0
        self.last_trade_size = 0.0
        self.total_volume = 0.0
    
    @property
    def best_bid(self) -> Optional[float]:
        """Get best bid price."""
        if not self.bids:
            return None
        return max(self.bids.keys())
    
    @property
    def best_ask(self) -> Optional[float]:
        """Get best ask price."""
        if not self.asks:
            return None
        return min(self.asks.keys())
    
    @property
    def mid_price(self) -> Optional[float]:
        """Get mid price."""
        bid = self.best_bid
        ask = self.best_ask
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2
    
    @property
    def spread(self) -> Optional[float]:
        """Get bid-ask spread."""
        bid = self.best_bid
        ask = self.best_ask
        if bid is None or ask is None:
            return None
        return ask - bid
    
    @property
    def spread_bps(self) -> Optional[float]:
        """Get spread in basis points."""
        mid = self.mid_price
        spread = self.spread
        if mid is None or spread is None or mid == 0:
            return None
        return (spread / mid) * 10000
    
    def add_order(self, order: Order) -> Order:
        """Add order to book."""
        order.order_id = self._next_order_id
        self._next_order_id += 1
        order.status = OrderStatus.OPEN
        
        self.orders[order.order_id] = order
        
        # Add to appropriate side
        if order.side == Side.BID:
            if order.price not in self.bids:
                self.bids[order.price] = PriceLevel(order.price)
            self.bids[order.price].add_order(order)
        else:
            if order.price not in self.asks:
                self.asks[order.price] = PriceLevel(order.price)
            self.asks[order.price].add_order(order)
        
        return order
    
    def cancel_order(self, order_id: int) -> Optional[Order]:
        """Cancel order by ID."""
        if order_id not in self.orders:
            return None
        
        order = self.orders[order_id]
        
        if order.is_complete:
            return None
        
        # Remove from book
        if order.side == Side.BID:
            if order.price in self.bids:
                self.bids[order.price].remove_order(order_id)
                if not self.bids[order.price].orders:
                    del self.bids[order.price]
        else:
            if order.price in self.asks:
                self.asks[order.price].remove_order(order_id)
                if not self.asks[order.price].orders:
                    del self.asks[order.price]
        
        order.status = OrderStatus.CANCELLED
        return order
    
    def execute_market_order(
        self,
        side: Side,
        size: float,
        timestamp_ns: int = None
    ) -> Tuple[float, float, List[Tuple[Order, float]]]:
        """
        Execute market order against the book.
        
        Args:
            side: BID (buy) or ASK (sell)
            size: Order size
            timestamp_ns: Execution timestamp
            
        Returns:
            Tuple of (filled_size, average_price, fills)
        """
        if timestamp_ns is None:
            timestamp_ns = int(time.time() * 1e9)
        
        fills = []
        remaining = size
        total_value = 0.0
        
        # Buy order matches against asks (lowest first)
        # Sell order matches against bids (highest first)
        if side == Side.BID:
            levels = sorted(self.asks.keys())
        else:
            levels = sorted(self.bids.keys(), reverse=True)
        
        for price in levels:
            if remaining <= 0:
                break
            
            if side == Side.BID:
                level = self.asks[price]
            else:
                level = self.bids[price]
            
            remaining, level_fills = level.match(remaining, price)
            
            for order, fill_size in level_fills:
                fills.append((order, fill_size))
                total_value += fill_size * price
            
            # Remove empty level
            if level.total_size <= 0:
                if side == Side.BID:
                    del self.asks[price]
                else:
                    del self.bids[price]
        
        filled_size = size - remaining
        avg_price = total_value / filled_size if filled_size > 0 else 0.0
        
        if filled_size > 0:
            self.last_trade_price = avg_price
            self.last_trade_size = filled_size
            self.total_volume += filled_size
        
        return filled_size, avg_price, fills
    
    def get_depth(self, levels: int = 10) -> Dict[str, List[Tuple[float, float]]]:
        """
        Get order book depth.
        
        Returns:
            Dict with 'bids' and 'asks' as lists of (price, size) tuples
        """
        bid_prices = sorted(self.bids.keys(), reverse=True)[:levels]
        ask_prices = sorted(self.asks.keys())[:levels]
        
        return {
            'bids': [(p, self.bids[p].total_size) for p in bid_prices],
            'asks': [(p, self.asks[p].total_size) for p in ask_prices]
        }
    
    def get_queue_position(self, order_id: int) -> Optional[int]:
        """
        Get queue position for an order.
        
        Position is number of shares ahead in queue at same price.
        """
        if order_id not in self.orders:
            return None
        
        order = self.orders[order_id]
        
        if order.side == Side.BID:
            if order.price not in self.bids:
                return None
            level = self.bids[order.price]
        else:
            if order.price not in self.asks:
                return None
            level = self.asks[order.price]
        
        position = 0
        for o in level.orders:
            if o.order_id == order_id:
                return position
            position += o.remaining_size
        
        return None
    
    def estimate_market_impact(self, side: Side, size: float) -> float:
        """
        Estimate market impact of a market order.
        
        Returns expected average fill price.
        """
        if side == Side.BID:
            levels = sorted(self.asks.keys())
        else:
            levels = sorted(self.bids.keys(), reverse=True)
        
        remaining = size
        total_value = 0.0
        
        for price in levels:
            if remaining <= 0:
                break
            
            if side == Side.BID:
                level_size = self.asks[price].total_size
            else:
                level_size = self.bids[price].total_size
            
            fill_at_level = min(remaining, level_size)
            total_value += fill_at_level * price
            remaining -= fill_at_level
        
        if size - remaining > 0:
            return total_value / (size - remaining)
        return 0.0
    
    def initialize_from_snapshot(
        self,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        timestamp_ns: int = None
    ) -> None:
        """
        Initialize book from depth snapshot.
        
        Args:
            bids: List of (price, size) tuples
            asks: List of (price, size) tuples
        """
        if timestamp_ns is None:
            timestamp_ns = int(time.time() * 1e9)
        
        self.bids.clear()
        self.asks.clear()
        
        for price, size in bids:
            order = Order(
                order_id=self._next_order_id,
                side=Side.BID,
                price=price,
                size=size,
                timestamp_ns=timestamp_ns
            )
            self.add_order(order)
        
        for price, size in asks:
            order = Order(
                order_id=self._next_order_id,
                side=Side.ASK,
                price=price,
                size=size,
                timestamp_ns=timestamp_ns
            )
            self.add_order(order)
    
    def __repr__(self) -> str:
        bid = self.best_bid or 0
        ask = self.best_ask or 0
        spread = self.spread_bps or 0
        return f"OrderBook({self.symbol}: bid={bid:.2f}, ask={ask:.2f}, spread={spread:.1f}bps)"
