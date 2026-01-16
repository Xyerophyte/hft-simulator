"""
Matching Engine for HFT Simulation.

FIFO price-time priority matching engine with
realistic fill simulation.
"""

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from enum import Enum

from .order_book import OrderBook, Order, OrderType, Side, OrderStatus
from .tick_data import Tick, TickType


@dataclass
class Fill:
    """Represents a trade execution (fill)."""
    fill_id: int
    order_id: int
    side: Side
    price: float
    size: float
    timestamp_ns: int
    maker_order_id: Optional[int] = None
    taker_order_id: Optional[int] = None
    fee: float = 0.0


@dataclass
class OrderResult:
    """Result of order submission."""
    order: Order
    fills: List[Fill]
    status: str
    message: str = ""


class MatchingEngine:
    """
    Price-time priority matching engine.
    
    Features:
    - FIFO matching at each price level
    - Limit and market orders
    - Partial fills
    - Trade recording
    """
    
    def __init__(
        self,
        order_book: Optional[OrderBook] = None,
        maker_fee: float = -0.0001,  # Maker rebate
        taker_fee: float = 0.0003    # Taker fee
    ):
        self.order_book = order_book or OrderBook()
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        
        self._next_fill_id = 1
        self.fills: List[Fill] = []
        self.trades: List[dict] = []
    
    def submit_order(
        self,
        side: Side,
        size: float,
        price: Optional[float] = None,
        order_type: OrderType = OrderType.LIMIT,
        timestamp_ns: Optional[int] = None
    ) -> OrderResult:
        """
        Submit order to matching engine.
        
        Args:
            side: BID or ASK
            size: Order size
            price: Limit price (None for market)
            order_type: LIMIT or MARKET
            timestamp_ns: Order timestamp
            
        Returns:
            OrderResult with order and fills
        """
        if timestamp_ns is None:
            timestamp_ns = int(time.time() * 1e9)
        
        if order_type == OrderType.MARKET or price is None:
            return self._execute_market_order(side, size, timestamp_ns)
        else:
            return self._execute_limit_order(side, size, price, timestamp_ns)
    
    def _execute_market_order(
        self,
        side: Side,
        size: float,
        timestamp_ns: int
    ) -> OrderResult:
        """Execute market order."""
        # Create order
        order = Order(
            order_id=0,  # Will be assigned
            side=side,
            price=0.0,  # Market order
            size=size,
            timestamp_ns=timestamp_ns,
            order_type=OrderType.MARKET
        )
        
        # Execute against book
        filled_size, avg_price, book_fills = self.order_book.execute_market_order(
            side, size, timestamp_ns
        )
        
        # Create fill records
        fills = []
        for maker_order, fill_size in book_fills:
            fill = Fill(
                fill_id=self._next_fill_id,
                order_id=order.order_id,
                side=side,
                price=maker_order.price,
                size=fill_size,
                timestamp_ns=timestamp_ns,
                maker_order_id=maker_order.order_id,
                taker_order_id=order.order_id,
                fee=fill_size * maker_order.price * self.taker_fee
            )
            self._next_fill_id += 1
            fills.append(fill)
            self.fills.append(fill)
            
            # Record trade
            self.trades.append({
                'timestamp_ns': timestamp_ns,
                'price': maker_order.price,
                'size': fill_size,
                'side': 'buy' if side == Side.BID else 'sell',
                'maker_order_id': maker_order.order_id,
                'taker_order_id': order.order_id
            })
        
        # Update order
        order.filled_size = filled_size
        order.average_fill_price = avg_price
        
        if filled_size >= size:
            order.status = OrderStatus.FILLED
            status = "filled"
        elif filled_size > 0:
            order.status = OrderStatus.PARTIAL
            status = "partial"
        else:
            order.status = OrderStatus.REJECTED
            status = "rejected"
        
        return OrderResult(order=order, fills=fills, status=status)
    
    def _execute_limit_order(
        self,
        side: Side,
        size: float,
        price: float,
        timestamp_ns: int
    ) -> OrderResult:
        """Execute limit order."""
        # Create order
        order = Order(
            order_id=0,
            side=side,
            price=price,
            size=size,
            timestamp_ns=timestamp_ns,
            order_type=OrderType.LIMIT
        )
        
        fills = []
        remaining = size
        
        # Check if order crosses the spread (marketable limit)
        if side == Side.BID:
            # Buy order - check if price >= best ask
            while remaining > 0 and self.order_book.best_ask is not None:
                if price >= self.order_book.best_ask:
                    # Execute against asks
                    fill_size = min(remaining, self._get_size_at_best_ask())
                    if fill_size > 0:
                        filled, avg_px, book_fills = self.order_book.execute_market_order(
                            Side.BID, fill_size, timestamp_ns
                        )
                        remaining -= filled
                        
                        for maker_order, fs in book_fills:
                            fill = Fill(
                                fill_id=self._next_fill_id,
                                order_id=order.order_id,
                                side=side,
                                price=maker_order.price,
                                size=fs,
                                timestamp_ns=timestamp_ns,
                                maker_order_id=maker_order.order_id,
                                taker_order_id=order.order_id,
                                fee=fs * maker_order.price * self.taker_fee
                            )
                            self._next_fill_id += 1
                            fills.append(fill)
                            self.fills.append(fill)
                            
                            self.trades.append({
                                'timestamp_ns': timestamp_ns,
                                'price': maker_order.price,
                                'size': fs,
                                'side': 'buy',
                                'maker_order_id': maker_order.order_id,
                                'taker_order_id': order.order_id
                            })
                    else:
                        break
                else:
                    break
        else:
            # Sell order - check if price <= best bid
            while remaining > 0 and self.order_book.best_bid is not None:
                if price <= self.order_book.best_bid:
                    fill_size = min(remaining, self._get_size_at_best_bid())
                    if fill_size > 0:
                        filled, avg_px, book_fills = self.order_book.execute_market_order(
                            Side.ASK, fill_size, timestamp_ns
                        )
                        remaining -= filled
                        
                        for maker_order, fs in book_fills:
                            fill = Fill(
                                fill_id=self._next_fill_id,
                                order_id=order.order_id,
                                side=side,
                                price=maker_order.price,
                                size=fs,
                                timestamp_ns=timestamp_ns,
                                maker_order_id=maker_order.order_id,
                                taker_order_id=order.order_id,
                                fee=fs * maker_order.price * self.taker_fee
                            )
                            self._next_fill_id += 1
                            fills.append(fill)
                            self.fills.append(fill)
                            
                            self.trades.append({
                                'timestamp_ns': timestamp_ns,
                                'price': maker_order.price,
                                'size': fs,
                                'side': 'sell',
                                'maker_order_id': maker_order.order_id,
                                'taker_order_id': order.order_id
                            })
                    else:
                        break
                else:
                    break
        
        # Place remaining as limit order if any
        if remaining > 0:
            order.size = remaining
            order = self.order_book.add_order(order)
        
        # Update order stats
        filled_size = size - remaining
        if filled_size > 0:
            total_value = sum(f.size * f.price for f in fills)
            order.filled_size = filled_size
            order.average_fill_price = total_value / filled_size
        
        if remaining <= 0:
            order.status = OrderStatus.FILLED
            status = "filled"
        elif filled_size > 0:
            order.status = OrderStatus.PARTIAL
            status = "partial_open"
        else:
            order.status = OrderStatus.OPEN
            status = "open"
        
        return OrderResult(order=order, fills=fills, status=status)
    
    def _get_size_at_best_ask(self) -> float:
        """Get size at best ask."""
        ask = self.order_book.best_ask
        if ask is None or ask not in self.order_book.asks:
            return 0.0
        return self.order_book.asks[ask].total_size
    
    def _get_size_at_best_bid(self) -> float:
        """Get size at best bid."""
        bid = self.order_book.best_bid
        if bid is None or bid not in self.order_book.bids:
            return 0.0
        return self.order_book.bids[bid].total_size
    
    def cancel_order(self, order_id: int) -> Optional[Order]:
        """Cancel an order."""
        return self.order_book.cancel_order(order_id)
    
    def process_tick(self, tick: Tick) -> None:
        """
        Process incoming tick to update book state.
        
        Used when simulating with historical tick data.
        """
        if tick.tick_type == TickType.TRADE:
            # Trade tick - update last price
            self.order_book.last_trade_price = tick.price
            self.order_book.last_trade_size = tick.size
            self.order_book.total_volume += tick.size
        
        elif tick.tick_type == TickType.BID:
            # Bid quote update
            if tick.price not in self.order_book.bids:
                order = Order(
                    order_id=0,
                    side=Side.BID,
                    price=tick.price,
                    size=tick.size,
                    timestamp_ns=tick.timestamp_ns
                )
                self.order_book.add_order(order)
        
        elif tick.tick_type == TickType.ASK:
            # Ask quote update
            if tick.price not in self.order_book.asks:
                order = Order(
                    order_id=0,
                    side=Side.ASK,
                    price=tick.price,
                    size=tick.size,
                    timestamp_ns=tick.timestamp_ns
                )
                self.order_book.add_order(order)
    
    def get_stats(self) -> Dict:
        """Get engine statistics."""
        return {
            'total_fills': len(self.fills),
            'total_trades': len(self.trades),
            'total_volume': sum(f.size for f in self.fills),
            'total_value': sum(f.size * f.price for f in self.fills),
            'avg_trade_size': sum(f.size for f in self.fills) / len(self.fills) if self.fills else 0
        }
