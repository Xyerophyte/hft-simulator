"""
Matching engine for simulating order execution with price-time priority.
Handles market and limit orders with basic slippage modeling.
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass
import time

try:
    from .orderbook import OrderBook, Order, OrderSide, OrderType
except ImportError:
    from orderbook import OrderBook, Order, OrderSide, OrderType


@dataclass
class Trade:
    """Represents an executed trade."""
    trade_id: str
    timestamp: float
    buyer_order_id: str
    seller_order_id: str
    price: float
    quantity: float
    buyer_is_taker: bool  # True if buyer initiated (market/aggressive order)
    
    @property
    def value(self) -> float:
        """Trade value (price * quantity)."""
        return self.price * self.quantity


@dataclass
class Fill:
    """Represents a partial or complete order fill."""
    order_id: str
    price: float
    quantity: float
    timestamp: float
    fee: float = 0.0
    
    @property
    def total_cost(self) -> float:
        """Total cost including fees."""
        return (self.price * self.quantity) + self.fee


class MatchingEngine:
    """
    Matches orders in the order book with price-time priority.
    Supports market and limit orders with configurable fees and slippage.
    """
    
    def __init__(
        self,
        maker_fee: float = 0.0001,  # 0.01% maker fee
        taker_fee: float = 0.0002,  # 0.02% taker fee
        slippage_pct: float = 0.0005  # 0.05% slippage for market orders
    ):
        """
        Initialize matching engine.
        
        Args:
            maker_fee: Fee for providing liquidity (as decimal)
            taker_fee: Fee for taking liquidity (as decimal)
            slippage_pct: Slippage percentage for market orders
        """
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.slippage_pct = slippage_pct
        
        self.trade_counter = 0
        self.trades: List[Trade] = []
    
    def generate_trade_id(self) -> str:
        """Generate unique trade ID."""
        self.trade_counter += 1
        return f"TRD_{self.trade_counter}_{int(time.time() * 1000000)}"
    
    def match_market_order(
        self,
        order: Order,
        order_book: OrderBook
    ) -> List[Fill]:
        """
        Match a market order against the book.
        
        Args:
            order: Market order to execute
            order_book: Current order book
            
        Returns:
            List of fills for the order
        """
        if order.order_type != OrderType.MARKET:
            raise ValueError("Order must be a market order")
        
        fills = []
        remaining_qty = order.quantity
        
        # Determine which side of the book to match against
        opposite_side_name = "asks" if order.side == OrderSide.BUY else "bids"
        
        # Get book side
        book_side = getattr(order_book, opposite_side_name)
        
        if not book_side:
            # No liquidity - order cannot be filled
            return fills
        
        # Sort price levels (ascending for asks, descending for bids)
        sorted_prices = sorted(
            book_side.keys(),
            reverse=(order.side == OrderSide.SELL)
        )
        
        # Match against each price level
        for price in sorted_prices:
            if remaining_qty <= 0:
                break
            
            price_level = book_side[price]
            
            # Apply slippage to execution price
            execution_price = self._apply_slippage(price, order.side)
            
            # Match against orders at this level (FIFO)
            for book_order in list(price_level.orders):
                if remaining_qty <= 0:
                    break
                
                # Calculate fill quantity
                fill_qty = min(remaining_qty, book_order.remaining_quantity)
                
                # Create fill
                fill = Fill(
                    order_id=order.order_id,
                    price=execution_price,
                    quantity=fill_qty,
                    timestamp=time.time(),
                    fee=self._calculate_fee(execution_price, fill_qty, is_taker=True)
                )
                fills.append(fill)
                
                # Update order quantities
                order.fill(fill_qty)
                book_order.fill(fill_qty)
                
                # Record trade
                trade = Trade(
                    trade_id=self.generate_trade_id(),
                    timestamp=fill.timestamp,
                    buyer_order_id=order.order_id if order.side == OrderSide.BUY else book_order.order_id,
                    seller_order_id=book_order.order_id if order.side == OrderSide.BUY else order.order_id,
                    price=execution_price,
                    quantity=fill_qty,
                    buyer_is_taker=(order.side == OrderSide.BUY)
                )
                self.trades.append(trade)
                
                remaining_qty -= fill_qty
                
                # Remove fully filled book order
                if book_order.is_filled:
                    order_book.cancel_order(book_order.order_id)
        
        return fills
    
    def match_limit_order(
        self,
        order: Order,
        order_book: OrderBook
    ) -> List[Fill]:
        """
        Match a limit order against the book.
        Only executes if price crosses the spread.
        
        Args:
            order: Limit order to execute
            order_book: Current order book
            
        Returns:
            List of fills for the order (empty if no match)
        """
        if order.order_type != OrderType.LIMIT:
            raise ValueError("Order must be a limit order")
        
        fills = []
        
        # Check if order crosses the spread
        if order.side == OrderSide.BUY:
            best_ask = order_book.get_best_ask()
            if best_ask is None or order.price < best_ask:
                # No match - add to book
                order_book.add_order(order)
                return fills
        else:  # SELL
            best_bid = order_book.get_best_bid()
            if best_bid is None or order.price > best_bid:
                # No match - add to book
                order_book.add_order(order)
                return fills
        
        # Order crosses spread - match as aggressive order
        remaining_qty = order.quantity
        opposite_side_name = "asks" if order.side == OrderSide.BUY else "bids"
        book_side = getattr(order_book, opposite_side_name)
        
        # Sort price levels
        sorted_prices = sorted(
            book_side.keys(),
            reverse=(order.side == OrderSide.SELL)
        )
        
        # Match against each price level
        for price in sorted_prices:
            if remaining_qty <= 0:
                break
            
            # Check if we've exhausted prices within our limit
            if order.side == OrderSide.BUY and price > order.price:
                break
            if order.side == OrderSide.SELL and price < order.price:
                break
            
            price_level = book_side[price]
            
            # Match against orders at this level
            for book_order in list(price_level.orders):
                if remaining_qty <= 0:
                    break
                
                fill_qty = min(remaining_qty, book_order.remaining_quantity)
                
                # Create fill at matched price
                fill = Fill(
                    order_id=order.order_id,
                    price=price,  # Match at book price
                    quantity=fill_qty,
                    timestamp=time.time(),
                    fee=self._calculate_fee(price, fill_qty, is_taker=True)
                )
                fills.append(fill)
                
                # Update quantities
                order.fill(fill_qty)
                book_order.fill(fill_qty)
                
                # Record trade
                trade = Trade(
                    trade_id=self.generate_trade_id(),
                    timestamp=fill.timestamp,
                    buyer_order_id=order.order_id if order.side == OrderSide.BUY else book_order.order_id,
                    seller_order_id=book_order.order_id if order.side == OrderSide.BUY else order.order_id,
                    price=price,
                    quantity=fill_qty,
                    buyer_is_taker=(order.side == OrderSide.BUY)
                )
                self.trades.append(trade)
                
                remaining_qty -= fill_qty
                
                # Remove fully filled book order
                if book_order.is_filled:
                    order_book.cancel_order(book_order.order_id)
        
        # Add remaining quantity to book if not fully filled
        if not order.is_filled:
            order_book.add_order(order)
        
        return fills
    
    def _apply_slippage(self, price: float, side: OrderSide) -> float:
        """
        Apply slippage to execution price.
        
        Args:
            price: Base price
            side: Order side
            
        Returns:
            Price with slippage
        """
        slippage = price * self.slippage_pct
        
        if side == OrderSide.BUY:
            # Pay more when buying
            return price + slippage
        else:
            # Receive less when selling
            return price - slippage
    
    def _calculate_fee(
        self,
        price: float,
        quantity: float,
        is_taker: bool
    ) -> float:
        """
        Calculate trading fee.
        
        Args:
            price: Execution price
            quantity: Trade quantity
            is_taker: Whether order takes liquidity
            
        Returns:
            Fee amount
        """
        notional = price * quantity
        fee_rate = self.taker_fee if is_taker else self.maker_fee
        return notional * fee_rate
    
    def get_average_fill_price(self, fills: List[Fill]) -> Optional[float]:
        """
        Calculate volume-weighted average fill price.
        
        Args:
            fills: List of fills
            
        Returns:
            Average price or None if no fills
        """
        if not fills:
            return None
        
        total_value = sum(f.price * f.quantity for f in fills)
        total_quantity = sum(f.quantity for f in fills)
        
        if total_quantity == 0:
            return None
        
        return total_value / total_quantity
    
    def get_total_fees(self, fills: List[Fill]) -> float:
        """Calculate total fees from fills."""
        return sum(f.fee for f in fills)
    
    def get_trade_statistics(self) -> dict:
        """
        Get statistics about executed trades.
        
        Returns:
            Dictionary with trade statistics
        """
        if not self.trades:
            return {
                'total_trades': 0,
                'total_volume': 0.0,
                'total_value': 0.0,
                'avg_trade_size': 0.0
            }
        
        total_volume = sum(t.quantity for t in self.trades)
        total_value = sum(t.value for t in self.trades)
        
        return {
            'total_trades': len(self.trades),
            'total_volume': total_volume,
            'total_value': total_value,
            'avg_trade_size': total_volume / len(self.trades),
            'avg_trade_price': total_value / total_volume if total_volume > 0 else 0.0
        }


# Example usage
if __name__ == "__main__":
    from orderbook import OrderBook, Order, OrderSide, OrderType
    
    # Create order book and engine
    book = OrderBook()
    engine = MatchingEngine()
    
    # Add some limit orders to the book
    for i in range(5):
        bid = Order(
            order_id=book.generate_order_id(),
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            price=100.0 - i * 0.5,
            quantity=10.0,
            timestamp=time.time()
        )
        book.add_order(bid)
    
    for i in range(5):
        ask = Order(
            order_id=book.generate_order_id(),
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            price=101.0 + i * 0.5,
            quantity=10.0,
            timestamp=time.time()
        )
        book.add_order(ask)
    
    print("Initial order book:")
    print(book)
    print()
    
    # Execute a market buy order
    market_order = Order(
        order_id=book.generate_order_id(),
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        price=None,
        quantity=25.0,
        timestamp=time.time()
    )
    
    fills = engine.match_market_order(market_order, book)
    
    print(f"Market order fills: {len(fills)}")
    for fill in fills:
        print(f"  Filled {fill.quantity} @ ${fill.price:.2f} (fee: ${fill.fee:.4f})")
    
    print(f"\nAverage fill price: ${engine.get_average_fill_price(fills):.2f}")
    print(f"Total fees: ${engine.get_total_fees(fills):.4f}")
    
    print("\nOrder book after execution:")
    print(book)
    
    print("\nTrade statistics:")
    stats = engine.get_trade_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")