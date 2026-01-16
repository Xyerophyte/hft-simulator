"""
Market Making Strategy for HFT.

The core HFT strategy: continuously quote bid/ask prices
to earn the spread while managing inventory risk.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
from enum import Enum
import time

from ..order_book import OrderBook, Order, Side, OrderType
from ..matching_engine import MatchingEngine, Fill
from ..latency import LatencyModel


@dataclass
class Quote:
    """Represents a two-sided quote."""
    bid_price: float
    bid_size: float
    ask_price: float
    ask_size: float
    timestamp_ns: int
    
    @property
    def spread(self) -> float:
        return self.ask_price - self.bid_price
    
    @property
    def mid_price(self) -> float:
        return (self.bid_price + self.ask_price) / 2
    
    @property
    def spread_bps(self) -> float:
        return (self.spread / self.mid_price) * 10000


@dataclass
class MarketMakerConfig:
    """Market maker configuration."""
    # Quote parameters
    default_spread_bps: float = 2.0       # 2 bps default spread
    min_spread_bps: float = 0.5           # Minimum spread
    max_spread_bps: float = 10.0          # Maximum spread
    
    # Size parameters
    base_size: float = 1.0                # Base quote size
    max_position: float = 10.0            # Maximum inventory
    
    # Inventory management
    inventory_skew_factor: float = 0.1    # Skew per unit inventory
    max_inventory_spread_adjust: float = 5.0  # Max spread adjustment
    
    # Risk parameters
    max_loss_per_trade: float = 100.0     # Stop loss
    daily_loss_limit: float = 1000.0      # Daily loss limit
    
    # Quoting behavior
    quote_refresh_ns: int = 100_000       # 100 microseconds
    cancel_on_fill: bool = True           # Cancel opposite on fill
    
    # Fee structure
    maker_rebate_bps: float = 0.5         # Maker rebate


class InventoryManager:
    """Manages market maker inventory and risk."""
    
    def __init__(self, max_position: float = 10.0):
        self.max_position = max_position
        self.position = 0.0
        self.cash = 0.0
        self.pnl = 0.0
        self.trades: List[Dict] = []
    
    @property
    def is_long(self) -> bool:
        return self.position > 0
    
    @property
    def is_short(self) -> bool:
        return self.position < 0
    
    @property
    def position_pct(self) -> float:
        """Position as percentage of max."""
        if self.max_position == 0:
            return 0.0
        return self.position / self.max_position
    
    def can_buy(self, size: float) -> bool:
        """Check if we can buy more."""
        return self.position + size <= self.max_position
    
    def can_sell(self, size: float) -> bool:
        """Check if we can sell more."""
        return self.position - size >= -self.max_position
    
    def update(self, side: Side, size: float, price: float) -> Dict:
        """Update position from a fill."""
        if side == Side.BID:  # Bought
            self.position += size
            self.cash -= size * price
        else:  # Sold
            self.position -= size
            self.cash += size * price
        
        trade = {
            'side': side.value,
            'size': size,
            'price': price,
            'position': self.position,
            'cash': self.cash,
            'timestamp': time.time()
        }
        self.trades.append(trade)
        return trade
    
    def calculate_pnl(self, current_price: float) -> float:
        """Calculate current PnL."""
        self.pnl = self.cash + self.position * current_price
        return self.pnl
    
    def get_statistics(self) -> Dict:
        """Get inventory statistics."""
        return {
            'position': self.position,
            'position_pct': self.position_pct,
            'cash': self.cash,
            'pnl': self.pnl,
            'num_trades': len(self.trades),
            'is_flat': abs(self.position) < 0.001
        }


class MarketMaker:
    """
    Market Making Strategy.
    
    Core HFT strategy that:
    1. Continuously quotes bid/ask prices
    2. Earns the bid-ask spread
    3. Manages inventory risk
    4. Adjusts quotes based on inventory
    
    Key concepts:
    - Adverse selection: informed traders pick off stale quotes
    - Inventory risk: directional exposure from imbalanced fills
    - Quote skewing: adjusting quotes to manage inventory
    """
    
    def __init__(
        self,
        config: Optional[MarketMakerConfig] = None,
        latency_model: Optional[LatencyModel] = None,
        seed: int = 42
    ):
        self.config = config or MarketMakerConfig()
        self.latency = latency_model or LatencyModel()
        self.inventory = InventoryManager(self.config.max_position)
        self.rng = np.random.default_rng(seed)
        
        # Quote management
        self.current_quote: Optional[Quote] = None
        self.bid_order: Optional[Order] = None
        self.ask_order: Optional[Order] = None
        
        # Performance tracking
        self.quotes_sent = 0
        self.fills_received = 0
        self.spread_captured = 0.0
        self.adverse_selection_count = 0
        
        # State
        self.is_active = False
        self.last_quote_ns = 0
    
    def calculate_quote(
        self,
        mid_price: float,
        volatility: float = 0.001,
        order_flow_imbalance: float = 0.0
    ) -> Quote:
        """
        Calculate optimal quote.
        
        Args:
            mid_price: Current mid price
            volatility: Recent price volatility
            order_flow_imbalance: Buy/sell imbalance (-1 to 1)
            
        Returns:
            Quote with bid/ask prices and sizes
        """
        # Base half-spread
        half_spread = mid_price * (self.config.default_spread_bps / 10000 / 2)
        
        # Adjust for volatility (wider spread = higher vol)
        vol_adjust = volatility * mid_price * 0.5
        half_spread += vol_adjust
        
        # Inventory skew: if long, lower bid/raise ask to reduce inventory
        inventory_skew = self.inventory.position_pct * self.config.inventory_skew_factor
        
        # Order flow adjustment: if buyers dominate, raise prices
        flow_adjust = order_flow_imbalance * half_spread * 0.2
        
        # Calculate prices
        bid_price = mid_price - half_spread * (1 + inventory_skew) - flow_adjust
        ask_price = mid_price + half_spread * (1 - inventory_skew) - flow_adjust
        
        # Ensure minimum spread
        min_spread = mid_price * (self.config.min_spread_bps / 10000)
        if ask_price - bid_price < min_spread:
            adjustment = (min_spread - (ask_price - bid_price)) / 2
            bid_price -= adjustment
            ask_price += adjustment
        
        # Size based on inventory
        if self.inventory.can_buy(self.config.base_size):
            bid_size = self.config.base_size
        else:
            bid_size = max(0, self.config.max_position - self.inventory.position)
        
        if self.inventory.can_sell(self.config.base_size):
            ask_size = self.config.base_size
        else:
            ask_size = max(0, self.config.max_position + self.inventory.position)
        
        return Quote(
            bid_price=bid_price,
            bid_size=bid_size,
            ask_price=ask_price,
            ask_size=ask_size,
            timestamp_ns=int(time.time() * 1e9)
        )
    
    def submit_quotes(
        self,
        engine: MatchingEngine,
        mid_price: float,
        volatility: float = 0.001,
        order_flow_imbalance: float = 0.0
    ) -> Quote:
        """
        Submit two-sided quote to the market.
        
        Args:
            engine: Matching engine to submit to
            mid_price: Current mid price
            volatility: Recent volatility
            order_flow_imbalance: Order flow imbalance
            
        Returns:
            Submitted quote
        """
        # Cancel existing quotes
        self.cancel_quotes(engine)
        
        # Calculate new quote
        quote = self.calculate_quote(mid_price, volatility, order_flow_imbalance)
        self.current_quote = quote
        
        # Submit bid
        if quote.bid_size > 0:
            result = engine.submit_order(
                side=Side.BID,
                size=quote.bid_size,
                price=quote.bid_price,
                order_type=OrderType.LIMIT,
                timestamp_ns=quote.timestamp_ns
            )
            self.bid_order = result.order
        
        # Submit ask
        if quote.ask_size > 0:
            result = engine.submit_order(
                side=Side.ASK,
                size=quote.ask_size,
                price=quote.ask_price,
                order_type=OrderType.LIMIT,
                timestamp_ns=quote.timestamp_ns
            )
            self.ask_order = result.order
        
        self.quotes_sent += 1
        self.last_quote_ns = quote.timestamp_ns
        self.is_active = True
        
        return quote
    
    def cancel_quotes(self, engine: MatchingEngine) -> None:
        """Cancel all outstanding quotes."""
        if self.bid_order and not self.bid_order.is_complete:
            engine.cancel_order(self.bid_order.order_id)
            self.bid_order = None
        
        if self.ask_order and not self.ask_order.is_complete:
            engine.cancel_order(self.ask_order.order_id)
            self.ask_order = None
    
    def on_fill(self, fill: Fill) -> Dict:
        """
        Handle a fill.
        
        Updates inventory and calculates PnL.
        """
        self.fills_received += 1
        
        # Update inventory
        trade = self.inventory.update(fill.side, fill.size, fill.price)
        
        # Calculate spread captured
        if self.current_quote:
            if fill.side == Side.BID:
                # Bought on bid
                spread_captured = self.current_quote.ask_price - fill.price
            else:
                # Sold on ask
                spread_captured = fill.price - self.current_quote.bid_price
            
            self.spread_captured += spread_captured * fill.size
        
        return trade
    
    def check_adverse_selection(
        self,
        fill: Fill,
        current_mid: float,
        threshold_bps: float = 5.0
    ) -> bool:
        """
        Check if a fill was adversely selected.
        
        Adverse selection occurs when we get filled and the
        price immediately moves against us.
        """
        if fill.side == Side.BID:
            # Bought - check if price dropped
            if (fill.price - current_mid) / fill.price * 10000 > threshold_bps:
                self.adverse_selection_count += 1
                return True
        else:
            # Sold - check if price increased
            if (current_mid - fill.price) / fill.price * 10000 > threshold_bps:
                self.adverse_selection_count += 1
                return True
        
        return False
    
    def should_requote(
        self,
        current_ns: int,
        mid_price: float,
        volatility: float
    ) -> bool:
        """Determine if we should update quotes."""
        if not self.is_active:
            return True
        
        # Time-based refresh
        if current_ns - self.last_quote_ns > self.config.quote_refresh_ns:
            return True
        
        # Price moved significantly
        if self.current_quote:
            price_change = abs(mid_price - self.current_quote.mid_price)
            if price_change / mid_price > volatility:
                return True
        
        return False
    
    def get_performance(self, current_price: float) -> Dict:
        """Get market maker performance statistics."""
        pnl = self.inventory.calculate_pnl(current_price)
        
        return {
            'quotes_sent': self.quotes_sent,
            'fills_received': self.fills_received,
            'fill_rate': self.fills_received / max(1, self.quotes_sent),
            'spread_captured': self.spread_captured,
            'adverse_selection_count': self.adverse_selection_count,
            'adverse_selection_rate': self.adverse_selection_count / max(1, self.fills_received),
            'position': self.inventory.position,
            'cash': self.inventory.cash,
            'pnl': pnl,
            'pnl_per_fill': pnl / max(1, self.fills_received)
        }


class SimpleMarketMaker:
    """
    Simplified market maker for quick simulations.
    
    Focuses on the core concept without all the bells and whistles.
    """
    
    def __init__(
        self,
        spread_bps: float = 2.0,
        size: float = 1.0,
        max_inventory: float = 10.0
    ):
        self.spread_bps = spread_bps
        self.size = size
        self.max_inventory = max_inventory
        self.inventory = 0.0
        self.pnl = 0.0
        self.trades: List[Dict] = []
    
    def make_market(
        self,
        mid_price: float,
        incoming_side: Optional[Side] = None,
        incoming_size: float = 0.0
    ) -> Dict:
        """
        Process market making logic.
        
        If incoming order matches our quote, execute trade.
        
        Args:
            mid_price: Current mid price
            incoming_side: Side of incoming order (if any)
            incoming_size: Size of incoming order
            
        Returns:
            Result dict with quote and trade info
        """
        # Calculate quote
        half_spread = mid_price * (self.spread_bps / 10000 / 2)
        
        # Skew based on inventory
        skew = (self.inventory / self.max_inventory) * half_spread * 0.5
        
        bid = mid_price - half_spread - skew
        ask = mid_price + half_spread - skew
        
        result = {
            'bid': bid,
            'ask': ask,
            'spread_bps': self.spread_bps,
            'inventory': self.inventory,
            'mid': mid_price,
            'traded': False
        }
        
        # Process incoming order
        if incoming_side == Side.BID and incoming_size > 0:
            # Someone buying at our ask
            if abs(self.inventory - incoming_size) <= self.max_inventory:
                trade_size = min(incoming_size, self.size)
                self.inventory -= trade_size
                self.pnl += trade_size * (ask - mid_price)
                result['traded'] = True
                result['trade_side'] = 'sell'
                result['trade_price'] = ask
                result['trade_size'] = trade_size
                
        elif incoming_side == Side.ASK and incoming_size > 0:
            # Someone selling at our bid
            if abs(self.inventory + incoming_size) <= self.max_inventory:
                trade_size = min(incoming_size, self.size)
                self.inventory += trade_size
                self.pnl += trade_size * (mid_price - bid)
                result['traded'] = True
                result['trade_side'] = 'buy'
                result['trade_price'] = bid
                result['trade_size'] = trade_size
        
        return result
