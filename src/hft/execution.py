"""
Execution Simulation for HFT.

Realistic order execution with slippage, partial fills,
and fill probability modeling.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
from enum import Enum
import time

from .order_book import OrderBook, Order, Side, OrderStatus
from .latency import LatencyModel


class FillModel(Enum):
    """Fill simulation models."""
    INSTANT = "instant"          # Immediate fill at mid
    REALISTIC = "realistic"      # Based on book depth
    AGGRESSIVE = "aggressive"    # Assumes full fill
    CONSERVATIVE = "conservative"  # Assumes worst case


@dataclass
class ExecutionConfig:
    """Execution simulation configuration."""
    # Slippage model
    base_slippage_bps: float = 0.5       # 0.5 bps base
    volume_impact_factor: float = 0.1    # Impact per 1% of ADV
    
    # Fill probability
    min_fill_probability: float = 0.8    # Min prob for market orders
    limit_fill_decay: float = 0.1        # Decay per tick from mid
    
    # Partial fills
    allow_partial_fills: bool = True
    min_partial_pct: float = 0.1         # Min 10% fill
    
    # Fees
    maker_fee_bps: float = -0.5          # Maker rebate
    taker_fee_bps: float = 2.0           # Taker fee


@dataclass
class ExecutionResult:
    """Result of execution simulation."""
    filled: bool
    filled_size: float
    fill_price: float
    slippage_bps: float
    fees: float
    latency_ns: int
    timestamp_ns: int
    partial: bool = False
    reason: str = ""


class ExecutionSimulator:
    """
    Simulates realistic order execution.
    
    Features:
    - Market impact modeling
    - Slippage estimation
    - Partial fill simulation
    - Fill probability based on book depth
    """
    
    def __init__(
        self,
        order_book: Optional[OrderBook] = None,
        latency_model: Optional[LatencyModel] = None,
        config: Optional[ExecutionConfig] = None,
        seed: int = 42
    ):
        self.order_book = order_book or OrderBook()
        self.latency = latency_model or LatencyModel()
        self.config = config or ExecutionConfig()
        self.rng = np.random.default_rng(seed)
        
        # Execution history
        self.executions: List[ExecutionResult] = []
        self.total_volume = 0.0
        self.total_fees = 0.0
    
    def execute_market_order(
        self,
        side: Side,
        size: float,
        timestamp_ns: Optional[int] = None
    ) -> ExecutionResult:
        """
        Execute market order with realistic simulation.
        
        Args:
            side: BID (buy) or ASK (sell)
            size: Order size
            timestamp_ns: Execution timestamp
            
        Returns:
            ExecutionResult with fill details
        """
        if timestamp_ns is None:
            timestamp_ns = int(time.time() * 1e9)
        
        # Get latency
        latency = self.latency.get_total_order_latency()
        exec_timestamp = timestamp_ns + latency
        
        # Check available liquidity
        if side == Side.BID:
            available = self._get_total_ask_liquidity()
        else:
            available = self._get_total_bid_liquidity()
        
        if available <= 0:
            return ExecutionResult(
                filled=False,
                filled_size=0,
                fill_price=0,
                slippage_bps=0,
                fees=0,
                latency_ns=latency,
                timestamp_ns=exec_timestamp,
                reason="No liquidity"
            )
        
        # Calculate fill
        fill_size = min(size, available)
        if self.config.allow_partial_fills and fill_size < size:
            partial = True
        else:
            partial = False
            fill_size = size if available >= size else 0
        
        if fill_size <= 0:
            return ExecutionResult(
                filled=False,
                filled_size=0,
                fill_price=0,
                slippage_bps=0,
                fees=0,
                latency_ns=latency,
                timestamp_ns=exec_timestamp,
                reason="Insufficient liquidity"
            )
        
        # Calculate fill price with market impact
        fill_price = self._calculate_fill_price(side, fill_size)
        
        # Calculate slippage
        mid = self.order_book.mid_price or fill_price
        if side == Side.BID:
            slippage_bps = (fill_price - mid) / mid * 10000
        else:
            slippage_bps = (mid - fill_price) / mid * 10000
        
        # Calculate fees (taker)
        fees = fill_size * fill_price * (self.config.taker_fee_bps / 10000)
        
        result = ExecutionResult(
            filled=True,
            filled_size=fill_size,
            fill_price=fill_price,
            slippage_bps=slippage_bps,
            fees=fees,
            latency_ns=latency,
            timestamp_ns=exec_timestamp,
            partial=partial,
            reason="Filled"
        )
        
        self._record_execution(result)
        return result
    
    def execute_limit_order(
        self,
        side: Side,
        size: float,
        price: float,
        timestamp_ns: Optional[int] = None,
        max_wait_ns: int = 60_000_000_000  # 60 seconds
    ) -> ExecutionResult:
        """
        Simulate limit order execution.
        
        Args:
            side: BID or ASK
            size: Order size
            price: Limit price
            max_wait_ns: Maximum wait time
            
        Returns:
            ExecutionResult
        """
        if timestamp_ns is None:
            timestamp_ns = int(time.time() * 1e9)
        
        # Get queue position
        queue_position = self._estimate_queue_position(side, price)
        
        # Get latency including queue wait
        latency = self.latency.get_total_order_latency(queue_position)
        
        # Estimate fill probability
        fill_prob = self._estimate_fill_probability(side, price, size)
        
        # Simulate fill
        if self.rng.random() < fill_prob:
            # Calculate partial fill if applicable
            if self.config.allow_partial_fills:
                fill_pct = self.rng.uniform(self.config.min_partial_pct, 1.0)
            else:
                fill_pct = 1.0
            
            fill_size = size * fill_pct
            partial = fill_pct < 1.0
            
            # Maker fees (rebate)
            fees = fill_size * price * (self.config.maker_fee_bps / 10000)
            
            # No slippage for limit orders (fill at limit price)
            result = ExecutionResult(
                filled=True,
                filled_size=fill_size,
                fill_price=price,
                slippage_bps=0,
                fees=fees,
                latency_ns=latency,
                timestamp_ns=timestamp_ns + latency,
                partial=partial,
                reason="Limit filled"
            )
        else:
            result = ExecutionResult(
                filled=False,
                filled_size=0,
                fill_price=0,
                slippage_bps=0,
                fees=0,
                latency_ns=latency,
                timestamp_ns=timestamp_ns + latency,
                reason="Limit not filled"
            )
        
        self._record_execution(result)
        return result
    
    def _calculate_fill_price(self, side: Side, size: float) -> float:
        """Calculate average fill price with market impact."""
        if side == Side.BID:
            return self.order_book.estimate_market_impact(Side.BID, size)
        else:
            return self.order_book.estimate_market_impact(Side.ASK, size)
    
    def _get_total_bid_liquidity(self) -> float:
        """Get total bid liquidity."""
        return sum(level.total_size for level in self.order_book.bids.values())
    
    def _get_total_ask_liquidity(self) -> float:
        """Get total ask liquidity."""
        return sum(level.total_size for level in self.order_book.asks.values())
    
    def _estimate_queue_position(self, side: Side, price: float) -> int:
        """Estimate queue position for a limit order."""
        if side == Side.BID:
            if price in self.order_book.bids:
                return int(self.order_book.bids[price].total_size)
            return 0
        else:
            if price in self.order_book.asks:
                return int(self.order_book.asks[price].total_size)
            return 0
    
    def _estimate_fill_probability(
        self,
        side: Side,
        price: float,
        size: float
    ) -> float:
        """Estimate probability of limit order fill."""
        mid = self.order_book.mid_price
        if mid is None:
            return 0.5
        
        # Distance from mid in ticks (assume 0.01 tick)
        if side == Side.BID:
            distance = (mid - price) / mid
        else:
            distance = (price - mid) / mid
        
        # Base probability decreases with distance
        base_prob = np.exp(-distance / self.config.limit_fill_decay)
        
        # Adjust for size
        available = self._get_total_bid_liquidity() if side == Side.ASK else self._get_total_ask_liquidity()
        size_factor = min(1.0, available / (size + 1))
        
        return base_prob * size_factor
    
    def _record_execution(self, result: ExecutionResult) -> None:
        """Record execution for statistics."""
        self.executions.append(result)
        if result.filled:
            self.total_volume += result.filled_size
            self.total_fees += result.fees
    
    def get_statistics(self) -> Dict:
        """Get execution statistics."""
        if not self.executions:
            return {}
        
        filled = [e for e in self.executions if e.filled]
        
        if not filled:
            return {
                'total_executions': len(self.executions),
                'filled_executions': 0,
                'fill_rate': 0.0
            }
        
        return {
            'total_executions': len(self.executions),
            'filled_executions': len(filled),
            'fill_rate': len(filled) / len(self.executions),
            'total_volume': self.total_volume,
            'total_fees': self.total_fees,
            'avg_slippage_bps': np.mean([e.slippage_bps for e in filled]),
            'avg_latency_us': np.mean([e.latency_ns / 1000 for e in filled]),
            'partial_fill_rate': sum(1 for e in filled if e.partial) / len(filled)
        }


class MarketImpactModel:
    """
    Models market impact for large orders.
    
    Uses square-root impact model common in HFT research.
    """
    
    def __init__(
        self,
        daily_volume: float = 1_000_000,
        volatility: float = 0.02,
        impact_coefficient: float = 0.1
    ):
        self.daily_volume = daily_volume
        self.volatility = volatility
        self.impact_coefficient = impact_coefficient
    
    def estimate_impact(self, order_size: float, mid_price: float) -> float:
        """
        Estimate market impact using square-root model.
        
        Impact = coefficient * volatility * sqrt(size / ADV) * price
        """
        participation_rate = order_size / self.daily_volume
        
        impact = (
            self.impact_coefficient *
            self.volatility *
            np.sqrt(participation_rate) *
            mid_price
        )
        
        return impact
    
    def estimate_execution_cost(
        self,
        order_size: float,
        mid_price: float,
        spread: float
    ) -> Dict:
        """
        Estimate total execution cost.
        
        Returns:
            Dict with spread cost, impact cost, and total
        """
        # Half spread cost (crossing the spread)
        spread_cost = (spread / 2) * order_size
        
        # Market impact cost
        impact = self.estimate_impact(order_size, mid_price)
        impact_cost = impact * order_size
        
        return {
            'spread_cost': spread_cost,
            'impact_cost': impact_cost,
            'total_cost': spread_cost + impact_cost,
            'cost_bps': (spread_cost + impact_cost) / (order_size * mid_price) * 10000
        }
