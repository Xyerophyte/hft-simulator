"""
HFT Module - High-Frequency Trading Components

This package contains true HFT simulation components:
- Tick data processing
- Order book simulation (L2/L3)
- Matching engine
- Latency modeling
- Market making strategies
"""

from .tick_data import Tick, TickStream, TickAggregator
from .order_book import OrderBook, Order, OrderType, Side
from .matching_engine import MatchingEngine
from .latency import LatencyModel
from .execution import ExecutionSimulator

__all__ = [
    'Tick', 'TickStream', 'TickAggregator',
    'OrderBook', 'Order', 'OrderType', 'Side',
    'MatchingEngine',
    'LatencyModel',
    'ExecutionSimulator'
]
