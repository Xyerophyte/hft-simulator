"""
Event-Driven HFT Simulator.

Processes tick-by-tick data with realistic timing
and strategy execution.
"""

import time
import heapq
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable, Any
from enum import Enum
import numpy as np

from .tick_data import Tick, TickStream, TickType
from .order_book import OrderBook, Order, Side
from .matching_engine import MatchingEngine, Fill
from .latency import LatencyModel
from .execution import ExecutionSimulator
from .strategies.market_maker import MarketMaker, SimpleMarketMaker


class EventType(Enum):
    """Types of events in the simulation."""
    TICK = "tick"                    # Market data tick
    ORDER_SUBMIT = "order_submit"    # Order submission
    ORDER_FILL = "order_fill"        # Order filled
    ORDER_CANCEL = "order_cancel"    # Order cancelled
    QUOTE_UPDATE = "quote_update"    # Quote refresh
    STRATEGY = "strategy"            # Strategy action


@dataclass(order=True)
class Event:
    """Event in the simulation."""
    timestamp_ns: int
    event_type: EventType = field(compare=False)
    data: Dict = field(default_factory=dict, compare=False)
    callback: Optional[Callable] = field(default=None, compare=False)


class EventLoop:
    """
    Priority queue based event loop.
    
    Events are processed in timestamp order to maintain
    causality in the simulation.
    """
    
    def __init__(self):
        self.queue: List[Event] = []
        self.current_time_ns = 0
        self.events_processed = 0
    
    def schedule(self, event: Event) -> None:
        """Schedule an event."""
        heapq.heappush(self.queue, event)
    
    def schedule_at(
        self,
        timestamp_ns: int,
        event_type: EventType,
        data: Dict = None,
        callback: Callable = None
    ) -> None:
        """Schedule event at specific time."""
        event = Event(
            timestamp_ns=timestamp_ns,
            event_type=event_type,
            data=data or {},
            callback=callback
        )
        self.schedule(event)
    
    def next_event(self) -> Optional[Event]:
        """Get next event."""
        if not self.queue:
            return None
        
        event = heapq.heappop(self.queue)
        self.current_time_ns = event.timestamp_ns
        self.events_processed += 1
        return event
    
    def has_events(self) -> bool:
        return len(self.queue) > 0
    
    def clear(self) -> None:
        self.queue.clear()
        self.events_processed = 0


@dataclass
class SimulationConfig:
    """HFT simulation configuration."""
    # Simulation settings
    warmup_ticks: int = 100           # Ticks before trading
    
    # Order book initialization
    initial_mid_price: float = 50000.0
    initial_spread_bps: float = 2.0
    initial_depth_levels: int = 10
    initial_size_per_level: float = 10.0
    
    # Latency profile
    use_hft_latency: bool = True
    
    # Strategy
    strategy_type: str = "market_maker"  # market_maker, stat_arb, latency_arb
    
    # Risk limits
    max_position: float = 10.0
    daily_loss_limit: float = 1000.0


class HFTSimulator:
    """
    Complete HFT simulation environment.
    
    Features:
    - Tick-by-tick data processing
    - Realistic order book simulation
    - Latency modeling
    - Multiple strategy support
    - Performance analytics
    """
    
    def __init__(
        self,
        config: Optional[SimulationConfig] = None,
        seed: int = 42
    ):
        self.config = config or SimulationConfig()
        self.rng = np.random.default_rng(seed)
        
        # Core components
        self.order_book = OrderBook()
        self.engine = MatchingEngine(self.order_book)
        self.latency = LatencyModel()
        self.execution = ExecutionSimulator(self.order_book, self.latency)
        self.event_loop = EventLoop()
        
        # Strategy
        self.strategy = None
        self._init_strategy()
        
        # State
        self.current_tick: Optional[Tick] = None
        self.ticks_processed = 0
        self.is_warmup = True
        
        # Performance tracking
        self.tick_times: List[int] = []
        self.processing_times: List[float] = []
    
    def _init_strategy(self) -> None:
        """Initialize strategy based on config."""
        if self.config.strategy_type == "market_maker":
            self.strategy = SimpleMarketMaker(
                spread_bps=self.config.initial_spread_bps,
                max_inventory=self.config.max_position
            )
        # Add other strategies as needed
    
    def initialize_book(self, mid_price: Optional[float] = None) -> None:
        """Initialize order book with synthetic depth."""
        mid = mid_price or self.config.initial_mid_price
        half_spread = mid * (self.config.initial_spread_bps / 10000 / 2)
        
        bids = []
        asks = []
        
        for i in range(self.config.initial_depth_levels):
            tick_offset = i * (half_spread * 0.1)  # 10% of half-spread per level
            size = self.config.initial_size_per_level * (1 + i * 0.1)  # Increasing size
            
            bid_price = mid - half_spread - tick_offset
            ask_price = mid + half_spread + tick_offset
            
            bids.append((bid_price, size))
            asks.append((ask_price, size))
        
        self.order_book.initialize_from_snapshot(bids, asks)
    
    def process_tick(self, tick: Tick) -> Dict:
        """
        Process a single tick.
        
        Returns processing result with any trades/signals.
        """
        start_time = time.perf_counter_ns()
        
        self.current_tick = tick
        self.ticks_processed += 1
        
        result = {
            'tick': tick.to_dict(),
            'trades': [],
            'signals': [],
            'book_state': None
        }
        
        # Update order book with tick
        if tick.tick_type == TickType.TRADE:
            # Simulate market order hitting the book
            if tick.side == 'buy':
                side = Side.BID
            elif tick.side == 'sell':
                side = Side.ASK
            else:
                side = Side.BID if self.rng.random() > 0.5 else Side.ASK
            
            # Execute against book
            filled, avg_price, fills = self.order_book.execute_market_order(
                side, tick.size, tick.timestamp_ns
            )
            
            for order, fill_size in fills:
                result['trades'].append({
                    'price': order.price,
                    'size': fill_size,
                    'side': side.value
                })
        
        # Check if warmup complete
        if self.ticks_processed >= self.config.warmup_ticks:
            self.is_warmup = False
        
        # Run strategy if not in warmup
        if not self.is_warmup and self.strategy:
            mid = self.order_book.mid_price or tick.price
            
            if isinstance(self.strategy, SimpleMarketMaker):
                # Determine incoming order side from tick
                if tick.side == 'buy':
                    incoming_side = Side.BID
                elif tick.side == 'sell':
                    incoming_side = Side.ASK
                else:
                    incoming_side = None
                
                mm_result = self.strategy.make_market(
                    mid_price=mid,
                    incoming_side=incoming_side,
                    incoming_size=tick.size
                )
                
                if mm_result.get('traded'):
                    result['signals'].append(mm_result)
        
        # Record book state
        result['book_state'] = {
            'mid': self.order_book.mid_price,
            'spread_bps': self.order_book.spread_bps,
            'best_bid': self.order_book.best_bid,
            'best_ask': self.order_book.best_ask
        }
        
        end_time = time.perf_counter_ns()
        self.processing_times.append((end_time - start_time) / 1000)  # microseconds
        self.tick_times.append(tick.timestamp_ns)
        
        return result
    
    def run(self, tick_stream: TickStream) -> Dict:
        """
        Run simulation on tick stream.
        
        Returns simulation results.
        """
        print(f"Starting HFT simulation with {len(tick_stream)} ticks...")
        
        # Initialize book from first tick
        first_tick = tick_stream.ticks[0] if tick_stream.ticks else None
        if first_tick:
            self.initialize_book(first_tick.price)
        
        results = {
            'ticks_processed': 0,
            'trades': [],
            'strategy_trades': [],
            'final_pnl': 0.0,
            'final_position': 0.0,
            'processing_stats': {}
        }
        
        # Process each tick
        for tick in tick_stream:
            tick_result = self.process_tick(tick)
            results['ticks_processed'] += 1
            
            if tick_result['trades']:
                results['trades'].extend(tick_result['trades'])
            
            if tick_result['signals']:
                results['strategy_trades'].extend(tick_result['signals'])
        
        # Get final stats
        if isinstance(self.strategy, SimpleMarketMaker):
            results['final_pnl'] = self.strategy.pnl
            results['final_position'] = self.strategy.inventory
        
        # Processing stats
        if self.processing_times:
            results['processing_stats'] = {
                'mean_us': np.mean(self.processing_times),
                'median_us': np.median(self.processing_times),
                'p99_us': np.percentile(self.processing_times, 99),
                'max_us': np.max(self.processing_times),
                'total_ticks': len(self.processing_times)
            }
        
        return results
    
    def get_stats(self) -> Dict:
        """Get simulation statistics."""
        stats = {
            'ticks_processed': self.ticks_processed,
            'is_warmup': self.is_warmup,
            'book_mid': self.order_book.mid_price,
            'book_spread_bps': self.order_book.spread_bps
        }
        
        if isinstance(self.strategy, SimpleMarketMaker):
            stats['strategy_pnl'] = self.strategy.pnl
            stats['strategy_inventory'] = self.strategy.inventory
        
        if self.processing_times:
            stats['avg_processing_us'] = np.mean(self.processing_times)
        
        return stats


def run_quick_demo():
    """Run a quick HFT simulation demo."""
    import pandas as pd
    
    print("=" * 60)
    print("HFT SIMULATOR DEMO")
    print("=" * 60)
    
    # Generate synthetic tick data
    print("\n1. Generating synthetic tick data...")
    n_ticks = 1000
    
    np.random.seed(42)
    base_price = 50000
    prices = base_price + np.cumsum(np.random.randn(n_ticks) * 10)
    sizes = np.abs(np.random.exponential(1.0, n_ticks))
    sides = np.random.choice(['buy', 'sell'], n_ticks)
    
    stream = TickStream()
    base_ns = int(time.time() * 1e9)
    
    for i in range(n_ticks):
        tick = Tick(
            timestamp_ns=base_ns + i * 100_000,  # 100us between ticks
            price=prices[i],
            size=sizes[i],
            tick_type=TickType.TRADE,
            side=sides[i]
        )
        stream.add_tick(tick)
    
    print(f"   Generated {n_ticks} ticks")
    
    # Run simulation
    print("\n2. Running HFT simulation...")
    config = SimulationConfig(
        warmup_ticks=100,
        initial_mid_price=base_price,
        strategy_type="market_maker"
    )
    
    sim = HFTSimulator(config)
    results = sim.run(stream)
    
    # Print results
    print("\n3. Results:")
    print(f"   Ticks processed: {results['ticks_processed']}")
    print(f"   Market trades: {len(results['trades'])}")
    print(f"   Strategy trades: {len(results['strategy_trades'])}")
    print(f"   Final PnL: ${results['final_pnl']:.2f}")
    print(f"   Final position: {results['final_position']:.4f}")
    
    if results['processing_stats']:
        stats = results['processing_stats']
        print(f"\n   Processing latency:")
        print(f"     Mean: {stats['mean_us']:.2f} µs")
        print(f"     P99: {stats['p99_us']:.2f} µs")
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    run_quick_demo()
