"""
Latency Modeling for HFT Simulation.

Models realistic network and exchange latencies that are
critical to HFT performance.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable
from enum import Enum
import time


class LatencyType(Enum):
    """Types of latency in HFT systems."""
    NETWORK = "network"          # Network round-trip
    EXCHANGE = "exchange"        # Exchange processing
    QUEUE = "queue"              # Queue position wait
    INTERNAL = "internal"        # Internal processing


@dataclass
class LatencyConfig:
    """
    Latency configuration for different participant types.
    
    All values in nanoseconds.
    """
    # Network latency (one-way)
    network_mean_ns: int = 50_000          # 50 microseconds
    network_std_ns: int = 10_000           # 10 microseconds jitter
    
    # Exchange processing time
    exchange_mean_ns: int = 5_000          # 5 microseconds
    exchange_std_ns: int = 1_000           # 1 microsecond jitter
    
    # Internal processing
    internal_mean_ns: int = 1_000          # 1 microsecond
    internal_std_ns: int = 500             # 0.5 microsecond jitter
    
    # Co-location advantage (reduces network latency)
    is_colocated: bool = False
    colocation_factor: float = 0.1         # 10x faster if colocated


@dataclass
class LatencyProfile:
    """Latency profile for different market participants."""
    
    # Retail trader (high latency)
    RETAIL = LatencyConfig(
        network_mean_ns=50_000_000,        # 50ms
        network_std_ns=10_000_000,         # 10ms jitter
        exchange_mean_ns=100_000,          # 100us
        exchange_std_ns=20_000,
        internal_mean_ns=1_000_000,        # 1ms
        internal_std_ns=200_000,
        is_colocated=False
    )
    
    # Institutional trader
    INSTITUTIONAL = LatencyConfig(
        network_mean_ns=1_000_000,         # 1ms
        network_std_ns=200_000,            # 200us jitter
        exchange_mean_ns=50_000,           # 50us
        exchange_std_ns=10_000,
        internal_mean_ns=100_000,          # 100us
        internal_std_ns=20_000,
        is_colocated=False
    )
    
    # HFT firm (co-located)
    HFT = LatencyConfig(
        network_mean_ns=5_000,             # 5 microseconds
        network_std_ns=1_000,              # 1us jitter
        exchange_mean_ns=2_000,            # 2us
        exchange_std_ns=500,
        internal_mean_ns=500,              # 0.5us
        internal_std_ns=100,
        is_colocated=True,
        colocation_factor=0.1
    )


class LatencyModel:
    """
    Models realistic latency for HFT simulation.
    
    Features:
    - Network round-trip simulation
    - Exchange processing delays
    - Queue position delays
    - Co-location advantage
    """
    
    def __init__(
        self,
        config: Optional[LatencyConfig] = None,
        seed: int = 42
    ):
        self.config = config or LatencyConfig()
        self.rng = np.random.default_rng(seed)
        
        # Latency statistics
        self.total_latency_ns = 0
        self.latency_samples: list = []
    
    def get_network_latency(self) -> int:
        """
        Get network latency in nanoseconds.
        
        Models round-trip time with jitter.
        """
        base = self.config.network_mean_ns
        
        if self.config.is_colocated:
            base = int(base * self.config.colocation_factor)
        
        # Add jitter (log-normal for realistic tail)
        jitter = self.rng.lognormal(
            mean=0,
            sigma=self.config.network_std_ns / self.config.network_mean_ns
        )
        
        latency = int(base * jitter)
        self._record_latency(latency, LatencyType.NETWORK)
        return latency
    
    def get_exchange_latency(self) -> int:
        """Get exchange processing latency in nanoseconds."""
        latency = int(self.rng.normal(
            self.config.exchange_mean_ns,
            self.config.exchange_std_ns
        ))
        latency = max(100, latency)  # Minimum 100ns
        self._record_latency(latency, LatencyType.EXCHANGE)
        return latency
    
    def get_internal_latency(self) -> int:
        """Get internal processing latency in nanoseconds."""
        latency = int(self.rng.normal(
            self.config.internal_mean_ns,
            self.config.internal_std_ns
        ))
        latency = max(10, latency)  # Minimum 10ns
        self._record_latency(latency, LatencyType.INTERNAL)
        return latency
    
    def get_queue_latency(self, queue_position: int, fill_rate: float = 1000.0) -> int:
        """
        Get queue position latency in nanoseconds.
        
        Args:
            queue_position: Shares ahead in queue
            fill_rate: Expected fills per second at this price
            
        Returns:
            Expected wait time in nanoseconds
        """
        if queue_position <= 0:
            return 0
        
        # Expected time = position / fill_rate (in seconds)
        expected_seconds = queue_position / fill_rate
        
        # Add randomness (exponential distribution)
        actual_seconds = self.rng.exponential(expected_seconds)
        
        latency = int(actual_seconds * 1e9)
        self._record_latency(latency, LatencyType.QUEUE)
        return latency
    
    def get_total_order_latency(self, queue_position: int = 0) -> int:
        """
        Get total latency for order execution.
        
        Includes: internal + network (to exchange) + exchange + 
                  queue + network (response)
        """
        total = 0
        
        # Internal processing
        total += self.get_internal_latency()
        
        # Network to exchange
        total += self.get_network_latency() // 2  # One-way
        
        # Exchange processing
        total += self.get_exchange_latency()
        
        # Queue wait (if limit order)
        if queue_position > 0:
            total += self.get_queue_latency(queue_position)
        
        # Network response
        total += self.get_network_latency() // 2  # One-way
        
        return total
    
    def simulate_race_condition(
        self,
        our_latency: int,
        competitor_latencies: list[int]
    ) -> bool:
        """
        Simulate race condition against competitors.
        
        Returns True if we win (arrive first).
        """
        our_arrival = our_latency
        
        for comp_latency in competitor_latencies:
            if comp_latency < our_arrival:
                return False
        
        return True
    
    def _record_latency(self, latency: int, latency_type: LatencyType) -> None:
        """Record latency sample for statistics."""
        self.total_latency_ns += latency
        self.latency_samples.append({
            'latency_ns': latency,
            'type': latency_type.value,
            'timestamp': time.time()
        })
    
    def get_statistics(self) -> dict:
        """Get latency statistics."""
        if not self.latency_samples:
            return {}
        
        latencies = [s['latency_ns'] for s in self.latency_samples]
        
        return {
            'mean_ns': np.mean(latencies),
            'median_ns': np.median(latencies),
            'std_ns': np.std(latencies),
            'min_ns': np.min(latencies),
            'max_ns': np.max(latencies),
            'p99_ns': np.percentile(latencies, 99),
            'total_samples': len(latencies),
            'total_latency_ns': self.total_latency_ns
        }
    
    def reset_statistics(self) -> None:
        """Reset latency statistics."""
        self.total_latency_ns = 0
        self.latency_samples.clear()


class LatencySimulator:
    """
    Simulates latency impact on trading decisions.
    
    Models scenarios where latency matters:
    - Quote stuffing detection
    - Stale quote arbitrage
    - Race conditions
    """
    
    def __init__(self, latency_model: LatencyModel):
        self.latency = latency_model
    
    def is_quote_stale(
        self,
        quote_age_ns: int,
        threshold_ns: int = 1_000_000  # 1ms default
    ) -> bool:
        """Check if a quote is stale."""
        return quote_age_ns > threshold_ns
    
    def can_beat_to_book(
        self,
        signal_age_ns: int,
        our_total_latency_ns: int,
        competitor_advantage_ns: int = 0
    ) -> bool:
        """
        Estimate if we can beat competitors to the book.
        
        Args:
            signal_age_ns: Age of the signal we're reacting to
            our_total_latency_ns: Our total order latency
            competitor_advantage_ns: Competitor's latency advantage
        """
        # We need to arrive before the opportunity disappears
        opportunity_window_ns = 10_000_000  # 10ms typical
        
        time_to_arrive = signal_age_ns + our_total_latency_ns
        competitor_time = signal_age_ns + our_total_latency_ns - competitor_advantage_ns
        
        if time_to_arrive > opportunity_window_ns:
            return False
        
        if competitor_advantage_ns > 0 and competitor_time < time_to_arrive:
            return False
        
        return True
    
    def estimate_fill_probability(
        self,
        our_latency_ns: int,
        queue_position: int,
        order_size: float,
        available_liquidity: float
    ) -> float:
        """
        Estimate probability of getting filled.
        
        Considers:
        - Latency (faster = better chance)
        - Queue position
        - Order size vs available liquidity
        """
        # Base probability from liquidity
        if available_liquidity <= 0:
            return 0.0
        
        base_prob = min(1.0, available_liquidity / (order_size + available_liquidity))
        
        # Latency penalty (slower = lower prob)
        latency_penalty = 1.0 / (1.0 + our_latency_ns / 1_000_000_000)
        
        # Queue penalty
        queue_penalty = 1.0 / (1.0 + queue_position / 1000)
        
        return base_prob * latency_penalty * queue_penalty
