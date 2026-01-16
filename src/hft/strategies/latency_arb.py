"""
Latency Arbitrage Strategy for HFT.

Exploits speed advantage to trade on information
before slower participants.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict
from enum import Enum
import time

from ..order_book import OrderBook, Side
from ..latency import LatencyModel, LatencyConfig


@dataclass
class LatencyArbConfig:
    """Latency arbitrage configuration."""
    # Speed advantage requirements
    min_advantage_ns: int = 10_000        # 10 microseconds minimum edge
    
    # Signal detection
    min_price_move_bps: float = 1.0       # Minimum move to trade on
    signal_decay_ns: int = 1_000_000      # Signal valid for 1ms
    
    # Position sizing
    base_size: float = 1.0
    max_position: float = 5.0             # Smaller for latency arb
    
    # Risk
    max_loss_per_trade: float = 10.0
    stop_loss_bps: float = 5.0


@dataclass
class LatencySignal:
    """Represents a latency arbitrage signal."""
    timestamp_ns: int
    signal_type: str  # 'stale_quote', 'news', 'order_flow'
    direction: Side
    target_price: float
    stale_price: float
    expected_profit_bps: float
    time_edge_ns: int
    confidence: float


class StaleQuoteDetector:
    """
    Detects stale quotes that can be picked off.
    
    In real HFT, slower market makers may have quotes
    that don't reflect the latest information.
    """
    
    def __init__(
        self,
        quote_lifetime_ns: int = 100_000,  # 100 microseconds
        min_edge_bps: float = 0.5
    ):
        self.quote_lifetime_ns = quote_lifetime_ns
        self.min_edge_bps = min_edge_bps
        
        # Track quote timestamps
        self.quote_timestamps: Dict[str, int] = {}
        self.quote_prices: Dict[str, float] = {}
    
    def update_quote(
        self,
        venue: str,
        side: Side,
        price: float,
        timestamp_ns: int
    ) -> None:
        """Update quote for a venue."""
        key = f"{venue}_{side.value}"
        self.quote_timestamps[key] = timestamp_ns
        self.quote_prices[key] = price
    
    def find_stale_quotes(
        self,
        fair_value: float,
        current_ns: int
    ) -> List[Dict]:
        """
        Find quotes that are stale relative to fair value.
        
        Returns list of stale quote opportunities.
        """
        opportunities = []
        
        for key, quote_time in self.quote_timestamps.items():
            quote_age = current_ns - quote_time
            
            if quote_age > self.quote_lifetime_ns:
                venue, side_str = key.rsplit('_', 1)
                side = Side.BID if side_str == 'bid' else Side.ASK
                quote_price = self.quote_prices[key]
                
                # Check if quote is mispriced
                if side == Side.BID:
                    # Bid should be below fair value
                    edge = (quote_price - fair_value) / fair_value * 10000
                    if edge > self.min_edge_bps:
                        # Stale bid above fair value - sell to it
                        opportunities.append({
                            'venue': venue,
                            'side': Side.ASK,  # We sell
                            'price': quote_price,
                            'edge_bps': edge,
                            'age_ns': quote_age
                        })
                else:
                    # Ask should be above fair value
                    edge = (fair_value - quote_price) / fair_value * 10000
                    if edge > self.min_edge_bps:
                        # Stale ask below fair value - buy from it
                        opportunities.append({
                            'venue': venue,
                            'side': Side.BID,  # We buy
                            'price': quote_price,
                            'edge_bps': edge,
                            'age_ns': quote_age
                        })
        
        return opportunities


class OrderFlowPredictor:
    """
    Predicts short-term price movement from order flow.
    
    HFT firms analyze order flow to predict price direction
    before it's fully reflected in the market.
    """
    
    def __init__(self, lookback: int = 100):
        self.lookback = lookback
        self.buy_volume: List[float] = []
        self.sell_volume: List[float] = []
        self.imbalance_history: List[float] = []
    
    def add_trade(self, side: Side, size: float) -> None:
        """Record a trade."""
        if side == Side.BID:
            self.buy_volume.append(size)
            self.sell_volume.append(0)
        else:
            self.buy_volume.append(0)
            self.sell_volume.append(size)
        
        # Calculate imbalance
        if len(self.buy_volume) > 0:
            recent_buy = sum(self.buy_volume[-self.lookback:])
            recent_sell = sum(self.sell_volume[-self.lookback:])
            total = recent_buy + recent_sell
            
            if total > 0:
                imbalance = (recent_buy - recent_sell) / total
            else:
                imbalance = 0.0
            
            self.imbalance_history.append(imbalance)
        
        # Trim history
        if len(self.buy_volume) > self.lookback * 2:
            self.buy_volume = self.buy_volume[-self.lookback:]
            self.sell_volume = self.sell_volume[-self.lookback:]
            self.imbalance_history = self.imbalance_history[-self.lookback:]
    
    @property
    def current_imbalance(self) -> float:
        """Get current order flow imbalance (-1 to 1)."""
        if not self.imbalance_history:
            return 0.0
        return self.imbalance_history[-1]
    
    def predict_direction(self, threshold: float = 0.3) -> Optional[Side]:
        """
        Predict price direction from order flow.
        
        Returns Side if confident prediction, else None.
        """
        imbalance = self.current_imbalance
        
        if imbalance > threshold:
            return Side.BID  # Buyers dominating, price going up
        elif imbalance < -threshold:
            return Side.ASK  # Sellers dominating, price going down
        
        return None


class LatencyArbitrage:
    """
    Latency Arbitrage Strategy.
    
    Core concept: Use speed advantage to:
    1. Detect stale quotes and pick them off
    2. React to information faster than competitors
    3. Front-run predictable order flow
    
    This is one of the most controversial HFT strategies
    as it profits from being faster than others.
    """
    
    def __init__(
        self,
        config: Optional[LatencyArbConfig] = None,
        latency_model: Optional[LatencyModel] = None,
        seed: int = 42
    ):
        self.config = config or LatencyArbConfig()
        self.latency = latency_model or LatencyModel()
        self.rng = np.random.default_rng(seed)
        
        # Detection components
        self.stale_detector = StaleQuoteDetector()
        self.flow_predictor = OrderFlowPredictor()
        
        # State
        self.position = 0.0
        self.pnl = 0.0
        
        # Performance
        self.signals: List[LatencySignal] = []
        self.trades: List[Dict] = []
        self.opportunities_detected = 0
        self.opportunities_captured = 0
    
    def process_quote(
        self,
        venue: str,
        side: Side,
        price: float,
        timestamp_ns: int
    ) -> None:
        """Process incoming quote update."""
        self.stale_detector.update_quote(venue, side, price, timestamp_ns)
    
    def process_trade(self, side: Side, size: float) -> None:
        """Process incoming trade for flow prediction."""
        self.flow_predictor.add_trade(side, size)
    
    def scan_opportunities(
        self,
        fair_value: float,
        current_ns: int
    ) -> List[LatencySignal]:
        """
        Scan for latency arbitrage opportunities.
        
        Returns list of signals sorted by expected profit.
        """
        signals = []
        
        # Check stale quotes
        stale_opps = self.stale_detector.find_stale_quotes(fair_value, current_ns)
        
        for opp in stale_opps:
            # Calculate if we can capture before quote refreshes
            our_latency = self.latency.get_total_order_latency()
            time_edge = opp['age_ns'] - our_latency
            
            if time_edge > self.config.min_advantage_ns:
                signal = LatencySignal(
                    timestamp_ns=current_ns,
                    signal_type='stale_quote',
                    direction=opp['side'],
                    target_price=fair_value,
                    stale_price=opp['price'],
                    expected_profit_bps=opp['edge_bps'],
                    time_edge_ns=time_edge,
                    confidence=min(0.9, time_edge / 100_000)  # Higher edge = higher confidence
                )
                signals.append(signal)
                self.opportunities_detected += 1
        
        # Check order flow prediction
        predicted_direction = self.flow_predictor.predict_direction()
        if predicted_direction:
            imbalance = abs(self.flow_predictor.current_imbalance)
            expected_move_bps = imbalance * 5  # Rough estimate
            
            if expected_move_bps > self.config.min_price_move_bps:
                signal = LatencySignal(
                    timestamp_ns=current_ns,
                    signal_type='order_flow',
                    direction=predicted_direction,
                    target_price=fair_value * (1 + expected_move_bps / 10000),
                    stale_price=fair_value,
                    expected_profit_bps=expected_move_bps,
                    time_edge_ns=self.config.signal_decay_ns,
                    confidence=min(0.8, imbalance)
                )
                signals.append(signal)
        
        # Sort by expected profit
        signals.sort(key=lambda s: s.expected_profit_bps, reverse=True)
        self.signals.extend(signals)
        
        return signals
    
    def execute_signal(
        self,
        signal: LatencySignal,
        execution_price: float
    ) -> Dict:
        """Execute a latency arbitrage signal."""
        size = self.config.base_size
        
        # Check position limits
        if signal.direction == Side.BID:
            if self.position + size > self.config.max_position:
                size = self.config.max_position - self.position
        else:
            if self.position - size < -self.config.max_position:
                size = self.position + self.config.max_position
        
        if size <= 0:
            return {'executed': False, 'reason': 'position_limit'}
        
        # Execute
        if signal.direction == Side.BID:
            self.position += size
            cost = size * execution_price
        else:
            self.position -= size
            cost = -size * execution_price
        
        # Calculate immediate PnL (for stale quote arb)
        if signal.signal_type == 'stale_quote':
            if signal.direction == Side.BID:
                # Bought at stale_price, worth target_price
                pnl = size * (signal.target_price - execution_price)
            else:
                # Sold at stale_price, worth target_price
                pnl = size * (execution_price - signal.target_price)
            
            self.pnl += pnl
        else:
            pnl = 0  # Will be realized later
        
        self.opportunities_captured += 1
        
        trade = {
            'signal_type': signal.signal_type,
            'direction': signal.direction.value,
            'size': size,
            'price': execution_price,
            'expected_profit_bps': signal.expected_profit_bps,
            'time_edge_ns': signal.time_edge_ns,
            'pnl': pnl,
            'timestamp': signal.timestamp_ns
        }
        self.trades.append(trade)
        
        return {'executed': True, 'trade': trade}
    
    def get_performance(self) -> Dict:
        """Get strategy performance."""
        return {
            'opportunities_detected': self.opportunities_detected,
            'opportunities_captured': self.opportunities_captured,
            'capture_rate': self.opportunities_captured / max(1, self.opportunities_detected),
            'total_trades': len(self.trades),
            'total_pnl': self.pnl,
            'position': self.position,
            'avg_profit_bps': np.mean([t['expected_profit_bps'] for t in self.trades]) if self.trades else 0,
            'avg_time_edge_us': np.mean([t['time_edge_ns'] / 1000 for t in self.trades]) if self.trades else 0
        }


class SimpleLatencyArb:
    """
    Simplified latency arbitrage for demos.
    
    Shows the core concept: faster reaction = profit.
    """
    
    def __init__(self, our_latency_ns: int = 10_000):
        self.our_latency_ns = our_latency_ns
        self.profits = []
    
    def race(
        self,
        opportunity_value: float,
        opportunity_duration_ns: int,
        competitor_latency_ns: int
    ) -> Dict:
        """
        Simulate race for an opportunity.
        
        Returns result of the race.
        """
        we_win = self.our_latency_ns < competitor_latency_ns
        we_arrive_in_time = self.our_latency_ns < opportunity_duration_ns
        
        if we_win and we_arrive_in_time:
            profit = opportunity_value
            self.profits.append(profit)
            return {
                'winner': 'us',
                'profit': profit,
                'our_time_ns': self.our_latency_ns,
                'competitor_time_ns': competitor_latency_ns,
                'edge_ns': competitor_latency_ns - self.our_latency_ns
            }
        elif not we_arrive_in_time:
            return {
                'winner': 'none',
                'profit': 0,
                'reason': 'opportunity_expired'
            }
        else:
            return {
                'winner': 'competitor',
                'profit': 0,
                'their_edge_ns': self.our_latency_ns - competitor_latency_ns
            }
