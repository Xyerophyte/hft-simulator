"""
Statistical Arbitrage Strategy for HFT.

Exploits short-term price discrepancies between related
instruments or venues.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from enum import Enum
import time

from ..order_book import OrderBook, Side
from ..latency import LatencyModel


@dataclass
class ArbitrageSignal:
    """Represents an arbitrage opportunity."""
    timestamp_ns: int
    instrument_1: str
    instrument_2: str
    spread: float
    z_score: float
    signal: str  # 'buy_1_sell_2', 'sell_1_buy_2', 'neutral'
    expected_profit: float
    confidence: float


@dataclass
class StatArbConfig:
    """Statistical arbitrage configuration."""
    # Z-score thresholds
    entry_z_threshold: float = 2.0      # Enter when |z-score| > 2
    exit_z_threshold: float = 0.5       # Exit when |z-score| < 0.5
    
    # Position sizing
    base_size: float = 1.0
    max_position: float = 10.0
    
    # Lookback for statistics
    lookback_periods: int = 100
    min_observations: int = 30
    
    # Risk limits
    max_loss_per_trade: float = 50.0
    stop_loss_z: float = 4.0            # Stop if z-score exceeds
    
    # Execution
    max_leg_delay_ns: int = 1_000_000   # 1ms max between legs


class PairStatistics:
    """Tracks statistics for a pair of instruments."""
    
    def __init__(self, lookback: int = 100):
        self.lookback = lookback
        self.spreads: List[float] = []
        self.prices_1: List[float] = []
        self.prices_2: List[float] = []
        
        self._mean = 0.0
        self._std = 1.0
    
    def update(self, price_1: float, price_2: float) -> None:
        """Update with new prices."""
        spread = price_1 - price_2  # Simple spread
        
        self.spreads.append(spread)
        self.prices_1.append(price_1)
        self.prices_2.append(price_2)
        
        # Keep only lookback period
        if len(self.spreads) > self.lookback:
            self.spreads = self.spreads[-self.lookback:]
            self.prices_1 = self.prices_1[-self.lookback:]
            self.prices_2 = self.prices_2[-self.lookback:]
        
        # Update statistics
        if len(self.spreads) >= 2:
            self._mean = np.mean(self.spreads)
            self._std = np.std(self.spreads)
            if self._std == 0:
                self._std = 1.0
    
    @property
    def mean(self) -> float:
        return self._mean
    
    @property
    def std(self) -> float:
        return self._std
    
    @property
    def current_spread(self) -> float:
        if not self.spreads:
            return 0.0
        return self.spreads[-1]
    
    @property
    def z_score(self) -> float:
        """Current z-score of spread."""
        return (self.current_spread - self._mean) / self._std
    
    @property
    def half_life(self) -> float:
        """Estimate mean reversion half-life."""
        if len(self.spreads) < 10:
            return float('inf')
        
        # Simple AR(1) estimation
        spreads = np.array(self.spreads)
        y = spreads[1:]
        x = spreads[:-1]
        
        if len(x) == 0 or np.std(x) == 0:
            return float('inf')
        
        # OLS coefficient
        phi = np.cov(x, y)[0, 1] / np.var(x)
        
        if phi >= 1 or phi <= 0:
            return float('inf')
        
        return -np.log(2) / np.log(phi)
    
    def is_cointegrated(self, threshold: float = 0.9) -> bool:
        """Simple check if pair shows mean reversion."""
        return self.half_life < len(self.spreads) * 0.5


class StatisticalArbitrage:
    """
    Statistical Arbitrage Strategy.
    
    Core concepts:
    1. Find pairs of related instruments
    2. Track spread between them
    3. When spread deviates from mean, trade for reversion
    4. Exit when spread normalizes
    
    Used in HFT for:
    - Cross-venue arbitrage (same asset, different exchanges)
    - ETF arbitrage (ETF vs basket)
    - Pair trading (related assets)
    """
    
    def __init__(
        self,
        config: Optional[StatArbConfig] = None,
        latency_model: Optional[LatencyModel] = None,
        seed: int = 42
    ):
        self.config = config or StatArbConfig()
        self.latency = latency_model or LatencyModel()
        self.rng = np.random.default_rng(seed)
        
        # Pair statistics
        self.pairs: Dict[str, PairStatistics] = {}
        
        # Position tracking
        self.positions: Dict[str, float] = {}  # instrument -> position
        self.entry_prices: Dict[str, float] = {}
        
        # Performance
        self.signals: List[ArbitrageSignal] = []
        self.trades: List[Dict] = []
        self.pnl = 0.0
    
    def add_pair(self, name: str, instrument_1: str, instrument_2: str) -> None:
        """Add a pair to track."""
        self.pairs[name] = PairStatistics(self.config.lookback_periods)
        self.positions[instrument_1] = 0.0
        self.positions[instrument_2] = 0.0
    
    def update_prices(
        self,
        pair_name: str,
        price_1: float,
        price_2: float,
        timestamp_ns: Optional[int] = None
    ) -> Optional[ArbitrageSignal]:
        """
        Update prices and generate signal if opportunity exists.
        
        Args:
            pair_name: Name of the pair
            price_1: Price of instrument 1
            price_2: Price of instrument 2
            timestamp_ns: Current timestamp
            
        Returns:
            ArbitrageSignal if opportunity detected, else None
        """
        if timestamp_ns is None:
            timestamp_ns = int(time.time() * 1e9)
        
        if pair_name not in self.pairs:
            return None
        
        stats = self.pairs[pair_name]
        stats.update(price_1, price_2)
        
        # Need minimum observations
        if len(stats.spreads) < self.config.min_observations:
            return None
        
        z = stats.z_score
        
        # Generate signal
        if abs(z) > self.config.entry_z_threshold:
            if z > 0:
                # Spread too high: sell 1, buy 2
                signal = 'sell_1_buy_2'
            else:
                # Spread too low: buy 1, sell 2
                signal = 'buy_1_sell_2'
            
            expected_profit = abs(z) * stats.std  # Expected profit = z * std
            confidence = min(0.95, abs(z) / 4.0)  # Cap at 95%
            
        elif abs(z) < self.config.exit_z_threshold:
            signal = 'neutral'
            expected_profit = 0.0
            confidence = 0.0
        else:
            return None
        
        arb_signal = ArbitrageSignal(
            timestamp_ns=timestamp_ns,
            instrument_1=pair_name.split('_')[0] if '_' in pair_name else 'A',
            instrument_2=pair_name.split('_')[1] if '_' in pair_name else 'B',
            spread=stats.current_spread,
            z_score=z,
            signal=signal,
            expected_profit=expected_profit,
            confidence=confidence
        )
        
        self.signals.append(arb_signal)
        return arb_signal
    
    def execute_signal(
        self,
        signal: ArbitrageSignal,
        price_1: float,
        price_2: float
    ) -> Dict:
        """
        Execute an arbitrage signal.
        
        Returns trade execution details.
        """
        size = self.config.base_size
        
        if signal.signal == 'buy_1_sell_2':
            # Long instrument 1, short instrument 2
            self.positions[signal.instrument_1] += size
            self.positions[signal.instrument_2] -= size
            self.entry_prices[signal.instrument_1] = price_1
            self.entry_prices[signal.instrument_2] = price_2
            
            trade = {
                'type': 'open',
                'signal': signal.signal,
                'leg_1': {'side': 'buy', 'price': price_1, 'size': size},
                'leg_2': {'side': 'sell', 'price': price_2, 'size': size},
                'z_score': signal.z_score,
                'timestamp': signal.timestamp_ns
            }
            
        elif signal.signal == 'sell_1_buy_2':
            self.positions[signal.instrument_1] -= size
            self.positions[signal.instrument_2] += size
            self.entry_prices[signal.instrument_1] = price_1
            self.entry_prices[signal.instrument_2] = price_2
            
            trade = {
                'type': 'open',
                'signal': signal.signal,
                'leg_1': {'side': 'sell', 'price': price_1, 'size': size},
                'leg_2': {'side': 'buy', 'price': price_2, 'size': size},
                'z_score': signal.z_score,
                'timestamp': signal.timestamp_ns
            }
            
        else:  # neutral - close position
            pnl = self._calculate_close_pnl(signal.instrument_1, signal.instrument_2, price_1, price_2)
            self.pnl += pnl
            
            self.positions[signal.instrument_1] = 0.0
            self.positions[signal.instrument_2] = 0.0
            
            trade = {
                'type': 'close',
                'signal': signal.signal,
                'pnl': pnl,
                'z_score': signal.z_score,
                'timestamp': signal.timestamp_ns
            }
        
        self.trades.append(trade)
        return trade
    
    def _calculate_close_pnl(
        self,
        inst_1: str,
        inst_2: str,
        price_1: float,
        price_2: float
    ) -> float:
        """Calculate PnL from closing position."""
        pnl = 0.0
        
        if inst_1 in self.entry_prices and inst_1 in self.positions:
            pos = self.positions[inst_1]
            entry = self.entry_prices[inst_1]
            pnl += pos * (price_1 - entry)
        
        if inst_2 in self.entry_prices and inst_2 in self.positions:
            pos = self.positions[inst_2]
            entry = self.entry_prices[inst_2]
            pnl += pos * (price_2 - entry)
        
        return pnl
    
    def get_performance(self) -> Dict:
        """Get strategy performance."""
        winning_trades = [t for t in self.trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in self.trades if t.get('pnl', 0) < 0]
        
        return {
            'total_signals': len(self.signals),
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / max(1, len(self.trades)),
            'total_pnl': self.pnl,
            'avg_z_score_entry': np.mean([s.z_score for s in self.signals]) if self.signals else 0
        }


class SimpleCrossVenueArb:
    """
    Simple cross-venue arbitrage.
    
    Exploits price differences for the same asset on different venues.
    """
    
    def __init__(self, min_profit_bps: float = 1.0):
        self.min_profit_bps = min_profit_bps
        self.opportunities: List[Dict] = []
        self.executions: List[Dict] = []
    
    def check_opportunity(
        self,
        venue_1_bid: float,
        venue_1_ask: float,
        venue_2_bid: float,
        venue_2_ask: float
    ) -> Optional[Dict]:
        """
        Check for cross-venue arbitrage opportunity.
        
        Opportunity exists if:
        - Venue 1 bid > Venue 2 ask (buy 2, sell 1)
        - Venue 2 bid > Venue 1 ask (buy 1, sell 2)
        """
        # Buy on venue 2, sell on venue 1
        profit_1 = venue_1_bid - venue_2_ask
        profit_1_bps = (profit_1 / venue_2_ask) * 10000
        
        # Buy on venue 1, sell on venue 2
        profit_2 = venue_2_bid - venue_1_ask
        profit_2_bps = (profit_2 / venue_1_ask) * 10000
        
        if profit_1_bps > self.min_profit_bps:
            opp = {
                'type': 'buy_2_sell_1',
                'buy_price': venue_2_ask,
                'sell_price': venue_1_bid,
                'profit_bps': profit_1_bps,
                'timestamp': time.time()
            }
            self.opportunities.append(opp)
            return opp
        
        if profit_2_bps > self.min_profit_bps:
            opp = {
                'type': 'buy_1_sell_2',
                'buy_price': venue_1_ask,
                'sell_price': venue_2_bid,
                'profit_bps': profit_2_bps,
                'timestamp': time.time()
            }
            self.opportunities.append(opp)
            return opp
        
        return None
