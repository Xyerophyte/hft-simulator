"""
Tick Data Processing for HFT Simulation.

Handles tick-by-tick trade data with microsecond precision.
Real HFT operates on individual trades, not candlesticks.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Iterator, Callable
from enum import Enum
from datetime import datetime
import time


class TickType(Enum):
    """Type of tick event."""
    TRADE = "trade"
    BID = "bid"
    ASK = "ask"
    BID_CANCEL = "bid_cancel"
    ASK_CANCEL = "ask_cancel"


@dataclass
class Tick:
    """
    Single tick representing a market event.
    
    In real HFT, each tick is an individual trade or quote update
    with microsecond/nanosecond precision timestamps.
    """
    timestamp_ns: int  # Nanosecond timestamp
    price: float
    size: float
    tick_type: TickType = TickType.TRADE
    side: Optional[str] = None  # 'buy' or 'sell' (aggressor side)
    trade_id: Optional[int] = None
    
    @property
    def timestamp(self) -> datetime:
        """Get timestamp as datetime."""
        return datetime.fromtimestamp(self.timestamp_ns / 1e9)
    
    @property
    def timestamp_us(self) -> int:
        """Get microsecond timestamp."""
        return self.timestamp_ns // 1000
    
    @property
    def timestamp_ms(self) -> int:
        """Get millisecond timestamp."""
        return self.timestamp_ns // 1_000_000
    
    def to_dict(self) -> dict:
        return {
            'timestamp_ns': self.timestamp_ns,
            'price': self.price,
            'size': self.size,
            'tick_type': self.tick_type.value,
            'side': self.side,
            'trade_id': self.trade_id
        }


class TickStream:
    """
    Stream of ticks for HFT simulation.
    
    Provides iterator interface for processing ticks sequentially,
    mimicking real-time data feed.
    """
    
    def __init__(self, ticks: Optional[List[Tick]] = None):
        self.ticks: List[Tick] = ticks or []
        self._index = 0
    
    def add_tick(self, tick: Tick) -> None:
        """Add a tick to the stream."""
        self.ticks.append(tick)
    
    def __iter__(self) -> Iterator[Tick]:
        self._index = 0
        return self
    
    def __next__(self) -> Tick:
        if self._index >= len(self.ticks):
            raise StopIteration
        tick = self.ticks[self._index]
        self._index += 1
        return tick
    
    def __len__(self) -> int:
        return len(self.ticks)
    
    def reset(self) -> None:
        """Reset stream to beginning."""
        self._index = 0
    
    @classmethod
    def from_trades_df(cls, df: pd.DataFrame) -> 'TickStream':
        """
        Create tick stream from trades DataFrame.
        
        Expected columns: timestamp, price, size, side (optional)
        """
        stream = cls()
        
        for idx, row in df.iterrows():
            # Convert timestamp to nanoseconds
            if isinstance(idx, pd.Timestamp):
                ts_ns = int(idx.timestamp() * 1e9)
            elif isinstance(row.get('timestamp'), (int, float)):
                ts_ns = int(row['timestamp'] * 1e9)
            else:
                ts_ns = int(time.time() * 1e9)
            
            tick = Tick(
                timestamp_ns=ts_ns,
                price=float(row['price']),
                size=float(row.get('size', row.get('volume', 1.0))),
                tick_type=TickType.TRADE,
                side=row.get('side', None),
                trade_id=row.get('trade_id', idx if isinstance(idx, int) else None)
            )
            stream.add_tick(tick)
        
        return stream
    
    @classmethod
    def generate_from_ohlcv(
        cls,
        df: pd.DataFrame,
        ticks_per_bar: int = 100,
        seed: int = 42
    ) -> 'TickStream':
        """
        Generate synthetic tick stream from OHLCV data.
        
        Creates realistic tick distribution within each bar.
        """
        np.random.seed(seed)
        stream = cls()
        trade_id = 0
        
        for idx, row in df.iterrows():
            bar_open = row['open']
            bar_high = row['high']
            bar_low = row['low']
            bar_close = row['close']
            bar_volume = row['volume']
            
            # Get bar start/end time
            if isinstance(idx, pd.Timestamp):
                bar_start_ns = int(idx.timestamp() * 1e9)
            else:
                bar_start_ns = int(time.time() * 1e9)
            
            # Assume 1-minute bars
            bar_duration_ns = 60 * 1_000_000_000
            
            # Generate price path within bar
            # Random walk from open to close, touching high and low
            prices = cls._generate_intrabar_prices(
                bar_open, bar_high, bar_low, bar_close, 
                ticks_per_bar
            )
            
            # Distribute volume across ticks
            sizes = np.random.exponential(bar_volume / ticks_per_bar, ticks_per_bar)
            sizes = sizes * (bar_volume / sizes.sum())  # Normalize
            
            # Generate timestamps within bar
            timestamps = np.sort(np.random.randint(
                bar_start_ns, 
                bar_start_ns + bar_duration_ns,
                ticks_per_bar
            ))
            
            # Alternate buy/sell based on price direction
            for i in range(ticks_per_bar):
                if i == 0:
                    side = 'buy' if bar_close > bar_open else 'sell'
                else:
                    side = 'buy' if prices[i] > prices[i-1] else 'sell'
                
                tick = Tick(
                    timestamp_ns=int(timestamps[i]),
                    price=prices[i],
                    size=sizes[i],
                    tick_type=TickType.TRADE,
                    side=side,
                    trade_id=trade_id
                )
                stream.add_tick(tick)
                trade_id += 1
        
        return stream
    
    @staticmethod
    def _generate_intrabar_prices(
        open_: float, high: float, low: float, close: float,
        n_ticks: int
    ) -> np.ndarray:
        """Generate realistic price path within a bar."""
        prices = np.zeros(n_ticks)
        prices[0] = open_
        prices[-1] = close
        
        # Random walk with drift toward close
        for i in range(1, n_ticks - 1):
            drift = (close - prices[i-1]) / (n_ticks - i)
            noise = np.random.normal(0, (high - low) / 10)
            prices[i] = prices[i-1] + drift + noise
            # Clip to high/low
            prices[i] = max(low, min(high, prices[i]))
        
        return prices
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert tick stream to DataFrame."""
        data = [t.to_dict() for t in self.ticks]
        df = pd.DataFrame(data)
        if not df.empty:
            df['datetime'] = pd.to_datetime(df['timestamp_ns'], unit='ns')
            df = df.set_index('datetime')
        return df


class TickAggregator:
    """
    Aggregates ticks into OHLCV bars.
    
    Supports time-based and volume-based aggregation.
    """
    
    def __init__(self, bar_size_ns: int = 60_000_000_000):
        """
        Initialize aggregator.
        
        Args:
            bar_size_ns: Bar size in nanoseconds (default 1 minute)
        """
        self.bar_size_ns = bar_size_ns
        self.current_bar_start = None
        self.current_open = None
        self.current_high = None
        self.current_low = None
        self.current_close = None
        self.current_volume = 0.0
        self.current_trades = 0
        self.bars: List[dict] = []
    
    def process_tick(self, tick: Tick) -> Optional[dict]:
        """
        Process a tick and return completed bar if applicable.
        
        Args:
            tick: Incoming tick
            
        Returns:
            Completed bar dict if bar closed, else None
        """
        if tick.tick_type != TickType.TRADE:
            return None
        
        # Determine bar start
        bar_start = (tick.timestamp_ns // self.bar_size_ns) * self.bar_size_ns
        
        # Check if new bar
        if self.current_bar_start is None:
            self._start_new_bar(bar_start, tick)
            return None
        
        if bar_start > self.current_bar_start:
            # Complete current bar
            completed_bar = self._complete_bar()
            self._start_new_bar(bar_start, tick)
            return completed_bar
        
        # Update current bar
        self.current_high = max(self.current_high, tick.price)
        self.current_low = min(self.current_low, tick.price)
        self.current_close = tick.price
        self.current_volume += tick.size
        self.current_trades += 1
        
        return None
    
    def _start_new_bar(self, bar_start: int, tick: Tick) -> None:
        """Start a new bar."""
        self.current_bar_start = bar_start
        self.current_open = tick.price
        self.current_high = tick.price
        self.current_low = tick.price
        self.current_close = tick.price
        self.current_volume = tick.size
        self.current_trades = 1
    
    def _complete_bar(self) -> dict:
        """Complete current bar and return it."""
        bar = {
            'timestamp_ns': self.current_bar_start,
            'open': self.current_open,
            'high': self.current_high,
            'low': self.current_low,
            'close': self.current_close,
            'volume': self.current_volume,
            'trades': self.current_trades
        }
        self.bars.append(bar)
        return bar
    
    def finalize(self) -> Optional[dict]:
        """Finalize any remaining bar."""
        if self.current_bar_start is not None:
            return self._complete_bar()
        return None
    
    def get_bars_df(self) -> pd.DataFrame:
        """Get all completed bars as DataFrame."""
        if not self.bars:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.bars)
        df['datetime'] = pd.to_datetime(df['timestamp_ns'], unit='ns')
        df = df.set_index('datetime')
        return df


class VolumeBarAggregator:
    """
    Volume-based bar aggregation (used in some HFT strategies).
    
    Creates bars based on traded volume, not time.
    """
    
    def __init__(self, volume_per_bar: float = 100.0):
        self.volume_per_bar = volume_per_bar
        self.accumulated_volume = 0.0
        self.current_open = None
        self.current_high = None
        self.current_low = None
        self.current_close = None
        self.bar_start_ns = None
        self.bars: List[dict] = []
    
    def process_tick(self, tick: Tick) -> Optional[dict]:
        """Process tick and return bar if volume threshold reached."""
        if tick.tick_type != TickType.TRADE:
            return None
        
        if self.current_open is None:
            self._start_new_bar(tick)
        
        # Update bar
        self.current_high = max(self.current_high, tick.price)
        self.current_low = min(self.current_low, tick.price)
        self.current_close = tick.price
        self.accumulated_volume += tick.size
        
        # Check if bar complete
        if self.accumulated_volume >= self.volume_per_bar:
            return self._complete_bar(tick.timestamp_ns)
        
        return None
    
    def _start_new_bar(self, tick: Tick) -> None:
        self.bar_start_ns = tick.timestamp_ns
        self.current_open = tick.price
        self.current_high = tick.price
        self.current_low = tick.price
        self.current_close = tick.price
        self.accumulated_volume = 0.0
    
    def _complete_bar(self, end_ns: int) -> dict:
        bar = {
            'timestamp_ns': self.bar_start_ns,
            'end_timestamp_ns': end_ns,
            'open': self.current_open,
            'high': self.current_high,
            'low': self.current_low,
            'close': self.current_close,
            'volume': self.accumulated_volume
        }
        self.bars.append(bar)
        self.current_open = None
        return bar
