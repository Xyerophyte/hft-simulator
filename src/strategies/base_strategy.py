"""
Base strategy interface for all trading strategies.
Provides common interface and shared functionality.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, List, Any
import pandas as pd
import numpy as np


class Signal(Enum):
    """Trading signal enumeration."""
    BUY = 1
    SELL = -1
    HOLD = 0


@dataclass
class TradeSignal:
    """Represents a trading signal with metadata."""
    timestamp: pd.Timestamp
    signal: Signal
    confidence: float  # 0.0 to 1.0
    price: float
    reason: str
    strategy_name: str = "unknown"
    ml_probability: Optional[float] = None
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'signal': self.signal.name,
            'confidence': self.confidence,
            'price': self.price,
            'reason': self.reason,
            'strategy_name': self.strategy_name,
            'ml_probability': self.ml_probability,
            'target_price': self.target_price,
            'stop_loss': self.stop_loss
        }


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    All strategies must implement:
    - generate_signal(): Generate trading signal from market data
    - get_name(): Return strategy name
    """
    
    def __init__(self):
        self.signals_generated: List[TradeSignal] = []
        self._is_initialized = False
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the strategy name."""
        pass
    
    @abstractmethod
    def generate_signal(
        self,
        row: pd.Series,
        ml_prediction: Optional[float] = None
    ) -> TradeSignal:
        """
        Generate trading signal from market data.
        
        Args:
            row: DataFrame row with market data and features
            ml_prediction: Optional ML model prediction (0-1 probability)
            
        Returns:
            TradeSignal object
        """
        pass
    
    def initialize(self, df: pd.DataFrame) -> None:
        """
        Initialize strategy with historical data.
        Override if strategy needs warmup.
        
        Args:
            df: Historical data for initialization
        """
        self._is_initialized = True
    
    def generate_signals_batch(
        self,
        df: pd.DataFrame,
        ml_predictions: Optional[np.ndarray] = None
    ) -> List[TradeSignal]:
        """
        Generate signals for entire dataframe.
        
        Args:
            df: DataFrame with market data
            ml_predictions: Optional array of ML predictions
            
        Returns:
            List of TradeSignal objects
        """
        signals = []
        
        for i, (idx, row) in enumerate(df.iterrows()):
            ml_pred = ml_predictions[i] if ml_predictions is not None and i < len(ml_predictions) else None
            signal = self.generate_signal(row, ml_pred)
            signals.append(signal)
        
        self.signals_generated.extend(signals)
        return signals
    
    def get_signal_statistics(self) -> Dict[str, Any]:
        """Get statistics about generated signals."""
        if not self.signals_generated:
            return {
                'total_signals': 0,
                'buy_signals': 0,
                'sell_signals': 0,
                'hold_signals': 0,
                'buy_pct': 0.0,
                'sell_pct': 0.0,
                'hold_pct': 0.0,
                'avg_confidence': 0.0
            }
        
        total = len(self.signals_generated)
        buys = sum(1 for s in self.signals_generated if s.signal == Signal.BUY)
        sells = sum(1 for s in self.signals_generated if s.signal == Signal.SELL)
        holds = total - buys - sells
        
        avg_conf = np.mean([s.confidence for s in self.signals_generated 
                          if s.signal != Signal.HOLD])
        
        return {
            'total_signals': total,
            'buy_signals': buys,
            'sell_signals': sells,
            'hold_signals': holds,
            'buy_pct': buys / total * 100 if total > 0 else 0,
            'sell_pct': sells / total * 100 if total > 0 else 0,
            'hold_pct': holds / total * 100 if total > 0 else 0,
            'avg_confidence': avg_conf if not np.isnan(avg_conf) else 0.0
        }
    
    def reset(self) -> None:
        """Reset strategy state."""
        self.signals_generated = []
        self._is_initialized = False
    
    def _create_signal(
        self,
        row: pd.Series,
        signal: Signal,
        confidence: float,
        reason: str,
        ml_probability: Optional[float] = None,
        target_price: Optional[float] = None,
        stop_loss: Optional[float] = None
    ) -> TradeSignal:
        """Helper to create TradeSignal objects."""
        ts = TradeSignal(
            timestamp=row.name if hasattr(row, 'name') else pd.Timestamp.now(),
            signal=signal,
            confidence=confidence,
            price=row['close'],
            reason=reason,
            strategy_name=self.get_name(),
            ml_probability=ml_probability,
            target_price=target_price,
            stop_loss=stop_loss
        )
        return ts
    
    def _get_feature(self, row: pd.Series, name: str, default: float = 0.0) -> float:
        """Safely get feature value."""
        val = row.get(name, default)
        if pd.isna(val):
            return default
        return float(val)
