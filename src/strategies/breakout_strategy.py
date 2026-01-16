"""
Breakout Trading Strategy.
Detects price breakouts from consolidation ranges.
Trades in the direction of the breakout with volume confirmation.
"""

import numpy as np
import pandas as pd
from typing import Optional
from .base_strategy import BaseStrategy, Signal, TradeSignal


class BreakoutStrategy(BaseStrategy):
    """
    Breakout strategy detecting price movements outside recent ranges.
    
    Signals:
    - BUY: Price breaks above recent high with volume
    - SELL: Price breaks below recent low with volume
    
    Parameters:
        lookback: Period for calculating range (default: 20)
        breakout_threshold: Min breakout % above/below range (default: 0.001)
        volume_threshold: Volume multiplier for confirmation (default: 1.3)
        atr_multiplier: ATR multiplier for targets (default: 2.0)
    """
    
    def __init__(
        self,
        lookback: int = 20,
        breakout_threshold: float = 0.001,
        volume_threshold: float = 1.3,
        atr_multiplier: float = 2.0,
        require_trend: bool = True
    ):
        super().__init__()
        self.lookback = lookback
        self.breakout_threshold = breakout_threshold
        self.volume_threshold = volume_threshold
        self.atr_multiplier = atr_multiplier
        self.require_trend = require_trend
    
    def get_name(self) -> str:
        return "Breakout"
    
    def generate_signal(
        self,
        row: pd.Series,
        ml_prediction: Optional[float] = None
    ) -> TradeSignal:
        """Generate breakout signal."""
        
        close = row['close']
        high = row.get('high', close)
        low = row.get('low', close)
        
        # Get range levels
        recent_high = self._get_feature(row, f'rolling_high_{self.lookback}', 
                                        self._get_feature(row, 'rolling_high_20', high))
        recent_low = self._get_feature(row, f'rolling_low_{self.lookback}',
                                       self._get_feature(row, 'rolling_low_20', low))
        
        # Fallback to high/low from row if no rolling
        if recent_high == high:
            recent_high = self._get_feature(row, 'high', close * 1.01)
        if recent_low == low:
            recent_low = self._get_feature(row, 'low', close * 0.99)
        
        # Volume
        volume_ratio = self._get_feature(row, 'volume_ratio_10', 1.0)
        
        # Trend indicator
        sma_short = self._get_feature(row, 'sma_10', close)
        sma_long = self._get_feature(row, 'sma_20', close)
        trend_up = sma_short > sma_long
        trend_down = sma_short < sma_long
        
        # ATR for targets
        atr = self._get_feature(row, 'atr_14', close * 0.01)
        
        # Momentum
        momentum = self._get_feature(row, 'momentum_10', 0.0)
        
        # Calculate breakout levels
        breakout_up_level = recent_high * (1 + self.breakout_threshold)
        breakout_down_level = recent_low * (1 - self.breakout_threshold)
        
        # Volume confirmation
        volume_confirmed = volume_ratio >= self.volume_threshold
        
        # Initialize
        signal = Signal.HOLD
        confidence = 0.0
        reason = "No breakout"
        target = None
        stop = None
        
        # Check for upside breakout
        if close > breakout_up_level:
            # Calculate breakout strength
            breakout_pct = (close - recent_high) / recent_high
            
            # Trend alignment
            trend_aligned = not self.require_trend or trend_up
            
            if volume_confirmed and trend_aligned:
                signal = Signal.BUY
                
                # Confidence based on breakout strength and momentum
                conf_breakout = min(breakout_pct / 0.01, 0.4)
                conf_volume = min((volume_ratio - 1) / 2, 0.3)
                conf_momentum = min(abs(momentum) / 0.002, 0.3) if momentum > 0 else 0
                confidence = conf_breakout + conf_volume + conf_momentum
                confidence = min(confidence, 0.95)
                
                # Targets
                target = close + self.atr_multiplier * atr
                stop = recent_high * 0.995  # Just below breakout level
                
                reason = f"Breakout UP: {breakout_pct*100:.2f}% above range, Vol×{volume_ratio:.1f}"
            
            elif volume_confirmed:
                # Weak signal - breakout without trend
                signal = Signal.BUY
                confidence = 0.45
                target = close + 1.5 * atr
                stop = recent_low
                reason = f"Weak breakout UP (no trend): {breakout_pct*100:.2f}%"
        
        # Check for downside breakout
        elif close < breakout_down_level:
            breakout_pct = (recent_low - close) / recent_low
            trend_aligned = not self.require_trend or trend_down
            
            if volume_confirmed and trend_aligned:
                signal = Signal.SELL
                
                conf_breakout = min(breakout_pct / 0.01, 0.4)
                conf_volume = min((volume_ratio - 1) / 2, 0.3)
                conf_momentum = min(abs(momentum) / 0.002, 0.3) if momentum < 0 else 0
                confidence = conf_breakout + conf_volume + conf_momentum
                confidence = min(confidence, 0.95)
                
                target = close - self.atr_multiplier * atr
                stop = recent_low * 1.005
                
                reason = f"Breakout DOWN: {breakout_pct*100:.2f}% below range, Vol×{volume_ratio:.1f}"
            
            elif volume_confirmed:
                signal = Signal.SELL
                confidence = 0.45
                target = close - 1.5 * atr
                stop = recent_high
                reason = f"Weak breakout DOWN (no trend): {breakout_pct*100:.2f}%"
        
        # Incorporate ML prediction if available
        if ml_prediction is not None and signal != Signal.HOLD:
            if signal == Signal.BUY and ml_prediction > 0.5:
                confidence = min(confidence + (ml_prediction - 0.5) * 0.3, 0.98)
            elif signal == Signal.SELL and ml_prediction < 0.5:
                confidence = min(confidence + (0.5 - ml_prediction) * 0.3, 0.98)
            elif (signal == Signal.BUY and ml_prediction < 0.4) or \
                 (signal == Signal.SELL and ml_prediction > 0.6):
                # ML disagrees - reduce confidence
                confidence *= 0.6
        
        return self._create_signal(
            row, signal, confidence, reason,
            ml_probability=ml_prediction,
            target_price=target,
            stop_loss=stop
        )
