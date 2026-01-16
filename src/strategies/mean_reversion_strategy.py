"""
Mean Reversion Trading Strategy.
Buys on oversold conditions, sells on overbought conditions.
Based on RSI, Bollinger Bands, and price deviation from mean.
"""

import numpy as np
import pandas as pd
from typing import Optional
from .base_strategy import BaseStrategy, Signal, TradeSignal


class MeanReversionStrategy(BaseStrategy):
    """
    Mean reversion strategy using multiple indicators.
    
    Signals:
    - BUY: Price oversold (RSI low, below lower BB, far from mean)
    - SELL: Price overbought (RSI high, above upper BB, far from mean)
    
    Parameters:
        rsi_oversold: RSI level for oversold (default: 30)
        rsi_overbought: RSI level for overbought (default: 70)
        bb_threshold: Min distance from BB for signal (default: 0.0)
        mean_deviation_threshold: Min deviation from SMA (default: 0.02)
        volume_confirmation: Require volume confirmation (default: True)
    """
    
    def __init__(
        self,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        bb_threshold: float = 0.0,
        mean_deviation_threshold: float = 0.02,
        volume_confirmation: bool = True,
        volume_threshold: float = 1.1
    ):
        super().__init__()
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.bb_threshold = bb_threshold
        self.mean_deviation_threshold = mean_deviation_threshold
        self.volume_confirmation = volume_confirmation
        self.volume_threshold = volume_threshold
    
    def get_name(self) -> str:
        return "MeanReversion"
    
    def generate_signal(
        self,
        row: pd.Series,
        ml_prediction: Optional[float] = None
    ) -> TradeSignal:
        """Generate mean reversion signal."""
        
        # Get indicators
        rsi = self._get_feature(row, 'rsi_14', 50.0)
        close = row['close']
        
        # Bollinger Bands
        bb_upper = self._get_feature(row, 'bb_upper', close * 1.02)
        bb_lower = self._get_feature(row, 'bb_lower', close * 0.98)
        bb_mid = self._get_feature(row, 'bb_mid', close)
        if bb_mid == 0:
            bb_mid = self._get_feature(row, 'sma_20', close)
        
        # Moving average
        sma = self._get_feature(row, 'sma_20', close)
        
        # Volume
        volume_ratio = self._get_feature(row, 'volume_ratio_10', 1.0)
        
        # Calculate signals
        signals = []
        confidence = 0.0
        reasons = []
        
        # 1. RSI signal
        if rsi < self.rsi_oversold:
            signals.append(1)
            conf = (self.rsi_oversold - rsi) / self.rsi_oversold
            confidence += conf * 0.4
            reasons.append(f"RSI={rsi:.1f}")
        elif rsi > self.rsi_overbought:
            signals.append(-1)
            conf = (rsi - self.rsi_overbought) / (100 - self.rsi_overbought)
            confidence += conf * 0.4
            reasons.append(f"RSI={rsi:.1f}")
        else:
            signals.append(0)
        
        # 2. Bollinger Band signal
        if close < bb_lower:
            signals.append(1)
            bb_dev = (bb_lower - close) / bb_lower
            confidence += min(bb_dev / 0.02, 0.3)
            reasons.append("Below BB")
        elif close > bb_upper:
            signals.append(-1)
            bb_dev = (close - bb_upper) / bb_upper
            confidence += min(bb_dev / 0.02, 0.3)
            reasons.append("Above BB")
        else:
            signals.append(0)
        
        # 3. Mean deviation signal
        if sma > 0:
            mean_dev = (close - sma) / sma
            if mean_dev < -self.mean_deviation_threshold:
                signals.append(1)
                confidence += 0.2
                reasons.append(f"Deviation={mean_dev*100:.1f}%")
            elif mean_dev > self.mean_deviation_threshold:
                signals.append(-1)
                confidence += 0.2
                reasons.append(f"Deviation={mean_dev*100:.1f}%")
            else:
                signals.append(0)
        else:
            signals.append(0)
        
        # Volume confirmation
        volume_ok = (not self.volume_confirmation or 
                    volume_ratio >= self.volume_threshold)
        
        if not volume_ok:
            confidence *= 0.7
        
        # Aggregate signals
        signal_sum = sum(signals)
        non_zero = sum(1 for s in signals if s != 0)
        
        # Calculate target and stop loss
        atr = self._get_feature(row, 'atr_14', close * 0.01)
        
        if signal_sum >= 2 and non_zero >= 2:
            # Strong BUY (oversold)
            final_signal = Signal.BUY
            confidence = min(confidence, 0.95)
            target = close + 2 * atr
            stop = close - 1.5 * atr
            reason = f"Mean Rev BUY: {' + '.join(reasons)}"
        elif signal_sum <= -2 and non_zero >= 2:
            # Strong SELL (overbought)
            final_signal = Signal.SELL
            confidence = min(confidence, 0.95)
            target = close - 2 * atr
            stop = close + 1.5 * atr
            reason = f"Mean Rev SELL: {' + '.join(reasons)}"
        elif signal_sum == 1 and non_zero >= 1 and volume_ok:
            # Weak BUY
            final_signal = Signal.BUY
            confidence = min(confidence * 0.7, 0.65)
            target = close + 1.5 * atr
            stop = close - 1 * atr
            reason = f"Weak Mean Rev BUY: {' + '.join(reasons)}"
        elif signal_sum == -1 and non_zero >= 1 and volume_ok:
            # Weak SELL
            final_signal = Signal.SELL
            confidence = min(confidence * 0.7, 0.65)
            target = close - 1.5 * atr
            stop = close + 1 * atr
            reason = f"Weak Mean Rev SELL: {' + '.join(reasons)}"
        else:
            final_signal = Signal.HOLD
            confidence = 0.0
            target = None
            stop = None
            reason = "No mean reversion signal"
        
        return self._create_signal(
            row, final_signal, confidence, reason,
            ml_probability=ml_prediction,
            target_price=target,
            stop_loss=stop
        )
