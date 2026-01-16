"""
Basic momentum trading strategy using ML signals.
Combines technical indicators with ML predictions for trading decisions.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List
from dataclasses import dataclass
from enum import Enum


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
    ml_probability: Optional[float] = None


class MomentumStrategy:
    """
    Momentum-based trading strategy combining ML predictions
    with technical indicators.
    """
    
    def __init__(
        self,
        ml_threshold: float = 0.55,  # Minimum ML probability for signal
        momentum_threshold: float = 0.0005,  # 0.05% momentum threshold
        volume_threshold: float = 1.2,  # Volume must be 1.2x average
        confidence_scaling: bool = True
    ):
        """
        Initialize momentum strategy.
        
        Args:
            ml_threshold: Minimum ML probability to generate signal
            momentum_threshold: Minimum momentum percentage
            volume_threshold: Minimum volume ratio to average
            confidence_scaling: Scale position size by confidence
        """
        self.ml_threshold = ml_threshold
        self.momentum_threshold = momentum_threshold
        self.volume_threshold = volume_threshold
        self.confidence_scaling = confidence_scaling
        
        self.signals: List[TradeSignal] = []
    
    def get_name(self) -> str:
        """Return strategy name."""
        return "Momentum"
        
    def generate_signal(
        self,
        row: pd.Series,
        ml_prediction: Optional[float] = None
    ) -> TradeSignal:
        """
        Generate trading signal from market data and ML prediction.
        
        Args:
            row: DataFrame row with market data and features
            ml_prediction: ML model prediction (probability of up move)
            
        Returns:
            TradeSignal object
        """
        timestamp = row.name
        price = row['close']
        
        # Default to HOLD
        signal = Signal.HOLD
        confidence = 0.0
        reason = "No clear signal"
        
        # Check if we have required features
        required_features = ['momentum_10', 'volume_ratio_20', 'volatility_20']
        if not all(f in row.index for f in required_features):
            return TradeSignal(
                timestamp=timestamp,
                signal=signal,
                confidence=confidence,
                price=price,
                reason="Missing required features",
                ml_probability=ml_prediction
            )
        
        # Get momentum and volume indicators
        momentum = row['momentum_10'] / price  # Normalize by price
        volume_ratio = row['volume_ratio_20']
        volatility = row['volatility_20']
        
        # ML-based signal
        ml_signal = Signal.HOLD
        ml_confidence = 0.0
        
        if ml_prediction is not None:
            if ml_prediction > self.ml_threshold:
                ml_signal = Signal.BUY
                ml_confidence = ml_prediction
            elif ml_prediction < (1 - self.ml_threshold):
                ml_signal = Signal.SELL
                ml_confidence = 1 - ml_prediction
        
        # Momentum-based signal
        momentum_signal = Signal.HOLD
        momentum_confidence = 0.0
        
        if abs(momentum) > self.momentum_threshold:
            if momentum > 0:
                momentum_signal = Signal.BUY
                momentum_confidence = min(abs(momentum) / self.momentum_threshold, 1.0)
            else:
                momentum_signal = Signal.SELL
                momentum_confidence = min(abs(momentum) / self.momentum_threshold, 1.0)
        
        # Volume confirmation
        volume_confirmed = volume_ratio >= self.volume_threshold
        
        # Combine signals
        if ml_signal == momentum_signal and ml_signal != Signal.HOLD:
            # Both agree - strong signal
            signal = ml_signal
            confidence = (ml_confidence + momentum_confidence) / 2
            
            if volume_confirmed:
                confidence *= 1.2  # Boost confidence with volume
                reason = f"ML + Momentum + Volume ({signal.name})"
            else:
                confidence *= 0.8  # Reduce confidence without volume
                reason = f"ML + Momentum, weak volume ({signal.name})"
                
        elif ml_signal != Signal.HOLD and volume_confirmed:
            # ML signal with volume confirmation
            signal = ml_signal
            confidence = ml_confidence * 0.7
            reason = f"ML + Volume ({signal.name})"
            
        elif momentum_signal != Signal.HOLD and volume_confirmed:
            # Momentum signal with volume confirmation
            signal = momentum_signal
            confidence = momentum_confidence * 0.6
            reason = f"Momentum + Volume ({signal.name})"
        
        # Cap confidence at 1.0
        confidence = min(confidence, 1.0)
        
        # Create signal
        trade_signal = TradeSignal(
            timestamp=timestamp,
            signal=signal,
            confidence=confidence,
            price=price,
            reason=reason,
            ml_probability=ml_prediction
        )
        
        self.signals.append(trade_signal)
        return trade_signal
    
    def generate_signals_batch(
        self,
        df: pd.DataFrame,
        ml_predictions: Optional[np.ndarray] = None
    ) -> List[TradeSignal]:
        """
        Generate signals for entire dataframe.
        
        Args:
            df: DataFrame with market data
            ml_predictions: Array of ML predictions (optional)
            
        Returns:
            List of TradeSignal objects
        """
        signals = []
        
        for i, (idx, row) in enumerate(df.iterrows()):
            ml_pred = ml_predictions[i] if ml_predictions is not None else None
            signal = self.generate_signal(row, ml_pred)
            signals.append(signal)
        
        return signals
    
    def get_position_size(
        self,
        signal: TradeSignal,
        account_balance: float,
        current_position: float = 0.0,
        max_position_pct: float = 0.5
    ) -> float:
        """
        Calculate position size based on signal and risk parameters.
        
        Args:
            signal: Trading signal
            account_balance: Current account balance
            current_position: Current position value
            max_position_pct: Maximum position as % of balance
            
        Returns:
            Position size to take (positive for buy, negative for sell)
        """
        if signal.signal == Signal.HOLD:
            return 0.0
        
        # Calculate max position value
        max_position_value = account_balance * max_position_pct
        
        # Scale by confidence if enabled
        if self.confidence_scaling:
            target_value = max_position_value * signal.confidence
        else:
            target_value = max_position_value
        
        # Calculate position change needed
        if signal.signal == Signal.BUY:
            # Want to be long
            if current_position < target_value:
                return target_value - current_position
            else:
                return 0.0
        
        else:  # SELL
            # Want to be flat or short
            if current_position > 0:
                # Close long position
                return -current_position
            else:
                # Could go short here if desired
                return 0.0
        
    def get_signal_statistics(self) -> Dict:
        """
        Get statistics about generated signals.
        
        Returns:
            Dictionary with signal statistics
        """
        if not self.signals:
            return {
                'total_signals': 0,
                'buy_signals': 0,
                'sell_signals': 0,
                'hold_signals': 0,
                'avg_confidence': 0.0
            }
        
        buy_signals = sum(1 for s in self.signals if s.signal == Signal.BUY)
        sell_signals = sum(1 for s in self.signals if s.signal == Signal.SELL)
        hold_signals = sum(1 for s in self.signals if s.signal == Signal.HOLD)
        
        confidences = [s.confidence for s in self.signals if s.signal != Signal.HOLD]
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        return {
            'total_signals': len(self.signals),
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'hold_signals': hold_signals,
            'avg_confidence': avg_confidence,
            'buy_pct': buy_signals / len(self.signals) * 100,
            'sell_pct': sell_signals / len(self.signals) * 100,
            'hold_pct': hold_signals / len(self.signals) * 100
        }


# Example usage
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    
    from data.fetcher import BinanceDataFetcher
    from ml.features import FeatureEngineer
    
    # Fetch data
    print("Fetching data...")
    fetcher = BinanceDataFetcher("BTCUSDT")
    df = fetcher.fetch_klines(interval="1m", limit=1000)
    
    if not df.empty:
        # Create features
        print("Creating features...")
        engineer = FeatureEngineer()
        df_features = engineer.create_all_features(df)
        
        # Create strategy
        strategy = MomentumStrategy(
            ml_threshold=0.55,
            momentum_threshold=0.0005,
            volume_threshold=1.2
        )
        
        # Generate mock ML predictions (random for demo)
        np.random.seed(42)
        ml_predictions = np.random.uniform(0.3, 0.7, len(df_features))
        
        # Generate signals
        print("\nGenerating signals...")
        signals = strategy.generate_signals_batch(df_features, ml_predictions)
        
        # Get statistics
        stats = strategy.get_signal_statistics()
        print("\nSignal Statistics:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        # Show some signals
        print("\nSample Signals (non-HOLD):")
        action_signals = [s for s in signals if s.signal != Signal.HOLD][:10]
        for sig in action_signals:
            print(f"  {sig.timestamp}: {sig.signal.name} @ ${sig.price:.2f}")
            print(f"    Confidence: {sig.confidence:.2f}, Reason: {sig.reason}")
            if sig.ml_probability:
                print(f"    ML Prob: {sig.ml_probability:.2f}")