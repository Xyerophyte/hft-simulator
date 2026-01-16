"""
Ensemble Strategy that combines multiple trading strategies.
Uses voting or weighted average to generate final signals.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any
from .base_strategy import BaseStrategy, Signal, TradeSignal


class EnsembleStrategy(BaseStrategy):
    """
    Ensemble strategy combining multiple sub-strategies.
    
    Combines signals using:
    - Unanimous voting (all must agree)
    - Majority voting (>50% agree)
    - Weighted average (confidence-weighted)
    
    Parameters:
        strategies: List of strategy instances
        method: Combination method ('majority', 'unanimous', 'weighted')
        min_confidence: Minimum confidence to generate signal
        weights: Optional weights for each strategy (for weighted method)
    """
    
    def __init__(
        self,
        strategies: List[BaseStrategy],
        method: str = 'majority',
        min_confidence: float = 0.5,
        weights: Optional[List[float]] = None
    ):
        super().__init__()
        self.strategies = strategies
        self.method = method
        self.min_confidence = min_confidence
        
        # Normalize weights
        if weights:
            total = sum(weights)
            self.weights = [w / total for w in weights]
        else:
            self.weights = [1.0 / len(strategies)] * len(strategies)
    
    def get_name(self) -> str:
        names = [s.get_name() for s in self.strategies]
        return f"Ensemble({'+'.join(names)})"
    
    @property
    def signals(self):
        """Alias for signals_generated for backtester compatibility."""
        return self.signals_generated
    
    def add_strategy(self, strategy: BaseStrategy, weight: float = 1.0) -> None:
        """Add a strategy to the ensemble."""
        self.strategies.append(strategy)
        self.weights.append(weight)
        # Renormalize
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]
    
    def generate_signal(
        self,
        row: pd.Series,
        ml_prediction: Optional[float] = None
    ) -> TradeSignal:
        """Generate combined signal from all strategies."""
        
        # Get signals from all strategies
        sub_signals = []
        for strategy in self.strategies:
            signal = strategy.generate_signal(row, ml_prediction)
            sub_signals.append(signal)
        
        # Combine based on method
        if self.method == 'unanimous':
            return self._combine_unanimous(row, sub_signals, ml_prediction)
        elif self.method == 'weighted':
            return self._combine_weighted(row, sub_signals, ml_prediction)
        else:  # majority
            return self._combine_majority(row, sub_signals, ml_prediction)
    
    def _combine_majority(
        self,
        row: pd.Series,
        signals: List[TradeSignal],
        ml_prediction: Optional[float]
    ) -> TradeSignal:
        """Combine using majority voting."""
        
        buy_votes = sum(1 for s in signals if s.signal == Signal.BUY)
        sell_votes = sum(1 for s in signals if s.signal == Signal.SELL)
        total = len(signals)
        
        # Aggregate confidence
        buy_conf = np.mean([s.confidence for s in signals if s.signal == Signal.BUY]) if buy_votes > 0 else 0
        sell_conf = np.mean([s.confidence for s in signals if s.signal == Signal.SELL]) if sell_votes > 0 else 0
        
        # Get targets from agreeing strategies
        buy_targets = [s.target_price for s in signals if s.signal == Signal.BUY and s.target_price]
        buy_stops = [s.stop_loss for s in signals if s.signal == Signal.BUY and s.stop_loss]
        sell_targets = [s.target_price for s in signals if s.signal == Signal.SELL and s.target_price]
        sell_stops = [s.stop_loss for s in signals if s.signal == Signal.SELL and s.stop_loss]
        
        # Determine final signal
        if buy_votes > total / 2 and buy_conf >= self.min_confidence:
            signal = Signal.BUY
            confidence = buy_conf * (buy_votes / total)
            voters = [s.strategy_name for s in signals if s.signal == Signal.BUY]
            reason = f"Majority BUY ({buy_votes}/{total}): {', '.join(voters)}"
            target = np.mean(buy_targets) if buy_targets else None
            stop = np.mean(buy_stops) if buy_stops else None
        
        elif sell_votes > total / 2 and sell_conf >= self.min_confidence:
            signal = Signal.SELL
            confidence = sell_conf * (sell_votes / total)
            voters = [s.strategy_name for s in signals if s.signal == Signal.SELL]
            reason = f"Majority SELL ({sell_votes}/{total}): {', '.join(voters)}"
            target = np.mean(sell_targets) if sell_targets else None
            stop = np.mean(sell_stops) if sell_stops else None
        
        else:
            signal = Signal.HOLD
            confidence = 0.0
            reason = f"No majority (BUY:{buy_votes}, SELL:{sell_votes})"
            target = None
            stop = None
        
        return self._create_signal(
            row, signal, min(confidence, 0.95), reason,
            ml_probability=ml_prediction,
            target_price=target,
            stop_loss=stop
        )
    
    def _combine_unanimous(
        self,
        row: pd.Series,
        signals: List[TradeSignal],
        ml_prediction: Optional[float]
    ) -> TradeSignal:
        """Combine using unanimous voting."""
        
        non_hold = [s for s in signals if s.signal != Signal.HOLD]
        
        if not non_hold:
            return self._create_signal(row, Signal.HOLD, 0.0, "All strategies hold")
        
        # Check if all agree
        first_signal = non_hold[0].signal
        all_agree = all(s.signal == first_signal for s in non_hold)
        
        if all_agree:
            avg_conf = np.mean([s.confidence for s in non_hold])
            targets = [s.target_price for s in non_hold if s.target_price]
            stops = [s.stop_loss for s in non_hold if s.stop_loss]
            
            return self._create_signal(
                row, first_signal, min(avg_conf, 0.98),
                f"Unanimous {first_signal.name}: All {len(non_hold)} strategies agree",
                ml_probability=ml_prediction,
                target_price=np.mean(targets) if targets else None,
                stop_loss=np.mean(stops) if stops else None
            )
        
        return self._create_signal(
            row, Signal.HOLD, 0.0, "Strategies disagree"
        )
    
    def _combine_weighted(
        self,
        row: pd.Series,
        signals: List[TradeSignal],
        ml_prediction: Optional[float]
    ) -> TradeSignal:
        """Combine using weighted average."""
        
        # Calculate weighted scores (-1 to 1)
        total_score = 0.0
        total_weight = 0.0
        
        for signal, weight in zip(signals, self.weights):
            if signal.signal == Signal.BUY:
                score = signal.confidence
            elif signal.signal == Signal.SELL:
                score = -signal.confidence
            else:
                score = 0.0
            
            total_score += score * weight
            total_weight += weight
        
        if total_weight > 0:
            weighted_score = total_score / total_weight
        else:
            weighted_score = 0.0
        
        # Convert to signal
        if weighted_score > self.min_confidence / 2:
            signal = Signal.BUY
            confidence = min(weighted_score, 0.95)
            reason = f"Weighted BUY (score={weighted_score:.2f})"
        elif weighted_score < -self.min_confidence / 2:
            signal = Signal.SELL
            confidence = min(abs(weighted_score), 0.95)
            reason = f"Weighted SELL (score={weighted_score:.2f})"
        else:
            signal = Signal.HOLD
            confidence = 0.0
            reason = f"Weighted neutral (score={weighted_score:.2f})"
        
        # Get average targets from matching signals
        if signal == Signal.BUY:
            targets = [s.target_price for s in signals if s.signal == Signal.BUY and s.target_price]
            stops = [s.stop_loss for s in signals if s.signal == Signal.BUY and s.stop_loss]
        elif signal == Signal.SELL:
            targets = [s.target_price for s in signals if s.signal == Signal.SELL and s.target_price]
            stops = [s.stop_loss for s in signals if s.signal == Signal.SELL and s.stop_loss]
        else:
            targets = []
            stops = []
        
        return self._create_signal(
            row, signal, confidence, reason,
            ml_probability=ml_prediction,
            target_price=np.mean(targets) if targets else None,
            stop_loss=np.mean(stops) if stops else None
        )
    
    def get_strategy_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get individual strategy performance within ensemble."""
        performance = {}
        for strategy in self.strategies:
            stats = strategy.get_signal_statistics()
            performance[strategy.get_name()] = stats
        return performance
