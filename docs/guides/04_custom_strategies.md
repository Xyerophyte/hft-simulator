# Building Custom Strategies

This guide shows how to create your own trading strategies.

## Strategy Interface

All strategies should implement signal generation:

```python
from strategies.momentum_strategy import Signal, TradeSignal
import pandas as pd
from typing import Optional

class CustomStrategy:
    """Base structure for a custom strategy."""
    
    def generate_signal(
        self,
        row: pd.Series,
        ml_prediction: Optional[float] = None
    ) -> TradeSignal:
        """
        Generate trading signal from current data.
        
        Args:
            row: Current bar data with features
            ml_prediction: Optional ML model output (0-1)
            
        Returns:
            TradeSignal with decision and confidence
        """
        # Your logic here
        pass
```

---

## Example 1: Simple Moving Average Crossover

Classic strategy: Buy when short MA crosses above long MA.

```python
class SMACrossoverStrategy:
    """
    Simple Moving Average Crossover Strategy.
    
    - BUY: Short MA > Long MA (uptrend)
    - SELL: Short MA < Long MA (downtrend)
    """
    
    def __init__(
        self,
        short_window: int = 10,
        long_window: int = 30,
        min_spread: float = 0.001
    ):
        self.short_window = short_window
        self.long_window = long_window
        self.min_spread = min_spread
        self.signals_generated = []
        
    def generate_signal(self, row: pd.Series, ml_prediction=None) -> TradeSignal:
        # Get MA values from features
        short_ma = row.get(f'sma_{self.short_window}', row.get('sma_10'))
        long_ma = row.get(f'sma_{self.long_window}', row.get('sma_20'))
        
        if pd.isna(short_ma) or pd.isna(long_ma):
            return self._create_signal(row, Signal.HOLD, 0.0, "Missing MA data")
        
        # Calculate spread
        spread = (short_ma - long_ma) / long_ma
        
        # Generate signal
        if spread > self.min_spread:
            confidence = min(abs(spread) / self.min_spread * 0.5, 1.0)
            return self._create_signal(row, Signal.BUY, confidence, 
                                       f"Short MA above Long MA by {spread*100:.2f}%")
                                       
        elif spread < -self.min_spread:
            confidence = min(abs(spread) / self.min_spread * 0.5, 1.0)
            return self._create_signal(row, Signal.SELL, confidence,
                                       f"Short MA below Long MA by {abs(spread)*100:.2f}%")
        else:
            return self._create_signal(row, Signal.HOLD, 0.0, "MAs too close")
    
    def _create_signal(self, row, signal, confidence, reason):
        ts = TradeSignal(
            timestamp=row.name if hasattr(row, 'name') else pd.Timestamp.now(),
            signal=signal,
            confidence=confidence,
            price=row['close'],
            reason=reason,
            ml_probability=None
        )
        self.signals_generated.append(ts)
        return ts
```

---

## Example 2: RSI Mean Reversion

Buy oversold conditions, sell overbought conditions.

```python
class RSIMeanReversionStrategy:
    """
    RSI-based mean reversion strategy.
    
    - BUY: RSI < oversold (expecting bounce)
    - SELL: RSI > overbought (expecting drop)
    """
    
    def __init__(
        self,
        oversold: float = 30.0,
        overbought: float = 70.0,
        extreme_oversold: float = 20.0,
        extreme_overbought: float = 80.0
    ):
        self.oversold = oversold
        self.overbought = overbought
        self.extreme_oversold = extreme_oversold
        self.extreme_overbought = extreme_overbought
        self.signals_generated = []
        
    def generate_signal(self, row: pd.Series, ml_prediction=None) -> TradeSignal:
        rsi = row.get('rsi_14', 50.0)
        
        if pd.isna(rsi):
            return self._create_signal(row, Signal.HOLD, 0.0, "No RSI data")
        
        # Extreme oversold = strong BUY
        if rsi < self.extreme_oversold:
            return self._create_signal(row, Signal.BUY, 0.9,
                                       f"Extreme oversold RSI={rsi:.1f}")
        
        # Oversold = moderate BUY
        elif rsi < self.oversold:
            confidence = (self.oversold - rsi) / (self.oversold - self.extreme_oversold)
            return self._create_signal(row, Signal.BUY, 0.5 + 0.4 * confidence,
                                       f"Oversold RSI={rsi:.1f}")
        
        # Extreme overbought = strong SELL
        elif rsi > self.extreme_overbought:
            return self._create_signal(row, Signal.SELL, 0.9,
                                       f"Extreme overbought RSI={rsi:.1f}")
        
        # Overbought = moderate SELL
        elif rsi > self.overbought:
            confidence = (rsi - self.overbought) / (self.extreme_overbought - self.overbought)
            return self._create_signal(row, Signal.SELL, 0.5 + 0.4 * confidence,
                                       f"Overbought RSI={rsi:.1f}")
        
        # Neutral
        return self._create_signal(row, Signal.HOLD, 0.0, f"Neutral RSI={rsi:.1f}")
    
    def _create_signal(self, row, signal, confidence, reason):
        ts = TradeSignal(
            timestamp=row.name if hasattr(row, 'name') else pd.Timestamp.now(),
            signal=signal,
            confidence=confidence,
            price=row['close'],
            reason=reason,
            ml_probability=None
        )
        self.signals_generated.append(ts)
        return ts
```

---

## Example 3: Breakout Strategy

Trade when price breaks out of recent range.

```python
class BreakoutStrategy:
    """
    Breakout strategy: Trade when price moves outside recent range.
    
    - BUY: Price breaks above recent high
    - SELL: Price breaks below recent low
    """
    
    def __init__(
        self,
        lookback: int = 20,
        breakout_threshold: float = 0.001
    ):
        self.lookback = lookback
        self.breakout_threshold = breakout_threshold
        self.signals_generated = []
        
    def generate_signal(self, row: pd.Series, ml_prediction=None) -> TradeSignal:
        # Get recent high/low from features
        recent_high = row.get('rolling_high_20', row.get('high'))
        recent_low = row.get('rolling_low_20', row.get('low'))
        current_price = row['close']
        
        if pd.isna(recent_high) or pd.isna(recent_low):
            return self._create_signal(row, Signal.HOLD, 0.0, "Missing range data")
        
        # Calculate range position
        range_size = recent_high - recent_low
        if range_size == 0:
            return self._create_signal(row, Signal.HOLD, 0.0, "No range")
            
        # Check for breakout
        if current_price > recent_high * (1 + self.breakout_threshold):
            breakout_pct = (current_price - recent_high) / recent_high
            confidence = min(breakout_pct / 0.01, 1.0)
            return self._create_signal(row, Signal.BUY, confidence,
                                       f"Breakout above {recent_high:.2f}")
        
        elif current_price < recent_low * (1 - self.breakout_threshold):
            breakdown_pct = (recent_low - current_price) / recent_low
            confidence = min(breakdown_pct / 0.01, 1.0)
            return self._create_signal(row, Signal.SELL, confidence,
                                       f"Breakdown below {recent_low:.2f}")
        
        return self._create_signal(row, Signal.HOLD, 0.0, "In range")
    
    def _create_signal(self, row, signal, confidence, reason):
        ts = TradeSignal(
            timestamp=row.name if hasattr(row, 'name') else pd.Timestamp.now(),
            signal=signal,
            confidence=confidence,
            price=row['close'],
            reason=reason,
            ml_probability=None
        )
        self.signals_generated.append(ts)
        return ts
```

---

## Example 4: Combined Strategy

Use multiple signals together.

```python
class CombinedStrategy:
    """
    Combine multiple strategy signals.
    
    Votes from: SMA crossover, RSI, Volume.
    Trade when majority agree.
    """
    
    def __init__(
        self,
        sma_short: int = 10,
        sma_long: int = 30,
        rsi_oversold: float = 35,
        rsi_overbought: float = 65,
        volume_threshold: float = 1.3
    ):
        self.sma_short = sma_short
        self.sma_long = sma_long
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.volume_threshold = volume_threshold
        self.signals_generated = []
        
    def generate_signal(self, row: pd.Series, ml_prediction=None) -> TradeSignal:
        votes = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        reasons = []
        
        # Signal 1: SMA Crossover
        sma_short = row.get('sma_10', row.get('close'))
        sma_long = row.get('sma_20', row.get('close'))
        
        if sma_short > sma_long:
            votes['BUY'] += 1
            reasons.append("SMA↑")
        elif sma_short < sma_long:
            votes['SELL'] += 1
            reasons.append("SMA↓")
        else:
            votes['HOLD'] += 1
        
        # Signal 2: RSI
        rsi = row.get('rsi_14', 50)
        
        if rsi < self.rsi_oversold:
            votes['BUY'] += 1
            reasons.append(f"RSI={rsi:.0f}")
        elif rsi > self.rsi_overbought:
            votes['SELL'] += 1
            reasons.append(f"RSI={rsi:.0f}")
        else:
            votes['HOLD'] += 1
        
        # Signal 3: Volume confirmation
        volume_ratio = row.get('volume_ratio_10', 1.0)
        volume_confirmed = volume_ratio >= self.volume_threshold
        
        if volume_confirmed:
            reasons.append(f"Vol×{volume_ratio:.1f}")
        
        # Determine final signal
        max_votes = max(votes.values())
        if max_votes >= 2:  # Majority (at least 2 of 3)
            if votes['BUY'] == max_votes and volume_confirmed:
                confidence = max_votes / 3 + 0.2
                return self._create_signal(row, Signal.BUY, min(confidence, 1.0),
                                          " + ".join(reasons))
                                          
            elif votes['SELL'] == max_votes and volume_confirmed:
                confidence = max_votes / 3 + 0.2
                return self._create_signal(row, Signal.SELL, min(confidence, 1.0),
                                          " + ".join(reasons))
        
        return self._create_signal(row, Signal.HOLD, 0.0, "No consensus")
    
    def _create_signal(self, row, signal, confidence, reason):
        ts = TradeSignal(
            timestamp=row.name if hasattr(row, 'name') else pd.Timestamp.now(),
            signal=signal,
            confidence=confidence,
            price=row['close'],
            reason=reason,
            ml_probability=None
        )
        self.signals_generated.append(ts)
        return ts
```

---

## Using Custom Strategies

### With Backtester

```python
from backtest.backtester import Backtester, BacktestConfig

# Create your custom strategy
strategy = RSIMeanReversionStrategy(
    oversold=25,
    overbought=75
)

# Configure and run
config = BacktestConfig(initial_capital=100000)
backtester = Backtester(strategy, config)
results = backtester.run(df_features)

print(f"Return: {results['summary']['total_return_pct']:.2f}%")
```

---

## Adding Custom Features

If your strategy needs specific features:

```python
def add_custom_features(df):
    """Add custom features for your strategy."""
    
    # Rolling high/low
    df['rolling_high_20'] = df['high'].rolling(20).max()
    df['rolling_low_20'] = df['low'].rolling(20).min()
    
    # ATR for volatility-adjusted entries
    df['atr_pct'] = df['atr_14'] / df['close']
    
    # Custom indicator: Price vs VWAP
    df['price_vs_vwap'] = (df['close'] - df.get('vwap', df['close'])) / df['close']
    
    # Custom oscillator
    fast = df['close'].ewm(span=5).mean()
    slow = df['close'].ewm(span=15).mean()
    df['custom_oscillator'] = (fast - slow) / slow * 100
    
    return df.dropna()

# Use it
df_features = engineer.create_all_features(df)
df_features = add_custom_features(df_features)
```

---

## Best Practices

### 1. Start Simple

```python
# Good first strategy
class SimpleStrategy:
    def generate_signal(self, row, ml_prediction=None):
        if row['momentum_10'] > 0.001:
            return Signal.BUY
        elif row['momentum_10'] < -0.001:
            return Signal.SELL
        return Signal.HOLD
```

### 2. Add Complexity Gradually

```python
# Version 2: Add filter
if row['momentum_10'] > 0.001 and row['volume_ratio_10'] > 1.2:
    return Signal.BUY
    
# Version 3: Add confirmation
if row['momentum_10'] > 0.001 and row['rsi_14'] < 70:
    return Signal.BUY
```

### 3. Log Everything

```python
def generate_signal(self, row, ml_prediction=None):
    # Keep track of why decisions were made
    self.decision_log.append({
        'time': row.name,
        'price': row['close'],
        'momentum': row.get('momentum_10'),
        'rsi': row.get('rsi_14'),
        'signal': signal.name,
        'reason': reason
    })
```

### 4. Test Edge Cases

```python
# Handle missing data
if pd.isna(row.get('indicator')):
    return Signal.HOLD
    
# Handle extreme values
if not (0 <= rsi <= 100):
    return Signal.HOLD
```

---

## Template

Here's a template to start from:

```python
from strategies.momentum_strategy import Signal, TradeSignal
import pandas as pd
from typing import Optional

class MyCustomStrategy:
    """
    Description of your strategy here.
    """
    
    def __init__(self, param1=0.5, param2=20):
        self.param1 = param1
        self.param2 = param2
        self.signals_generated = []
        
    def generate_signal(
        self,
        row: pd.Series,
        ml_prediction: Optional[float] = None
    ) -> TradeSignal:
        """Generate trading signal."""
        
        # Your logic here
        signal = Signal.HOLD
        confidence = 0.0
        reason = "Default"
        
        # Example condition
        if some_condition:
            signal = Signal.BUY
            confidence = 0.7
            reason = "Condition met"
        
        return TradeSignal(
            timestamp=row.name if hasattr(row, 'name') else pd.Timestamp.now(),
            signal=signal,
            confidence=confidence,
            price=row['close'],
            reason=reason,
            ml_probability=ml_prediction
        )
    
    def get_signal_statistics(self):
        """Return statistics about generated signals."""
        total = len(self.signals_generated)
        if total == 0:
            return {}
        
        buys = sum(1 for s in self.signals_generated if s.signal == Signal.BUY)
        sells = sum(1 for s in self.signals_generated if s.signal == Signal.SELL)
        
        return {
            'total_signals': total,
            'buy_signals': buys,
            'sell_signals': sells,
            'hold_signals': total - buys - sells,
            'buy_pct': buys / total * 100,
            'sell_pct': sells / total * 100
        }
```

---

## Next Steps

- [Troubleshooting](05_troubleshooting.md) - Common issues
- [API Reference](../api/04_strategies.md) - Detailed strategy API
