# Custom Indicators Example

This example shows how to add your own technical indicators.

## Adding Custom Indicators

### Method 1: Extend FeatureEngineer

```python
from ml.features import FeatureEngineer
import pandas as pd
import numpy as np

class ExtendedFeatureEngineer(FeatureEngineer):
    """Extended feature engineer with custom indicators."""
    
    def create_all_features(self, df, windows=[5, 10, 20]):
        """Create all features including custom ones."""
        # Get base features
        df = super().create_all_features(df, windows)
        
        # Add custom features
        df = self.add_custom_indicators(df)
        
        return df
    
    def add_custom_indicators(self, df):
        """Add your custom indicators here."""
        
        # Custom 1: MACD
        df = self._add_macd(df)
        
        # Custom 2: Stochastic Oscillator
        df = self._add_stochastic(df)
        
        # Custom 3: ADX (Average Directional Index)
        df = self._add_adx(df)
        
        # Custom 4: Custom momentum score
        df = self._add_momentum_score(df)
        
        return df.dropna()
    
    def _add_macd(self, df, fast=12, slow=26, signal=9):
        """Add MACD indicator."""
        exp_fast = df['close'].ewm(span=fast, adjust=False).mean()
        exp_slow = df['close'].ewm(span=slow, adjust=False).mean()
        
        df['macd'] = exp_fast - exp_slow
        df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Normalized MACD
        df['macd_pct'] = df['macd'] / df['close'] * 100
        
        return df
    
    def _add_stochastic(self, df, k_period=14, d_period=3):
        """Add Stochastic Oscillator."""
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        
        df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min)
        df['stoch_d'] = df['stoch_k'].rolling(window=d_period).mean()
        
        # Overbought/oversold signals
        df['stoch_overbought'] = (df['stoch_k'] > 80).astype(int)
        df['stoch_oversold'] = (df['stoch_k'] < 20).astype(int)
        
        return df
    
    def _add_adx(self, df, period=14):
        """Add ADX trend strength indicator."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed values
        atr = pd.Series(tr).rolling(period).mean()
        plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / atr
        
        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        df['adx'] = dx.rolling(period).mean()
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di
        
        # Strong trend indicator
        df['strong_trend'] = (df['adx'] > 25).astype(int)
        
        return df
    
    def _add_momentum_score(self, df):
        """Custom composite momentum score."""
        # Combine multiple momentum indicators
        score = 0
        
        # RSI contribution
        if 'rsi_14' in df.columns:
            rsi = df['rsi_14']
            rsi_score = (rsi - 50) / 50  # -1 to 1
            score += rsi_score * 0.3
        
        # MACD contribution
        if 'macd_pct' in df.columns:
            macd_score = df['macd_pct'].clip(-1, 1)
            score += macd_score * 0.3
        
        # Momentum contribution
        if 'momentum_10' in df.columns:
            mom_score = (df['momentum_10'] * 100).clip(-1, 1)
            score += mom_score * 0.4
        
        df['momentum_score'] = score
        df['momentum_buy'] = (df['momentum_score'] > 0.3).astype(int)
        df['momentum_sell'] = (df['momentum_score'] < -0.3).astype(int)
        
        return df


# Usage
engineer = ExtendedFeatureEngineer()
df_features = engineer.create_all_features(df_processed)

print(f"Total features: {len(df_features.columns)}")
# Now has MACD, Stochastic, ADX, and custom momentum score
```

---

## Method 2: Add Indicators Before Feature Engineering

```python
def add_custom_indicators(df):
    """Add custom indicators before feature engineering."""
    
    # Ichimoku Cloud
    df = add_ichimoku(df)
    
    # Keltner Channels
    df = add_keltner(df)
    
    # Williams %R
    df = add_williams_r(df)
    
    return df


def add_ichimoku(df, tenkan=9, kijun=26, senkou=52):
    """Add Ichimoku Cloud components."""
    high = df['high']
    low = df['low']
    close = df['close']
    
    # Tenkan-sen (Conversion Line)
    tenkan_high = high.rolling(tenkan).max()
    tenkan_low = low.rolling(tenkan).min()
    df['tenkan_sen'] = (tenkan_high + tenkan_low) / 2
    
    # Kijun-sen (Base Line)
    kijun_high = high.rolling(kijun).max()
    kijun_low = low.rolling(kijun).min()
    df['kijun_sen'] = (kijun_high + kijun_low) / 2
    
    # Senkou Span A (Leading Span A)
    df['senkou_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(kijun)
    
    # Senkou Span B (Leading Span B)
    senkou_high = high.rolling(senkou).max()
    senkou_low = low.rolling(senkou).min()
    df['senkou_b'] = ((senkou_high + senkou_low) / 2).shift(kijun)
    
    # Price position relative to cloud
    cloud_top = df[['senkou_a', 'senkou_b']].max(axis=1)
    cloud_bottom = df[['senkou_a', 'senkou_b']].min(axis=1)
    
    df['above_cloud'] = (close > cloud_top).astype(int)
    df['below_cloud'] = (close < cloud_bottom).astype(int)
    df['in_cloud'] = (~df['above_cloud'].astype(bool) & 
                      ~df['below_cloud'].astype(bool)).astype(int)
    
    return df


def add_keltner(df, period=20, multiplier=2):
    """Add Keltner Channels."""
    typical = (df['high'] + df['low'] + df['close']) / 3
    ma = typical.rolling(period).mean()
    atr = df['atr_14'] if 'atr_14' in df.columns else \
          (df['high'] - df['low']).rolling(period).mean()
    
    df['keltner_upper'] = ma + multiplier * atr
    df['keltner_lower'] = ma - multiplier * atr
    df['keltner_mid'] = ma
    
    # Price position
    df['above_keltner'] = (df['close'] > df['keltner_upper']).astype(int)
    df['below_keltner'] = (df['close'] < df['keltner_lower']).astype(int)
    
    return df


def add_williams_r(df, period=14):
    """Add Williams %R."""
    high = df['high'].rolling(period).max()
    low = df['low'].rolling(period).min()
    
    df['williams_r'] = -100 * (high - df['close']) / (high - low)
    
    # Signals
    df['williams_overbought'] = (df['williams_r'] > -20).astype(int)
    df['williams_oversold'] = (df['williams_r'] < -80).astype(int)
    
    return df


# Usage
df_with_indicators = add_custom_indicators(df_processed)
engineer = FeatureEngineer()
df_features = engineer.create_all_features(df_with_indicators)
```

---

## Method 3: Create Strategy with Custom Indicators

```python
class CustomIndicatorStrategy:
    """Strategy that uses custom indicators."""
    
    def __init__(self):
        self.signals_generated = []
    
    def generate_signal(self, row, ml_prediction=None):
        # Use MACD
        macd_signal = self._macd_signal(row)
        
        # Use Stochastic
        stoch_signal = self._stochastic_signal(row)
        
        # Use ADX for trend strength
        trend_strong = row.get('adx', 0) > 25
        
        # Combine signals
        if macd_signal == 'BUY' and stoch_signal == 'BUY' and trend_strong:
            return self._create_signal(row, Signal.BUY, 0.8, "MACD + Stoch + Trend")
        elif macd_signal == 'SELL' and stoch_signal == 'SELL':
            return self._create_signal(row, Signal.SELL, 0.7, "MACD + Stoch")
        
        return self._create_signal(row, Signal.HOLD, 0.0, "No signal")
    
    def _macd_signal(self, row):
        macd = row.get('macd', 0)
        macd_signal = row.get('macd_signal', 0)
        
        if macd > macd_signal and macd > 0:
            return 'BUY'
        elif macd < macd_signal and macd < 0:
            return 'SELL'
        return 'HOLD'
    
    def _stochastic_signal(self, row):
        stoch_k = row.get('stoch_k', 50)
        stoch_d = row.get('stoch_d', 50)
        
        if stoch_k < 20 and stoch_k > stoch_d:
            return 'BUY'
        elif stoch_k > 80 and stoch_k < stoch_d:
            return 'SELL'
        return 'HOLD'
```

---

## Complete Example

```python
#!/usr/bin/env python3
"""Custom Indicators Example"""

import sys
sys.path.insert(0, 'src')

from data.fetcher import BinanceDataFetcher
from data.preprocessor import DataPreprocessor
from ml.features import FeatureEngineer
from backtest.backtester import Backtester, BacktestConfig

# Define custom indicator functions
def add_macd(df, fast=12, slow=26, signal=9):
    exp_fast = df['close'].ewm(span=fast).mean()
    exp_slow = df['close'].ewm(span=slow).mean()
    df['macd'] = exp_fast - exp_slow
    df['macd_signal'] = df['macd'].ewm(span=signal).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    return df

def add_stochastic(df, k=14, d=3):
    low_min = df['low'].rolling(k).min()
    high_max = df['high'].rolling(k).max()
    df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min)
    df['stoch_d'] = df['stoch_k'].rolling(d).mean()
    return df

# Fetch and process data
fetcher = BinanceDataFetcher()
df = fetcher.fetch_klines(limit=2000)

preprocessor = DataPreprocessor()
df_processed = preprocessor.preprocess_pipeline(df)

# Add custom indicators
df_processed = add_macd(df_processed)
df_processed = add_stochastic(df_processed)

# Create standard features
engineer = FeatureEngineer()
df_features = engineer.create_all_features(df_processed)

# Check new indicators exist
custom_cols = ['macd', 'macd_signal', 'macd_histogram', 'stoch_k', 'stoch_d']
for col in custom_cols:
    if col in df_features.columns:
        print(f"✓ {col} added")
    else:
        print(f"✗ {col} missing")

print(f"\nTotal features: {len(df_features.columns)}")

# Run backtest with custom features
from strategies.momentum_strategy import MomentumStrategy

strategy = MomentumStrategy()
config = BacktestConfig(initial_capital=100000)

backtester = Backtester(strategy, config)
results = backtester.run(df_features)

print(f"\nReturn: {results['summary']['total_return_pct']:.2f}%")
```

---

## Popular Indicators to Add

| Indicator | What It Does | Typical Usage |
|-----------|--------------|---------------|
| **MACD** | Momentum and trend | Crossover signals |
| **Stochastic** | Overbought/oversold | Mean reversion |
| **ADX** | Trend strength | Filter weak trends |
| **Ichimoku** | Multi-purpose | Trend + S/R |
| **Williams %R** | Overbought/oversold | Momentum |
| **Keltner** | Volatility channels | Breakouts |
| **CCI** | Commodity Channel | Trend + reversal |
| **OBV** | On-Balance Volume | Volume confirmation |
| **MFI** | Money Flow | Volume-based RSI |

---

## Tips

1. **Drop NaN rows** after adding indicators
2. **Normalize** indicators for ML (0-1 or z-score)
3. **Test individually** before combining
4. **Avoid overfitting** with too many indicators
