# Troubleshooting Guide

This guide covers common issues and their solutions.

## Import Errors

### Issue: `ModuleNotFoundError: No module named 'src'`

**Cause:** Python can't find the project modules.

**Solution:**
```python
# Add to the top of your script
import sys
sys.path.insert(0, 'path/to/hft-sim/src')

# Or run from project root
cd hft-sim
python your_script.py
```

### Issue: `ImportError: cannot import name 'X' from 'Y'`

**Cause:** Module name changed or doesn't exist.

**Solution:**
```python
# Check what's available
from data import fetcher
print(dir(fetcher))

# Common correct imports
from data.fetcher import BinanceDataFetcher
from data.preprocessor import DataPreprocessor
from ml.features import FeatureEngineer
from ml.models import PriceLSTM
from strategies.momentum_strategy import MomentumStrategy
from backtest.backtester import Backtester, BacktestConfig
```

---

## Data Fetching Issues

### Issue: `ConnectionError` when fetching from Binance

**Cause:** Network issue or API problem.

**Solutions:**

1. **Check internet connection**
   ```python
   import requests
   response = requests.get('https://api.binance.com/api/v3/ping')
   print(response.status_code)  # Should be 200
   ```

2. **Use cached data**
   ```python
   from data.cache import DataCache
   
   cache = DataCache()
   if cache.exists('BTCUSDT', '1m'):
       df = cache.load('BTCUSDT', '1m')
   else:
       # Try fetching
       pass
   ```

3. **Add retry logic**
   ```python
   import time
   
   def fetch_with_retry(fetcher, max_retries=3):
       for attempt in range(max_retries):
           try:
               return fetcher.fetch_klines(limit=1000)
           except Exception as e:
               print(f"Attempt {attempt + 1} failed: {e}")
               time.sleep(2 ** attempt)  # Exponential backoff
       raise Exception("All retries failed")
   ```

### Issue: Empty DataFrame returned

**Cause:** API returned no data.

**Solutions:**

1. **Check symbol is valid**
   ```python
   # Correct
   fetcher = BinanceDataFetcher("BTCUSDT")
   
   # Wrong
   fetcher = BinanceDataFetcher("BTC-USD")  # Wrong format
   ```

2. **Check limit parameter**
   ```python
   # API maximum is 1000 per call
   df = fetcher.fetch_klines(limit=500)  # Safe
   ```

---

## Training Issues

### Issue: `CUDA out of memory`

**Cause:** GPU doesn't have enough memory.

**Solutions:**

1. **Reduce batch size**
   ```python
   history = model.train_model(X, y, batch_size=16)  # Instead of 32
   ```

2. **Use CPU**
   ```python
   import os
   os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU
   ```

3. **Reduce sequence length**
   ```python
   X, y, _ = engineer.prepare_training_data(df, lookback=20)  # Instead of 60
   ```

### Issue: Training loss is NaN

**Cause:** Numerical instability.

**Solutions:**

1. **Normalize features**
   ```python
   from sklearn.preprocessing import StandardScaler
   
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
   ```

2. **Reduce learning rate**
   ```python
   history = model.train_model(X, y, learning_rate=0.0001)
   ```

3. **Check for infinite values**
   ```python
   import numpy as np
   
   print(f"NaN in X: {np.isnan(X).sum()}")
   print(f"Inf in X: {np.isinf(X).sum()}")
   
   # Remove problematic rows
   X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
   ```

### Issue: Validation accuracy stuck at ~50%

**Cause:** Model not learning meaningful patterns.

**Solutions:**

1. **Check class balance**
   ```python
   print(f"Class 0: {(y == 0).sum()}")
   print(f"Class 1: {(y == 1).sum()}")
   # Should be roughly balanced
   ```

2. **Try more training data**
   ```python
   df = fetcher.fetch_historical_data(days=30)  # More data
   ```

3. **Adjust model architecture**
   ```python
   model = PriceLSTM(
       input_size=n_features,
       hidden_size=64,  # Try 128
       num_layers=3     # Try more layers
   )
   ```

---

## Backtesting Issues

### Issue: No trades executed

**Cause:** Signal thresholds too strict or missing features.

**Solutions:**

1. **Check signal statistics**
   ```python
   print(f"Total signals: {results['num_signals']}")
   print(f"BUY signals: {results['signal_stats']['buy_signals']}")
   print(f"SELL signals: {results['signal_stats']['sell_signals']}")
   ```

2. **Lower thresholds**
   ```python
   strategy = MomentumStrategy(
       ml_threshold=0.52,        # Lower (default 0.55)
       momentum_threshold=0.0001  # Lower (default 0.0005)
   )
   ```

3. **Check required features exist**
   ```python
   required = ['close', 'momentum_10', 'volume_ratio_10', 'rsi_14']
   for col in required:
       print(f"{col}: {'✓' if col in df.columns else '✗'}")
   ```

### Issue: Unrealistic returns (too good to be true)

**Cause:** Look-ahead bias or bug.

**Solutions:**

1. **Check for future data leakage**
   ```python
   # Wrong: Using shift without direction
   df['target'] = df['close'].shift(-1)  # Uses future!
   
   # Correct: Shift direction matters
   df['prev_close'] = df['close'].shift(1)  # Uses past
   ```

2. **Add transaction costs**
   ```python
   config = BacktestConfig(
       fee_rate=0.001  # Ensure fees are realistic
   )
   ```

3. **Check data ordering**
   ```python
   assert df.index.is_monotonic_increasing, "Data not sorted by time!"
   ```

---

## Memory Issues

### Issue: `MemoryError`

**Cause:** Too much data loaded at once.

**Solutions:**

1. **Process in chunks**
   ```python
   chunk_size = 1000
   for i in range(0, len(df), chunk_size):
       chunk = df.iloc[i:i + chunk_size]
       # Process chunk
   ```

2. **Use efficient data types**
   ```python
   for col in df.select_dtypes('float64'):
       df[col] = df[col].astype('float32')
   ```

3. **Clear unused variables**
   ```python
   import gc
   del large_dataframe
   gc.collect()
   ```

---

## Visualization Issues

### Issue: Plots not showing

**Cause:** Matplotlib backend issue.

**Solutions:**

```python
# In Jupyter notebook
%matplotlib inline

# In regular Python
import matplotlib
matplotlib.use('TkAgg')  # or 'Agg' for headless
import matplotlib.pyplot as plt

plt.show()
```

### Issue: Blank or empty plots

**Cause:** No data or wrong data format.

**Solutions:**

```python
# Check data before plotting
print(f"Equity curve length: {len(equity_curve)}")
print(f"Empty: {equity_curve.empty}")
print(equity_curve.head())
```

---

## Performance Issues

### Issue: Backtest running slowly

**Cause:** Large dataset or inefficient operations.

**Solutions:**

1. **Use fewer data points**
   ```python
   df = df.iloc[-5000:]  # Last 5000 points only
   ```

2. **Disable unused features**
   ```python
   config = BacktestConfig(
       use_ml_model=False  # Skip ML predictions
   )
   ```

3. **Profile the code**
   ```python
   import time
   
   start = time.time()
   results = backtester.run(df)
   print(f"Runtime: {time.time() - start:.2f}s")
   ```

---

## Common Error Messages

### `ValueError: cannot reindex from a duplicate axis`

**Cause:** Duplicate timestamps in data.

**Solution:**
```python
# Remove duplicates
df = df[~df.index.duplicated(keep='last')]
```

### `KeyError: 'column_name'`

**Cause:** Expected column doesn't exist.

**Solution:**
```python
# Check available columns
print(df.columns.tolist())

# Use safe access
value = df.get('column_name', default_value)
```

### `TypeError: 'NoneType' object is not subscriptable`

**Cause:** Function returned None unexpectedly.

**Solution:**
```python
# Add None checks
result = some_function()
if result is not None:
    process(result)
else:
    print("Got None, handling gracefully")
```

---

## Debugging Tips

### 1. Check Data at Each Step

```python
print("=== Step 1: Raw data ===")
print(f"Shape: {df.shape}")
print(df.head())

print("\n=== Step 2: After preprocessing ===")
print(f"Shape: {df_proc.shape}")
print(df_proc.head())

print("\n=== Step 3: Features ===")
print(f"Shape: {df_features.shape}")
print(f"NaN count: {df_features.isna().sum().sum()}")
```

### 2. Use Assertions

```python
assert len(df) > 100, "Need more data"
assert 'close' in df.columns, "Missing close price"
assert not df['close'].isna().any(), "NaN in close prices"
```

### 3. Catch and Log Errors

```python
try:
    results = backtester.run(df)
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
```

---

## Getting Help

1. **Check documentation**
   - API references in `/docs/api/`
   - Guides in `/docs/guides/`

2. **Check tests**
   - See how functions are used in `/tests/`

3. **Print intermediate values**
   - Add print statements to narrow down the issue

4. **Create minimal reproduction**
   - Simplify code to isolate the problem
