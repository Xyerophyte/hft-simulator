# Data Pipeline API Reference

This document covers the data pipeline components.

---

## BinanceDataFetcher

**Module:** `src/data/fetcher.py`

Fetches market data from the Binance REST API.

### Constructor

```python
from data.fetcher import BinanceDataFetcher

fetcher = BinanceDataFetcher(symbol: str = "BTCUSDT")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `symbol` | str | "BTCUSDT" | Trading pair symbol |

### Methods

#### `fetch_klines()`

Fetch candlestick (OHLCV) data.

```python
df = fetcher.fetch_klines(
    interval: str = "1m",
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = 1000
) -> pd.DataFrame
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `interval` | str | "1m" | Candle interval (1m, 5m, 15m, 1h, 4h, 1d) |
| `start_time` | datetime | None | Start time for data |
| `end_time` | datetime | None | End time for data |
| `limit` | int | 1000 | Maximum number of candles |

**Returns:** DataFrame with columns: `open`, `high`, `low`, `close`, `volume`

**Example:**
```python
fetcher = BinanceDataFetcher("BTCUSDT")
df = fetcher.fetch_klines(interval="1m", limit=1000)
print(df.head())
```

#### `fetch_historical_data()`

Fetch extended historical data with automatic pagination.

```python
df = fetcher.fetch_historical_data(
    interval: str = "1m",
    days: int = 90,
    end_date: Optional[datetime] = None
) -> pd.DataFrame
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `interval` | str | "1m" | Candle interval |
| `days` | int | 90 | Number of days to fetch |
| `end_date` | datetime | None | End date (defaults to now) |

**Returns:** DataFrame with OHLCV data

**Example:**
```python
# Fetch 30 days of historical data
df = fetcher.fetch_historical_data(interval="1m", days=30)
print(f"Fetched {len(df)} candles")
```

#### `get_current_price()`

Get the current market price.

```python
price = fetcher.get_current_price() -> Optional[float]
```

**Returns:** Current price as float, or None if request fails

---

## DataCache

**Module:** `src/data/cache.py`

Caches market data to disk for offline use.

### Constructor

```python
from data.cache import DataCache

cache = DataCache(cache_dir: str = "data/raw")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cache_dir` | str | "data/raw" | Directory for cached files |

### Methods

#### `save()`

Save DataFrame to cache.

```python
success = cache.save(
    data: pd.DataFrame,
    symbol: str,
    interval: str,
    file_format: str = "parquet"
) -> bool
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | DataFrame | Data to save |
| `symbol` | str | Trading symbol (e.g., "BTCUSDT") |
| `interval` | str | Time interval (e.g., "1m") |
| `file_format` | str | "parquet" or "csv" |

**Returns:** True if successful

**Example:**
```python
cache = DataCache()
cache.save(df, symbol="BTCUSDT", interval="1m", file_format="parquet")
```

#### `load()`

Load DataFrame from cache.

```python
df = cache.load(
    symbol: str,
    interval: str,
    file_format: str = "parquet"
) -> Optional[pd.DataFrame]
```

**Returns:** DataFrame or None if not found

#### `exists()`

Check if cached data exists.

```python
exists = cache.exists(
    symbol: str,
    interval: str,
    file_format: str = "parquet"
) -> bool
```

#### `get_cache_info()`

Get information about cached files.

```python
info = cache.get_cache_info(symbol: str, interval: str) -> dict
```

**Returns:** Dict with `exists`, `size_mb`, `modified_time`, `rows`

#### `clear_cache()`

Clear cached data.

```python
success = cache.clear_cache(
    symbol: Optional[str] = None,
    interval: Optional[str] = None
) -> bool
```

If no parameters, clears all cache.

---

## DataPreprocessor

**Module:** `src/data/preprocessor.py`

Cleans and preprocesses market data.

### Constructor

```python
from data.preprocessor import DataPreprocessor

preprocessor = DataPreprocessor()
```

### Methods

#### `clean_data()`

Clean raw data (handle NaN, validate OHLC).

```python
df_clean = preprocessor.clean_data(data: pd.DataFrame) -> pd.DataFrame
```

Operations:
- Forward fill missing values
- Remove duplicate indices
- Validate OHLC relationships (high >= low, etc.)

#### `add_technical_indicators()`

Add technical indicators to data.

```python
df = preprocessor.add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame
```

**Added Indicators:**

| Indicator | Description |
|-----------|-------------|
| `returns` | Simple returns |
| `log_returns` | Log returns |
| `volatility_*` | Rolling volatility (10, 20 periods) |
| `sma_*` | Simple moving averages (10, 20, 50) |
| `ema_*` | Exponential moving averages (10, 20) |
| `rsi_14` | Relative Strength Index (14 period) |
| `bb_upper/lower` | Bollinger Bands |
| `atr_14` | Average True Range |
| `roc_*` | Rate of Change |
| `momentum_*` | Price momentum |
| `volume_ma_*` | Volume moving averages |

#### `normalize_features()`

Normalize feature columns.

```python
df = preprocessor.normalize_features(
    df: pd.DataFrame,
    columns: Optional[list] = None,
    method: str = "minmax"
) -> pd.DataFrame
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `columns` | list | Columns to normalize (None = all numeric) |
| `method` | str | "minmax" or "zscore" |

#### `preprocess_pipeline()`

Run complete preprocessing pipeline.

```python
df = preprocessor.preprocess_pipeline(
    df: pd.DataFrame,
    add_features: bool = True
) -> pd.DataFrame
```

This runs: clean_data → add_technical_indicators → drop NaN rows

---

## Usage Examples

### Complete Data Pipeline

```python
from data.fetcher import BinanceDataFetcher
from data.cache import DataCache
from data.preprocessor import DataPreprocessor

# Initialize components
fetcher = BinanceDataFetcher("BTCUSDT")
cache = DataCache("data/cache")
preprocessor = DataPreprocessor()

# Check cache first
if cache.exists("BTCUSDT", "1m"):
    print("Loading from cache...")
    df = cache.load("BTCUSDT", "1m")
else:
    print("Fetching from API...")
    df = fetcher.fetch_historical_data(days=7)
    cache.save(df, "BTCUSDT", "1m")

# Preprocess
df_processed = preprocessor.preprocess_pipeline(df)

print(f"Data shape: {df_processed.shape}")
print(f"Columns: {list(df_processed.columns)}")
```

### Updating Cache

```python
# Append new data to existing cache
existing = cache.load("BTCUSDT", "1m")
new_data = fetcher.fetch_klines(limit=100)

# Combine and deduplicate
combined = pd.concat([existing, new_data])
combined = combined[~combined.index.duplicated(keep='last')]

# Save updated data
cache.save(combined, "BTCUSDT", "1m")
```

---

## Next Steps

- [Market Simulation API](02_market_simulation.md)
- [Machine Learning API](03_machine_learning.md)
