# Machine Learning API Reference

This document covers the machine learning components.

---

## FeatureEngineer

**Module:** `src/ml/features.py`

Creates features from market data for ML models.

### Constructor

```python
from ml.features import FeatureEngineer

engineer = FeatureEngineer()
```

### Methods

#### `create_all_features()`

Create all features from OHLCV data.

```python
df = engineer.create_all_features(
    df: pd.DataFrame,
    windows: List[int] = [5, 10, 20]
) -> pd.DataFrame
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | DataFrame | - | OHLCV data |
| `windows` | List[int] | [5, 10, 20] | Lookback windows for rolling features |

**Returns:** DataFrame with 90+ features added

**Feature Categories:**

| Category | Features | Description |
|----------|----------|-------------|
| Price | ~15 | Returns, momentum, log returns |
| Volatility | ~12 | Rolling std, ATR, Bollinger |
| Volume | ~12 | VWAP, volume ratio, OBV |
| Candle | ~8 | Body size, shadows, doji |
| Order Flow | ~9 | VPIN, buy/sell volume |
| Time | ~4 | Hour, day, is_weekend |

#### `create_features()`

Alias for `create_all_features()`.

```python
df = engineer.create_features(df: pd.DataFrame) -> pd.DataFrame
```

#### `prepare_training_data()`

Prepare sequences for LSTM training.

```python
X, y, feature_names = engineer.prepare_training_data(
    df: pd.DataFrame,
    target_col: str = 'close',
    lookback: int = 60,
    prediction_horizon: int = 1
) -> Tuple[np.ndarray, np.ndarray, List[str]]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | DataFrame | - | Data with features |
| `target_col` | str | 'close' | Column for prediction target |
| `lookback` | int | 60 | Sequence length (timesteps) |
| `prediction_horizon` | int | 1 | Steps ahead to predict |

**Returns:**
- `X`: Array of shape (n_samples, lookback, n_features)
- `y`: Array of shape (n_samples,) with 0/1 labels
- `feature_names`: List of feature column names

**Example:**
```python
engineer = FeatureEngineer()
df_features = engineer.create_all_features(df)

X, y, names = engineer.prepare_training_data(
    df_features,
    lookback=30,
    prediction_horizon=1
)

print(f"X shape: {X.shape}")  # (n_samples, 30, n_features)
print(f"y shape: {y.shape}")  # (n_samples,)
```

#### `select_top_features()`

Select features by correlation with target.

```python
top_features = engineer.select_top_features(
    df: pd.DataFrame,
    target_col: str = 'close',
    n_features: int = 20
) -> List[str]
```

---

## PriceLSTM

**Module:** `src/ml/models.py`

LSTM model for price direction prediction.

### Constructor

```python
from ml.models import PriceLSTM

model = PriceLSTM(
    input_size: int,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    bidirectional: bool = False
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_size` | int | - | Number of input features (required) |
| `hidden_size` | int | 64 | LSTM hidden state size |
| `num_layers` | int | 2 | Number of LSTM layers |
| `dropout` | float | 0.2 | Dropout probability |
| `bidirectional` | bool | False | Use bidirectional LSTM |

### Architecture

```
Input (batch, seq_len, features)
    ↓
LSTM Layers (num_layers × hidden_size)
    ↓
Last Timestep Output
    ↓
FC Layer (hidden_size → 32)
    ↓
ReLU + Dropout
    ↓
FC Layer (32 → 1)
    ↓
Sigmoid
    ↓
Output (batch, 1) - probability of price going up
```

### Methods

#### `train_model()`

Train the model with data.

```python
history = model.train_model(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 50,
    batch_size: int = 32,
    validation_split: float = 0.2,
    learning_rate: float = 0.001,
    early_stopping_patience: int = 10
) -> dict
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X` | ndarray | - | Training features (samples, seq, features) |
| `y` | ndarray | - | Training labels (samples,) |
| `epochs` | int | 50 | Maximum training epochs |
| `batch_size` | int | 32 | Mini-batch size |
| `validation_split` | float | 0.2 | Fraction for validation |
| `learning_rate` | float | 0.001 | Adam optimizer learning rate |
| `early_stopping_patience` | int | 10 | Stop after N epochs without improvement |

**Returns:**
```python
{
    'train_loss': List[float],     # Training loss per epoch
    'val_loss': List[float],       # Validation loss per epoch
    'val_accuracy': List[float]    # Validation accuracy per epoch
}
```

**Example:**
```python
model = PriceLSTM(input_size=90, hidden_size=64)

history = model.train_model(
    X, y,
    epochs=20,
    batch_size=32,
    validation_split=0.2
)

print(f"Final val accuracy: {history['val_accuracy'][-1]:.4f}")
```

#### `predict()`

Make predictions on new data.

```python
predictions = model.predict(X: np.ndarray) -> np.ndarray
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `X` | ndarray | Features (samples, seq, features) |

**Returns:** Array of probabilities (samples, 1)

**Example:**
```python
# Get predictions
probs = model.predict(X_test)

# Convert to binary
predictions = (probs > 0.5).astype(int)
```

#### `forward()`

PyTorch forward pass (for advanced use).

```python
output = model.forward(x: torch.Tensor) -> torch.Tensor
```

---

## ModelTrainer

**Module:** `src/ml/models.py`

Alternative trainer class with more control.

### Constructor

```python
from ml.models import ModelTrainer

trainer = ModelTrainer(
    input_size: int,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    learning_rate: float = 0.001
)
```

### Methods

#### `train()`

Train with explicit train/validation split.

```python
history = trainer.train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    sequence_length: int = 20,
    epochs: int = 50,
    batch_size: int = 32,
    early_stopping_patience: int = 10
) -> dict
```

#### `predict()`

Make predictions.

```python
predictions = trainer.predict(
    X: np.ndarray,
    sequence_length: int = 20
) -> np.ndarray
```

#### `save_model()` / `load_model()`

Save and load trained models.

```python
trainer.save_model(filepath: str)
trainer.load_model(filepath: str)
```

**Example:**
```python
# Save
trainer.save_model('models/saved/lstm_model.pt')

# Load
trainer.load_model('models/saved/lstm_model.pt')
```

---

## Usage Examples

### Complete ML Pipeline

```python
from data.fetcher import BinanceDataFetcher
from data.preprocessor import DataPreprocessor
from ml.features import FeatureEngineer
from ml.models import PriceLSTM

# 1. Get data
fetcher = BinanceDataFetcher()
df = fetcher.fetch_klines(limit=5000)

# 2. Preprocess
preprocessor = DataPreprocessor()
df = preprocessor.preprocess_pipeline(df)

# 3. Create features
engineer = FeatureEngineer()
df_features = engineer.create_all_features(df)

# 4. Prepare training data
X, y, feature_names = engineer.prepare_training_data(
    df_features,
    lookback=30
)

# 5. Train model
model = PriceLSTM(input_size=X.shape[2])
history = model.train_model(X, y, epochs=20)

# 6. Make predictions
predictions = model.predict(X[-100:])
print(f"Predicted up moves: {(predictions > 0.5).sum()}/{len(predictions)}")
```

### Evaluating Model Performance

```python
from sklearn.metrics import accuracy_score, classification_report

# Split data
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Train
model = PriceLSTM(input_size=X.shape[2])
model.train_model(X_train, y_train, validation_split=0.2)

# Predict
probs = model.predict(X_test)
preds = (probs.flatten() > 0.5).astype(int)

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
print(classification_report(y_test, preds))
```

---

## Next Steps

- [Trading Strategies API](04_strategies.md)
- [Backtesting API](05_backtesting.md)
