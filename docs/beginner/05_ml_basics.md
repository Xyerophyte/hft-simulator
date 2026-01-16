# Machine Learning Basics

This guide explains how machine learning is used to predict price movements.

## What is Machine Learning?

**Machine Learning (ML)** is teaching a computer to find patterns in data and make predictions.

### Analogy: Learning to Recognize Cats

How do you teach a child to recognize cats?

1. Show them many pictures of cats
2. Show them many pictures of NOT cats
3. Eventually they learn what makes a cat a cat
4. Now they can identify cats they've never seen before

ML works the same way for price prediction:

1. Show the computer many examples of price going UP
2. Show it many examples of price going DOWN
3. It learns patterns that predict price movement
4. Now it can predict new price movements

---

## Our ML Model: LSTM

We use an **LSTM** (Long Short-Term Memory) neural network.

### Why LSTM?

LSTMs are good at learning **sequences** - patterns that happen over time.

```
Time:     t=1    t=2    t=3    t=4    t=5    → Predict t=6
Prices:   $100   $102   $101   $103   $105   → ???
```

The LSTM looks at the sequence of past prices and patterns to predict what comes next.

### Simple Explanation

Think of LSTM as having:
- **Memory**: Remembers important information from the past
- **Forget Gate**: Knows what to ignore
- **Attention**: Focuses on what matters most

---

## What Are Features?

**Features** are the patterns we show the computer.

### Raw Data vs Features

**Raw data** (just the price):
```
Time 1: $50,000
Time 2: $50,100
Time 3: $50,050
```

**Features** (patterns extracted from price):
```
Time 3 Features:
- Price: $50,050
- Change from time 2: -$50 (-0.1%)
- Change from time 1: +$50 (+0.1%)
- Average of last 2: $50,050
- Is price above average? Yes
- Volatility: 0.1%
- Volume: 100 BTC
- ... and 83 more features!
```

### Our 90 Features

We create 90 different features in 6 categories:

| Category | Examples |
|----------|----------|
| **Price** | Returns, momentum, log returns |
| **Volatility** | How much price is jumping around |
| **Volume** | How much is being traded |
| **Candles** | Shape of price bars |
| **Order Flow** | Buy vs sell pressure |
| **Time** | Hour, day of week |

---

## How Training Works

### Step 1: Prepare Data

```
For each moment in time, create:
- Input: Last 30 minutes of features (30 × 90 = 2700 numbers)
- Label: Did price go UP or DOWN in the next minute?
```

### Step 2: Show Examples

```
Example 1: These 30 minutes of features → Price went UP
Example 2: These 30 minutes of features → Price went DOWN
Example 3: These 30 minutes of features → Price went UP
... (thousands of examples)
```

### Step 3: Learn Patterns

The computer adjusts its internal numbers to get better at predicting.

```
Epoch 1: 45% accuracy (almost random guessing)
Epoch 5: 50% accuracy (getting better)
Epoch 10: 52% accuracy (finding patterns)
Epoch 20: 55% accuracy (learning useful patterns)
```

### Step 4: Validate

Test on data the model has never seen:
```
Training data: Jan - Oct (learn from this)
Test data: Nov - Dec (check if learning is real)
```

If test accuracy is similar to training accuracy = good learning!
If test accuracy is much worse = **overfitting** (memorizing, not learning)

---

## Understanding Predictions

### What the Model Outputs

```
Input: Last 30 minutes of features
Output: Probability of price going UP (0.0 to 1.0)
```

**Examples:**
- Output = 0.85 → 85% confident price will go UP
- Output = 0.50 → 50% confident (coin flip)
- Output = 0.20 → 80% confident price will go DOWN

### How We Use Predictions

```
If prediction > 0.55:
    Signal = BUY (we think price goes up)
    
If prediction < 0.45:
    Signal = SELL (we think price goes down)
    
If 0.45 <= prediction <= 0.55:
    Signal = HOLD (not confident enough)
```

---

## Why We Can't Predict Perfectly

### The Market is Hard

1. **Random Events**: News, tweets, world events
2. **Other Traders**: Everyone is trying to predict
3. **Self-Defeating**: If everyone knew, prices would already move
4. **Noise**: Small fluctuations are random

### Realistic Expectations

| Metric | Random | Our Goal | Professional |
|--------|--------|----------|--------------|
| Accuracy | 50% | 53-55% | 52-58% |
| Edge per trade | 0% | 0.05-0.1% | 0.03-0.2% |

**Key insight**: Even 52% accuracy can be profitable if:
- You make many trades
- Your winners are bigger than losers
- Fees are low

---

## The Training Process

### Code Overview

```python
from ml.features import FeatureEngineer
from ml.models import PriceLSTM

# 1. Create features
engineer = FeatureEngineer()
df_features = engineer.create_all_features(df)

# 2. Prepare training data
X, y, feature_names = engineer.prepare_training_data(
    df_features,
    lookback=30  # Look at last 30 minutes
)

# 3. Create model
model = PriceLSTM(
    input_size=90,    # 90 features
    hidden_size=64,   # 64 memory units
    num_layers=2      # 2 LSTM layers
)

# 4. Train
history = model.train_model(X, y, epochs=20)

# 5. Predict
predictions = model.predict(X_new)
```

### Training Output

```
Training on cpu with 3000 samples...
Epoch 5/20 - Loss: 0.6932, Val Loss: 0.6928, Val Acc: 0.5123
Epoch 10/20 - Loss: 0.6890, Val Loss: 0.6915, Val Acc: 0.5234
Epoch 15/20 - Loss: 0.6850, Val Loss: 0.6900, Val Acc: 0.5356
Epoch 20/20 - Loss: 0.6810, Val Loss: 0.6890, Val Acc: 0.5412

Training complete!
Final validation accuracy: 54.12%
```

---

## Overfitting: The Big Danger

### What is Overfitting?

**Overfitting** = Model memorizes training data instead of learning patterns

**Analogy**: A student who memorizes answers to practice tests but can't solve new problems.

### How We Prevent It

1. **Validation Split**: Keep some data hidden during training
2. **Early Stopping**: Stop when validation stops improving
3. **Dropout**: Randomly turn off neurons during training
4. **Regularization**: Penalize complex patterns

### Signs of Overfitting

```
Training accuracy: 75%  ← Very high
Validation accuracy: 51% ← Much lower

This is overfitting! The model memorized training data.
```

### Good Training

```
Training accuracy: 54%
Validation accuracy: 53%

This is good! Performance is similar on new data.
```

---

## Key Takeaways

1. **ML finds patterns** in historical data to predict future prices
2. **Features** are the patterns we extract from raw data (90 features)
3. **LSTM** is good for sequence data like price history
4. **Training** shows the model many examples to learn from
5. **Predictions** are probabilities (0.7 = 70% confident of UP)
6. **Accuracy** of 52-55% is realistic and can be profitable
7. **Overfitting** is the main danger - always validate!

---

## Next Steps

- [Strategy Logic](06_strategy_logic.md) - How ML predictions are used
- [API Reference](../api/03_machine_learning.md) - Detailed ML API docs
