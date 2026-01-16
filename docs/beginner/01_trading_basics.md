# Trading Basics for Absolute Beginners

## What is Trading?

**Trading** is buying and selling things to make a profit. In this project, we trade **Bitcoin** (a digital currency).

### Simple Example

Imagine you're trading Pokemon cards:
- You **buy** a Charizard card for $50
- Later, you **sell** it for $60
- Your **profit** is $10 ($60 - $50)

Trading Bitcoin works the same way:
- You **buy** 0.1 Bitcoin for $9,000
- Later, you **sell** it for $9,500
- Your **profit** is $500

## What is Algorithmic Trading?

**Algorithmic trading** means using a computer program to trade automatically instead of clicking buttons yourself.

### Why Use a Computer?

1. **Speed** - Computers react in milliseconds
2. **Emotionless** - No panic or greed
3. **Consistency** - Follows rules exactly
4. **24/7** - Never sleeps
5. **Data analysis** - Processes thousands of data points instantly

### Real-World Analogy

Think of it like a **self-driving car**:
- **Manual trading** = Driving yourself (you make every decision)
- **Algorithmic trading** = Self-driving car (program makes decisions based on sensors and rules)

## What is High-Frequency Trading (HFT)?

**High-Frequency Trading** is algorithmic trading that happens **very fast** - sometimes thousands of trades per second.

### Speed Comparison

- **Human trader**: 1-2 seconds to make a decision
- **Regular algo**: 100-500 milliseconds
- **HFT**: 0.001 milliseconds (1 microsecond!)

### Our Simulator

Our simulator is designed for **HFT research**, meaning:
- It processes market data at microsecond precision
- It can simulate thousands of trades quickly
- It models realistic market conditions

## Key Trading Concepts

### 1. Price

**Price** is what something costs right now.

- Bitcoin price: $90,000 means 1 Bitcoin = $90,000
- If you have $10,000, you can buy 0.111 Bitcoin ($10,000 ÷ $90,000)

### 2. Volume

**Volume** is how much was traded.

Example:
- "Volume: 100 BTC" means 100 Bitcoins were traded
- High volume = Lots of people trading (usually means price is more stable)
- Low volume = Few people trading (price can jump around)

### 3. Buy vs Sell

- **Buy (Long)**: You think price will go UP
  - Buy at $90,000, sell at $95,000 → Profit $5,000
  
- **Sell (Short)**: You think price will go DOWN
  - Sell at $90,000, buy back at $85,000 → Profit $5,000
  - (In our simulator, we only do regular buying/selling, not advanced shorting)

### 4. Position

Your **position** is what you currently own.

Examples:
- "I have a position of 0.5 BTC" = You own 0.5 Bitcoin
- "I have no position" = You own nothing (all cash)
- "Long position" = You bought and haven't sold yet

### 5. Profit and Loss (PnL)

**PnL** is your total profit or loss.

Example:
- You start with $100,000
- You buy 1 BTC at $90,000 (you have $10,000 cash + 1 BTC)
- Bitcoin rises to $95,000 (your 1 BTC is now worth $95,000)
- Your total value: $10,000 cash + $95,000 BTC = $105,000
- Your PnL: $105,000 - $100,000 = **+$5,000** (profit!)

### 6. Commission/Fees

Exchanges charge **fees** for every trade.

Example:
- You buy $10,000 worth of Bitcoin
- Exchange charges 0.1% fee
- Fee: $10,000 × 0.001 = **$10**
- You actually pay: $10,010

**Important**: Fees add up! If you trade 100 times, that's $1,000 in fees.

## Market Data Terms

### OHLCV Data

This is the most common market data format. Each "bar" or "candle" shows:

- **O**pen: Price at the start of the period
- **H**igh: Highest price during the period
- **L**ow: Lowest price during the period
- **C**lose: Price at the end of the period
- **V**olume: Total amount traded

**Example (1-minute bar)**:
```
Time: 10:00-10:01
Open: $90,000
High: $90,200 (highest it reached)
Low: $89,900 (lowest it dropped)
Close: $90,100
Volume: 5.2 BTC traded
```

### Visualizing OHLCV

```
        $90,200 ← High
           |
   ┌───────┴───────┐
   │               │
$90,000 → Open     │  ← Close $90,100
   │               │
   └───────┬───────┘
           |
        $89,900 ← Low
```

## What is a Strategy?

A **strategy** is a set of rules that tell you when to buy and sell.

### Simple Strategy Example

**"Momentum Strategy"**:
```
IF price went up 2% in last hour:
    BUY (thinking it will keep going up)
    
IF price went down 2% after we bought:
    SELL (stop our losses)
    
IF price went up 5% after we bought:
    SELL (take our profit)
```

### Our Simulator's Strategy

Our strategy is more sophisticated and uses:
1. **Machine Learning** (AI) to predict if price will go up or down
2. **Momentum** checking if price is moving strongly
3. **Volume** checking if lots of people are trading
4. **Risk limits** to prevent losing too much money

## What is Backtesting?

**Backtesting** means testing your strategy on old data to see if it would have made money.

### The Process

```
Step 1: Get historical data (e.g., Bitcoin prices from last year)
Step 2: Run your strategy on this old data
Step 3: See if you would have made profit
Step 4: If profitable → Maybe it's a good strategy!
        If not profitable → Change strategy and try again
```

### Important Warning

Just because a strategy worked in the past doesn't mean it will work in the future!

**Analogy**: Like checking weather from last year to predict tomorrow - it helps, but isn't perfect.

## What is Machine Learning in Trading?

**Machine Learning (ML)** is teaching a computer to recognize patterns and make predictions.

### How It Works (Simple Version)

1. **Show the computer** 10,000 examples of Bitcoin prices
2. **Tell the computer** which times the price went up vs down
3. **Computer learns patterns** (e.g., "when volume is high and price is rising, it usually keeps rising")
4. **Computer predicts** what will happen next time it sees similar patterns

### Real-World Analogy

It's like teaching a child to recognize animals:
- Show them 100 pictures of cats
- Show them 100 pictures of dogs
- After enough examples, they can identify new animals they've never seen

### In Our Simulator

Our ML model:
- Looks at 90 different features (patterns in data)
- Predicts if Bitcoin price will go up, down, or stay flat
- Gets about 44% accuracy (better than random guessing!)

## Risk Management

**Risk Management** is protecting yourself from losing too much money.

### Key Rules

1. **Position Limit**: Never risk more than 30% of your money on one trade
   - If you have $100,000, max position = $30,000
   
2. **Stop Loss**: Auto-sell if you lose more than 2%
   - Buy at $90,000, stop loss at $88,200 (2% loss)
   
3. **Max Drawdown**: Stop trading if you lose more than 10% total
   - Started with $100,000
   - If value drops to $90,000 → STOP TRADING

### Why This Matters

**Without risk management**:
- One bad trade could lose 50% of your money
- Three bad trades could wipe you out

**With risk management**:
- Maximum loss per trade: 2%
- Would take 50 bad trades in a row to lose everything (very unlikely!)

## Performance Metrics

These numbers tell you if your strategy is good or bad.

### 1. Total Return

**How much money you made (or lost)**.

```
Return = (Final Value - Starting Value) / Starting Value × 100%

Example:
Started: $100,000
Ended: $105,000
Return: ($105,000 - $100,000) / $100,000 = 5%
```

### 2. Sharpe Ratio

**How much profit per unit of risk** (higher is better).

- Sharpe > 2.0 = Excellent
- Sharpe 1.0-2.0 = Good
- Sharpe < 1.0 = Needs improvement

**Simple explanation**: Would you rather:
- Make $10 with certainty? (High Sharpe)
- Make $10 but might lose $20? (Low Sharpe)

### 3. Maximum Drawdown

**Biggest loss from peak to bottom**.

Example:
- Your account grows: $100k → $110k → $120k
- Then drops: $120k → $105k
- Max drawdown: ($120k - $105k) / $120k = **12.5%**

Lower is better! (means smaller losses)

### 4. Win Rate

**Percentage of trades that made money**.

```
Win Rate = Winning Trades / Total Trades × 100%

Example:
100 trades total
65 were profitable
35 lost money
Win Rate = 65/100 = 65%
```

**Note**: High win rate doesn't always mean profitable!
- You could win 90% of trades (making $1 each)
- But lose 10% of trades (losing $20 each)
- Overall: Lost money despite 90% win rate!

## Common Terms Glossary

| Term | Simple Definition | Example |
|------|------------------|---------|
| **Asset** | What you're trading | Bitcoin, stocks, gold |
| **Bid** | Highest price someone will pay | "I'll buy at $89,999" |
| **Ask** | Lowest price someone will sell | "I'll sell at $90,001" |
| **Spread** | Difference between bid and ask | Ask $90,001 - Bid $89,999 = $2 |
| **Liquidity** | How easy to buy/sell without moving price | High = easy, Low = hard |
| **Volatility** | How much price jumps around | High = big swings, Low = stable |
| **Slippage** | Price difference between order and execution | Want $90,000, get $90,005 |
| **Leverage** | Borrowing money to trade bigger | 2x leverage = trade double your money |
| **Hedge** | Trade to reduce risk | Own Bitcoin, sell futures to protect |

## What This Simulator Does

Our HFT Simulator is a complete system that:

1. **Gets Real Data** - Fetches Bitcoin prices from Binance exchange
2. **Prepares Data** - Calculates technical indicators (moving averages, etc.)
3. **Creates Features** - Extracts 90 patterns for machine learning
4. **Trains AI Model** - Teaches computer to predict price movements
5. **Generates Signals** - Decides when to buy/sell based on AI + momentum
6. **Simulates Market** - Creates realistic order book and execution
7. **Manages Risk** - Enforces position limits and stop losses
8. **Executes Trades** - Buys and sells based on strategy
9. **Tracks Performance** - Records all trades and calculates metrics
10. **Visualizes Results** - Creates charts showing performance

## Why This Matters for Your Portfolio

This project demonstrates **multiple valuable skills**:

### 1. Technical Skills
- Python programming
- Machine learning (PyTorch)
- Data analysis (Pandas, NumPy)
- System design
- Testing and validation

### 2. Domain Knowledge
- Financial markets
- Quantitative trading
- Risk management
- Performance analysis

### 3. Software Engineering
- Modular architecture
- Event-driven design
- Performance optimization
- Production-grade code

### 4. Research Mindset
- Hypothesis testing
- Statistical validation
- Avoiding overfitting
- Documentation

## Next Steps

Now that you understand the basics, you can:

1. **[Understand How It Works](02_how_it_works.md)** - See the system overview
2. **[Learn About Data Flow](03_data_flow.md)** - Follow data through the system
3. **[Run Your First Backtest](../guides/01_getting_started.md)** - Hands-on tutorial

---

## Key Takeaways

✅ **Trading** = Buying low, selling high to make profit

✅ **Algorithmic trading** = Computer program trades automatically

✅ **HFT** = Very fast trading (microseconds)

✅ **Strategy** = Rules for when to buy/sell

✅ **Backtesting** = Testing strategy on historical data

✅ **ML** = Computer learns patterns to predict prices

✅ **Risk Management** = Protecting against big losses

✅ **Metrics** = Numbers that show if strategy is good

---

**Ready to dive deeper?** Continue to [How It Works](02_how_it_works.md) →