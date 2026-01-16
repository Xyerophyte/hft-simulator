"""
Backtesting framework for trading strategies.
Event-driven backtesting with historical data replay.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from dataclasses import dataclass
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from strategies.momentum_strategy import MomentumStrategy, Signal
from strategies.portfolio import Portfolio
from strategies.risk_manager import RiskManager, RiskLimits
from ml.models import PriceLSTM
import torch


@dataclass
class BacktestConfig:
    """Backtesting configuration."""
    initial_capital: float = 100000.0
    fee_rate: float = 0.001
    position_size_pct: float = 0.3
    use_risk_manager: bool = True
    use_ml_model: bool = False


class Backtester:
    """
    Event-driven backtester for trading strategies.
    """
    
    def __init__(
        self,
        strategy: MomentumStrategy,
        config: Optional[BacktestConfig] = None
    ):
        """
        Initialize backtester.
        
        Args:
            strategy: Trading strategy to backtest
            config: Backtesting configuration
        """
        self.strategy = strategy
        self.config = config or BacktestConfig()
        
        # Initialize components
        self.portfolio = Portfolio(
            initial_capital=self.config.initial_capital,
            fee_rate=self.config.fee_rate
        )
        
        if self.config.use_risk_manager:
            self.risk_manager = RiskManager(
                RiskLimits(max_position_pct=self.config.position_size_pct)
            )
        else:
            self.risk_manager = None
        
        self.ml_model = None
        self.trades_executed = 0
        
    def load_ml_model(self, model_path: str, input_size: int, device: str = 'cpu'):
        """Load trained ML model for predictions."""
        self.ml_model = PriceLSTM(input_size=input_size)
        self.ml_model.load_state_dict(torch.load(model_path, map_location=device))
        self.ml_model.eval()
        self.ml_model.to(device)
    
    def run(
        self,
        df: pd.DataFrame,
        symbol: str = 'BTC'
    ) -> Dict:
        """
        Run backtest on historical data.
        
        Args:
            df: DataFrame with OHLCV data and features
            symbol: Trading symbol
            
        Returns:
            Dictionary with backtest results
        """
        print(f"Starting backtest with {len(df)} bars...")
        print(f"Initial capital: ${self.config.initial_capital:,.2f}")
        print()
        
        # Generate ML predictions if model available
        ml_predictions = None
        if self.config.use_ml_model and self.ml_model is not None:
            print("Generating ML predictions...")
            ml_predictions = self._generate_ml_predictions(df)
        
        # Generate signals
        print("Generating trading signals...")
        signals = self.strategy.generate_signals_batch(df, ml_predictions)
        
        # Execute trades
        print("Executing trades...")
        for i, (idx, row) in enumerate(df.iterrows()):
            signal = signals[i]
            current_price = row['close']
            
            # Update portfolio prices
            if symbol in self.portfolio.positions:
                self.portfolio.update_prices({symbol: current_price}, idx)
            
            # Check risk limits if using risk manager
            if self.risk_manager:
                current_equity = self.portfolio.cash + sum(
                    pos.market_value for pos in self.portfolio.positions.values()
                )
                
                # Check drawdown
                if not self.risk_manager.check_drawdown(
                    current_equity, self.config.initial_capital
                ):
                    print(f"  Drawdown limit exceeded at {idx}")
                    continue
                
                # Check daily loss
                if not self.risk_manager.check_daily_loss(current_equity):
                    print(f"  Daily loss limit exceeded at {idx}")
                    continue
            
            # Execute signal
            if signal.signal != Signal.HOLD:
                self._execute_signal(signal, symbol, current_price, idx)
        
        # Final update
        final_price = df.iloc[-1]['close']
        self.portfolio.update_prices({symbol: final_price}, df.index[-1])
        
        # Get results
        results = self._generate_results(df)
        return results
    
    def _execute_signal(
        self,
        signal,
        symbol: str,
        price: float,
        timestamp: pd.Timestamp
    ):
        """Execute a trading signal."""
        current_equity = self.portfolio.cash + sum(
            pos.market_value for pos in self.portfolio.positions.values()
        )
        
        current_pos = self.portfolio.get_position(symbol)
        current_pos_value = current_pos.market_value if current_pos else 0.0
        
        # Calculate position size
        if signal.signal == Signal.BUY:
            # Calculate target position size
            target_value = current_equity * self.config.position_size_pct * signal.confidence
            
            # Apply risk manager if enabled
            if self.risk_manager:
                target_qty = target_value / price
                target_qty = self.risk_manager.calculate_position_size(
                    target_qty, current_equity, price
                )
                target_value = target_qty * price
            
            # Only buy if we don't already have a position or want to add
            if current_pos_value < target_value:
                quantity = (target_value - current_pos_value) / price
                
                # Check if we have enough cash
                cost = quantity * price * (1 + self.config.fee_rate)
                if cost <= self.portfolio.cash:
                    self.portfolio.execute_trade(symbol, quantity, price, timestamp)
                    self.trades_executed += 1
        
        elif signal.signal == Signal.SELL:
            # Close position if we have one
            if current_pos and current_pos.quantity > 0:
                self.portfolio.execute_trade(symbol, -current_pos.quantity, price, timestamp)
                self.trades_executed += 1
    
    def _generate_ml_predictions(self, df: pd.DataFrame) -> np.ndarray:
        """Generate ML predictions for the dataset."""
        # Placeholder - would need actual feature preparation
        return np.random.uniform(0.4, 0.6, len(df))
    
    def _generate_results(self, df: pd.DataFrame) -> Dict:
        """Generate backtest results."""
        equity_curve = self.portfolio.get_equity_curve()
        trades = self.portfolio.get_trade_history()
        summary = self.portfolio.get_summary()
        
        # Calculate returns
        if not equity_curve.empty:
            returns = equity_curve['equity'].pct_change().dropna()
        else:
            returns = pd.Series()
        
        results = {
            'summary': summary,
            'equity_curve': equity_curve,
            'trades': trades,
            'returns': returns,
            'num_signals': len(self.strategy.signals),
            'trades_executed': self.trades_executed,
            'signal_stats': self.strategy.get_signal_statistics()
        }
        
        return results


# Example usage
if __name__ == "__main__":
    from data.fetcher import BinanceDataFetcher
    from ml.features import FeatureEngineer
    
    print("Fetching data...")
    fetcher = BinanceDataFetcher("BTCUSDT")
    df = fetcher.fetch_klines(interval="1m", limit=1000)
    
    if not df.empty:
        print("Creating features...")
        engineer = FeatureEngineer()
        df_features = engineer.create_all_features(df)
        
        print("\nInitializing backtest...")
        strategy = MomentumStrategy(
            ml_threshold=0.55,
            momentum_threshold=0.0005,
            volume_threshold=1.2
        )
        
        config = BacktestConfig(
            initial_capital=100000,
            fee_rate=0.001,
            position_size_pct=0.3,
            use_risk_manager=True
        )
        
        backtester = Backtester(strategy, config)
        
        print("\nRunning backtest...")
        results = backtester.run(df_features, symbol='BTC')
        
        print("\n" + "="*50)
        print("BACKTEST RESULTS")
        print("="*50)
        
        summary = results['summary']
        print(f"\nPortfolio Performance:")
        print(f"  Initial Capital: ${summary['initial_capital']:,.2f}")
        print(f"  Final Value: ${summary['current_value']:,.2f}")
        print(f"  Total Return: {summary['total_return_pct']:.2f}%")
        print(f"  Total PnL: ${summary['total_pnl']:,.2f}")
        print(f"  Realized PnL: ${summary['realized_pnl']:,.2f}")
        print(f"  Unrealized PnL: ${summary['unrealized_pnl']:,.2f}")
        
        print(f"\nTrading Activity:")
        print(f"  Total Signals: {results['num_signals']}")
        print(f"  Trades Executed: {results['trades_executed']}")
        print(f"  Total Fees: ${summary['total_fees']:,.2f}")
        
        signal_stats = results['signal_stats']
        print(f"\nSignal Distribution:")
        print(f"  BUY: {signal_stats['buy_signals']} ({signal_stats['buy_pct']:.1f}%)")
        print(f"  SELL: {signal_stats['sell_signals']} ({signal_stats['sell_pct']:.1f}%)")
        print(f"  HOLD: {signal_stats['hold_signals']} ({signal_stats['hold_pct']:.1f}%)")
        
        if not results['trades'].empty:
            trades = results['trades']
            winning_trades = trades[trades['pnl'] > 0]
            losing_trades = trades[trades['pnl'] < 0]
            
            print(f"\nTrade Statistics:")
            print(f"  Winning Trades: {len(winning_trades)}")
            print(f"  Losing Trades: {len(losing_trades)}")
            if len(trades) > 0:
                win_rate = len(winning_trades) / len(trades) * 100
                print(f"  Win Rate: {win_rate:.1f}%")
            
            if len(winning_trades) > 0:
                print(f"  Avg Win: ${winning_trades['pnl'].mean():,.2f}")
            if len(losing_trades) > 0:
                print(f"  Avg Loss: ${losing_trades['pnl'].mean():,.2f}")