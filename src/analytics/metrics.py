"""
Performance metrics calculator for trading strategies.
Calculates Sharpe ratio, drawdown, win rate, and other metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


class PerformanceMetrics:
    """
    Calculates comprehensive performance metrics for trading strategies.
    """
    
    @staticmethod
    def calculate_returns(equity_curve: pd.Series) -> pd.Series:
        """Calculate returns from equity curve."""
        return equity_curve.pct_change().dropna()
    
    @staticmethod
    def calculate_sharpe_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 365 * 24 * 60  # 1-minute bars
    ) -> float:
        """
        Calculate annualized Sharpe ratio.
        
        Args:
            returns: Series of returns
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of periods in a year
            
        Returns:
            Sharpe ratio
        """
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - (risk_free_rate / periods_per_year)
        
        if excess_returns.std() == 0:
            return 0.0
        
        sharpe = np.sqrt(periods_per_year) * (excess_returns.mean() / excess_returns.std())
        return sharpe
    
    @staticmethod
    def calculate_sortino_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 365 * 24 * 60
    ) -> float:
        """
        Calculate annualized Sortino ratio (uses downside deviation).
        
        Args:
            returns: Series of returns
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of periods in a year
            
        Returns:
            Sortino ratio
        """
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - (risk_free_rate / periods_per_year)
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        
        sortino = np.sqrt(periods_per_year) * (excess_returns.mean() / downside_returns.std())
        return sortino
    
    @staticmethod
    def calculate_max_drawdown(equity_curve: pd.Series) -> Dict:
        """
        Calculate maximum drawdown and related metrics.
        
        Args:
            equity_curve: Series of equity values
            
        Returns:
            Dictionary with drawdown metrics
        """
        if len(equity_curve) < 2:
            return {
                'max_drawdown': 0.0,
                'max_drawdown_pct': 0.0,
                'drawdown_duration': 0,
                'current_drawdown_pct': 0.0
            }
        
        # Calculate running maximum
        running_max = equity_curve.expanding().max()
        
        # Calculate drawdown
        drawdown = equity_curve - running_max
        drawdown_pct = (drawdown / running_max) * 100
        
        # Max drawdown
        max_dd = drawdown.min()
        max_dd_pct = drawdown_pct.min()
        
        # Drawdown duration
        is_drawdown = drawdown < 0
        drawdown_periods = is_drawdown.astype(int).groupby(
            (~is_drawdown).cumsum()
        ).cumsum()
        max_dd_duration = drawdown_periods.max() if len(drawdown_periods) > 0 else 0
        
        # Current drawdown
        current_dd_pct = drawdown_pct.iloc[-1]
        
        return {
            'max_drawdown': max_dd,
            'max_drawdown_pct': max_dd_pct,
            'drawdown_duration': max_dd_duration,
            'current_drawdown_pct': current_dd_pct
        }
    
    @staticmethod
    def calculate_win_rate(trades: pd.DataFrame) -> Dict:
        """
        Calculate win rate and trade statistics.
        
        Args:
            trades: DataFrame with trade information
            
        Returns:
            Dictionary with win rate metrics
        """
        if trades.empty or 'pnl' not in trades.columns:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0,
                'profit_factor': 0.0
            }
        
        total_trades = len(trades)
        winning_trades = trades[trades['pnl'] > 0]
        losing_trades = trades[trades['pnl'] < 0]
        
        num_wins = len(winning_trades)
        num_losses = len(losing_trades)
        
        win_rate = (num_wins / total_trades * 100) if total_trades > 0 else 0.0
        
        avg_win = winning_trades['pnl'].mean() if num_wins > 0 else 0.0
        avg_loss = losing_trades['pnl'].mean() if num_losses > 0 else 0.0
        
        largest_win = winning_trades['pnl'].max() if num_wins > 0 else 0.0
        largest_loss = losing_trades['pnl'].min() if num_losses > 0 else 0.0
        
        total_wins = winning_trades['pnl'].sum() if num_wins > 0 else 0.0
        total_losses = abs(losing_trades['pnl'].sum()) if num_losses > 0 else 0.0
        
        profit_factor = (total_wins / total_losses) if total_losses > 0 else 0.0
        
        return {
            'total_trades': total_trades,
            'winning_trades': num_wins,
            'losing_trades': num_losses,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'profit_factor': profit_factor
        }
    
    @staticmethod
    def calculate_calmar_ratio(
        returns: pd.Series,
        equity_curve: pd.Series,
        periods_per_year: int = 365 * 24 * 60
    ) -> float:
        """
        Calculate Calmar ratio (annual return / max drawdown).
        
        Args:
            returns: Series of returns
            equity_curve: Series of equity values
            periods_per_year: Number of periods in a year
            
        Returns:
            Calmar ratio
        """
        if len(returns) < 2:
            return 0.0
        
        annual_return = returns.mean() * periods_per_year * 100
        dd_metrics = PerformanceMetrics.calculate_max_drawdown(equity_curve)
        max_dd_pct = abs(dd_metrics['max_drawdown_pct'])
        
        if max_dd_pct == 0:
            return 0.0
        
        return annual_return / max_dd_pct
    
    @staticmethod
    def calculate_all_metrics(
        equity_curve: pd.DataFrame,
        trades: pd.DataFrame,
        initial_capital: float,
        risk_free_rate: float = 0.02
    ) -> Dict:
        """
        Calculate all performance metrics.
        
        Args:
            equity_curve: DataFrame with equity over time
            trades: DataFrame with trade history
            initial_capital: Initial capital
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Dictionary with all metrics
        """
        if equity_curve.empty:
            return {}
        
        equity_series = equity_curve['equity']
        returns = PerformanceMetrics.calculate_returns(equity_series)
        
        # Return metrics
        total_return = ((equity_series.iloc[-1] - initial_capital) / initial_capital) * 100
        
        # Calculate annualized return
        num_periods = len(equity_series)
        periods_per_year = 365 * 24 * 60  # 1-minute bars
        years = num_periods / periods_per_year
        
        if years > 0:
            cagr = (((equity_series.iloc[-1] / initial_capital) ** (1 / years)) - 1) * 100
        else:
            cagr = 0.0
        
        # Risk metrics
        sharpe = PerformanceMetrics.calculate_sharpe_ratio(returns, risk_free_rate)
        sortino = PerformanceMetrics.calculate_sortino_ratio(returns, risk_free_rate)
        
        # Drawdown metrics
        dd_metrics = PerformanceMetrics.calculate_max_drawdown(equity_series)
        
        # Calmar ratio
        calmar = PerformanceMetrics.calculate_calmar_ratio(returns, equity_series)
        
        # Win rate metrics
        trade_metrics = PerformanceMetrics.calculate_win_rate(trades)
        
        # Volatility
        annual_vol = returns.std() * np.sqrt(periods_per_year) * 100
        
        # Combine all metrics
        metrics = {
            'total_return_pct': total_return,
            'cagr_pct': cagr,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'annual_volatility_pct': annual_vol,
            **dd_metrics,
            **trade_metrics
        }
        
        return metrics


# Example usage
if __name__ == "__main__":
    # Create sample equity curve
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=1000, freq='1min')
    
    # Simulate equity curve with some drift and volatility
    returns = np.random.normal(0.0001, 0.01, 1000)
    equity = 100000 * (1 + returns).cumprod()
    
    equity_df = pd.DataFrame({
        'timestamp': dates,
        'equity': equity
    })
    equity_df.set_index('timestamp', inplace=True)
    
    # Create sample trades
    trades_data = {
        'timestamp': dates[:20],
        'pnl': np.random.normal(50, 200, 20)
    }
    trades_df = pd.DataFrame(trades_data)
    
    # Calculate metrics
    metrics = PerformanceMetrics.calculate_all_metrics(
        equity_df,
        trades_df,
        initial_capital=100000,
        risk_free_rate=0.02
    )
    
    print("Performance Metrics:")
    print("=" * 50)
    print(f"\nReturn Metrics:")
    print(f"  Total Return: {metrics['total_return_pct']:.2f}%")
    print(f"  CAGR: {metrics['cagr_pct']:.2f}%")
    
    print(f"\nRisk-Adjusted Metrics:")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"  Sortino Ratio: {metrics['sortino_ratio']:.2f}")
    print(f"  Calmar Ratio: {metrics['calmar_ratio']:.2f}")
    print(f"  Annual Volatility: {metrics['annual_volatility_pct']:.2f}%")
    
    print(f"\nDrawdown Metrics:")
    print(f"  Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
    print(f"  Current Drawdown: {metrics['current_drawdown_pct']:.2f}%")
    print(f"  Max Drawdown Duration: {metrics['drawdown_duration']} periods")
    
    print(f"\nTrade Metrics:")
    print(f"  Total Trades: {metrics['total_trades']}")
    print(f"  Win Rate: {metrics['win_rate']:.2f}%")
    print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"  Avg Win: ${metrics['avg_win']:.2f}")
    print(f"  Avg Loss: ${metrics['avg_loss']:.2f}")
    print(f"  Largest Win: ${metrics['largest_win']:.2f}")
    print(f"  Largest Loss: ${metrics['largest_loss']:.2f}")