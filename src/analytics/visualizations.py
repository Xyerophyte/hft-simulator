"""
Visualization utilities for trading analysis.
Creates plots for equity curves, trades, and performance metrics.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Optional, Dict
import os


class TradingVisualizer:
    """
    Creates visualizations for trading analysis.
    """
    
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        """
        Initialize visualizer.
        
        Args:
            style: Matplotlib style to use
        """
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
    
    def plot_equity_curve(
        self,
        equity_curve: pd.DataFrame,
        title: str = "Equity Curve",
        save_path: Optional[str] = None
    ):
        """
        Plot equity curve over time.
        
        Args:
            equity_curve: DataFrame with equity values
            title: Plot title
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(equity_curve.index, equity_curve['equity'], 
               label='Equity', linewidth=2, color='#2E86AB')
        
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Equity ($)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved equity curve to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_drawdown(
        self,
        equity_curve: pd.DataFrame,
        title: str = "Drawdown",
        save_path: Optional[str] = None
    ):
        """
        Plot drawdown over time.
        
        Args:
            equity_curve: DataFrame with equity values
            title: Plot title
            save_path: Path to save figure
        """
        equity = equity_curve['equity']
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max * 100
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.fill_between(drawdown.index, drawdown, 0, 
                       alpha=0.3, color='red', label='Drawdown')
        ax.plot(drawdown.index, drawdown, color='darkred', linewidth=1)
        
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved drawdown plot to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_pnl_distribution(
        self,
        trades: pd.DataFrame,
        title: str = "PnL Distribution",
        save_path: Optional[str] = None
    ):
        """
        Plot distribution of trade PnL.
        
        Args:
            trades: DataFrame with trade information
            title: Plot title
            save_path: Path to save figure
        """
        if trades.empty or 'pnl' not in trades.columns:
            print("No trade data available for PnL distribution")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        ax1.hist(trades['pnl'], bins=30, alpha=0.7, color='#2E86AB', edgecolor='black')
        ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='Break-even')
        ax1.set_xlabel('PnL ($)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('PnL Distribution', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Cumulative PnL
        cumulative_pnl = trades['pnl'].cumsum()
        ax2.plot(range(len(cumulative_pnl)), cumulative_pnl, 
                linewidth=2, color='#2E86AB')
        ax2.axhline(0, color='red', linestyle='--', linewidth=1)
        ax2.set_xlabel('Trade Number', fontsize=12)
        ax2.set_ylabel('Cumulative PnL ($)', fontsize=12)
        ax2.set_title('Cumulative PnL', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved PnL distribution to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_returns_distribution(
        self,
        returns: pd.Series,
        title: str = "Returns Distribution",
        save_path: Optional[str] = None
    ):
        """
        Plot distribution of returns.
        
        Args:
            returns: Series of returns
            title: Plot title
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(returns, bins=50, alpha=0.7, color='#2E86AB', edgecolor='black')
        ax.axvline(returns.mean(), color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {returns.mean():.4f}')
        ax.axvline(returns.median(), color='green', linestyle='--', linewidth=2,
                  label=f'Median: {returns.median():.4f}')
        
        ax.set_xlabel('Returns', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved returns distribution to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_summary_dashboard(
        self,
        equity_curve: pd.DataFrame,
        trades: pd.DataFrame,
        metrics: Dict,
        save_path: Optional[str] = None
    ):
        """
        Create a summary dashboard with multiple plots.
        
        Args:
            equity_curve: DataFrame with equity values
            trades: DataFrame with trades
            metrics: Dictionary with performance metrics
            save_path: Path to save figure
        """
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # Equity curve
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(equity_curve.index, equity_curve['equity'], 
                linewidth=2, color='#2E86AB')
        ax1.set_title('Equity Curve', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Equity ($)')
        ax1.grid(True, alpha=0.3)
        
        # Drawdown
        equity = equity_curve['equity']
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max * 100
        
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.fill_between(drawdown.index, drawdown, 0, 
                        alpha=0.3, color='red')
        ax2.plot(drawdown.index, drawdown, color='darkred', linewidth=1)
        ax2.set_title('Drawdown', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)
        
        # PnL distribution
        if not trades.empty and 'pnl' in trades.columns:
            ax3 = fig.add_subplot(gs[1, 1])
            ax3.hist(trades['pnl'], bins=20, alpha=0.7, color='#2E86AB', edgecolor='black')
            ax3.axvline(0, color='red', linestyle='--', linewidth=2)
            ax3.set_title('PnL Distribution', fontsize=12, fontweight='bold')
            ax3.set_xlabel('PnL ($)')
            ax3.set_ylabel('Frequency')
            ax3.grid(True, alpha=0.3)
        
        # Metrics table
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')
        
        metrics_text = f"""
        Performance Metrics
        
        Return: {metrics.get('total_return_pct', 0):.2f}%    Sharpe: {metrics.get('sharpe_ratio', 0):.2f}    Max DD: {metrics.get('max_drawdown_pct', 0):.2f}%
        
        Trades: {metrics.get('total_trades', 0)}    Win Rate: {metrics.get('win_rate', 0):.1f}%    Profit Factor: {metrics.get('profit_factor', 0):.2f}
        """
        
        ax4.text(0.5, 0.5, metrics_text, 
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=11,
                family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.suptitle('Trading Strategy Performance Dashboard', 
                    fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved dashboard to {save_path}")
        else:
            plt.show()
        
        plt.close()


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=1000, freq='1min')
    
    # Equity curve
    returns = np.random.normal(0.0001, 0.01, 1000)
    equity = 100000 * (1 + returns).cumprod()
    
    equity_df = pd.DataFrame({
        'equity': equity,
        'cash': equity * 0.3,
        'positions_value': equity * 0.7
    }, index=dates)
    
    # Trades
    trades_df = pd.DataFrame({
        'pnl': np.random.normal(50, 200, 50)
    })
    
    # Metrics
    metrics = {
        'total_return_pct': 27.82,
        'sharpe_ratio': 2.15,
        'max_drawdown_pct': -15.3,
        'total_trades': 50,
        'win_rate': 60.0,
        'profit_factor': 1.85
    }
    
    # Create visualizer
    viz = TradingVisualizer()
    
    # Create output directory
    os.makedirs('plots', exist_ok=True)
    
    print("Generating visualizations...")
    
    # Individual plots
    viz.plot_equity_curve(equity_df, save_path='plots/equity_curve.png')
    viz.plot_drawdown(equity_df, save_path='plots/drawdown.png')
    viz.plot_pnl_distribution(trades_df, save_path='plots/pnl_distribution.png')
    
    # Dashboard
    viz.plot_summary_dashboard(equity_df, trades_df, metrics, 
                               save_path='plots/dashboard.png')
    
    print("\nAll visualizations saved to 'plots/' directory")