"""
Risk management module for trading strategies.
Implements position limits, stop-loss, and risk checks.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List
from dataclasses import dataclass
from enum import Enum


class RiskViolation(Enum):
    """Types of risk violations."""
    MAX_POSITION = "max_position_exceeded"
    MAX_DRAWDOWN = "max_drawdown_exceeded"
    MAX_LOSS = "max_loss_exceeded"
    VOLATILITY_LIMIT = "volatility_too_high"
    CORRELATION_LIMIT = "correlation_too_high"


@dataclass
class RiskLimits:
    """Risk limit configuration."""
    max_position_pct: float = 0.5  # Max 50% of capital per position
    max_total_exposure: float = 1.0  # Max 100% total exposure
    max_drawdown_pct: float = 0.15  # Max 15% drawdown
    max_daily_loss_pct: float = 0.05  # Max 5% daily loss
    stop_loss_pct: float = 0.02  # 2% stop loss per position
    volatility_limit: float = 0.05  # 5% volatility limit
    min_sharpe_ratio: float = 0.5  # Minimum Sharpe ratio


@dataclass
class RiskMetrics:
    """Current risk metrics."""
    total_exposure_pct: float
    max_position_pct: float
    current_drawdown_pct: float
    daily_pnl_pct: float
    portfolio_volatility: float
    var_95: float  # Value at Risk 95%
    num_violations: int
    violations: List[RiskViolation]


class RiskManager:
    """
    Manages risk controls for trading strategies.
    Enforces position limits, stop-losses, and risk checks.
    """
    
    def __init__(self, limits: Optional[RiskLimits] = None):
        """
        Initialize risk manager.
        
        Args:
            limits: Risk limits configuration
        """
        self.limits = limits or RiskLimits()
        self.violations: List[RiskViolation] = []
        self.peak_equity = 0.0
        self.daily_start_equity = 0.0
    
    def check_position_limit(
        self,
        position_value: float,
        total_equity: float
    ) -> bool:
        """
        Check if position size is within limits.
        
        Args:
            position_value: Value of the position
            total_equity: Total portfolio equity
            
        Returns:
            True if within limits, False otherwise
        """
        if total_equity <= 0:
            return False
        
        position_pct = abs(position_value) / total_equity
        
        if position_pct > self.limits.max_position_pct:
            self.violations.append(RiskViolation.MAX_POSITION)
            return False
        
        return True
    
    def check_total_exposure(
        self,
        positions_value: float,
        total_equity: float
    ) -> bool:
        """
        Check if total exposure is within limits.
        
        Args:
            positions_value: Total value of all positions
            total_equity: Total portfolio equity
            
        Returns:
            True if within limits, False otherwise
        """
        if total_equity <= 0:
            return False
        
        exposure_pct = abs(positions_value) / total_equity
        
        return exposure_pct <= self.limits.max_total_exposure
    
    def check_drawdown(
        self,
        current_equity: float,
        initial_equity: float
    ) -> bool:
        """
        Check if drawdown exceeds limit.
        
        Args:
            current_equity: Current portfolio equity
            initial_equity: Initial portfolio equity
            
        Returns:
            True if within limits, False otherwise
        """
        # Update peak
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        if self.peak_equity <= 0:
            return True
        
        drawdown_pct = (self.peak_equity - current_equity) / self.peak_equity
        
        if drawdown_pct > self.limits.max_drawdown_pct:
            self.violations.append(RiskViolation.MAX_DRAWDOWN)
            return False
        
        return True
    
    def check_daily_loss(
        self,
        current_equity: float,
        daily_start_equity: Optional[float] = None
    ) -> bool:
        """
        Check if daily loss exceeds limit.
        
        Args:
            current_equity: Current portfolio equity
            daily_start_equity: Equity at start of day
            
        Returns:
            True if within limits, False otherwise
        """
        if daily_start_equity is None:
            daily_start_equity = self.daily_start_equity
        
        if daily_start_equity is None or daily_start_equity <= 0:
            self.daily_start_equity = current_equity
            return True
        
        daily_pnl_pct = (current_equity - daily_start_equity) / daily_start_equity
        
        if daily_pnl_pct < -self.limits.max_daily_loss_pct:
            self.violations.append(RiskViolation.MAX_LOSS)
            return False
        
        return True
    
    def check_stop_loss(
        self,
        entry_price: float,
        current_price: float,
        is_long: bool
    ) -> bool:
        """
        Check if stop loss is triggered.
        
        Args:
            entry_price: Position entry price
            current_price: Current market price
            is_long: True if long position
            
        Returns:
            True if stop loss triggered, False otherwise
        """
        if entry_price <= 0:
            return False
        
        pnl_pct = (current_price - entry_price) / entry_price
        
        if not is_long:
            pnl_pct = -pnl_pct
        
        return pnl_pct < -self.limits.stop_loss_pct
    
    def check_volatility(
        self,
        returns: pd.Series,
        window: int = 20
    ) -> bool:
        """
        Check if volatility exceeds limit.
        
        Args:
            returns: Series of returns
            window: Rolling window for volatility
            
        Returns:
            True if within limits, False otherwise
        """
        if len(returns) < window:
            return True
        
        volatility = returns.rolling(window).std().iloc[-1]
        
        if volatility > self.limits.volatility_limit:
            self.violations.append(RiskViolation.VOLATILITY_LIMIT)
            return False
        
        return True
    
    def calculate_position_size(
        self,
        signal_size: float,
        current_equity: float,
        current_price: float,
        volatility: Optional[float] = None
    ) -> float:
        """
        Calculate risk-adjusted position size.
        
        Args:
            signal_size: Desired position size from strategy
            current_equity: Current portfolio equity
            current_price: Current asset price
            volatility: Asset volatility (optional)
            
        Returns:
            Risk-adjusted position size
        """
        if current_equity <= 0 or current_price <= 0:
            return 0.0
        
        # Calculate max position value
        max_value = current_equity * self.limits.max_position_pct
        max_quantity = max_value / current_price
        
        # Apply volatility scaling if provided
        if volatility is not None and volatility > 0:
            # Reduce size in high volatility
            vol_scalar = min(self.limits.volatility_limit / volatility, 1.0)
            max_quantity *= vol_scalar
        
        # Return minimum of signal size and max allowed
        return min(abs(signal_size), max_quantity) * np.sign(signal_size)
    
    def calculate_var(
        self,
        returns: pd.Series,
        confidence: float = 0.95,
        window: int = 252
    ) -> float:
        """
        Calculate Value at Risk.
        
        Args:
            returns: Series of returns
            confidence: Confidence level (default 95%)
            window: Lookback window
            
        Returns:
            VaR value
        """
        if len(returns) < 2:
            return 0.0
        
        recent_returns = returns.tail(window)
        return np.percentile(recent_returns, (1 - confidence) * 100)
    
    def get_risk_metrics(
        self,
        current_equity: float,
        initial_equity: float,
        positions_value: float,
        returns: Optional[pd.Series] = None
    ) -> RiskMetrics:
        """
        Calculate current risk metrics.
        
        Args:
            current_equity: Current portfolio equity
            initial_equity: Initial portfolio equity
            positions_value: Total value of positions
            returns: Historical returns series
            
        Returns:
            RiskMetrics object
        """
        # Calculate exposure
        total_exposure_pct = (abs(positions_value) / current_equity * 100 
                             if current_equity > 0 else 0)
        
        max_position_pct = self.limits.max_position_pct * 100
        
        # Calculate drawdown
        if self.peak_equity == 0:
            self.peak_equity = initial_equity
        
        current_drawdown_pct = ((self.peak_equity - current_equity) / self.peak_equity * 100
                               if self.peak_equity > 0 else 0)
        
        # Calculate daily PnL
        if self.daily_start_equity == 0:
            self.daily_start_equity = initial_equity
        
        daily_pnl_pct = ((current_equity - self.daily_start_equity) / self.daily_start_equity * 100
                        if self.daily_start_equity > 0 else 0)
        
        # Calculate volatility and VaR
        portfolio_volatility = 0.0
        var_95 = 0.0
        
        if returns is not None and len(returns) > 1:
            portfolio_volatility = returns.std() * np.sqrt(252) * 100  # Annualized
            var_95 = self.calculate_var(returns) * 100
        
        return RiskMetrics(
            total_exposure_pct=total_exposure_pct,
            max_position_pct=max_position_pct,
            current_drawdown_pct=current_drawdown_pct,
            daily_pnl_pct=daily_pnl_pct,
            portfolio_volatility=portfolio_volatility,
            var_95=var_95,
            num_violations=len(self.violations),
            violations=self.violations.copy()
        )
    
    def reset_daily(self, current_equity: float):
        """Reset daily tracking metrics."""
        self.daily_start_equity = current_equity
    
    def reset_violations(self):
        """Clear violation history."""
        self.violations.clear()


# Example usage
if __name__ == "__main__":
    # Create risk manager
    limits = RiskLimits(
        max_position_pct=0.3,
        max_drawdown_pct=0.1,
        stop_loss_pct=0.02
    )
    
    risk_mgr = RiskManager(limits)
    
    print("Risk Limits:")
    print(f"  Max Position: {limits.max_position_pct * 100}%")
    print(f"  Max Drawdown: {limits.max_drawdown_pct * 100}%")
    print(f"  Stop Loss: {limits.stop_loss_pct * 100}%")
    print()
    
    # Test position limit
    total_equity = 100000
    position_value = 40000
    
    print(f"Checking position of ${position_value:,} (${total_equity:,} equity)...")
    within_limit = risk_mgr.check_position_limit(position_value, total_equity)
    print(f"  Within limit: {within_limit}")
    print()
    
    # Test drawdown
    initial_equity = 100000
    current_equity = 92000
    
    print(f"Checking drawdown (${initial_equity:,} -> ${current_equity:,})...")
    within_limit = risk_mgr.check_drawdown(current_equity, initial_equity)
    print(f"  Within limit: {within_limit}")
    print()
    
    # Test stop loss
    entry_price = 50000
    current_price = 49500
    
    print(f"Checking stop loss (entry ${entry_price:,}, current ${current_price:,})...")
    stop_triggered = risk_mgr.check_stop_loss(entry_price, current_price, is_long=True)
    print(f"  Stop triggered: {stop_triggered}")
    print()
    
    # Calculate risk-adjusted size
    signal_size = 2.0
    adjusted_size = risk_mgr.calculate_position_size(
        signal_size, total_equity, current_price
    )
    print(f"Position sizing:")
    print(f"  Signal size: {signal_size}")
    print(f"  Risk-adjusted: {adjusted_size:.4f}")
    print()
    
    # Generate sample returns for metrics
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.001, 0.02, 100))
    
    # Get risk metrics
    metrics = risk_mgr.get_risk_metrics(
        current_equity=current_equity,
        initial_equity=initial_equity,
        positions_value=position_value,
        returns=returns
    )
    
    print("Risk Metrics:")
    print(f"  Total Exposure: {metrics.total_exposure_pct:.2f}%")
    print(f"  Current Drawdown: {metrics.current_drawdown_pct:.2f}%")
    print(f"  Daily PnL: {metrics.daily_pnl_pct:.2f}%")
    print(f"  Portfolio Volatility: {metrics.portfolio_volatility:.2f}%")
    print(f"  VaR (95%): {metrics.var_95:.2f}%")
    print(f"  Violations: {metrics.num_violations}")