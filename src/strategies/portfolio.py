"""
Portfolio and position tracking with PnL calculation.
Tracks positions, trades, and calculates profit/loss metrics.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Position:
    """Represents a trading position."""
    symbol: str
    quantity: float  # Positive for long, negative for short
    entry_price: float
    entry_time: pd.Timestamp
    current_price: float = 0.0
    
    @property
    def market_value(self) -> float:
        """Current market value of position."""
        return self.quantity * self.current_price
    
    @property
    def cost_basis(self) -> float:
        """Original cost of position."""
        return abs(self.quantity) * self.entry_price
    
    @property
    def unrealized_pnl(self) -> float:
        """Unrealized profit/loss."""
        if self.quantity > 0:
            # Long position
            return self.quantity * (self.current_price - self.entry_price)
        else:
            # Short position
            return abs(self.quantity) * (self.entry_price - self.current_price)
    
    @property
    def unrealized_pnl_pct(self) -> float:
        """Unrealized PnL as percentage."""
        if self.cost_basis == 0:
            return 0.0
        return (self.unrealized_pnl / self.cost_basis) * 100


@dataclass
class Trade:
    """Represents a completed trade."""
    trade_id: int
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: float
    price: float
    timestamp: pd.Timestamp
    fee: float = 0.0
    pnl: float = 0.0
    
    @property
    def notional(self) -> float:
        """Trade notional value."""
        return self.quantity * self.price
    
    @property
    def total_cost(self) -> float:
        """Total cost including fees."""
        return self.notional + self.fee


@dataclass
class PortfolioState:
    """Snapshot of portfolio state at a point in time."""
    timestamp: pd.Timestamp
    cash: float
    positions_value: float
    total_value: float
    unrealized_pnl: float
    realized_pnl: float
    fees_paid: float
    num_positions: int
    
    @property
    def equity(self) -> float:
        """Total equity (cash + positions)."""
        return self.total_value


class Portfolio:
    """
    Portfolio manager tracking positions, trades, and PnL.
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        fee_rate: float = 0.001  # 0.1% per trade
    ):
        """
        Initialize portfolio.
        
        Args:
            initial_capital: Starting capital
            fee_rate: Trading fee as decimal
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.fee_rate = fee_rate
        
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.history: List[PortfolioState] = []
        
        self.realized_pnl = 0.0
        self.total_fees = 0.0
        self.trade_counter = 0
    
    def execute_trade(
        self,
        symbol: str,
        quantity: float,  # Positive for buy, negative for sell
        price: float,
        timestamp: pd.Timestamp
    ) -> Trade:
        """
        Execute a trade and update positions.
        
        Args:
            symbol: Trading symbol
            quantity: Quantity (positive=buy, negative=sell)
            price: Execution price
            timestamp: Trade timestamp
            
        Returns:
            Trade object
        """
        self.trade_counter += 1
        
        # Calculate fee
        notional = abs(quantity) * price
        fee = notional * self.fee_rate
        
        # Determine side
        side = 'BUY' if quantity > 0 else 'SELL'
        
        # Initialize PnL for this trade
        trade_pnl = 0.0
        
        # Update positions
        if symbol not in self.positions:
            # New position
            if quantity != 0:
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    entry_price=price,
                    entry_time=timestamp,
                    current_price=price
                )
            
        else:
            # Existing position
            pos = self.positions[symbol]
            
            # Check if closing or adding to position
            if (pos.quantity > 0 and quantity < 0) or (pos.quantity < 0 and quantity > 0):
                # Opposite direction - closing or reversing
                close_qty = min(abs(quantity), abs(pos.quantity))
                
                # Calculate PnL on closed portion
                if pos.quantity > 0:
                    # Closing long
                    trade_pnl = close_qty * (price - pos.entry_price)
                else:
                    # Closing short
                    trade_pnl = close_qty * (pos.entry_price - price)
                
                self.realized_pnl += trade_pnl
                
                # Update position
                new_qty = pos.quantity + quantity
                
                if abs(new_qty) < 1e-8:
                    # Position fully closed
                    del self.positions[symbol]
                else:
                    # Partial close or reversal
                    if np.sign(new_qty) == np.sign(pos.quantity):
                        # Partial close
                        pos.quantity = new_qty
                    else:
                        # Reversal - new position in opposite direction
                        self.positions[symbol] = Position(
                            symbol=symbol,
                            quantity=new_qty,
                            entry_price=price,
                            entry_time=timestamp,
                            current_price=price
                        )
            else:
                # Same direction - adding to position
                # Calculate new average entry price
                total_cost = (abs(pos.quantity) * pos.entry_price + 
                            abs(quantity) * price)
                total_qty = abs(pos.quantity) + abs(quantity)
                new_entry_price = total_cost / total_qty if total_qty > 0 else price
                
                pos.quantity += quantity
                pos.entry_price = new_entry_price
                pos.current_price = price
        
        # Update cash
        if side == 'BUY':
            self.cash -= (notional + fee)
        else:
            self.cash += (notional - fee)
        
        self.total_fees += fee
        
        # Create trade record
        trade = Trade(
            trade_id=self.trade_counter,
            symbol=symbol,
            side=side,
            quantity=abs(quantity),
            price=price,
            timestamp=timestamp,
            fee=fee,
            pnl=trade_pnl
        )
        
        self.trades.append(trade)
        
        return trade
    
    def update_prices(self, prices: Dict[str, float], timestamp: pd.Timestamp):
        """
        Update current prices for all positions.
        
        Args:
            prices: Dictionary of symbol -> price
            timestamp: Update timestamp
        """
        for symbol, pos in self.positions.items():
            if symbol in prices:
                pos.current_price = prices[symbol]
        
        # Record portfolio state
        self._record_state(timestamp)
    
    def _record_state(self, timestamp: pd.Timestamp):
        """Record current portfolio state."""
        positions_value = sum(pos.market_value for pos in self.positions.values())
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        total_value = self.cash + positions_value
        
        state = PortfolioState(
            timestamp=timestamp,
            cash=self.cash,
            positions_value=positions_value,
            total_value=total_value,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=self.realized_pnl,
            fees_paid=self.total_fees,
            num_positions=len(self.positions)
        )
        
        self.history.append(state)
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol."""
        return self.positions.get(symbol)
    
    def get_total_pnl(self) -> float:
        """Get total PnL (realized + unrealized)."""
        unrealized = sum(pos.unrealized_pnl for pos in self.positions.values())
        return self.realized_pnl + unrealized
    
    def get_total_return(self) -> float:
        """Get total return as percentage."""
        current_value = self.cash + sum(pos.market_value for pos in self.positions.values())
        return ((current_value - self.initial_capital) / self.initial_capital) * 100
    
    def get_equity_curve(self) -> pd.DataFrame:
        """
        Get equity curve from history.
        
        Returns:
            DataFrame with timestamp and equity values
        """
        if not self.history:
            return pd.DataFrame()
        
        data = {
            'timestamp': [s.timestamp for s in self.history],
            'equity': [s.equity for s in self.history],
            'cash': [s.cash for s in self.history],
            'positions_value': [s.positions_value for s in self.history],
            'unrealized_pnl': [s.unrealized_pnl for s in self.history],
            'realized_pnl': [s.realized_pnl for s in self.history]
        }
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df
    
    def get_trade_history(self) -> pd.DataFrame:
        """
        Get trade history as DataFrame.
        
        Returns:
            DataFrame with all trades
        """
        if not self.trades:
            return pd.DataFrame()
        
        data = {
            'trade_id': [t.trade_id for t in self.trades],
            'timestamp': [t.timestamp for t in self.trades],
            'symbol': [t.symbol for t in self.trades],
            'side': [t.side for t in self.trades],
            'quantity': [t.quantity for t in self.trades],
            'price': [t.price for t in self.trades],
            'notional': [t.notional for t in self.trades],
            'fee': [t.fee for t in self.trades],
            'pnl': [t.pnl for t in self.trades]
        }
        
        df = pd.DataFrame(data)
        df.set_index('trade_id', inplace=True)
        return df
    
    def get_summary(self) -> Dict:
        """
        Get portfolio summary statistics.
        
        Returns:
            Dictionary with summary metrics
        """
        positions_value = sum(pos.market_value for pos in self.positions.values())
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        total_value = self.cash + positions_value
        
        return {
            'initial_capital': self.initial_capital,
            'current_value': total_value,
            'cash': self.cash,
            'positions_value': positions_value,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': unrealized_pnl,
            'total_pnl': self.realized_pnl + unrealized_pnl,
            'total_return_pct': self.get_total_return(),
            'total_fees': self.total_fees,
            'num_trades': len(self.trades),
            'num_positions': len(self.positions)
        }


# Example usage
if __name__ == "__main__":
    # Create portfolio
    portfolio = Portfolio(initial_capital=100000, fee_rate=0.001)
    
    print("Initial Portfolio:")
    print(f"  Cash: ${portfolio.cash:,.2f}")
    print(f"  Total Value: ${portfolio.cash:,.2f}")
    print()
    
    # Simulate some trades
    timestamps = pd.date_range('2024-01-01', periods=5, freq='1h')
    
    # Buy BTC
    trade1 = portfolio.execute_trade('BTC', 1.0, 50000, timestamps[0])
    print(f"Trade 1: {trade1.side} {trade1.quantity} BTC @ ${trade1.price:,.2f}")
    print(f"  Fee: ${trade1.fee:.2f}")
    print(f"  Cash remaining: ${portfolio.cash:,.2f}")
    print()
    
    # Update price
    portfolio.update_prices({'BTC': 51000}, timestamps[1])
    pos = portfolio.get_position('BTC')
    print(f"After price update to $51,000:")
    print(f"  Unrealized PnL: ${pos.unrealized_pnl:,.2f} ({pos.unrealized_pnl_pct:.2f}%)")
    print()
    
    # Sell half
    trade2 = portfolio.execute_trade('BTC', -0.5, 52000, timestamps[2])
    print(f"Trade 2: {trade2.side} {trade2.quantity} BTC @ ${trade2.price:,.2f}")
    print(f"  PnL: ${trade2.pnl:,.2f}")
    print(f"  Cash: ${portfolio.cash:,.2f}")
    print()
    
    # Final update
    portfolio.update_prices({'BTC': 53000}, timestamps[3])
    
    # Get summary
    print("Final Portfolio Summary:")
    summary = portfolio.get_summary()
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:,.2f}")
        else:
            print(f"  {key}: {value}")