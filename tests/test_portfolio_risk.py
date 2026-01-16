"""
Unit tests for Portfolio and RiskManager classes.
"""

import unittest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from strategies.portfolio import Portfolio, Position
from strategies.risk_manager import RiskManager, RiskViolation


class TestPortfolio(unittest.TestCase):
    """Test Portfolio functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.portfolio = Portfolio(initial_cash=100000.0)
    
    def test_initialization(self):
        """Test portfolio initializes correctly."""
        self.assertEqual(self.portfolio.cash, 100000.0)
        self.assertEqual(self.portfolio.initial_cash, 100000.0)
        self.assertEqual(len(self.portfolio.positions), 0)
        self.assertEqual(len(self.portfolio.trades), 0)
    
    def test_open_long_position(self):
        """Test opening a long position."""
        self.portfolio.update_position('BTC', 1.0, 50000.0, 100.0)
        
        self.assertEqual(len(self.portfolio.positions), 1)
        self.assertIn('BTC', self.portfolio.positions)
        
        pos = self.portfolio.positions['BTC']
        self.assertEqual(pos.quantity, 1.0)
        self.assertEqual(pos.avg_entry_price, 50000.0)
        self.assertEqual(pos.fees_paid, 100.0)
    
    def test_open_short_position(self):
        """Test opening a short position."""
        self.portfolio.update_position('BTC', -0.5, 50000.0, 50.0)
        
        pos = self.portfolio.positions['BTC']
        self.assertEqual(pos.quantity, -0.5)
        self.assertTrue(pos.is_short())
    
    def test_increase_long_position(self):
        """Test increasing a long position."""
        self.portfolio.update_position('BTC', 1.0, 50000.0, 100.0)
        self.portfolio.update_position('BTC', 0.5, 51000.0, 51.0)
        
        pos = self.portfolio.positions['BTC']
        self.assertEqual(pos.quantity, 1.5)
        # Average entry price should be weighted
        expected_avg = (1.0 * 50000.0 + 0.5 * 51000.0) / 1.5
        self.assertAlmostEqual(pos.avg_entry_price, expected_avg, places=2)
    
    def test_close_position(self):
        """Test closing a position."""
        self.portfolio.update_position('BTC', 1.0, 50000.0, 100.0)
        self.portfolio.update_position('BTC', -1.0, 51000.0, 102.0)
        
        # Position should be closed
        self.assertNotIn('BTC', self.portfolio.positions)
        
        # Should have a trade recorded
        self.assertEqual(len(self.portfolio.trades), 1)
        trade = self.portfolio.trades[0]
        self.assertGreater(trade['pnl'], 0)  # Profitable trade
    
    def test_unrealized_pnl(self):
        """Test unrealized PnL calculation."""
        self.portfolio.update_position('BTC', 1.0, 50000.0, 100.0)
        
        unrealized = self.portfolio.get_unrealized_pnl({'BTC': 51000.0})
        # 1 BTC * (51000 - 50000) = 1000
        self.assertAlmostEqual(unrealized, 1000.0, places=2)
    
    def test_realized_pnl(self):
        """Test realized PnL calculation."""
        self.portfolio.update_position('BTC', 1.0, 50000.0, 100.0)
        self.portfolio.update_position('BTC', -1.0, 51000.0, 102.0)
        
        realized = self.portfolio.get_realized_pnl()
        # Profit = 1000, Fees = 100 + 102 = 202
        expected = 1000.0 - 202.0
        self.assertAlmostEqual(realized, expected, places=2)
    
    def test_total_equity(self):
        """Test total equity calculation."""
        self.portfolio.update_position('BTC', 1.0, 50000.0, 100.0)
        
        equity = self.portfolio.get_total_equity({'BTC': 51000.0})
        
        # Initial 100000, paid 50100 for BTC, now worth 51000
        # Cash = 100000 - 50000 - 100 = 49900
        # Position value = 51000
        # Total = 49900 + 51000 = 100900
        expected = 100000.0 + 1000.0 - 100.0
        self.assertAlmostEqual(equity, expected, places=2)
    
    def test_position_value(self):
        """Test position value calculation."""
        self.portfolio.update_position('BTC', 2.0, 50000.0, 100.0)
        
        value = self.portfolio.get_position_value('BTC', 51000.0)
        self.assertEqual(value, 2.0 * 51000.0)


class TestPosition(unittest.TestCase):
    """Test Position class."""
    
    def test_long_position(self):
        """Test long position."""
        pos = Position('BTC', 1.0, 50000.0)
        
        self.assertTrue(pos.is_long())
        self.assertFalse(pos.is_short())
        self.assertFalse(pos.is_flat())
    
    def test_short_position(self):
        """Test short position."""
        pos = Position('BTC', -1.0, 50000.0)
        
        self.assertFalse(pos.is_long())
        self.assertTrue(pos.is_short())
        self.assertFalse(pos.is_flat())
    
    def test_unrealized_pnl_long(self):
        """Test unrealized PnL for long position."""
        pos = Position('BTC', 1.0, 50000.0)
        
        pnl = pos.get_unrealized_pnl(51000.0)
        self.assertEqual(pnl, 1000.0)
    
    def test_unrealized_pnl_short(self):
        """Test unrealized PnL for short position."""
        pos = Position('BTC', -1.0, 50000.0)
        
        pnl = pos.get_unrealized_pnl(49000.0)
        # Short: profit when price goes down
        self.assertEqual(pnl, 1000.0)


class TestRiskManager(unittest.TestCase):
    """Test RiskManager functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.risk_mgr = RiskManager(
            max_position_size=0.3,
            max_drawdown=0.15,
            stop_loss_pct=0.05
        )
    
    def test_initialization(self):
        """Test risk manager initializes correctly."""
        self.assertEqual(self.risk_mgr.max_position_size, 0.3)
        self.assertEqual(self.risk_mgr.max_drawdown, 0.15)
        self.assertEqual(self.risk_mgr.stop_loss_pct, 0.05)
    
    def test_position_size_check_pass(self):
        """Test position size check passes."""
        portfolio_value = 100000.0
        position_value = 25000.0  # 25% of portfolio
        
        violation = self.risk_mgr.check_position_size(position_value, portfolio_value)
        self.assertIsNone(violation)
    
    def test_position_size_check_fail(self):
        """Test position size check fails."""
        portfolio_value = 100000.0
        position_value = 40000.0  # 40% of portfolio (exceeds 30% limit)
        
        violation = self.risk_mgr.check_position_size(position_value, portfolio_value)
        self.assertIsNotNone(violation)
        self.assertEqual(violation.violation_type, 'position_size')
    
    def test_drawdown_check_pass(self):
        """Test drawdown check passes."""
        equity_curve = [100000, 102000, 98000, 101000]
        
        violation = self.risk_mgr.check_drawdown(equity_curve)
        self.assertIsNone(violation)
    
    def test_drawdown_check_fail(self):
        """Test drawdown check fails."""
        equity_curve = [100000, 102000, 84000]  # 16% drawdown
        
        violation = self.risk_mgr.check_drawdown(equity_curve)
        self.assertIsNotNone(violation)
        self.assertEqual(violation.violation_type, 'drawdown')
    
    def test_stop_loss_check_pass(self):
        """Test stop loss check passes."""
        entry_price = 50000.0
        current_price = 49000.0  # 2% loss
        
        violation = self.risk_mgr.check_stop_loss(
            entry_price, current_price, is_long=True
        )
        self.assertIsNone(violation)
    
    def test_stop_loss_check_fail_long(self):
        """Test stop loss triggers for long position."""
        entry_price = 50000.0
        current_price = 47000.0  # 6% loss (exceeds 5% stop)
        
        violation = self.risk_mgr.check_stop_loss(
            entry_price, current_price, is_long=True
        )
        self.assertIsNotNone(violation)
        self.assertEqual(violation.violation_type, 'stop_loss')
    
    def test_stop_loss_check_fail_short(self):
        """Test stop loss triggers for short position."""
        entry_price = 50000.0
        current_price = 53000.0  # 6% loss on short (price went up)
        
        violation = self.risk_mgr.check_stop_loss(
            entry_price, current_price, is_long=False
        )
        self.assertIsNotNone(violation)
        self.assertEqual(violation.violation_type, 'stop_loss')
    
    def test_volatility_check(self):
        """Test volatility scaling."""
        returns = [0.01, -0.02, 0.015, -0.01, 0.02] * 20  # Some returns
        
        vol_adjusted_size = self.risk_mgr.calculate_volatility_adjusted_size(
            returns, base_size=1.0, target_vol=0.02
        )
        
        self.assertIsNotNone(vol_adjusted_size)
        self.assertGreater(vol_adjusted_size, 0)


class TestRiskViolation(unittest.TestCase):
    """Test RiskViolation class."""
    
    def test_violation_creation(self):
        """Test creating a risk violation."""
        violation = RiskViolation(
            violation_type='position_size',
            severity='high',
            message='Position exceeds limit',
            value=0.4
        )
        
        self.assertEqual(violation.violation_type, 'position_size')
        self.assertEqual(violation.severity, 'high')
        self.assertEqual(violation.message, 'Position exceeds limit')
        self.assertEqual(violation.value, 0.4)


if __name__ == '__main__':
    unittest.main()