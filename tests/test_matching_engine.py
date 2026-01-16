"""
Unit tests for MatchingEngine class.
"""

import unittest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from market.orderbook import Order, OrderSide, OrderType
from market.matching_engine import MatchingEngine, Fill


class TestMatchingEngine(unittest.TestCase):
    """Test MatchingEngine functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = MatchingEngine(maker_fee=0.001, taker_fee=0.002)
    
    def test_initialization(self):
        """Test engine initializes correctly."""
        self.assertEqual(self.engine.maker_fee, 0.001)
        self.assertEqual(self.engine.taker_fee, 0.002)
        self.assertIsNotNone(self.engine.orderbook)
    
    def test_market_buy_full_fill(self):
        """Test market buy order with full fill."""
        # Add ask liquidity
        self.engine.orderbook.add_order(
            Order("ask1", OrderSide.SELL, OrderType.LIMIT, 100.0, 10.0, 1000)
        )
        
        # Market buy
        fills = self.engine.match_market_order(OrderSide.BUY, 5.0, 2000)
        
        self.assertEqual(len(fills), 1)
        self.assertEqual(fills[0].quantity, 5.0)
        self.assertEqual(fills[0].price, 100.0)
        self.assertGreater(fills[0].fee, 0)
    
    def test_market_buy_partial_fill(self):
        """Test market buy with partial fill."""
        # Add limited ask liquidity
        self.engine.orderbook.add_order(
            Order("ask1", OrderSide.SELL, OrderType.LIMIT, 100.0, 3.0, 1000)
        )
        
        # Try to buy more than available
        fills = self.engine.match_market_order(OrderSide.BUY, 10.0, 2000)
        
        self.assertEqual(len(fills), 1)
        self.assertEqual(fills[0].quantity, 3.0)  # Only 3.0 available
    
    def test_market_sell_full_fill(self):
        """Test market sell order with full fill."""
        # Add bid liquidity
        self.engine.orderbook.add_order(
            Order("bid1", OrderSide.BUY, OrderType.LIMIT, 100.0, 10.0, 1000)
        )
        
        # Market sell
        fills = self.engine.match_market_order(OrderSide.SELL, 5.0, 2000)
        
        self.assertEqual(len(fills), 1)
        self.assertEqual(fills[0].quantity, 5.0)
        self.assertEqual(fills[0].price, 100.0)
    
    def test_market_order_multiple_levels(self):
        """Test market order matching across multiple price levels."""
        # Add liquidity at multiple levels
        self.engine.orderbook.add_order(
            Order("ask1", OrderSide.SELL, OrderType.LIMIT, 100.0, 3.0, 1000)
        )
        self.engine.orderbook.add_order(
            Order("ask2", OrderSide.SELL, OrderType.LIMIT, 101.0, 5.0, 1000)
        )
        
        # Market buy that sweeps both levels
        fills = self.engine.match_market_order(OrderSide.BUY, 7.0, 2000)
        
        self.assertEqual(len(fills), 2)
        self.assertEqual(fills[0].price, 100.0)
        self.assertEqual(fills[0].quantity, 3.0)
        self.assertEqual(fills[1].price, 101.0)
        self.assertEqual(fills[1].quantity, 4.0)
    
    def test_limit_buy_no_match(self):
        """Test limit buy order that doesn't match."""
        # Add ask at 102
        self.engine.orderbook.add_order(
            Order("ask1", OrderSide.SELL, OrderType.LIMIT, 102.0, 5.0, 1000)
        )
        
        # Limit buy at 100 (below ask)
        fills = self.engine.match_limit_order(
            OrderSide.BUY, 100.0, 3.0, 2000
        )
        
        self.assertEqual(len(fills), 0)
        # Order should be added to book
        self.assertIn(100.0, self.engine.orderbook.bids)
    
    def test_limit_buy_immediate_match(self):
        """Test limit buy that matches immediately."""
        # Add ask at 100
        self.engine.orderbook.add_order(
            Order("ask1", OrderSide.SELL, OrderType.LIMIT, 100.0, 5.0, 1000)
        )
        
        # Limit buy at 101 (crosses spread)
        fills = self.engine.match_limit_order(
            OrderSide.BUY, 101.0, 3.0, 2000
        )
        
        self.assertEqual(len(fills), 1)
        self.assertEqual(fills[0].quantity, 3.0)
        self.assertEqual(fills[0].price, 100.0)  # Gets filled at ask price
    
    def test_limit_sell_no_match(self):
        """Test limit sell order that doesn't match."""
        # Add bid at 98
        self.engine.orderbook.add_order(
            Order("bid1", OrderSide.BUY, OrderType.LIMIT, 98.0, 5.0, 1000)
        )
        
        # Limit sell at 100 (above bid)
        fills = self.engine.match_limit_order(
            OrderSide.SELL, 100.0, 3.0, 2000
        )
        
        self.assertEqual(len(fills), 0)
        self.assertIn(100.0, self.engine.orderbook.asks)
    
    def test_limit_sell_immediate_match(self):
        """Test limit sell that matches immediately."""
        # Add bid at 100
        self.engine.orderbook.add_order(
            Order("bid1", OrderSide.BUY, OrderType.LIMIT, 100.0, 5.0, 1000)
        )
        
        # Limit sell at 99 (crosses spread)
        fills = self.engine.match_limit_order(
            OrderSide.SELL, 99.0, 3.0, 2000
        )
        
        self.assertEqual(len(fills), 1)
        self.assertEqual(fills[0].quantity, 3.0)
        self.assertEqual(fills[0].price, 100.0)  # Gets filled at bid price
    
    def test_fee_calculation(self):
        """Test that fees are calculated correctly."""
        # Add liquidity
        self.engine.orderbook.add_order(
            Order("ask1", OrderSide.SELL, OrderType.LIMIT, 100.0, 10.0, 1000)
        )
        
        # Market order (taker)
        fills = self.engine.match_market_order(OrderSide.BUY, 5.0, 2000)
        
        expected_fee = 5.0 * 100.0 * 0.002
        self.assertAlmostEqual(fills[0].fee, expected_fee, places=4)
    
    def test_slippage_calculation(self):
        """Test slippage calculation."""
        # Add liquidity at multiple levels
        self.engine.orderbook.add_order(
            Order("ask1", OrderSide.SELL, OrderType.LIMIT, 100.0, 2.0, 1000)
        )
        self.engine.orderbook.add_order(
            Order("ask2", OrderSide.SELL, OrderType.LIMIT, 101.0, 5.0, 1000)
        )
        
        # Market buy that crosses levels
        fills = self.engine.match_market_order(OrderSide.BUY, 5.0, 2000)
        
        # Calculate average fill price
        total_value = sum(f.quantity * f.price for f in fills)
        total_qty = sum(f.quantity for f in fills)
        avg_price = total_value / total_qty
        
        # Slippage should be positive (worse than best price)
        best_price = 100.0
        slippage = (avg_price - best_price) / best_price
        self.assertGreater(slippage, 0)
    
    def test_empty_orderbook_market_order(self):
        """Test market order on empty orderbook."""
        fills = self.engine.match_market_order(OrderSide.BUY, 5.0, 1000)
        self.assertEqual(len(fills), 0)
    
    def test_price_time_priority_enforcement(self):
        """Test that price-time priority is enforced."""
        # Add orders at same price with different timestamps
        self.engine.orderbook.add_order(
            Order("ask1", OrderSide.SELL, OrderType.LIMIT, 100.0, 3.0, 1000)
        )
        self.engine.orderbook.add_order(
            Order("ask2", OrderSide.SELL, OrderType.LIMIT, 100.0, 5.0, 2000)
        )
        
        # Market buy that partially fills
        fills = self.engine.match_market_order(OrderSide.BUY, 4.0, 3000)
        
        # Should fill ask1 completely first (time priority)
        self.assertEqual(len(fills), 2)
        self.assertEqual(fills[0].quantity, 3.0)  # ask1 filled completely
        self.assertEqual(fills[1].quantity, 1.0)  # ask2 partially filled


class TestFill(unittest.TestCase):
    """Test Fill class."""
    
    def test_fill_creation(self):
        """Test creating a fill."""
        fill = Fill(
            order_id="test1",
            side=OrderSide.BUY,
            price=100.0,
            quantity=5.0,
            fee=1.0,
            timestamp=1000
        )
        
        self.assertEqual(fill.order_id, "test1")
        self.assertEqual(fill.side, OrderSide.BUY)
        self.assertEqual(fill.price, 100.0)
        self.assertEqual(fill.quantity, 5.0)
        self.assertEqual(fill.fee, 1.0)
        self.assertEqual(fill.timestamp, 1000)
    
    def test_fill_value(self):
        """Test fill value calculation."""
        fill = Fill("test1", OrderSide.BUY, 100.0, 5.0, 1.0, 1000)
        
        expected_value = 5.0 * 100.0
        self.assertEqual(fill.value, expected_value)


if __name__ == '__main__':
    unittest.main()