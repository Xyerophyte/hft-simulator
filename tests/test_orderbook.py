"""
Unit tests for OrderBook class.
"""

import unittest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from market.orderbook import OrderBook, Order, OrderSide, OrderType


class TestOrderBook(unittest.TestCase):
    """Test OrderBook functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.orderbook = OrderBook()
    
    def test_initialization(self):
        """Test orderbook initializes correctly."""
        self.assertEqual(len(self.orderbook.bids), 0)
        self.assertEqual(len(self.orderbook.asks), 0)
        self.assertIsNone(self.orderbook.best_bid)
        self.assertIsNone(self.orderbook.best_ask)
    
    def test_add_bid_order(self):
        """Test adding bid orders."""
        order = Order(
            order_id="bid1",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            price=100.0,
            quantity=10.0,
            timestamp=1000
        )
        
        self.orderbook.add_order(order)
        
        self.assertEqual(len(self.orderbook.bids), 1)
        self.assertEqual(self.orderbook.best_bid, 100.0)
        self.assertIn(100.0, self.orderbook.bids)
    
    def test_add_ask_order(self):
        """Test adding ask orders."""
        order = Order(
            order_id="ask1",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            price=102.0,
            quantity=5.0,
            timestamp=1000
        )
        
        self.orderbook.add_order(order)
        
        self.assertEqual(len(self.orderbook.asks), 1)
        self.assertEqual(self.orderbook.best_ask, 102.0)
        self.assertIn(102.0, self.orderbook.asks)
    
    def test_price_time_priority(self):
        """Test that orders at same price maintain time priority."""
        order1 = Order("o1", OrderSide.BUY, OrderType.LIMIT, 100.0, 5.0, 1000)
        order2 = Order("o2", OrderSide.BUY, OrderType.LIMIT, 100.0, 3.0, 2000)
        
        self.orderbook.add_order(order1)
        self.orderbook.add_order(order2)
        
        orders_at_price = self.orderbook.bids[100.0]
        self.assertEqual(len(orders_at_price), 2)
        self.assertEqual(orders_at_price[0].order_id, "o1")
        self.assertEqual(orders_at_price[1].order_id, "o2")
    
    def test_best_bid_ask(self):
        """Test best bid/ask calculation."""
        self.orderbook.add_order(Order("b1", OrderSide.BUY, OrderType.LIMIT, 100.0, 5.0, 1000))
        self.orderbook.add_order(Order("b2", OrderSide.BUY, OrderType.LIMIT, 99.0, 3.0, 1000))
        self.orderbook.add_order(Order("a1", OrderSide.SELL, OrderType.LIMIT, 102.0, 5.0, 1000))
        self.orderbook.add_order(Order("a2", OrderSide.SELL, OrderType.LIMIT, 103.0, 3.0, 1000))
        
        self.assertEqual(self.orderbook.best_bid, 100.0)
        self.assertEqual(self.orderbook.best_ask, 102.0)
    
    def test_spread_calculation(self):
        """Test spread calculation."""
        self.orderbook.add_order(Order("b1", OrderSide.BUY, OrderType.LIMIT, 100.0, 5.0, 1000))
        self.orderbook.add_order(Order("a1", OrderSide.SELL, OrderType.LIMIT, 102.0, 5.0, 1000))
        
        spread = self.orderbook.get_spread()
        self.assertEqual(spread, 2.0)
    
    def test_spread_bps(self):
        """Test spread in basis points."""
        self.orderbook.add_order(Order("b1", OrderSide.BUY, OrderType.LIMIT, 100.0, 5.0, 1000))
        self.orderbook.add_order(Order("a1", OrderSide.SELL, OrderType.LIMIT, 101.0, 5.0, 1000))
        
        spread_bps = self.orderbook.get_spread_bps()
        expected_bps = (1.0 / 100.5) * 10000
        self.assertAlmostEqual(spread_bps, expected_bps, places=2)
    
    def test_cancel_order(self):
        """Test order cancellation."""
        order = Order("b1", OrderSide.BUY, OrderType.LIMIT, 100.0, 5.0, 1000)
        self.orderbook.add_order(order)
        
        self.assertTrue(self.orderbook.cancel_order("b1"))
        self.assertEqual(len(self.orderbook.bids.get(100.0, [])), 0)
    
    def test_cancel_nonexistent_order(self):
        """Test canceling non-existent order."""
        self.assertFalse(self.orderbook.cancel_order("nonexistent"))
    
    def test_depth_calculation(self):
        """Test depth calculation."""
        self.orderbook.add_order(Order("b1", OrderSide.BUY, OrderType.LIMIT, 100.0, 5.0, 1000))
        self.orderbook.add_order(Order("b2", OrderSide.BUY, OrderType.LIMIT, 100.0, 3.0, 1000))
        self.orderbook.add_order(Order("b3", OrderSide.BUY, OrderType.LIMIT, 99.0, 7.0, 1000))
        
        bid_depth, ask_depth = self.orderbook.get_depth(levels=2)
        
        self.assertEqual(len(bid_depth), 2)
        self.assertEqual(bid_depth[0], (100.0, 8.0))  # 5 + 3
        self.assertEqual(bid_depth[1], (99.0, 7.0))
    
    def test_imbalance_calculation(self):
        """Test order book imbalance."""
        self.orderbook.add_order(Order("b1", OrderSide.BUY, OrderType.LIMIT, 100.0, 10.0, 1000))
        self.orderbook.add_order(Order("a1", OrderSide.SELL, OrderType.LIMIT, 102.0, 5.0, 1000))
        
        imbalance = self.orderbook.get_imbalance()
        expected = (10.0 - 5.0) / (10.0 + 5.0)
        self.assertAlmostEqual(imbalance, expected, places=4)


class TestOrder(unittest.TestCase):
    """Test Order class."""
    
    def test_order_creation(self):
        """Test creating an order."""
        order = Order(
            order_id="test1",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            price=100.0,
            quantity=5.0,
            timestamp=1000
        )
        
        self.assertEqual(order.order_id, "test1")
        self.assertEqual(order.side, OrderSide.BUY)
        self.assertEqual(order.order_type, OrderType.LIMIT)
        self.assertEqual(order.price, 100.0)
        self.assertEqual(order.quantity, 5.0)
        self.assertEqual(order.filled_quantity, 0.0)
        self.assertEqual(order.timestamp, 1000)
    
    def test_partial_fill(self):
        """Test partial order fill."""
        order = Order("test1", OrderSide.BUY, OrderType.LIMIT, 100.0, 10.0, 1000)
        
        filled = order.fill(3.0)
        
        self.assertEqual(filled, 3.0)
        self.assertEqual(order.filled_quantity, 3.0)
        self.assertEqual(order.remaining_quantity, 7.0)
        self.assertFalse(order.is_filled())
    
    def test_complete_fill(self):
        """Test complete order fill."""
        order = Order("test1", OrderSide.BUY, OrderType.LIMIT, 100.0, 10.0, 1000)
        
        filled = order.fill(10.0)
        
        self.assertEqual(filled, 10.0)
        self.assertEqual(order.filled_quantity, 10.0)
        self.assertEqual(order.remaining_quantity, 0.0)
        self.assertTrue(order.is_filled())
    
    def test_overfill_protection(self):
        """Test that order cannot be overfilled."""
        order = Order("test1", OrderSide.BUY, OrderType.LIMIT, 100.0, 10.0, 1000)
        
        filled = order.fill(15.0)
        
        self.assertEqual(filled, 10.0)
        self.assertEqual(order.filled_quantity, 10.0)
        self.assertTrue(order.is_filled())


if __name__ == '__main__':
    unittest.main()