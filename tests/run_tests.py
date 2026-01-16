"""
Simple test runner for HFT simulator.
Tests basic functionality of core components.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import time
import pandas as pd
import numpy as np
from market.orderbook import OrderBook, Order, OrderSide, OrderType
from market.matching_engine import MatchingEngine
from strategies.portfolio import Portfolio
from strategies.risk_manager import RiskManager, RiskLimits


def test_orderbook():
    """Test basic orderbook functionality."""
    print("Testing OrderBook...")
    
    book = OrderBook()
    
    # Add orders
    bid = Order("b1", OrderSide.BUY, OrderType.LIMIT, 100.0, 10.0, time.time())
    ask = Order("a1", OrderSide.SELL, OrderType.LIMIT, 102.0, 5.0, time.time())
    
    book.add_order(bid)
    book.add_order(ask)
    
    assert book.get_best_bid() == 100.0, "Best bid incorrect"
    assert book.get_best_ask() == 102.0, "Best ask incorrect"
    assert book.get_spread() == 2.0, "Spread incorrect"
    
    print("  [PASS] OrderBook tests passed")


def test_matching_engine():
    """Test matching engine."""
    print("Testing MatchingEngine...")
    
    book = OrderBook()
    engine = MatchingEngine()
    
    # Add liquidity
    ask = Order("a1", OrderSide.SELL, OrderType.LIMIT, 100.0, 10.0, time.time())
    book.add_order(ask)
    
    # Market buy
    market_order = Order("m1", OrderSide.BUY, OrderType.MARKET, None, 5.0, time.time())
    fills = engine.match_market_order(market_order, book)
    
    assert len(fills) > 0, "No fills generated"
    assert sum(f.quantity for f in fills) == 5.0, "Fill quantity incorrect"
    
    print("  [PASS] MatchingEngine tests passed")


def test_portfolio():
    """Test portfolio management."""
    print("Testing Portfolio...")
    
    portfolio = Portfolio(initial_capital=100000.0)
    
    assert portfolio.cash == 100000.0, "Initial cash incorrect"
    
    # Execute trade
    timestamp = pd.Timestamp.now()
    trade = portfolio.execute_trade('BTC', 1.0, 50000.0, timestamp)
    
    assert trade.side == 'BUY', "Trade side incorrect"
    assert 'BTC' in portfolio.positions, "Position not created"
    assert portfolio.cash < 100000.0, "Cash not updated"
    
    # Update prices
    portfolio.update_prices({'BTC': 51000.0}, timestamp)
    pos = portfolio.get_position('BTC')
    
    assert pos.unrealized_pnl > 0, "Unrealized PnL incorrect"
    
    print("  [PASS] Portfolio tests passed")


def test_risk_manager():
    """Test risk management."""
    print("Testing RiskManager...")
    
    limits = RiskLimits(max_position_pct=0.3)
    risk_mgr = RiskManager(limits)
    
    # Test position limit
    within_limit = risk_mgr.check_position_limit(25000, 100000)
    assert within_limit == True, "Position limit check failed"
    
    over_limit = risk_mgr.check_position_limit(40000, 100000)
    assert over_limit == False, "Position limit check failed"
    
    # Test stop loss
    triggered = risk_mgr.check_stop_loss(50000, 48000, is_long=True)
    assert triggered == True, "Stop loss check failed"
    
    print("  [PASS] RiskManager tests passed")


def test_integration():
    """Test full integration."""
    print("Testing Integration...")
    
    # Create components
    portfolio = Portfolio(initial_capital=100000.0)
    risk_mgr = RiskManager(RiskLimits(max_position_pct=0.3))
    
    # Simulate trading
    timestamp = pd.Timestamp.now()
    
    # Buy
    trade1 = portfolio.execute_trade('BTC', 1.0, 50000.0, timestamp)
    assert trade1.side == 'BUY', "Buy trade failed"
    
    # Check risk - position is 50% of equity (50k position / 100k total equity)
    pos_value = 50000.0
    total_equity = 100000.0  # Cash was reduced but total equity stays same
    within_limit = risk_mgr.check_position_limit(pos_value, total_equity)
    assert within_limit == False, "Risk check should fail for 50% position (limit is 30%)"
    
    # Test with smaller position that should pass
    within_limit_small = risk_mgr.check_position_limit(25000.0, total_equity)
    assert within_limit_small == True, "Risk check should pass for 25% position"
    
    # Update price
    portfolio.update_prices({'BTC': 52000.0}, timestamp)
    
    # Sell
    trade2 = portfolio.execute_trade('BTC', -0.5, 52000.0, timestamp)
    assert trade2.side == 'SELL', "Sell trade failed"
    assert trade2.pnl > 0, "PnL calculation failed"
    
    print("  [PASS] Integration tests passed")


def main():
    """Run all tests."""
    print("\n" + "="*50)
    print("Running HFT Simulator Tests")
    print("="*50 + "\n")
    
    try:
        test_orderbook()
        test_matching_engine()
        test_portfolio()
        test_risk_manager()
        test_integration()
        
        print("\n" + "="*50)
        print("[SUCCESS] All tests passed!")
        print("="*50 + "\n")
        return 0
        
    except AssertionError as e:
        print(f"\n[FAIL] Test failed: {e}\n")
        return 1
    except Exception as e:
        print(f"\n[ERROR] {e}\n")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())