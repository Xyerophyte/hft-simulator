#!/usr/bin/env python3
"""
HFT Simulator - Main Entry Point

Run true HFT simulation with:
- Tick-by-tick data processing
- Order book simulation
- Market making strategy
- Latency modeling

Usage:
    python run_hft.py              # Quick demo
    python run_hft.py --ticks 5000 # Custom tick count
    python run_hft.py --strategy stat_arb  # Different strategy
"""

import sys
import argparse
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import numpy as np


def main():
    parser = argparse.ArgumentParser(description='HFT Simulator')
    parser.add_argument('--ticks', type=int, default=1000,
                       help='Number of ticks to simulate')
    parser.add_argument('--strategy', type=str, default='market_maker',
                       choices=['market_maker', 'stat_arb', 'latency_arb'],
                       help='Strategy to use')
    parser.add_argument('--price', type=float, default=50000,
                       help='Initial price')
    parser.add_argument('--spread', type=float, default=2.0,
                       help='Initial spread in bps')
    parser.add_argument('--volatility', type=float, default=0.001,
                       help='Price volatility')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("                    HIGH-FREQUENCY TRADING SIMULATOR")
    print("=" * 70)
    
    from hft.tick_data import Tick, TickStream, TickType
    from hft.order_book import OrderBook, Side
    from hft.matching_engine import MatchingEngine
    from hft.latency import LatencyModel, LatencyProfile
    from hft.simulator import HFTSimulator, SimulationConfig
    from hft.strategies.market_maker import SimpleMarketMaker
    
    # Generate tick data
    print(f"\n📊 Generating {args.ticks} synthetic ticks...")
    
    np.random.seed(args.seed)
    
    # Realistic price path
    returns = np.random.normal(0, args.volatility, args.ticks)
    prices = args.price * np.exp(np.cumsum(returns))
    
    # Volume with fat tails
    sizes = np.abs(np.random.exponential(1.0, args.ticks))
    
    # Trade direction with momentum tendency
    momentum = np.sign(np.diff(prices, prepend=prices[0]))
    noise = np.random.uniform(-0.3, 0.3, args.ticks)
    sides = np.where(momentum + noise > 0, 'buy', 'sell')
    
    # Create tick stream
    stream = TickStream()
    base_ns = int(time.time() * 1e9)
    
    # Realistic inter-arrival times (exponential)
    inter_arrival = np.random.exponential(100_000, args.ticks)  # ~100us mean
    timestamps = base_ns + np.cumsum(inter_arrival).astype(int)
    
    for i in range(args.ticks):
        tick = Tick(
            timestamp_ns=int(timestamps[i]),
            price=prices[i],
            size=sizes[i],
            tick_type=TickType.TRADE,
            side=sides[i],
            trade_id=i
        )
        stream.add_tick(tick)
    
    print(f"   Price range: ${prices.min():,.2f} - ${prices.max():,.2f}")
    print(f"   Total volume: {sizes.sum():,.2f}")
    print(f"   Trade ratio: {(sides == 'buy').mean()*100:.1f}% buys")
    
    # Configure simulation
    print(f"\n⚙️  Configuring simulation...")
    config = SimulationConfig(
        warmup_ticks=min(100, args.ticks // 10),
        initial_mid_price=args.price,
        initial_spread_bps=args.spread,
        strategy_type=args.strategy,
        max_position=10.0
    )
    
    print(f"   Strategy: {args.strategy}")
    print(f"   Initial spread: {args.spread} bps")
    print(f"   Max position: {config.max_position}")
    
    # Run simulation
    print(f"\n🚀 Running HFT simulation...")
    start_time = time.perf_counter()
    
    sim = HFTSimulator(config, seed=args.seed)
    results = sim.run(stream)
    
    elapsed = time.perf_counter() - start_time
    
    # Print results
    print(f"\n📈 SIMULATION RESULTS")
    print("-" * 50)
    print(f"   Ticks processed:    {results['ticks_processed']:,}")
    print(f"   Simulation time:    {elapsed:.3f} seconds")
    print(f"   Throughput:         {results['ticks_processed']/elapsed:,.0f} ticks/sec")
    
    print(f"\n💰 TRADING RESULTS")
    print("-" * 50)
    print(f"   Market trades:      {len(results['trades'])}")
    print(f"   Strategy trades:    {len(results['strategy_trades'])}")
    print(f"   Final PnL:          ${results['final_pnl']:,.2f}")
    print(f"   Final position:     {results['final_position']:.4f}")
    
    if results['processing_stats']:
        stats = results['processing_stats']
        print(f"\n⏱️  LATENCY STATISTICS")
        print("-" * 50)
        print(f"   Mean processing:    {stats['mean_us']:.2f} µs")
        print(f"   Median processing:  {stats['median_us']:.2f} µs")
        print(f"   P99 processing:     {stats['p99_us']:.2f} µs")
        print(f"   Max processing:     {stats['max_us']:.2f} µs")
    
    # Market making specific stats
    if args.strategy == 'market_maker' and results['strategy_trades']:
        trades = results['strategy_trades']
        buys = [t for t in trades if t.get('trade_side') == 'buy']
        sells = [t for t in trades if t.get('trade_side') == 'sell']
        
        print(f"\n📊 MARKET MAKING STATS")
        print("-" * 50)
        print(f"   Buy fills:          {len(buys)}")
        print(f"   Sell fills:         {len(sells)}")
        print(f"   Fill rate:          {len(trades)/results['ticks_processed']*100:.2f}%")
    
    print("\n" + "=" * 70)
    print("                         SIMULATION COMPLETE")
    print("=" * 70 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
