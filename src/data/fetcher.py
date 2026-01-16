"""
Data fetcher module for retrieving historical market data from Binance API.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List
import time


class BinanceDataFetcher:
    """
    Fetches historical OHLCV data from Binance public API.
    
    API Documentation: https://binance-docs.github.io/apidocs/spot/en/
    """
    
    BASE_URL = "https://api.binance.com/api/v3"
    
    def __init__(self, symbol: str = "BTCUSDT"):
        """
        Initialize the data fetcher.
        
        Args:
            symbol: Trading pair symbol (default: BTCUSDT)
        """
        self.symbol = symbol
        
    def fetch_klines(
        self,
        interval: str = "1m",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch historical kline/candlestick data.
        
        Args:
            interval: Kline interval (1m, 3m, 5m, 15m, 30m, 1h, 4h, 1d, etc.)
            start_time: Start time for data retrieval
            end_time: End time for data retrieval
            limit: Number of candles to fetch (max 1000 per request)
            
        Returns:
            DataFrame with OHLCV data
        """
        endpoint = f"{self.BASE_URL}/klines"
        
        params = {
            "symbol": self.symbol,
            "interval": interval,
            "limit": limit
        }
        
        if start_time:
            params["startTime"] = int(start_time.timestamp() * 1000)
        if end_time:
            params["endTime"] = int(end_time.timestamp() * 1000)
            
        try:
            response = requests.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Convert types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
                
            df['trades'] = df['trades'].astype(int)
            
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            
            # Keep only essential columns
            df = df[['open', 'high', 'low', 'close', 'volume', 'trades']]
            
            return df
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from Binance: {e}")
            return pd.DataFrame()
    
    def fetch_historical_data(
        self,
        interval: str = "1m",
        days: int = 90,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Fetch multiple months of historical data by making multiple API calls.
        
        Args:
            interval: Kline interval
            days: Number of days of historical data to fetch
            end_date: End date for data retrieval (default: now)
            
        Returns:
            DataFrame with complete historical OHLCV data
        """
        if end_date is None:
            end_date = datetime.now()
            
        start_date = end_date - timedelta(days=days)
        
        all_data = []
        current_start = start_date
        
        print(f"Fetching {days} days of {interval} data for {self.symbol}...")
        
        while current_start < end_date:
            # Calculate end time for this batch (max 1000 candles)
            if interval == "1m":
                batch_end = min(current_start + timedelta(minutes=1000), end_date)
            elif interval == "5m":
                batch_end = min(current_start + timedelta(minutes=5000), end_date)
            elif interval == "1h":
                batch_end = min(current_start + timedelta(hours=1000), end_date)
            else:
                # Default to 1000 intervals
                batch_end = min(current_start + timedelta(days=1000), end_date)
            
            print(f"Fetching: {current_start.strftime('%Y-%m-%d %H:%M')} to {batch_end.strftime('%Y-%m-%d %H:%M')}")
            
            batch_data = self.fetch_klines(
                interval=interval,
                start_time=current_start,
                end_time=batch_end,
                limit=1000
            )
            
            if not batch_data.empty:
                all_data.append(batch_data)
            
            # Move to next batch
            current_start = batch_end
            
            # Rate limiting - Binance has weight limits
            time.sleep(0.5)
        
        if not all_data:
            print("No data fetched!")
            return pd.DataFrame()
        
        # Combine all batches
        combined_df = pd.concat(all_data)
        
        # Remove duplicates (can occur at batch boundaries)
        combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
        
        # Sort by timestamp
        combined_df.sort_index(inplace=True)
        
        print(f"Successfully fetched {len(combined_df)} candles")
        print(f"Date range: {combined_df.index[0]} to {combined_df.index[-1]}")
        
        return combined_df
    
    def get_current_price(self) -> Optional[float]:
        """
        Get the current price of the symbol.
        
        Returns:
            Current price as float, or None if error
        """
        endpoint = f"{self.BASE_URL}/ticker/price"
        params = {"symbol": self.symbol}
        
        try:
            response = requests.get(endpoint, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            return float(data['price'])
        except requests.exceptions.RequestException as e:
            print(f"Error fetching current price: {e}")
            return None


# Example usage
if __name__ == "__main__":
    # Initialize fetcher
    fetcher = BinanceDataFetcher(symbol="BTCUSDT")
    
    # Get current price
    current_price = fetcher.get_current_price()
    print(f"Current BTC-USD price: ${current_price:,.2f}")
    
    # Fetch recent 1-minute data (last 7 days for testing)
    df = fetcher.fetch_historical_data(interval="1m", days=7)
    
    if not df.empty:
        print("\nData sample:")
        print(df.head())
        print("\nData info:")
        print(df.info())
        print("\nPrice statistics:")
        print(df['close'].describe())