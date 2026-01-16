"""
Data caching module for storing and loading historical market data.
"""

import pandas as pd
import os
from pathlib import Path
from typing import Optional
from datetime import datetime


class DataCache:
    """
    Manages caching of market data to disk for offline use.
    Supports both CSV and Parquet formats.
    """
    
    def __init__(self, cache_dir: str = "data/raw"):
        """
        Initialize data cache.
        
        Args:
            cache_dir: Directory to store cached data files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_cache_filename(
        self,
        symbol: str,
        interval: str,
        file_format: str = "parquet"
    ) -> Path:
        """
        Generate cache filename based on parameters.
        
        Args:
            symbol: Trading pair symbol
            interval: Kline interval
            file_format: File format (csv or parquet)
            
        Returns:
            Path to cache file
        """
        filename = f"{symbol}_{interval}.{file_format}"
        return self.cache_dir / filename
    
    def save(
        self,
        data: pd.DataFrame,
        symbol: str,
        interval: str,
        file_format: str = "parquet"
    ) -> bool:
        """
        Save DataFrame to cache.
        
        Args:
            data: DataFrame to save
            symbol: Trading pair symbol
            interval: Kline interval
            file_format: File format (csv or parquet)
            
        Returns:
            True if successful, False otherwise
        """
        if data.empty:
            print("Warning: Attempting to save empty DataFrame")
            return False
            
        filepath = self._get_cache_filename(symbol, interval, file_format)
        
        try:
            if file_format == "parquet":
                data.to_parquet(filepath, compression='snappy')
            elif file_format == "csv":
                data.to_csv(filepath)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
                
            file_size = filepath.stat().st_size / (1024 * 1024)  # MB
            print(f"Saved {len(data)} rows to {filepath} ({file_size:.2f} MB)")
            return True
            
        except Exception as e:
            print(f"Error saving data to cache: {e}")
            return False
    
    def load(
        self,
        symbol: str,
        interval: str,
        file_format: str = "parquet"
    ) -> Optional[pd.DataFrame]:
        """
        Load DataFrame from cache.
        
        Args:
            symbol: Trading pair symbol
            interval: Kline interval
            file_format: File format (csv or parquet)
            
        Returns:
            DataFrame if found, None otherwise
        """
        filepath = self._get_cache_filename(symbol, interval, file_format)
        
        if not filepath.exists():
            print(f"Cache file not found: {filepath}")
            return None
            
        try:
            if file_format == "parquet":
                data = pd.read_parquet(filepath)
            elif file_format == "csv":
                data = pd.read_csv(filepath, index_col=0, parse_dates=True)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
                
            print(f"Loaded {len(data)} rows from {filepath}")
            
            # Display cache info
            if not data.empty:
                print(f"Date range: {data.index[0]} to {data.index[-1]}")
                
            return data
            
        except Exception as e:
            print(f"Error loading data from cache: {e}")
            return None
    
    def exists(
        self,
        symbol: str,
        interval: str,
        file_format: str = "parquet"
    ) -> bool:
        """
        Check if cached data exists.
        
        Args:
            symbol: Trading pair symbol
            interval: Kline interval
            file_format: File format (csv or parquet)
            
        Returns:
            True if cache exists, False otherwise
        """
        filepath = self._get_cache_filename(symbol, interval, file_format)
        return filepath.exists()
    
    def get_cache_info(
        self,
        symbol: str,
        interval: str,
        file_format: str = "parquet"
    ) -> dict:
        """
        Get information about cached data.
        
        Args:
            symbol: Trading pair symbol
            interval: Kline interval
            file_format: File format (csv or parquet)
            
        Returns:
            Dictionary with cache metadata
        """
        filepath = self._get_cache_filename(symbol, interval, file_format)
        
        if not filepath.exists():
            return {"exists": False}
            
        stat = filepath.stat()
        
        info = {
            "exists": True,
            "path": str(filepath),
            "size_mb": stat.st_size / (1024 * 1024),
            "modified": datetime.fromtimestamp(stat.st_mtime)
        }
        
        # Try to get data range without loading full file
        try:
            if file_format == "parquet":
                df = pd.read_parquet(filepath)
            else:
                df = pd.read_csv(filepath, index_col=0, parse_dates=True)
                
            if not df.empty:
                info["rows"] = len(df)
                info["start_date"] = df.index[0]
                info["end_date"] = df.index[-1]
                info["columns"] = list(df.columns)
                
        except Exception as e:
            info["error"] = str(e)
            
        return info
    
    def clear_cache(self, symbol: Optional[str] = None, interval: Optional[str] = None):
        """
        Clear cached data.
        
        Args:
            symbol: If provided, only clear this symbol
            interval: If provided, only clear this interval
        """
        if symbol and interval:
            # Clear specific cache
            for fmt in ["parquet", "csv"]:
                filepath = self._get_cache_filename(symbol, interval, fmt)
                if filepath.exists():
                    filepath.unlink()
                    print(f"Deleted {filepath}")
        else:
            # Clear all cache files
            for filepath in self.cache_dir.glob("*"):
                if filepath.is_file():
                    filepath.unlink()
                    print(f"Deleted {filepath}")
            print("Cache cleared")
    
    def list_cached_files(self) -> list:
        """
        List all cached data files.
        
        Returns:
            List of cached file information
        """
        files = []
        for filepath in self.cache_dir.glob("*"):
            if filepath.is_file():
                stat = filepath.stat()
                files.append({
                    "name": filepath.name,
                    "size_mb": stat.st_size / (1024 * 1024),
                    "modified": datetime.fromtimestamp(stat.st_mtime)
                })
        return files


# Example usage
if __name__ == "__main__":
    from fetcher import BinanceDataFetcher
    
    # Initialize cache and fetcher
    cache = DataCache()
    fetcher = BinanceDataFetcher("BTCUSDT")
    
    symbol = "BTCUSDT"
    interval = "1m"
    
    # Check if data exists in cache
    if cache.exists(symbol, interval):
        print("Loading data from cache...")
        df = cache.load(symbol, interval)
    else:
        print("Fetching fresh data from Binance...")
        df = fetcher.fetch_historical_data(interval=interval, days=7)
        
        if not df.empty:
            cache.save(df, symbol, interval)
    
    # Show cache info
    print("\nCached files:")
    for file_info in cache.list_cached_files():
        print(f"  {file_info['name']}: {file_info['size_mb']:.2f} MB (modified: {file_info['modified']})")