"""
Data preprocessing module for normalizing and preparing market data for analysis.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple


class DataPreprocessor:
    """
    Preprocesses market data for ML models and backtesting.
    Handles missing data, normalization, and feature calculation.
    """
    
    def __init__(self):
        """Initialize preprocessor."""
        self.scaler_params = {}
        
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean market data by handling missing values and outliers.
        
        Args:
            data: Raw OHLCV DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        df = data.copy()
        
        # Remove any duplicate timestamps
        df = df[~df.index.duplicated(keep='first')]
        
        # Sort by timestamp
        df.sort_index(inplace=True)
        
        # Handle missing values (forward fill then backward fill)
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        
        # Remove any remaining NaN values
        df.dropna(inplace=True)
        
        # Validate OHLC relationships
        df = self._validate_ohlc(df)
        
        return df
    
    def _validate_ohlc(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate OHLC data integrity.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            Validated DataFrame
        """
        # Ensure high >= low
        invalid_hl = df['high'] < df['low']
        if invalid_hl.any():
            print(f"Warning: Found {invalid_hl.sum()} rows where high < low. Fixing...")
            df.loc[invalid_hl, 'high'] = df.loc[invalid_hl, 'low']
        
        # Ensure high >= open, close
        df['high'] = df[['high', 'open', 'close']].max(axis=1)
        
        # Ensure low <= open, close
        df['low'] = df[['low', 'open', 'close']].min(axis=1)
        
        return df
    
    def add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add return calculations to DataFrame.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with added return columns
        """
        # Simple returns
        df['returns'] = df['close'].pct_change()
        
        # Log returns (more suitable for ML)
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        return df
    
    def add_volatility(
        self,
        df: pd.DataFrame,
        windows: list = [10, 20, 50]
    ) -> pd.DataFrame:
        """
        Add volatility indicators.
        
        Args:
            df: DataFrame with price data
            windows: List of rolling window sizes
            
        Returns:
            DataFrame with volatility columns
        """
        for window in windows:
            # Rolling standard deviation of returns
            df[f'volatility_{window}'] = df['returns'].rolling(window).std()
            
            # True Range for ATR
            df['high_low'] = df['high'] - df['low']
            df['high_close'] = abs(df['high'] - df['close'].shift(1))
            df['low_close'] = abs(df['low'] - df['close'].shift(1))
            
            df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
            df[f'atr_{window}'] = df['true_range'].rolling(window).mean()
        
        # Clean up temporary columns
        df.drop(['high_low', 'high_close', 'low_close', 'true_range'], axis=1, inplace=True, errors='ignore')
        
        return df
    
    def add_moving_averages(
        self,
        df: pd.DataFrame,
        windows: list = [10, 20, 50]
    ) -> pd.DataFrame:
        """
        Add moving average indicators.
        
        Args:
            df: DataFrame with price data
            windows: List of MA window sizes
            
        Returns:
            DataFrame with MA columns
        """
        for window in windows:
            # Simple Moving Average
            df[f'sma_{window}'] = df['close'].rolling(window).mean()
            
            # Exponential Moving Average
            df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()
        
        return df
    
    def add_price_momentum(
        self,
        df: pd.DataFrame,
        periods: list = [5, 10, 20]
    ) -> pd.DataFrame:
        """
        Add price momentum indicators.
        
        Args:
            df: DataFrame with price data
            periods: List of lookback periods
            
        Returns:
            DataFrame with momentum columns
        """
        for period in periods:
            # Price change over period
            df[f'momentum_{period}'] = df['close'].diff(period)
            
            # Rate of change
            df[f'roc_{period}'] = df['close'].pct_change(period)
        
        return df
    
    def add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volume-based features.
        
        Args:
            df: DataFrame with volume data
            
        Returns:
            DataFrame with volume features
        """
        # Volume moving averages
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        
        # Volume ratio (current volume vs average)
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        # Volume rate of change
        df['volume_roc'] = df['volume'].pct_change()
        
        return df
    
    def normalize_features(
        self,
        df: pd.DataFrame,
        columns: Optional[list] = None,
        method: str = "minmax"
    ) -> pd.DataFrame:
        """
        Normalize specified columns.
        
        Args:
            df: DataFrame to normalize
            columns: List of columns to normalize (None = all numeric)
            method: Normalization method ('minmax' or 'zscore')
            
        Returns:
            DataFrame with normalized columns
        """
        if columns is None:
            # Get all numeric columns except OHLCV
            columns = df.select_dtypes(include=[np.number]).columns
            columns = [c for c in columns if c not in ['open', 'high', 'low', 'close', 'volume']]
        
        for col in columns:
            if col not in df.columns:
                continue
                
            if method == "minmax":
                # Min-Max normalization to [0, 1]
                min_val = df[col].min()
                max_val = df[col].max()
                
                if max_val - min_val > 0:
                    df[f'{col}_norm'] = (df[col] - min_val) / (max_val - min_val)
                    self.scaler_params[col] = {'min': min_val, 'max': max_val}
                    
            elif method == "zscore":
                # Z-score normalization
                mean_val = df[col].mean()
                std_val = df[col].std()
                
                if std_val > 0:
                    df[f'{col}_norm'] = (df[col] - mean_val) / std_val
                    self.scaler_params[col] = {'mean': mean_val, 'std': std_val}
        
        return df
    
    def prepare_for_ml(
        self,
        df: pd.DataFrame,
        target_column: str = 'close',
        lookback: int = 1
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for ML training by creating features and labels.
        
        Args:
            df: DataFrame with market data
            target_column: Column to predict
            lookback: Number of periods ahead to predict
            
        Returns:
            Tuple of (features_df, labels_series)
        """
        # Create target (future price direction)
        df['target'] = (df[target_column].shift(-lookback) > df[target_column]).astype(int)
        
        # Remove rows with NaN in target
        df_clean = df.dropna(subset=['target'])
        
        # Separate features and labels
        feature_cols = [c for c in df_clean.columns if c not in ['target', 'open', 'high', 'low', 'close', 'volume']]
        
        X = df_clean[feature_cols]
        y = df_clean['target']
        
        return X, y
    
    def create_sequences(
        self,
        data: np.ndarray,
        sequence_length: int
    ) -> np.ndarray:
        """
        Create sequences for LSTM/RNN models.
        
        Args:
            data: Input data array
            sequence_length: Length of each sequence
            
        Returns:
            Array of sequences
        """
        sequences = []
        for i in range(len(data) - sequence_length):
            sequences.append(data[i:i + sequence_length])
        return np.array(sequences)
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to DataFrame.
        Alias for preprocess_pipeline with add_features=True.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with technical indicators added
        """
        return self.preprocess_pipeline(df, add_features=True)
    
    def preprocess_pipeline(
        self,
        df: pd.DataFrame,
        add_features: bool = True
    ) -> pd.DataFrame:
        """
        Full preprocessing pipeline.
        
        Args:
            df: Raw OHLCV data
            add_features: Whether to add technical features
            
        Returns:
            Preprocessed DataFrame
        """
        print("Starting preprocessing pipeline...")
        
        # Clean data
        df = self.clean_data(df)
        print(f"After cleaning: {len(df)} rows")
        
        if add_features:
            # Add returns
            df = self.add_returns(df)
            
            # Add technical indicators
            df = self.add_moving_averages(df)
            df = self.add_volatility(df)
            df = self.add_price_momentum(df)
            df = self.add_volume_features(df)
            
            print(f"Added {len(df.columns)} total columns")
        
        # Remove any rows with NaN created by indicators
        df.dropna(inplace=True)
        print(f"After removing NaN: {len(df)} rows")
        
        return df


# Example usage
if __name__ == "__main__":
    from fetcher import BinanceDataFetcher
    
    # Fetch some data
    fetcher = BinanceDataFetcher("BTCUSDT")
    df = fetcher.fetch_klines(interval="1m", limit=1000)
    
    if not df.empty:
        # Preprocess
        preprocessor = DataPreprocessor()
        df_processed = preprocessor.preprocess_pipeline(df)
        
        print("\nProcessed data sample:")
        print(df_processed.head())
        print("\nColumns:")
        print(df_processed.columns.tolist())
        print(f"\nData shape: {df_processed.shape}")