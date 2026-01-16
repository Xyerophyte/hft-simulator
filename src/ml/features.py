"""
Feature engineering module for machine learning models.
Creates features from market data for price prediction.
"""

import pandas as pd
import numpy as np
from typing import List, Optional


class FeatureEngineer:
    """
    Engineers features from market data for ML models.
    Focuses on price-based, volume-based, and technical indicators.
    """
    
    def __init__(self):
        """Initialize feature engineer."""
        self.feature_names = []
        
    def create_price_features(
        self,
        df: pd.DataFrame,
        windows: List[int] = [5, 10, 20]
    ) -> pd.DataFrame:
        """
        Create price-based features.
        
        Args:
            df: DataFrame with OHLCV data
            windows: List of lookback windows
            
        Returns:
            DataFrame with price features
        """
        # Returns at different horizons
        for window in windows:
            df[f'return_{window}'] = df['close'].pct_change(window)
            df[f'log_return_{window}'] = np.log(df['close'] / df['close'].shift(window))
        
        # Price momentum
        for window in windows:
            df[f'momentum_{window}'] = df['close'] - df['close'].shift(window)
            df[f'momentum_pct_{window}'] = df['close'].pct_change(window)
        
        # Moving averages and crossovers
        for window in windows:
            df[f'sma_{window}'] = df['close'].rolling(window).mean()
            df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()
            
            # Distance from MA
            df[f'dist_sma_{window}'] = (df['close'] - df[f'sma_{window}']) / df[f'sma_{window}']
            df[f'dist_ema_{window}'] = (df['close'] - df[f'ema_{window}']) / df[f'ema_{window}']
        
        # Price position in range
        for window in windows:
            rolling_min = df['low'].rolling(window).min()
            rolling_max = df['high'].rolling(window).max()
            price_range = rolling_max - rolling_min
            
            df[f'price_position_{window}'] = np.where(
                price_range > 0,
                (df['close'] - rolling_min) / price_range,
                0.5
            )
        
        return df
    
    def create_volatility_features(
        self,
        df: pd.DataFrame,
        windows: List[int] = [5, 10, 20]
    ) -> pd.DataFrame:
        """
        Create volatility-based features.
        
        Args:
            df: DataFrame with OHLCV data
            windows: List of lookback windows
            
        Returns:
            DataFrame with volatility features
        """
        # Calculate returns if not present
        if 'return_1' not in df.columns:
            df['return_1'] = df['close'].pct_change()
        
        # Rolling volatility
        for window in windows:
            df[f'volatility_{window}'] = df['return_1'].rolling(window).std()
            df[f'volatility_{window}_annualized'] = df[f'volatility_{window}'] * np.sqrt(252 * 24 * 60)
        
        # Parkinson volatility (uses high-low)
        for window in windows:
            hl_ratio = np.log(df['high'] / df['low']) ** 2
            df[f'parkinson_vol_{window}'] = np.sqrt(
                hl_ratio.rolling(window).sum() / (4 * window * np.log(2))
            )
        
        # Average True Range
        df['hl'] = df['high'] - df['low']
        df['hc'] = abs(df['high'] - df['close'].shift(1))
        df['lc'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['hl', 'hc', 'lc']].max(axis=1)
        
        for window in windows:
            df[f'atr_{window}'] = df['tr'].rolling(window).mean()
            df[f'atr_pct_{window}'] = df[f'atr_{window}'] / df['close']
        
        # Clean up temp columns
        df.drop(['hl', 'hc', 'lc', 'tr'], axis=1, inplace=True, errors='ignore')
        
        return df
    
    def create_volume_features(
        self,
        df: pd.DataFrame,
        windows: List[int] = [5, 10, 20]
    ) -> pd.DataFrame:
        """
        Create volume-based features.
        
        Args:
            df: DataFrame with volume data
            windows: List of lookback windows
            
        Returns:
            DataFrame with volume features
        """
        # Volume moving averages
        for window in windows:
            df[f'volume_sma_{window}'] = df['volume'].rolling(window).mean()
            df[f'volume_ratio_{window}'] = df['volume'] / df[f'volume_sma_{window}']
        
        # Volume momentum
        for window in windows:
            df[f'volume_change_{window}'] = df['volume'].pct_change(window)
        
        # Volume-weighted price
        for window in windows:
            df[f'vwap_{window}'] = (
                (df['close'] * df['volume']).rolling(window).sum() / 
                df['volume'].rolling(window).sum()
            )
            df[f'dist_vwap_{window}'] = (df['close'] - df[f'vwap_{window}']) / df[f'vwap_{window}']
        
        # On-Balance Volume
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()
        for window in windows:
            df[f'obv_ema_{window}'] = df['obv'].ewm(span=window, adjust=False).mean()
        
        return df
    
    def create_candle_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create candlestick pattern features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with candle features
        """
        # Candle body and wicks
        df['body'] = df['close'] - df['open']
        df['body_pct'] = df['body'] / df['open']
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        
        # Candle range
        df['range'] = df['high'] - df['low']
        df['range_pct'] = df['range'] / df['open']
        
        # Body to range ratio
        df['body_range_ratio'] = np.where(
            df['range'] > 0,
            abs(df['body']) / df['range'],
            0
        )
        
        # Candle direction
        df['is_bullish'] = (df['close'] > df['open']).astype(int)
        
        return df
    
    def create_orderflow_features(
        self,
        df: pd.DataFrame,
        windows: List[int] = [5, 10, 20]
    ) -> pd.DataFrame:
        """
        Create order flow and pressure features.
        
        Args:
            df: DataFrame with trades data
            windows: List of lookback windows
            
        Returns:
            DataFrame with order flow features
        """
        if 'trades' not in df.columns:
            # If no trade count, use volume as proxy
            df['trades'] = df['volume'] / df['close']
        
        # Trade intensity
        for window in windows:
            df[f'trade_intensity_{window}'] = df['trades'].rolling(window).mean()
        
        # Buy/sell pressure estimation (using close position in candle)
        df['buy_pressure'] = np.where(
            df['high'] - df['low'] > 0,
            (df['close'] - df['low']) / (df['high'] - df['low']),
            0.5
        )
        
        for window in windows:
            df[f'buy_pressure_ma_{window}'] = df['buy_pressure'].rolling(window).mean()
        
        return df
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features.
        
        Args:
            df: DataFrame with datetime index
            
        Returns:
            DataFrame with time features
        """
        # Hour of day (cyclical encoding)
        df['hour'] = df.index.hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Day of week (cyclical encoding)
        df['day_of_week'] = df.index.dayofweek
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Weekend indicator
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        return df
    
    def create_all_features(
        self,
        df: pd.DataFrame,
        windows: List[int] = [5, 10, 20]
    ) -> pd.DataFrame:
        """
        Create all features at once.
        
        Args:
            df: DataFrame with OHLCV data
            windows: List of lookback windows
            
        Returns:
            DataFrame with all engineered features
        """
        print("Creating features...")
        
        # Create all feature types
        df = self.create_price_features(df, windows)
        df = self.create_volatility_features(df, windows)
        df = self.create_volume_features(df, windows)
        df = self.create_candle_features(df)
        df = self.create_orderflow_features(df, windows)
        df = self.create_time_features(df)
        
        # Remove rows with NaN
        initial_rows = len(df)
        df = df.dropna()
        dropped_rows = initial_rows - len(df)
        
        print(f"Created {len(df.columns)} total columns")
        print(f"Dropped {dropped_rows} rows with NaN")
        print(f"Final dataset: {len(df)} rows")
        
        # Store feature names (excluding OHLCV)
        self.feature_names = [
            col for col in df.columns 
            if col not in ['open', 'high', 'low', 'close', 'volume']
        ]
        
        return df
    
    def get_feature_importance_proxy(
        self,
        df: pd.DataFrame,
        target_col: str = 'close'
    ) -> pd.DataFrame:
        """
        Calculate correlation-based feature importance.
        
        Args:
            df: DataFrame with features
            target_col: Target column name
            
        Returns:
            DataFrame with feature correlations
        """
        # Calculate future returns as target
        df['target'] = df[target_col].pct_change().shift(-1)
        
        # Get correlations
        correlations = df[self.feature_names].corrwith(df['target']).abs()
        
        # Sort by importance
        importance_df = pd.DataFrame({
            'feature': correlations.index,
            'correlation': correlations.values
        }).sort_values('correlation', ascending=False)
        
        return importance_df
    
    def select_top_features(
        self,
        df: pd.DataFrame,
        target_col: str = 'close',
        n_features: int = 20
    ) -> List[str]:
        """
        Select top N features by correlation.
        
        Args:
            df: DataFrame with features
            target_col: Target column
            n_features: Number of features to select
            
        Returns:
            List of top feature names
        """
        importance_df = self.get_feature_importance_proxy(df, target_col)
        return importance_df['feature'].head(n_features).tolist()
    
    def create_features(self, df: pd.DataFrame, windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """
        Create features from DataFrame.
        Alias for create_all_features for API compatibility.
        
        Args:
            df: DataFrame with OHLCV data
            windows: List of lookback windows
            
        Returns:
            DataFrame with all engineered features
        """
        return self.create_all_features(df, windows)
    
    def prepare_training_data(
        self,
        df: pd.DataFrame,
        target_col: str = 'close',
        lookback: int = 60,
        prediction_horizon: int = 1
    ) -> tuple:
        """
        Prepare data for ML training with sequences and labels.
        
        Args:
            df: DataFrame with features (output from create_features)
            target_col: Column to use for target calculation
            lookback: Number of timesteps to use as input sequence
            prediction_horizon: How many steps ahead to predict
            
        Returns:
            Tuple of (X, y, feature_names):
            - X: numpy array of shape (n_samples, lookback, n_features)
            - y: numpy array of shape (n_samples,) with binary labels (1=up, 0=down)
            - feature_names: list of feature column names
        """
        # Get feature columns (exclude OHLCV)
        feature_cols = [col for col in df.columns 
                       if col not in ['open', 'high', 'low', 'close', 'volume', 'trades']]
        self.feature_names = feature_cols
        
        # Create target (future price direction)
        df = df.copy()
        df['future_return'] = df[target_col].pct_change(prediction_horizon).shift(-prediction_horizon)
        df['target'] = (df['future_return'] > 0).astype(int)
        
        # Drop NaN rows
        df = df.dropna()
        
        # Extract features as numpy array
        features = df[feature_cols].values
        targets = df['target'].values
        
        # Create sequences
        X_sequences = []
        y_sequences = []
        
        for i in range(len(features) - lookback):
            X_sequences.append(features[i:i + lookback])
            y_sequences.append(targets[i + lookback])
        
        X = np.array(X_sequences)
        y = np.array(y_sequences)
        
        print(f"Prepared training data:")
        print(f"  X shape: {X.shape} (samples, timesteps, features)")
        print(f"  y shape: {y.shape}")
        print(f"  Features: {len(feature_cols)}")
        
        return X, y, feature_cols


# Example usage
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from data.fetcher import BinanceDataFetcher
    
    # Fetch data
    fetcher = BinanceDataFetcher("BTCUSDT")
    df = fetcher.fetch_klines(interval="1m", limit=1000)
    
    if not df.empty:
        # Create features
        engineer = FeatureEngineer()
        df_features = engineer.create_all_features(df)
        
        print("\nFeature sample:")
        print(df_features.head())
        
        print("\nTop 15 features by correlation:")
        top_features = engineer.select_top_features(df_features, n_features=15)
        for i, feat in enumerate(top_features, 1):
            print(f"  {i}. {feat}")