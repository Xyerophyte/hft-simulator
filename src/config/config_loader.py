"""
Configuration loader for HFT Simulator.
Loads YAML config with environment variable overrides.
"""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class MLConfig:
    """ML model configuration."""
    model_type: str = "ensemble"
    lookback: int = 30
    prediction_horizon: int = 1
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    d_model: int = 64
    nhead: int = 4
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 0.001
    validation_split: float = 0.2
    early_stopping_patience: int = 10


@dataclass
class StrategyConfig:
    """Strategy configuration."""
    type: str = "ensemble"
    
    # Momentum
    ml_threshold: float = 0.52
    momentum_threshold: float = 0.0003
    volume_threshold: float = 1.15
    
    # Mean Reversion
    rsi_oversold: float = 28.0
    rsi_overbought: float = 72.0
    mean_deviation_threshold: float = 0.015
    
    # Breakout
    breakout_threshold: float = 0.0008
    breakout_lookback: int = 20
    
    # Ensemble
    ensemble_method: str = "weighted"
    min_confidence: float = 0.45
    weights: Dict[str, float] = field(default_factory=lambda: {
        'momentum': 0.4, 'mean_reversion': 0.35, 'breakout': 0.25
    })


@dataclass
class RiskConfig:
    """Risk management configuration."""
    max_position_pct: float = 0.25
    max_drawdown_pct: float = 0.12
    stop_loss_pct: float = 0.015
    take_profit_pct: float = 0.03
    max_daily_loss_pct: float = 0.04
    position_sizing: str = "kelly"


@dataclass
class BacktestConfig:
    """Backtest configuration."""
    initial_capital: float = 100000.0
    fee_rate: float = 0.0008
    slippage_pct: float = 0.0002
    use_risk_manager: bool = True


@dataclass
class DataConfig:
    """Data configuration."""
    symbol: str = "BTCUSDT"
    interval: str = "1m"
    source: str = "binance"
    history_days: int = 7
    cache_enabled: bool = True
    cache_dir: str = "data/cache"


@dataclass
class Config:
    """Main configuration container."""
    data: DataConfig = field(default_factory=DataConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    results_dir: str = "results"
    log_level: str = "INFO"


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from YAML file with env var overrides.
    
    Args:
        config_path: Path to config file. Defaults to config/default.yaml
        
    Returns:
        Config object
    """
    # Default path
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config" / "default.yaml"
    else:
        config_path = Path(config_path)
    
    # Load YAML
    if config_path.exists():
        with open(config_path) as f:
            raw_config = yaml.safe_load(f) or {}
    else:
        raw_config = {}
    
    # Build config objects
    config = Config()
    
    # Data config
    if 'data' in raw_config:
        d = raw_config['data']
        config.data = DataConfig(
            symbol=d.get('symbol', config.data.symbol),
            interval=d.get('interval', config.data.interval),
            source=d.get('source', config.data.source),
            history_days=d.get('history_days', config.data.history_days),
            cache_enabled=d.get('cache_enabled', config.data.cache_enabled),
            cache_dir=d.get('cache_dir', config.data.cache_dir)
        )
    
    # ML config
    if 'ml' in raw_config:
        m = raw_config['ml']
        lstm = m.get('lstm', {})
        transformer = m.get('transformer', {})
        training = m.get('training', {})
        
        config.ml = MLConfig(
            model_type=m.get('model_type', config.ml.model_type),
            lookback=m.get('lookback', config.ml.lookback),
            prediction_horizon=m.get('prediction_horizon', config.ml.prediction_horizon),
            hidden_size=lstm.get('hidden_size', config.ml.hidden_size),
            num_layers=lstm.get('num_layers', config.ml.num_layers),
            dropout=lstm.get('dropout', config.ml.dropout),
            d_model=transformer.get('d_model', config.ml.d_model),
            nhead=transformer.get('nhead', config.ml.nhead),
            epochs=training.get('epochs', config.ml.epochs),
            batch_size=training.get('batch_size', config.ml.batch_size),
            learning_rate=training.get('learning_rate', config.ml.learning_rate),
            validation_split=training.get('validation_split', config.ml.validation_split),
            early_stopping_patience=training.get('early_stopping_patience', config.ml.early_stopping_patience)
        )
    
    # Strategy config
    if 'strategy' in raw_config:
        s = raw_config['strategy']
        mom = s.get('momentum', {})
        mr = s.get('mean_reversion', {})
        br = s.get('breakout', {})
        ens = s.get('ensemble', {})
        
        config.strategy = StrategyConfig(
            type=s.get('type', config.strategy.type),
            ml_threshold=mom.get('ml_threshold', config.strategy.ml_threshold),
            momentum_threshold=mom.get('momentum_threshold', config.strategy.momentum_threshold),
            volume_threshold=mom.get('volume_threshold', config.strategy.volume_threshold),
            rsi_oversold=mr.get('rsi_oversold', config.strategy.rsi_oversold),
            rsi_overbought=mr.get('rsi_overbought', config.strategy.rsi_overbought),
            mean_deviation_threshold=mr.get('mean_deviation_threshold', config.strategy.mean_deviation_threshold),
            breakout_threshold=br.get('breakout_threshold', config.strategy.breakout_threshold),
            breakout_lookback=br.get('lookback', config.strategy.breakout_lookback),
            ensemble_method=ens.get('method', config.strategy.ensemble_method),
            min_confidence=ens.get('min_confidence', config.strategy.min_confidence),
            weights=ens.get('weights', config.strategy.weights)
        )
    
    # Risk config
    if 'risk' in raw_config:
        r = raw_config['risk']
        config.risk = RiskConfig(
            max_position_pct=r.get('max_position_pct', config.risk.max_position_pct),
            max_drawdown_pct=r.get('max_drawdown_pct', config.risk.max_drawdown_pct),
            stop_loss_pct=r.get('stop_loss_pct', config.risk.stop_loss_pct),
            take_profit_pct=r.get('take_profit_pct', config.risk.take_profit_pct),
            max_daily_loss_pct=r.get('max_daily_loss_pct', config.risk.max_daily_loss_pct),
            position_sizing=r.get('position_sizing', config.risk.position_sizing)
        )
    
    # Backtest config
    if 'backtest' in raw_config:
        b = raw_config['backtest']
        config.backtest = BacktestConfig(
            initial_capital=b.get('initial_capital', config.backtest.initial_capital),
            fee_rate=b.get('fee_rate', config.backtest.fee_rate),
            slippage_pct=b.get('slippage_pct', config.backtest.slippage_pct),
            use_risk_manager=b.get('use_risk_manager', config.backtest.use_risk_manager)
        )
    
    # Output config
    if 'output' in raw_config:
        o = raw_config['output']
        config.results_dir = o.get('results_dir', config.results_dir)
        config.log_level = o.get('log_level', config.log_level)
    
    # Environment variable overrides
    config = _apply_env_overrides(config)
    
    return config


def _apply_env_overrides(config: Config) -> Config:
    """Apply environment variable overrides."""
    
    # Data
    if os.getenv('HFT_SYMBOL'):
        config.data.symbol = os.getenv('HFT_SYMBOL')
    if os.getenv('HFT_INTERVAL'):
        config.data.interval = os.getenv('HFT_INTERVAL')
    if os.getenv('HFT_HISTORY_DAYS'):
        config.data.history_days = int(os.getenv('HFT_HISTORY_DAYS'))
    
    # Backtest
    if os.getenv('HFT_INITIAL_CAPITAL'):
        config.backtest.initial_capital = float(os.getenv('HFT_INITIAL_CAPITAL'))
    if os.getenv('HFT_FEE_RATE'):
        config.backtest.fee_rate = float(os.getenv('HFT_FEE_RATE'))
    
    # Strategy
    if os.getenv('HFT_STRATEGY'):
        config.strategy.type = os.getenv('HFT_STRATEGY')
    
    # ML
    if os.getenv('HFT_MODEL_TYPE'):
        config.ml.model_type = os.getenv('HFT_MODEL_TYPE')
    if os.getenv('HFT_EPOCHS'):
        config.ml.epochs = int(os.getenv('HFT_EPOCHS'))
    
    # Output
    if os.getenv('HFT_RESULTS_DIR'):
        config.results_dir = os.getenv('HFT_RESULTS_DIR')
    if os.getenv('HFT_LOG_LEVEL'):
        config.log_level = os.getenv('HFT_LOG_LEVEL')
    
    return config


def get_default_config() -> Config:
    """Get default configuration without loading file."""
    return Config()
