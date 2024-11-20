import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Union, List, Tuple
from enum import Enum

class MarketRegime(Enum):
    BULL = 'bull'
    BEAR = 'bear'
    SIDEWAYS = 'sideways'
    VOLATILE = 'volatile'
    CRASH = 'crash'
    RECOVERY = 'recovery'
    BUBBLE = 'bubble'

class AssetClass(Enum):
    STOCK = 'stock'
    CRYPTO = 'crypto'
    FOREX = 'forex'
    COMMODITY = 'commodity'
    ETF = 'etf'
    BOND = 'bond'
    INDEX = 'index'

class MarketHours(Enum):
    US = 'us'          # 9:30-16:00 EST
    EUROPE = 'europe'  # 8:00-16:30 CET
    ASIA = 'asia'      # 9:00-15:00 JST
    CRYPTO = 'crypto'  # 24/7
    FOREX = 'forex'    # 24/5

class OHLCVGenerator:
    """Generator for OHLCV (Open, High, Low, Close, Volume) time series data."""
    
    def __init__(self, 
                 base_volatility: float = 0.02,
                 base_drift: float = 0.0001,
                 volume_mean: float = 1000000,
                 volume_std: float = 500000,
                 regime: MarketRegime = MarketRegime.SIDEWAYS,
                 asset_class: AssetClass = AssetClass.STOCK,
                 market_hours: MarketHours = MarketHours.US,
                 volatility_clustering: float = 0.7,
                 volume_price_corr: float = 0.4,
                 gap_probability: float = 0.05,
                 max_gap_size: float = 0.05,
                 seasonality_amplitude: float = 0.1,
                 mean_reversion_strength: float = 0.02,  
                 momentum_factor: float = 0.1,          
                 jump_intensity: float = 0.01,          
                 jump_size_mean: float = 0.05,          
                 jump_size_std: float = 0.02,           
                 volatility_of_volatility: float = 0.2, 
                 correlation_decay: float = 0.98,       
                 bid_ask_spread: float = 0.001,         
                 tick_size: Optional[float] = None,     
                 market_impact: float = 0.1,            
                 intraday_volatility_pattern: Optional[List[float]] = None):
        """
        Initialize the OHLCV generator with enhanced parameters.
        
        Args:
            base_volatility: Base daily volatility of the price
            base_drift: Base daily drift (trend) in the price
            volume_mean: Mean daily volume
            volume_std: Standard deviation of daily volume
            regime: Market regime (bull, bear, sideways, volatile)
            asset_class: Type of asset to simulate
            market_hours: Market hours (US, Europe, Asia, Crypto, Forex)
            volatility_clustering: Persistence of volatility (0 to 1)
            volume_price_corr: Correlation between volume and absolute price changes
            gap_probability: Probability of overnight gaps
            max_gap_size: Maximum size of price gaps
            seasonality_amplitude: Amplitude of seasonal patterns
            mean_reversion_strength: Strength of mean reversion
            momentum_factor: Momentum effect strength
            jump_intensity: Frequency of price jumps
            jump_size_mean: Average jump size
            jump_size_std: Jump size variation
            volatility_of_volatility: How much volatility itself varies
            correlation_decay: Autocorrelation decay
            bid_ask_spread: Bid-ask spread as fraction of price
            tick_size: Minimum price movement
            market_impact: Price impact of volume
            intraday_volatility_pattern: Optional list of 24 hourly volatility multipliers
        """
        self.base_volatility = base_volatility
        self.base_drift = base_drift
        self.volume_mean = volume_mean
        self.volume_std = volume_std
        self.regime = regime
        self.asset_class = asset_class
        self.market_hours = market_hours
        self.volatility_clustering = volatility_clustering
        self.volume_price_corr = volume_price_corr
        self.gap_probability = gap_probability
        self.max_gap_size = max_gap_size
        self.seasonality_amplitude = seasonality_amplitude
        self.mean_reversion_strength = mean_reversion_strength
        self.momentum_factor = momentum_factor
        self.jump_intensity = jump_intensity
        self.jump_size_mean = jump_size_mean
        self.jump_size_std = jump_size_std
        self.volatility_of_volatility = volatility_of_volatility
        self.correlation_decay = correlation_decay
        self.bid_ask_spread = bid_ask_spread
        self.tick_size = tick_size
        self.market_impact = market_impact
        
        # Default intraday volatility pattern (U-shaped)
        if intraday_volatility_pattern is None:
            self.intraday_volatility_pattern = self._default_intraday_pattern()
        else:
            self.intraday_volatility_pattern = intraday_volatility_pattern
            
        # Set regime-specific parameters
        self._set_regime_parameters()
        
        # Set asset-specific parameters
        self._set_asset_parameters()
        
        # Initialize state variables
        self.current_volatility = self.base_volatility
        
        # Store the most recently generated data
        self.data = None
        self.last_params = {}
    
    def _default_intraday_pattern(self) -> List[float]:
        """Generate default U-shaped intraday volatility pattern."""
        hours = np.arange(24)
        pattern = 1 + 0.5 * (np.exp(-((hours - 9.5) ** 2) / 20) + 
                            np.exp(-((hours - 15.5) ** 2) / 20))
        return pattern.tolist()
    
    def _set_regime_parameters(self):
        """Set drift and volatility adjustments based on market regime."""
        if self.regime == MarketRegime.BULL:
            self.drift_adjustment = 2.0
            self.volatility_adjustment = 0.8
            self.jump_intensity *= 0.5
        elif self.regime == MarketRegime.BEAR:
            self.drift_adjustment = -1.5
            self.volatility_adjustment = 1.2
            self.jump_intensity *= 1.5
        elif self.regime == MarketRegime.VOLATILE:
            self.drift_adjustment = 0.0
            self.volatility_adjustment = 2.0
            self.jump_intensity *= 2.0
        elif self.regime == MarketRegime.CRASH:
            self.drift_adjustment = -4.0
            self.volatility_adjustment = 3.0
            self.jump_intensity *= 4.0
            self.jump_size_mean *= 2.0
        elif self.regime == MarketRegime.RECOVERY:
            self.drift_adjustment = 3.0
            self.volatility_adjustment = 1.5
            self.mean_reversion_strength *= 2.0
        elif self.regime == MarketRegime.BUBBLE:
            self.drift_adjustment = 5.0
            self.volatility_adjustment = 2.5
            self.momentum_factor *= 3.0
        else:  # SIDEWAYS
            self.drift_adjustment = 0.0
            self.volatility_adjustment = 1.0
            
    def _set_asset_parameters(self):
        """Set parameters specific to asset class."""
        if self.asset_class == AssetClass.CRYPTO:
            self.base_volatility *= 3.0
            self.gap_probability *= 2.0
            self.volume_std *= 2.0
            self.jump_intensity *= 3.0
            self.bid_ask_spread *= 2.0
        elif self.asset_class == AssetClass.FOREX:
            self.base_volatility *= 0.5
            self.gap_probability *= 0.5
            self.bid_ask_spread *= 0.5
            self.tick_size = 0.0001  # 1 pip for major pairs
        elif self.asset_class == AssetClass.COMMODITY:
            self.base_volatility *= 1.5
            self.seasonality_amplitude *= 2.0
            self.mean_reversion_strength *= 1.5
        elif self.asset_class == AssetClass.ETF:
            self.base_volatility *= 0.8
            self.volume_std *= 1.5
            self.market_impact *= 0.5
        elif self.asset_class == AssetClass.BOND:
            self.base_volatility *= 0.3
            self.mean_reversion_strength *= 2.0
            self.jump_intensity *= 0.5
        elif self.asset_class == AssetClass.INDEX:
            self.base_volatility *= 0.7
            self.momentum_factor *= 1.5
            self.correlation_decay *= 1.2
            
    def _update_volatility(self, last_return: float) -> float:
        """Update volatility using GARCH-like dynamics."""
        target_vol = self.base_volatility * (1 + 5 * abs(last_return))
        self.current_volatility = (self.volatility_clustering * self.current_volatility +
                                 (1 - self.volatility_clustering) * target_vol)
        return self.current_volatility * self.volatility_adjustment
    
    def _apply_seasonality(self, dates: pd.DatetimeIndex, values: np.ndarray) -> np.ndarray:
        """Apply seasonal patterns to the data."""
        if self.seasonality_amplitude > 0:
            seasonal_factor = 1 + self.seasonality_amplitude * np.sin(2 * np.pi * 
                (dates.dayofyear / 365.25 + dates.hour / 24))
            return values * seasonal_factor
        return values
    
    def _generate_prices(self, 
                        start_price: float,
                        periods: int,
                        dates: pd.DatetimeIndex) -> Dict[str, np.ndarray]:
        """Generate daily OHLC prices using enhanced model."""
        # Initialize arrays
        returns = np.zeros(periods)
        close_prices = np.zeros(periods)
        open_prices = np.zeros(periods)
        high_prices = np.zeros(periods)
        low_prices = np.zeros(periods)
        
        # Set initial prices
        close_prices[0] = start_price
        open_prices[0] = start_price
        
        # Generate initial candle
        intraday_vol = self.base_volatility / np.sqrt(4)
        high_prices[0] = open_prices[0] * (1 + abs(np.random.normal(0, intraday_vol)))
        low_prices[0] = open_prices[0] * (1 - abs(np.random.normal(0, intraday_vol)))
        
        # Generate price series with volatility clustering and gaps
        for i in range(1, periods):
            # Add potential gap
            gap = 0
            if np.random.random() < self.gap_probability:
                gap = np.random.uniform(-self.max_gap_size, self.max_gap_size)
            
            # Generate return with current volatility
            volatility = self._update_volatility(returns[i-1])
            returns[i] = np.random.normal(
                self.base_drift * self.drift_adjustment,
                volatility
            ) + gap
            
            # Calculate close price
            close_prices[i] = close_prices[i-1] * np.exp(returns[i])
            
            # Set open price to previous close (with small possible gap)
            if gap != 0:
                open_prices[i] = close_prices[i-1] * np.exp(gap)
            else:
                open_prices[i] = close_prices[i-1]
            
            # Generate high and low with respect to open and close
            price_range = volatility * np.random.uniform(0.5, 1.5)
            if close_prices[i] > open_prices[i]:
                high_prices[i] = close_prices[i] * (1 + abs(np.random.normal(0, price_range)))
                low_prices[i] = open_prices[i] * (1 - abs(np.random.normal(0, price_range)))
            else:
                high_prices[i] = open_prices[i] * (1 + abs(np.random.normal(0, price_range)))
                low_prices[i] = close_prices[i] * (1 - abs(np.random.normal(0, price_range)))
            
            # Ensure high is highest and low is lowest
            high_prices[i] = max(high_prices[i], open_prices[i], close_prices[i])
            low_prices[i] = min(low_prices[i], open_prices[i], close_prices[i])
        
        # Apply seasonality
        close_prices = self._apply_seasonality(dates, close_prices)
        open_prices = self._apply_seasonality(dates, open_prices)
        high_prices = self._apply_seasonality(dates, high_prices)
        low_prices = self._apply_seasonality(dates, low_prices)
        
        return {
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'returns': returns
        }
    
    def _generate_volume(self, periods: int, returns: np.ndarray) -> np.ndarray:
        """Generate trading volume data with price-volume correlation."""
        # Base volume
        volume = np.maximum(
            np.random.normal(self.volume_mean, 
                           self.volume_std, 
                           size=periods),
            self.volume_mean * 0.1
        )
        
        # Add correlation with absolute returns
        if self.volume_price_corr > 0:
            abs_returns = np.abs(returns)
            volume = volume * (1 + self.volume_price_corr * 
                             (abs_returns - np.mean(abs_returns)) / np.std(abs_returns))
            
        return np.maximum(volume, self.volume_mean * 0.1).astype(int)
    
    def get_params_string(self) -> str:
        """Get a formatted string of current parameters."""
        params = [
            f"Regime: {self.regime.value}",
            f"Asset: {self.asset_class.value}",
            f"Market: {self.market_hours.value}",
            f"Base Vol: {self.base_volatility:.3f}",
            f"Drift: {self.base_drift*self.drift_adjustment:.3f}",
            f"Vol Cluster: {self.volatility_clustering:.1f}",
            f"Gap Prob: {self.gap_probability:.2f}",
            f"Momentum: {self.momentum_factor:.2f}",
            f"Mean Rev: {self.mean_reversion_strength:.2f}",
            f"Jump Int: {self.jump_intensity:.2f}"
        ]
        return "\n".join(params)

    def generate(self,
                periods: int = 252,  
                start_date: Optional[Union[str, datetime]] = None,
                start_price: float = 100.0,
                frequency: str = 'D',  
                symbol: str = 'SAMPLE') -> pd.DataFrame:
        """
        Generate OHLCV data with enhanced features.
        
        Args:
            periods: Number of periods to generate
            start_date: Starting date (defaults to today if None)
            start_price: Initial price
            frequency: Time series frequency ('D' for daily, 'H' for hourly, 'min' for minutes)
            symbol: Stock symbol or identifier
            
        Returns:
            DataFrame with OHLCV data
        """
        # Store generation parameters
        self.last_params = {
            'periods': periods,
            'start_date': start_date,
            'start_price': start_price,
            'frequency': frequency,
            'symbol': symbol
        }
        
        # Handle start date
        if start_date is None:
            start_date = datetime.now()
        elif isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
            
        # Generate date range based on frequency
        if frequency == 'D':
            dates = pd.date_range(start=start_date, 
                                periods=periods, 
                                freq='B')  # Business days
        elif frequency == 'H':
            dates = pd.date_range(start=start_date,
                                periods=periods,
                                freq='H')
        elif frequency.endswith('min'):
            try:
                minutes = int(frequency[:-3])  # Extract number from '5min', '15min', etc.
                dates = pd.date_range(start=start_date,
                                    periods=periods,
                                    freq=f'{minutes}min')
            except ValueError:
                raise ValueError(f"Invalid minute frequency format: {frequency}. Use format like '5min', '15min', etc.")
        else:
            raise ValueError(f"Unsupported frequency: {frequency}. Use 'D', 'H', or '[number]min'")
            
        # Scale volatility and drift based on frequency
        if frequency != 'D':
            # Get number of periods per day
            if frequency == 'H':
                periods_per_day = 24
            else:
                minutes = int(frequency[:-3])
                periods_per_day = 24 * 60 // minutes
                
            # Scale parameters
            vol_scale = 1 / np.sqrt(periods_per_day)
            drift_scale = 1 / periods_per_day
            
            orig_vol = self.base_volatility
            orig_drift = self.base_drift
            
            self.base_volatility *= vol_scale
            self.base_drift *= drift_scale
            
        # Generate OHLCV data
        prices = self._generate_prices(start_price, periods, dates)
        volume = self._generate_volume(periods, prices['returns'])
        
        # Restore original parameters if they were scaled
        if frequency != 'D':
            self.base_volatility = orig_vol
            self.base_drift = orig_drift
        
        # Create DataFrame
        df = pd.DataFrame({
            'symbol': symbol,
            'datetime': dates,
            'open': prices['open'],
            'high': prices['high'],
            'low': prices['low'],
            'close': prices['close'],
            'volume': volume
        })
        
        # Store the generated data
        self.data = df.set_index(['datetime', 'symbol'])
        
        return self.data

    def save(self, filename: str, directory: str = "output") -> None:
        """Save the most recently generated data to a CSV file.
        
        Args:
            filename: Name of the file (without .csv extension)
            directory: Directory to save the file in (default: "output")
        
        Raises:
            ValueError: If no data has been generated yet
        """
        if self.data is None:
            raise ValueError("No data has been generated yet. Call generate() first.")
        
        # Create the output directory if it doesn't exist
        import os
        os.makedirs(directory, exist_ok=True)
        
        # Prepare the data for saving
        data_to_save = self.data.copy()
        data_to_save.index = data_to_save.index.get_level_values(0)  # Get datetime from MultiIndex
        data_to_save = data_to_save.reset_index()
        data_to_save.rename(columns={'index': 'datetime'}, inplace=True)
        
        # Save to CSV
        filepath = os.path.join(directory, f"{filename}")
        data_to_save[['datetime', 'open', 'high', 'low', 'close', 'volume']].to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")
