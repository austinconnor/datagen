import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Union, List, Tuple
from enum import Enum
from tqdm import tqdm

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

class TickGenerator:
    """Generator for tick-level financial data with realistic microstructure."""
    
    def __init__(self, 
                 base_volatility: float = 0.002,
                 base_drift: float = 0.00001,
                 tick_volume_mean: float = 100,
                 tick_volume_std: float = 50,
                 regime: MarketRegime = MarketRegime.SIDEWAYS,
                 asset_class: AssetClass = AssetClass.STOCK,
                 market_hours: MarketHours = MarketHours.US,
                 volatility_clustering: float = 0.7,
                 volume_price_corr: float = 0.4,
                 gap_probability: float = 0.001,
                 max_gap_size: float = 0.005,
                 mean_reversion_strength: float = 0.002,
                 momentum_factor: float = 0.02,
                 jump_intensity: float = 0.0005,
                 jump_size_mean: float = 0.005,
                 jump_size_std: float = 0.002,
                 volatility_of_volatility: float = 0.2,
                 correlation_decay: float = 0.98,
                 bid_ask_spread: float = 0.0001,
                 tick_size: Optional[float] = None,
                 market_impact: float = 0.01,
                 ticks_per_day: int = 390000):
        """Initialize the tick generator with market microstructure parameters."""
        self.base_volatility = base_volatility
        self.base_drift = base_drift
        self.tick_volume_mean = tick_volume_mean
        self.tick_volume_std = tick_volume_std
        self.regime = regime
        self.asset_class = asset_class
        self.market_hours = market_hours
        self.volatility_clustering = volatility_clustering
        self.volume_price_corr = volume_price_corr
        self.gap_probability = gap_probability
        self.max_gap_size = max_gap_size
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
        self.ticks_per_day = ticks_per_day
        
        # Set regime-specific parameters
        self._set_regime_parameters()
        
        # Set asset-specific parameters
        self._set_asset_parameters()
        
        # Initialize state variables
        self.current_volatility = self.base_volatility
        self.current_spread = self.bid_ask_spread
        
        # Store the most recently generated data
        self.tick_data = None
    
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
            self.tick_volume_std *= 2.0
            self.jump_intensity *= 3.0
            self.bid_ask_spread *= 2.0
        elif self.asset_class == AssetClass.FOREX:
            self.base_volatility *= 0.5
            self.gap_probability *= 0.5
            self.bid_ask_spread *= 0.5
            self.tick_size = 0.0001  # 1 pip for major pairs
        elif self.asset_class == AssetClass.COMMODITY:
            self.base_volatility *= 1.5
            self.mean_reversion_strength *= 1.5
        elif self.asset_class == AssetClass.ETF:
            self.base_volatility *= 0.8
            self.tick_volume_std *= 1.5
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
    
    def _generate_tick_prices(self, 
                            start_price: float,
                            num_ticks: int,
                            dates: pd.DatetimeIndex) -> Dict[str, np.ndarray]:
        """Generate tick-by-tick price data with microstructure effects."""
        # Initialize arrays
        mid_prices = np.zeros(num_ticks)
        bid_prices = np.zeros(num_ticks)
        ask_prices = np.zeros(num_ticks)
        spreads = np.zeros(num_ticks)
        volumes = np.zeros(num_ticks)
        
        # Set initial prices
        mid_prices[0] = start_price
        spreads[0] = self.current_spread * start_price
        bid_prices[0] = start_price - spreads[0]/2
        ask_prices[0] = start_price + spreads[0]/2
        
        # Generate tick-by-tick data
        for i in tqdm(range(1, num_ticks), desc="Generating tick data"):
            # Update volatility with bounds
            returns = (mid_prices[i-1] - mid_prices[max(0, i-2)]) / mid_prices[max(0, i-2)]
            returns = np.clip(returns, -0.005, 0.005)
            tick_vol = self._update_volatility(returns)
            tick_vol = min(tick_vol, 0.005)
            
            # Generate price innovation with mean reversion and momentum
            price_change = np.random.normal(
                self.base_drift * self.drift_adjustment / self.ticks_per_day,
                tick_vol / np.sqrt(self.ticks_per_day)
            )
            
            # Add mean reversion
            if i > 10:  # Need at least 10 points for mean reversion
                window = mid_prices[i-10:i]
                returns = np.diff(window) / window[:-1]
                returns = np.clip(returns, -0.005, 0.005)
                cumulative_return = np.sum(returns)
                mean_reversion = -self.mean_reversion_strength * cumulative_return
                mean_reversion = np.clip(mean_reversion, -0.001, 0.001)
                price_change += mean_reversion
            
            # Add momentum
            if i > 10:  # Need at least 10 points for momentum
                recent_return = (mid_prices[i-1] - mid_prices[i-10]) / mid_prices[i-10]
                recent_return = np.clip(recent_return, -0.005, 0.005)
                momentum = self.momentum_factor * recent_return
                momentum = np.clip(momentum, -0.001, 0.001)
                price_change += momentum
            
            # Add jumps
            if np.random.random() < self.jump_intensity / self.ticks_per_day:
                jump = np.random.normal(self.jump_size_mean, self.jump_size_std)
                jump = np.clip(jump, -0.005, 0.005)
                price_change += jump
            
            # Limit total price change
            price_change = np.clip(price_change, -0.005, 0.005)
            
            # Update mid price
            mid_prices[i] = mid_prices[i-1] * (1 + price_change)
            
            # Ensure price doesn't go too low
            mid_prices[i] = max(mid_prices[i], 0.01)
            
            # Update spread based on volatility and volume
            self.current_spread = np.clip(
                max(self.bid_ask_spread, self.bid_ask_spread * (1 + 5 * tick_vol)),
                0.00001,
                0.001
            )
            spreads[i] = self.current_spread * mid_prices[i]
            
            # Calculate bid/ask prices
            bid_prices[i] = mid_prices[i] - spreads[i]/2
            ask_prices[i] = mid_prices[i] + spreads[i]/2
            
            # Round to tick size if specified
            if self.tick_size is not None:
                bid_prices[i] = np.round(bid_prices[i] / self.tick_size) * self.tick_size
                ask_prices[i] = np.round(ask_prices[i] / self.tick_size) * self.tick_size
                mid_prices[i] = (bid_prices[i] + ask_prices[i]) / 2
            
            # Generate volume for this tick (ensure it's positive)
            vol_multiplier = max(1 + 3 * abs(price_change), 0.1)
            volumes[i] = max(1, int(np.random.normal(
                self.tick_volume_mean, 
                self.tick_volume_std * vol_multiplier
            )))
        
        return {
            'datetime': dates,
            'mid': mid_prices,
            'bid': bid_prices,
            'ask': ask_prices,
            'spread': spreads,
            'volume': volumes.astype(int)
        }
    
    def _aggregate_to_ohlcv(self, tick_data: pd.DataFrame, freq: str) -> pd.DataFrame:
        """Aggregate tick data to OHLCV format at specified frequency."""
        # Resample to desired frequency without filling gaps
        ohlc = tick_data.groupby(pd.Grouper(key='datetime', freq=freq, label='left')).agg({
            'mid': ['first', 'max', 'min', 'last'],
            'volume': 'sum'
        }).dropna()  # Remove any NaN rows
        
        # Flatten column names
        ohlc.columns = ['open', 'high', 'low', 'close', 'volume']
        
        return ohlc.reset_index()
    
    def generate(self,
                days: int = 1,
                start_date: Optional[Union[str, datetime]] = None,
                start_price: float = 100.0,
                symbol: str = 'APPL') -> pd.DataFrame:
        """
        Generate tick data for the specified number of days.
        
        Args:
            days: Number of days to generate
            start_date: Starting date (defaults to today if None)
            start_price: Initial price
            symbol: Stock symbol or identifier
            
        Returns:
            DataFrame with tick data
        """
        # Handle start date
        if start_date is None:
            start_date = datetime.now()
        elif isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        
        # Calculate total number of ticks
        num_ticks = int(days * self.ticks_per_day)
        
        # Generate timestamps based on market hours and volume patterns
        all_timestamps = []
        current_date = start_date
        
        for _ in range(days):
            # US Market hours: 9:30 AM - 4:00 PM EST
            market_open = pd.Timestamp(current_date).replace(hour=9, minute=30)
            market_close = pd.Timestamp(current_date).replace(hour=16, minute=0)
            
            # Create base timestamps for the day
            day_timestamps = pd.date_range(market_open, market_close, periods=self.ticks_per_day//days)
            
            # Create volume profile (U-shaped pattern)
            time_of_day = (day_timestamps - market_open).total_seconds() / (market_close - market_open).total_seconds()
            volume_profile = 1 + 2 * (1 - 4 * (time_of_day - 0.5)**2)  # U-shaped curve
            
            # Add some randomness to volume profile
            volume_profile *= np.random.normal(1, 0.2, len(volume_profile))
            volume_profile = np.clip(volume_profile, 0.1, 3)  # Limit the range
            
            # Adjust timestamp density based on volume profile
            time_deltas = np.random.exponential(1/volume_profile)
            time_deltas = time_deltas / np.sum(time_deltas) * (market_close - market_open).total_seconds()
            day_timestamps = pd.to_datetime(market_open.value + (time_deltas.cumsum() * 1e9).astype(np.int64))
            
            # Filter out timestamps after market close
            day_timestamps = day_timestamps[day_timestamps <= market_close]
            
            all_timestamps.extend(day_timestamps)
            current_date += pd.Timedelta(days=1)
            
            # Skip weekends
            while current_date.weekday() > 4:  # 5 = Saturday, 6 = Sunday
                current_date += pd.Timedelta(days=1)
        
        timestamps = pd.DatetimeIndex(all_timestamps)
        num_ticks = len(timestamps)
        
        # Generate tick data
        tick_data = self._generate_tick_prices(start_price, num_ticks, timestamps)
        
        # Create DataFrame
        self.tick_data = pd.DataFrame(tick_data)
        
        return self.tick_data
    
    def save(self, 
             filename: str, 
             directory: str = "output",
             freq: Optional[str] = None) -> None:
        """
        Save the generated data to a CSV file.
        
        Args:
            filename: Name of the file (without .csv extension)
            directory: Directory to save the file in (default: "output")
            freq: Frequency to aggregate data to (e.g., '1min', '5min', '1H', '1D')
                 If None, saves raw tick data
        
        Raises:
            ValueError: If no data has been generated yet
        """
        if self.tick_data is None:
            raise ValueError("No data has been generated yet. Call generate() first.")
        
        # Create the output directory if it doesn't exist
        import os
        os.makedirs(directory, exist_ok=True)
        
        # Prepare the data for saving
        if freq is not None:
            # Aggregate to specified frequency
            data_to_save = self._aggregate_to_ohlcv(self.tick_data, freq)
        else:
            # Save raw tick data
            data_to_save = self.tick_data.copy()
        
        # Save to CSV
        filepath = os.path.join(directory, f"{filename}.csv")
        data_to_save.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")
