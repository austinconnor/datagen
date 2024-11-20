import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Union, List

class OHLCVGenerator:
    """Generator for OHLCV (Open, High, Low, Close, Volume) time series data."""
    
    def __init__(self, 
                 volatility: float = 0.02,
                 drift: float = 0.0001,
                 volume_mean: float = 1000000,
                 volume_std: float = 500000):
        """
        Initialize the OHLCV generator.
        
        Args:
            volatility: Daily volatility of the price
            drift: Daily drift (trend) in the price
            volume_mean: Mean daily volume
            volume_std: Standard deviation of daily volume
        """
        self.volatility = volatility
        self.drift = drift
        self.volume_mean = volume_mean
        self.volume_std = volume_std
    
    def _generate_prices(self, 
                        start_price: float,
                        periods: int) -> Dict[str, np.ndarray]:
        """Generate daily OHLC prices using geometric Brownian motion."""
        # Generate daily returns with drift and volatility
        daily_returns = np.random.normal(self.drift, 
                                       self.volatility, 
                                       size=periods)
        
        # Generate close prices
        close_prices = start_price * np.exp(np.cumsum(daily_returns))
        
        # Generate intraday variations for open, high, low
        intraday_vol = self.volatility / np.sqrt(4)  # reduced volatility for intraday
        open_var = np.random.normal(0, intraday_vol, size=periods)
        high_var = np.abs(np.random.normal(0, intraday_vol, size=periods))
        low_var = -np.abs(np.random.normal(0, intraday_vol, size=periods))
        
        # Calculate OHLC prices
        open_prices = close_prices * np.exp(open_var)
        high_prices = close_prices * np.exp(np.maximum(high_var, np.maximum(open_var, 0)))
        low_prices = close_prices * np.exp(np.minimum(low_var, np.minimum(open_var, 0)))
        
        return {
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices
        }
    
    def _generate_volume(self, periods: int) -> np.ndarray:
        """Generate trading volume data."""
        return np.maximum(
            np.random.normal(self.volume_mean, 
                           self.volume_std, 
                           size=periods),
            self.volume_mean * 0.1  # Ensure minimum volume
        ).astype(int)
    
    def generate(self,
                periods: int = 252,  # One trading year
                start_date: Optional[Union[str, datetime]] = None,
                start_price: float = 100.0,
                frequency: str = 'D',  # D for daily
                symbol: str = 'SAMPLE') -> pd.DataFrame:
        """
        Generate OHLCV data.
        
        Args:
            periods: Number of periods to generate
            start_date: Starting date (defaults to today if None)
            start_price: Initial price
            frequency: Time series frequency ('D' for daily, 'H' for hourly)
            symbol: Stock symbol or identifier
            
        Returns:
            DataFrame with OHLCV data
        """
        # Handle start date
        if start_date is None:
            start_date = datetime.now()
        elif isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
            
        # Generate date range
        if frequency == 'D':
            dates = pd.date_range(start=start_date, 
                                periods=periods, 
                                freq='B')  # Business days
        elif frequency == 'H':
            dates = pd.date_range(start=start_date,
                                periods=periods,
                                freq='H')
        else:
            raise ValueError(f"Unsupported frequency: {frequency}")
            
        # Generate OHLCV data
        prices = self._generate_prices(start_price, periods)
        volume = self._generate_volume(periods)
        
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
        
        return df.set_index(['datetime', 'symbol'])
