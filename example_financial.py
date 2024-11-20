from datagen import OHLCVGenerator
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

def plot_ohlc(df: pd.DataFrame, title: str):
    """Plot OHLC data with volume."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1], sharex=True)
    
    # Plot OHLC
    df['close'].plot(ax=ax1, label='Close', color='black', alpha=0.7)
    ax1.fill_between(df.index.get_level_values(0), df['low'], df['high'], alpha=0.3, color='gray')
    ax1.set_title(title)
    ax1.set_ylabel('Price')
    ax1.grid(True)
    
    # Plot volume
    ax2.bar(df.index.get_level_values(0), df['volume'], alpha=0.5, color='blue')
    ax2.set_ylabel('Volume')
    ax2.grid(True)
    
    plt.tight_layout()
    return fig

def main():
    # Example 1: Generate daily data for one stock
    generator = OHLCVGenerator(
        volatility=0.02,  # 2% daily volatility
        drift=0.0001,     # slight upward trend
        volume_mean=1000000
    )
    
    daily_data = generator.generate(
        periods=252,  # One trading year
        start_date='2023-01-01',
        start_price=100.0,
        symbol='AAPL'
    )
    
    print("\nGenerated daily OHLCV data:")
    print(daily_data.head())
    
    # Plot daily data
    fig1 = plot_ohlc(daily_data, 'Daily OHLCV Data - AAPL')
    
    # Example 2: Generate hourly data for multiple stocks
    # Create generators with different characteristics
    generators = {
        'TECH': OHLCVGenerator(volatility=0.015, drift=0.0002),  # Growth stock
        'UTIL': OHLCVGenerator(volatility=0.008, drift=0.0001),  # Stable stock
        'SPEC': OHLCVGenerator(volatility=0.03, drift=-0.0001)   # Volatile stock
    }
    
    # Generate data for each stock
    start_date = datetime.now() - timedelta(days=5)
    hourly_dfs = []
    
    for symbol, gen in generators.items():
        df = gen.generate(
            periods=24 * 5,  # 5 days of hourly data
            start_date=start_date,
            frequency='H',
            start_price=100.0,
            symbol=symbol
        )
        hourly_dfs.append(df)
    
    # Combine all stocks
    hourly_data = pd.concat(hourly_dfs)
    
    print("\nGenerated hourly OHLCV data:")
    print(hourly_data.head())
    
    # Plot hourly data for one stock
    tech_data = hourly_data.xs('TECH', level='symbol')
    fig2 = plot_ohlc(tech_data, 'Hourly OHLCV Data - TECH Stock')
    
    plt.show()

if __name__ == '__main__':
    main()
