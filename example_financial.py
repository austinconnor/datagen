from datagen.financial import OHLCVGenerator, MarketRegime, AssetClass, MarketHours

def main():
    # Example configurations
    regimes = {
        'bull': MarketRegime.BULL,
        'bear': MarketRegime.BEAR,
        'crash': MarketRegime.CRASH,
        'bubble': MarketRegime.BUBBLE,
        'recovery': MarketRegime.RECOVERY
    }
    
    timeframes = ['D', 'H', '15min', '5min', '1min']
    
    # Calculate periods for each timeframe to generate roughly a month of data
    periods = {
        'D': 30,            # 30 days
        'H': 30 * 24,       # 30 days of hourly data
        '15min': 30 * 24 * 4,   # 30 days of 15-min data
        '5min': 30 * 24 * 12,   # 30 days of 5-min data
        '1min': 30 * 24 * 60    # 30 days of 1-min data
    }
    
    # Generate data for each regime and timeframe
    for regime_name, regime in regimes.items():
        print(f"\nGenerating {regime_name} market data:")
        for timeframe in timeframes:
            # Create generator with minimal parameters
            gen = OHLCVGenerator(
                regime=regime,
                asset_class=AssetClass.STOCK,
                market_hours=MarketHours.US
            )
            
            # Generate data and save to CSV
            gen.generate(
                periods=periods[timeframe],
                start_date="2024-01-01",
                start_price=100,
                frequency=timeframe
            )
            
            # Save using the generator's save method
            filename = f"{regime_name}_market_{timeframe}".lower().replace('.', '')
            gen.save_to_csv(filename)

if __name__ == "__main__":
    main()
