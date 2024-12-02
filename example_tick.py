from datagen.financial import TickGenerator, MarketRegime, AssetClass, MarketHours
import os

def main():
    """Generate and save tick data for a bear market."""
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    # Create generator with bear market parameters
    gen = TickGenerator(
        regime=MarketRegime.BEAR,
        asset_class=AssetClass.STOCK,
        market_hours=MarketHours.US,
        base_volatility=0.02,
        base_drift=-0.0001,  # Increased 10x, negative for bear market
        ticks_per_day=390000,  # About 1000 ticks per minute for 6.5 hours
        mean_reversion_strength=0.002,  # Increased 2x
        momentum_factor=0.02,  # Increased 2x
        jump_intensity=0.0005,  # Increased 5x
        jump_size_mean=-0.005,  # Increased 5x, negative for bear market
        jump_size_std=0.002,  # Increased 4x
        bid_ask_spread=0.0001,
        volatility_of_volatility=0.2  # Added for more dynamic volatility
    )
    
    # Generate 10 days of tick data
    gen.generate(days=10, start_date="2024-01-01", start_price=100.0)
    
    # Save raw tick data
    gen.save("bear_market_ticks")
    
    # Save at various timeframes
    for freq in ["1min", "5min", "15min", "30min", "1H", "1D"]:
        print(f"Saving {freq} data...")
        gen.save(f"bear_market_{freq}", freq=freq)

if __name__ == "__main__":
    main()
