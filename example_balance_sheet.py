from datagen.financial_docs import BalanceSheetGenerator, ReportingPeriod
from datetime import date

def main():
    # Create a balance sheet generator for a technology company
    # Starting with $100M in revenue
    generator = BalanceSheetGenerator(
        base_revenue=100_000_000,  # $100M
        industry='technology',
        growth_rate=0.20,  # 20% growth
        seed=42  # for reproducibility
    )
    
    # Generate quarterly balance sheets for the past year
    statements = generator.generate(
        periods=4,
        period_type=ReportingPeriod.QUARTERLY,
        start_date=date.today()
    )
    
    # Save as PDF
    generator.to_pdf("tech_company_balance_sheet")
    
    print("Generated balance sheet in PDF format in the output directory")
    
    # Print some key metrics from the latest period
    latest = statements.iloc[-1]
    print("\nKey Metrics (Latest Period):")
    print(f"Total Assets: {generator.format_number(latest['total_assets'])}")
    print(f"Current Ratio: {latest['current_ratio']:.2f}")
    print(f"Debt/Equity Ratio: {latest['debt_to_equity']:.2f}")
    print(f"Asset Turnover: {latest['asset_turnover']:.2f}")

if __name__ == "__main__":
    main()
