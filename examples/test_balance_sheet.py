from datagen.financial_docs import BalanceSheetGenerator, ReportingPeriod

# Create a balance sheet generator for a technology company
generator = BalanceSheetGenerator(
    base_revenue=10000000,  # $10M base revenue
    industry='technology',
    growth_rate=0.15,  # 15% growth rate
    volatility=0.1,
    seed=42  # For reproducibility
)

# Generate the balance sheet data
generator.generate(
    periods=4,
    period_type=ReportingPeriod.QUARTERLY,
    start_date='2023-01-01'
)

# Export to PDF
generator.to_pdf('sample_balance_sheet.pdf')
