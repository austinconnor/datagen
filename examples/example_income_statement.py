from datagen.financial_docs import IncomeStatementGenerator, ReportingPeriod
from datetime import date

def main():
    # Create an income statement generator for a technology company
    # Starting with $100M in revenue, 20% growth rate
    generator = IncomeStatementGenerator(
        base_revenue=100_000_000,  # $100M
        industry='technology',
        growth_rate=0.20,  # 20% growth
        seed=42  # for reproducibility
    )
    
    # Generate quarterly statements for the past year
    statements = generator.generate(
        periods=4,
        period_type=ReportingPeriod.QUARTERLY,
        start_date=date.today()
    )
    
    # Save as PDF
    generator.to_pdf("tech_company_income_statement")
    
    # Also save as Excel for comparison
    generator.to_excel("tech_company_income_statement")
    
    print("Generated income statements in both PDF and Excel formats in the output directory")

if __name__ == "__main__":
    main()
