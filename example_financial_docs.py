from datagen.financial_docs import IncomeStatementGenerator, ReportingPeriod
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 120)
pd.set_option('display.float_format', lambda x: f'${x:,.0f}' if abs(x) >= 1 else f'${x:.3f}')

def main():
    # Create generators for different industries
    industries = ['technology', 'retail', 'manufacturing', 'healthcare']
    base_revenues = {
        'technology': 5000000000,    # $5B - Large tech company
        'retail': 20000000000,       # $20B - Large retailer
        'manufacturing': 8000000000,  # $8B - Manufacturing company
        'healthcare': 12000000000     # $12B - Healthcare company
    }
    
    for industry in industries:
        print(f"\n=== {industry.title()} Company Income Statement ===")
        
        # Create generator with industry-specific parameters
        generator = IncomeStatementGenerator(
            base_revenue=base_revenues[industry],
            industry=industry,
            volatility=0.05,
            seed=42
        )
        
        # Generate quarterly statements for 2 years
        quarterly_data = generator.generate(
            periods=8,
            period_type=ReportingPeriod.QUARTERLY,
            start_date='2024-01-01'
        )
        
        # Display key metrics
        print("\nQuarterly Performance:")
        metrics_display = quarterly_data[[
            'period', 'revenue', 'gross_profit', 'operating_income', 
            'net_income', 'gross_margin', 'operating_margin', 'net_margin'
        ]]
        print(metrics_display.to_string())
        
        # Save to both CSV and Excel
        generator.save(f"{industry}_income_statement")
        generator.to_excel(f"{industry}_income_statement")
        
        # Generate annual statements for comparison
        print("\nAnnual Performance:")
        annual_data = generator.generate(
            periods=3,
            period_type=ReportingPeriod.ANNUAL,
            start_date='2024-01-01'
        )
        
        metrics_display = annual_data[[
            'period', 'revenue', 'gross_profit', 'operating_income', 
            'net_income', 'gross_margin', 'operating_margin', 'net_margin'
        ]]
        print(metrics_display.to_string())

if __name__ == '__main__':
    main()
