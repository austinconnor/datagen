from typing import Dict, Optional, Union
from datetime import date
import random
import os
import pandas as pd

from .base import FinancialDocument, ReportingPeriod
from .pdf_utils import (
    create_pdf_doc, get_title_style, create_footnote,
    style_financial_table, Paragraph, Table, Spacer, inch
)

class BalanceSheetGenerator(FinancialDocument):
    """Generator for realistic balance sheets."""
    
    def _calculate_metrics(self, revenue: float) -> Dict[str, float]:
        """Calculate various balance sheet metrics based on revenue."""
        # Get balance sheet ratios from industry profile
        current_ratio_min, current_ratio_max = self.metrics.profile['current_ratio']
        debt_equity_min, debt_equity_max = self.metrics.profile['debt_to_equity']
        inventory_turnover_min, inventory_turnover_max = self.metrics.profile['inventory_turnover']
        asset_turnover_min, asset_turnover_max = self.metrics.profile['asset_turnover']
        
        # Calculate key ratios with some random variation
        current_ratio = random.uniform(current_ratio_min, current_ratio_max)
        debt_to_equity = random.uniform(debt_equity_min, debt_equity_max)
        inventory_turnover = random.uniform(inventory_turnover_min, inventory_turnover_max)
        asset_turnover = random.uniform(asset_turnover_min, asset_turnover_max)
        
        # Calculate assets
        total_assets = revenue / asset_turnover
        
        # Current Assets
        inventory = (revenue / inventory_turnover) * (1 + random.uniform(-0.1, 0.1))
        accounts_receivable = revenue * random.uniform(0.15, 0.25)  # 15-25% of revenue
        cash = total_assets * random.uniform(0.1, 0.2)  # 10-20% of total assets
        other_current_assets = total_assets * random.uniform(0.05, 0.1)
        total_current_assets = cash + accounts_receivable + inventory + other_current_assets
        
        # Non-current Assets
        ppe = total_assets * random.uniform(0.3, 0.5)  # 30-50% of total assets
        intangible_assets = total_assets * random.uniform(0.1, 0.2)
        other_non_current_assets = total_assets - total_current_assets - ppe - intangible_assets
        total_non_current_assets = ppe + intangible_assets + other_non_current_assets
        
        # Verify total assets
        total_assets = total_current_assets + total_non_current_assets
        
        # Calculate liabilities based on debt-to-equity ratio
        equity = total_assets / (1 + debt_to_equity)
        total_liabilities = total_assets - equity
        
        # Current Liabilities (based on current ratio)
        total_current_liabilities = total_current_assets / current_ratio
        accounts_payable = total_current_liabilities * random.uniform(0.3, 0.4)
        short_term_debt = total_current_liabilities * random.uniform(0.2, 0.3)
        other_current_liabilities = total_current_liabilities - accounts_payable - short_term_debt
        
        # Non-current Liabilities
        long_term_debt = total_liabilities - total_current_liabilities
        
        # Equity components
        retained_earnings = equity * random.uniform(0.6, 0.8)
        common_stock = equity - retained_earnings
        
        return {
            'cash': cash,
            'accounts_receivable': accounts_receivable,
            'inventory': inventory,
            'other_current_assets': other_current_assets,
            'total_current_assets': total_current_assets,
            'ppe': ppe,
            'intangible_assets': intangible_assets,
            'other_non_current_assets': other_non_current_assets,
            'total_non_current_assets': total_non_current_assets,
            'total_assets': total_assets,
            'accounts_payable': accounts_payable,
            'short_term_debt': short_term_debt,
            'other_current_liabilities': other_current_liabilities,
            'total_current_liabilities': total_current_liabilities,
            'long_term_debt': long_term_debt,
            'total_liabilities': total_liabilities,
            'common_stock': common_stock,
            'retained_earnings': retained_earnings,
            'total_equity': equity,
            'current_ratio': current_ratio,
            'debt_to_equity': debt_to_equity
        }
    
    def generate(self,
                periods: int = 4,
                period_type: ReportingPeriod = ReportingPeriod.QUARTERLY,
                start_date: Optional[Union[str, date]] = None):
        """
        Generate a balance sheet time series.
        
        Args:
            periods: Number of periods to generate
            period_type: Type of reporting period (quarterly/annual)
            start_date: Starting date for the time series
            
        Returns:
            DataFrame with balance sheet data
        """
        # Generate base revenue
        revenue = self._generate_revenue(periods, period_type)
        
        # Calculate metrics for each period
        data = []
        for period_revenue in revenue:
            metrics = self._calculate_metrics(period_revenue)
            data.append(metrics)
        
        # Convert to DataFrame
        self.data = pd.DataFrame(data)
        
        # Add period labels
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        elif start_date is None:
            start_date = pd.Timestamp.now()
        
        if period_type == ReportingPeriod.QUARTERLY:
            dates = pd.date_range(start=start_date, periods=periods, freq='Q')
            self.data['period'] = [f"Q{d.quarter} {d.year}" for d in dates]
        else:
            dates = pd.date_range(start=start_date, periods=periods, freq='Y')
            self.data['period'] = [str(d.year) for d in dates]
        
        self.data['date'] = dates
        
        # Store parameters
        self.last_params = {
            'period_type': period_type,
            'start_date': start_date
        }
        
        return self.data
    
    def to_pdf(self, filename: str, directory: str = "output") -> None:
        """Save the balance sheet to a professionally formatted PDF."""
        if self.data is None:
            raise ValueError("No data has been generated yet. Call generate() first.")
        
        # Create the output directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, f"{filename}.pdf")
        
        # Create document
        doc = create_pdf_doc(filepath)
        
        # Create content
        content = []
        
        # Title
        period_type = self.last_params.get('period_type', ReportingPeriod.QUARTERLY)
        title = f"Balance Sheet ({period_type.value.title()})"
        content.append(Paragraph(title, get_title_style()))
        
        # Headers and sections for the two tables
        assets_sections = {
            'Current Assets': [
                'Cash',
                'A/R',
                'Inv.',
                'Other CA',
                'Tot. CA'
            ],
            'Non-Current Assets': [
                'PPE',
                'Intang.',
                'Other NCA',
                'Tot. NCA',
                'Tot. Assets'
            ]
        }
        
        liab_equity_sections = {
            'Current Liab.': [
                'A/P',
                'ST Debt',
                'Other CL',
                'Tot. CL'
            ],
            'Non-Current Liab.': [
                'LT Debt',
                'Tot. Liab.'
            ],
            'Equity': [
                'Stock',
                'Ret. Earn.',
                'Tot. Equity'
            ]
        }
        
        # Field mappings for both tables
        field_mapping = {
            'Cash': 'cash',
            'A/R': 'accounts_receivable',
            'Inv.': 'inventory',
            'Other CA': 'other_current_assets',
            'Tot. CA': 'total_current_assets',
            'PPE': 'ppe',
            'Intang.': 'intangible_assets',
            'Other NCA': 'other_non_current_assets',
            'Tot. NCA': 'total_non_current_assets',
            'Tot. Assets': 'total_assets',
            'A/P': 'accounts_payable',
            'ST Debt': 'short_term_debt',
            'Other CL': 'other_current_liabilities',
            'Tot. CL': 'total_current_liabilities',
            'LT Debt': 'long_term_debt',
            'Tot. Liab.': 'total_liabilities',
            'Stock': 'common_stock',
            'Ret. Earn.': 'retained_earnings',
            'Tot. Equity': 'total_equity'
        }

        def create_table_data(sections, include_period=True):
            """Helper function to create table data for a section."""
            headers = ['Period'] if include_period else []
            for section_fields in sections.values():
                headers.extend(section_fields)
            
            table_data = [headers]
            
            for _, row in self.data.sort_values('date').iterrows():
                data_row = [row['period']] if include_period else []
                for section_fields in sections.values():
                    for field in section_fields:
                        field_name = field_mapping[field]
                        value = row[field_name]
                        data_row.append(self.format_number(value))
                table_data.append(data_row)
            
            return table_data

        # Create Assets table
        assets_data = create_table_data(assets_sections)
        assets_col_widths = [0.8*inch] + [0.9*inch] * (len(assets_data[0]) - 1)
        assets_table = Table(assets_data, repeatRows=1, colWidths=assets_col_widths)
        assets_table = style_financial_table(assets_table, assets_data, assets_sections)
        
        # Create Liabilities & Equity table
        liab_equity_data = create_table_data(liab_equity_sections)
        liab_equity_col_widths = [0.8*inch] + [0.9*inch] * (len(liab_equity_data[0]) - 1)
        liab_equity_table = Table(liab_equity_data, repeatRows=1, colWidths=liab_equity_col_widths)
        liab_equity_table = style_financial_table(liab_equity_table, liab_equity_data, liab_equity_sections)
        
        # Add tables to content
        content.append(assets_table)
        content.append(Spacer(1, 20))  # Add space between tables
        content.append(liab_equity_table)
        
        # Add margins
        content.append(Spacer(1, 20))
        
        # Add footnote
        content.append(create_footnote())
        
        # Build PDF
        doc.build(content)
        print(f"Balance sheet saved to {filepath}")
