import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Union
from datetime import datetime, date
import random
from enum import Enum

class ReportingPeriod(Enum):
    """Enum for different reporting periods."""
    ANNUAL = 'annual'
    QUARTERLY = 'quarterly'
    TTM = 'ttm'  # Trailing Twelve Months

class DocumentType(Enum):
    """Enum for different financial document types."""
    INCOME_STATEMENT = 'income_statement'
    BALANCE_SHEET = 'balance_sheet'
    CASH_FLOW = 'cash_flow'

class FinancialMetrics:
    """Base class for industry-specific financial metrics and patterns."""
    
    def __init__(self, industry: str):
        self.industry = industry.lower()
        self.profile = self.get_profile(industry)
    
    @staticmethod
    def get_profile(industry: str) -> Dict[str, Dict[str, tuple]]:
        """Get comprehensive financial profile for a specific industry."""
        PROFILES = {
            'technology': {
                # Income Statement Metrics
                'gross_margin': (0.50, 0.85),
                'rd_to_revenue': (0.10, 0.25),
                'sales_growth': (0.15, 0.40),
                'operating_margin': (0.15, 0.35),
                # Balance Sheet Metrics
                'current_ratio': (1.5, 3.0),
                'debt_to_equity': (0.1, 0.5),
                'inventory_turnover': (8, 12),
                'asset_turnover': (0.7, 1.2),
                # Cash Flow Metrics
                'operating_cash_flow_margin': (0.15, 0.30),
                'capex_to_revenue': (0.05, 0.15),
                'fcf_margin': (0.10, 0.25)
            },
            'retail': {
                # Income Statement Metrics
                'gross_margin': (0.25, 0.45),
                'rd_to_revenue': (0.01, 0.03),
                'sales_growth': (0.02, 0.10),
                'operating_margin': (0.05, 0.12),
                # Balance Sheet Metrics
                'current_ratio': (1.2, 2.0),
                'debt_to_equity': (0.3, 0.8),
                'inventory_turnover': (4, 8),
                'asset_turnover': (1.5, 2.5),
                # Cash Flow Metrics
                'operating_cash_flow_margin': (0.05, 0.12),
                'capex_to_revenue': (0.02, 0.08),
                'fcf_margin': (0.03, 0.08)
            },
            'manufacturing': {
                # Income Statement Metrics
                'gross_margin': (0.20, 0.40),
                'rd_to_revenue': (0.03, 0.08),
                'sales_growth': (0.05, 0.15),
                'operating_margin': (0.08, 0.18),
                # Balance Sheet Metrics
                'current_ratio': (1.3, 2.5),
                'debt_to_equity': (0.4, 1.0),
                'inventory_turnover': (6, 10),
                'asset_turnover': (0.8, 1.5),
                # Cash Flow Metrics
                'operating_cash_flow_margin': (0.08, 0.18),
                'capex_to_revenue': (0.05, 0.12),
                'fcf_margin': (0.05, 0.12)
            }
        }
        return PROFILES.get(industry.lower(), PROFILES['technology'])

class FinancialDocument:
    """Base class for financial document generation."""
    
    def __init__(self, 
                 base_revenue: float = 1000000,
                 industry: str = 'technology',
                 growth_rate: Optional[float] = None,
                 volatility: float = 0.1,
                 seed: Optional[int] = None):
        """
        Initialize the financial document generator.
        
        Args:
            base_revenue: Starting annual revenue
            industry: Industry type (affects metrics and patterns)
            growth_rate: Override industry default growth rate
            volatility: Variation in financial metrics
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        self.base_revenue = base_revenue
        self.industry = industry.lower()
        self.metrics = FinancialMetrics(industry)
        self.volatility = volatility
        
        # Set growth rate
        if growth_rate is not None:
            self.growth_rate = growth_rate
        else:
            min_growth, max_growth = self.metrics.profile['sales_growth']
            self.growth_rate = random.uniform(min_growth, max_growth)
        
        # Store generated data
        self.data = None
        self.last_params = {}
    
    def _generate_revenue(self, 
                         periods: int,
                         period_type: ReportingPeriod) -> np.ndarray:
        """Generate revenue with growth and seasonality."""
        # Base growth pattern
        t = np.arange(periods)
        growth_factor = (1 + self.growth_rate) ** (t / 4 if period_type == ReportingPeriod.QUARTERLY else t)
        revenue = self.base_revenue * growth_factor
        
        # Add seasonality for quarterly data
        if period_type == ReportingPeriod.QUARTERLY:
            seasonality = 1 + np.sin(2 * np.pi * (t % 4) / 4) * 0.1
            revenue *= seasonality
        
        # Add random variation
        noise = np.random.normal(1, self.volatility, periods)
        revenue *= noise
        
        return revenue
    
    def format_number(self, x: float) -> str:
        """Format large numbers into readable strings with B/M suffix."""
        if x is None:
            return "-"
        
        # Format with parentheses for negative numbers
        is_negative = x < 0
        abs_x = abs(x)
        
        if abs_x >= 1e9:
            formatted = f"${abs_x/1e9:,.1f}B"
        elif abs_x >= 1e6:
            formatted = f"${abs_x/1e6:,.1f}M"
        else:
            formatted = f"${abs_x:,.0f}"
        
        return f"({formatted})" if is_negative else formatted
        
    def to_excel(self, filename: str, directory: str = "output") -> None:
        """Save the income statement to Excel with formatting."""
        if self.data is None:
            raise ValueError("No data has been generated yet. Call generate() first.")
            
        # Create the output directory if it doesn't exist
        import os
        os.makedirs(directory, exist_ok=True)
        
        # Create Excel writer
        filepath = os.path.join(directory, f"{filename}.xlsx")
        writer = pd.ExcelWriter(filepath, engine='xlsxwriter')
        
        # Write data
        self.data.to_excel(writer, sheet_name='Income Statement', index=False)
        
        # Get workbook and worksheet objects
        workbook = writer.book
        worksheet = writer.sheets['Income Statement']
        
        # Add formats
        header_format = workbook.add_format({
            'bold': True,
            'font_size': 12,
            'bg_color': '#D3D3D3',
            'border': 1
        })
        
        number_format = workbook.add_format({
            'num_format': '#,##0',
            'border': 1
        })
        
        percent_format = workbook.add_format({
            'num_format': '0.0%',
            'border': 1
        })
        
        # Apply formats
        for col_num, value in enumerate(self.data.columns.values):
            worksheet.write(0, col_num, value, header_format)
            
        # Set column formats
        for col_num, col in enumerate(self.data.columns):
            if col in ['period', 'date']:
                continue
            elif col.endswith('margin'):
                worksheet.set_column(col_num, col_num, 12, percent_format)
            else:
                worksheet.set_column(col_num, col_num, 15, number_format)
                
        # Adjust column widths
        worksheet.set_column('A:B', 15)  # period and date columns
        
        # Save
        writer.close()
        print(f"Income statement saved to {filepath}")

    def to_pdf(self, filename: str, directory: str = "output") -> None:
        """Save the income statement to a professionally formatted PDF."""
        if self.data is None:
            raise ValueError("No data has been generated yet. Call generate() first.")
            
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter, landscape
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.enums import TA_RIGHT, TA_CENTER
        import os
        
        # Create the output directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, f"{filename}.pdf")
        
        # Create document
        doc = SimpleDocTemplate(
            filepath,
            pagesize=landscape(letter),
            rightMargin=inch/2,
            leftMargin=inch/2,
            topMargin=inch/3,
            bottomMargin=inch/3
        )
        
        # Styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        number_style = ParagraphStyle(
            'Numbers',
            parent=styles['Normal'],
            alignment=TA_RIGHT
        )
        
        # Create content
        content = []
        
        # Title
        period_type = self.last_params.get('period_type', ReportingPeriod.QUARTERLY)
        title = f"Income Statement ({period_type.value.title()})"
        content.append(Paragraph(title, title_style))
        
        # Prepare data for table
        table_data = []
        
        # Headers
        headers = ['Period']
        numeric_columns = [
            'Revenue',
            'COGS',
            'Gross Profit',
            'R&D',
            'S&M',
            'G&A',
            'Op. Income',
            'Net Income'
        ]
        headers.extend(numeric_columns)
        table_data.append(headers)
        
        # Format numbers
        def format_number(x):
            if abs(x) >= 1e9:
                return f"${x/1e9:,.1f}B"
            elif abs(x) >= 1e6:
                return f"${x/1e6:,.1f}M"
            else:
                return f"${x:,.0f}"
        
        # Add data rows
        for i in range(len(self.data)):
            data_row = [self.data.iloc[i]['period']]
            data_row.extend([
                format_number(self.data.iloc[i]['revenue']),
                format_number(self.data.iloc[i]['cost_of_revenue']),
                format_number(self.data.iloc[i]['gross_profit']),
                format_number(self.data.iloc[i]['research_development']),
                format_number(self.data.iloc[i]['sales_marketing']),
                format_number(self.data.iloc[i]['general_admin']),
                format_number(self.data.iloc[i]['operating_income']),
                format_number(self.data.iloc[i]['net_income'])
            ])
            table_data.append(data_row)
            
        # Create table
        period_width = 0.8*inch
        data_col_width = 0.65*inch
        col_widths = [period_width] + [data_col_width] * len(numeric_columns)
        table = Table(table_data, repeatRows=1, colWidths=col_widths)
        
        # Style the table
        style = TableStyle([
            # Headers
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 4),
            ('TOPPADDING', (0, 0), (-1, 0), 4),
            ('WORDWRAP', (0, 0), (-1, 0), True),
            # Data
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            # Grid
            ('GRID', (0, 0), (-1, -1), 0.25, colors.black),
            ('LINEBELOW', (0, 0), (-1, 0), 0.5, colors.black),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            # Alternating row colors
            *[('BACKGROUND', (0, i), (-1, i), colors.Color(0.93, 0.93, 0.93)) 
              for i in range(2, len(table_data), 2)],
            # Subtotal lines
            ('LINEBELOW', (0, -1), (-1, -1), 0.5, colors.black),
        ])
        table.setStyle(style)
        
        # Add table to content
        content.append(table)
        
        # Add margins
        content.append(Spacer(1, 20))
        
        # Add footnote
        footnote_style = ParagraphStyle(
            'Footnote',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.grey,
            alignment=1  # 1 = center alignment
        )
        footnote = Paragraph(
            f"Generated on {datetime.now().strftime('%Y-%m-%d')} | For illustrative purposes only",
            footnote_style
        )
        content.append(footnote)
        
        # Build PDF
        doc.build(content)
        print(f"Income statement saved to {filepath}")

class IncomeStatementGenerator(FinancialDocument):
    """Generator for realistic income statements."""
    
    def _calculate_metrics(self, revenue: float) -> Dict[str, float]:
        """Calculate various income statement metrics based on revenue."""
        # Get margin ranges from industry profile
        gm_min, gm_max = self.metrics.profile['gross_margin']
        op_min, op_max = self.metrics.profile['operating_margin']
        rd_min, rd_max = self.metrics.profile['rd_to_revenue']
        
        # Calculate margins with some random variation but ensure they make sense
        gross_margin = random.uniform(gm_min, gm_max)
        rd_ratio = random.uniform(rd_min, rd_max)
        
        # Calculate main components
        cost_of_revenue = revenue * (1 - gross_margin)
        gross_profit = revenue - cost_of_revenue
        
        # Operating expenses (as percentage of revenue)
        rd_expense = revenue * rd_ratio
        sales_marketing = revenue * random.uniform(0.10, 0.20)
        general_admin = revenue * random.uniform(0.05, 0.12)
        
        total_opex = rd_expense + sales_marketing + general_admin
        
        # Operating income
        operating_income = gross_profit - total_opex
        operating_margin = operating_income / revenue
        
        # Ensure operating margin is within industry profile
        if operating_margin < op_min or operating_margin > op_max:
            # Adjust expenses to meet target operating margin
            target_operating_margin = random.uniform(op_min, op_max)
            target_operating_income = revenue * target_operating_margin
            adjustment_factor = (gross_profit - target_operating_income) / total_opex
            
            # Adjust each expense proportionally
            rd_expense *= adjustment_factor
            sales_marketing *= adjustment_factor
            general_admin *= adjustment_factor
            total_opex = rd_expense + sales_marketing + general_admin
            operating_income = gross_profit - total_opex
        
        # Other income/expense (keep these relatively small)
        interest_expense = revenue * random.uniform(0.005, 0.015)
        other_income = revenue * random.uniform(-0.005, 0.01)
        
        # Pre-tax income and taxes
        pretax_income = operating_income - interest_expense + other_income
        
        # Progressive tax rate based on pretax income
        if pretax_income > 0:
            if pretax_income > revenue * 0.3:
                tax_rate = 0.30
            elif pretax_income > revenue * 0.15:
                tax_rate = 0.25
            else:
                tax_rate = 0.20
            income_tax = pretax_income * tax_rate
        else:
            # Tax benefit for losses (simplified)
            income_tax = pretax_income * 0.20
        
        # Net income
        net_income = pretax_income - income_tax
        
        # Validate all calculations
        assert abs(gross_profit - (revenue - cost_of_revenue)) < 1.0, "Gross profit calculation error"
        assert abs(operating_income - (gross_profit - total_opex)) < 1.0, "Operating income calculation error"
        assert abs(pretax_income - (operating_income - interest_expense + other_income)) < 1.0, "Pretax income calculation error"
        assert abs(net_income - (pretax_income - income_tax)) < 1.0, "Net income calculation error"
        
        return {
            'revenue': revenue,
            'cost_of_revenue': cost_of_revenue,
            'gross_profit': gross_profit,
            'research_development': rd_expense,
            'sales_marketing': sales_marketing,
            'general_admin': general_admin,
            'total_operating_expenses': total_opex,
            'operating_income': operating_income,
            'interest_expense': interest_expense,
            'other_income': other_income,
            'pretax_income': pretax_income,
            'income_tax': income_tax,
            'net_income': net_income
        }
        
    def generate(self,
                periods: int = 4,
                period_type: ReportingPeriod = ReportingPeriod.QUARTERLY,
                start_date: Optional[Union[str, date]] = None) -> pd.DataFrame:
        """
        Generate an income statement time series.
        
        Args:
            periods: Number of periods to generate
            period_type: Type of reporting period (quarterly/annual)
            start_date: Starting date for the time series
            
        Returns:
            DataFrame with income statement data
        """
        # Store generation parameters
        self.last_params = {
            'periods': periods,
            'period_type': period_type,
            'start_date': start_date
        }
        
        # Handle start date
        if start_date is None:
            start_date = date.today()
        elif isinstance(start_date, str):
            start_date = pd.to_datetime(start_date).date()
            
        # Generate revenue for all periods
        revenues = self._generate_revenue(periods, period_type)
        
        # Generate full income statements
        data = []
        for i in range(periods):
            metrics = self._calculate_metrics(revenues[i])
            
            # Add period information
            if period_type == ReportingPeriod.QUARTERLY:
                period_date = pd.Timestamp(start_date) - pd.DateOffset(months=3*i)
                period_label = f"Q{(period_date.month-1)//3 + 1} {period_date.year}"
            else:
                period_date = pd.Timestamp(start_date) - pd.DateOffset(years=i)
                period_label = str(period_date.year)
                
            metrics['period'] = period_label
            metrics['date'] = period_date
            data.append(metrics)
            
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Add some common financial metrics
        df['gross_margin'] = df['gross_profit'] / df['revenue']
        df['operating_margin'] = df['operating_income'] / df['revenue']
        df['net_margin'] = df['net_income'] / df['revenue']
        
        # Sort by date
        df = df.sort_values('date')
        
        # Store the generated data
        self.data = df
        
        return df
        
    def save(self, filename: str, directory: str = "output") -> None:
        """Save the income statement to CSV."""
        if self.data is None:
            raise ValueError("No data has been generated yet. Call generate() first.")
            
        # Create the output directory if it doesn't exist
        import os
        os.makedirs(directory, exist_ok=True)
        
        # Save to CSV
        filepath = os.path.join(directory, f"{filename}.csv")
        self.data.to_csv(filepath, index=False)
        print(f"Income statement saved to {filepath}")
        
    def to_excel(self, filename: str, directory: str = "output") -> None:
        """Save the income statement to Excel with formatting."""
        if self.data is None:
            raise ValueError("No data has been generated yet. Call generate() first.")
            
        # Create the output directory if it doesn't exist
        import os
        os.makedirs(directory, exist_ok=True)
        
        # Create Excel writer
        filepath = os.path.join(directory, f"{filename}.xlsx")
        writer = pd.ExcelWriter(filepath, engine='xlsxwriter')
        
        # Write data
        self.data.to_excel(writer, sheet_name='Income Statement', index=False)
        
        # Get workbook and worksheet objects
        workbook = writer.book
        worksheet = writer.sheets['Income Statement']
        
        # Add formats
        header_format = workbook.add_format({
            'bold': True,
            'font_size': 12,
            'bg_color': '#D3D3D3',
            'border': 1
        })
        
        number_format = workbook.add_format({
            'num_format': '#,##0',
            'border': 1
        })
        
        percent_format = workbook.add_format({
            'num_format': '0.0%',
            'border': 1
        })
        
        # Apply formats
        for col_num, value in enumerate(self.data.columns.values):
            worksheet.write(0, col_num, value, header_format)
            
        # Set column formats
        for col_num, col in enumerate(self.data.columns):
            if col in ['period', 'date']:
                continue
            elif col.endswith('margin'):
                worksheet.set_column(col_num, col_num, 12, percent_format)
            else:
                worksheet.set_column(col_num, col_num, 15, number_format)
                
        # Adjust column widths
        worksheet.set_column('A:B', 15)  # period and date columns
        
        # Save
        writer.close()
        print(f"Income statement saved to {filepath}")

    def to_pdf(self, filename: str, directory: str = "output") -> None:
        """Save the income statement to a professionally formatted PDF."""
        if self.data is None:
            raise ValueError("No data has been generated yet. Call generate() first.")
            
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter, landscape
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.enums import TA_RIGHT, TA_CENTER
        import os
        
        # Create the output directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, f"{filename}.pdf")
        
        # Create document
        doc = SimpleDocTemplate(
            filepath,
            pagesize=landscape(letter),
            rightMargin=inch/2,
            leftMargin=inch/2,
            topMargin=inch/3,
            bottomMargin=inch/3
        )
        
        # Styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        number_style = ParagraphStyle(
            'Numbers',
            parent=styles['Normal'],
            alignment=TA_RIGHT
        )
        
        # Create content
        content = []
        
        # Title
        period_type = self.last_params.get('period_type', ReportingPeriod.QUARTERLY)
        title = f"Income Statement ({period_type.value.title()})"
        content.append(Paragraph(title, title_style))
        
        # Prepare data for table
        table_data = []
        
        # Headers
        headers = ['Period']
        numeric_columns = [
            'Revenue',
            'COGS',
            'Gross Profit',
            'R&D',
            'S&M',
            'G&A',
            'Op. Income',
            'Net Income'
        ]
        headers.extend(numeric_columns)
        table_data.append(headers)
        
        # Format numbers
        def format_number(x):
            if abs(x) >= 1e9:
                return f"${x/1e9:,.1f}B"
            elif abs(x) >= 1e6:
                return f"${x/1e6:,.1f}M"
            else:
                return f"${x:,.0f}"
        
        # Add data rows
        for i in range(len(self.data)):
            data_row = [self.data.iloc[i]['period']]
            data_row.extend([
                format_number(self.data.iloc[i]['revenue']),
                format_number(self.data.iloc[i]['cost_of_revenue']),
                format_number(self.data.iloc[i]['gross_profit']),
                format_number(self.data.iloc[i]['research_development']),
                format_number(self.data.iloc[i]['sales_marketing']),
                format_number(self.data.iloc[i]['general_admin']),
                format_number(self.data.iloc[i]['operating_income']),
                format_number(self.data.iloc[i]['net_income'])
            ])
            table_data.append(data_row)
            
        # Create table
        period_width = 0.8*inch
        data_col_width = 0.65*inch
        col_widths = [period_width] + [data_col_width] * len(numeric_columns)
        table = Table(table_data, repeatRows=1, colWidths=col_widths)
        
        # Style the table
        style = TableStyle([
            # Headers
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 4),
            ('TOPPADDING', (0, 0), (-1, 0), 4),
            ('WORDWRAP', (0, 0), (-1, 0), True),
            # Data
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            # Grid
            ('GRID', (0, 0), (-1, -1), 0.25, colors.black),
            ('LINEBELOW', (0, 0), (-1, 0), 0.5, colors.black),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            # Alternating row colors
            *[('BACKGROUND', (0, i), (-1, i), colors.Color(0.93, 0.93, 0.93)) 
              for i in range(2, len(table_data), 2)],
            # Subtotal lines
            ('LINEBELOW', (0, -1), (-1, -1), 0.5, colors.black),
        ])
        table.setStyle(style)
        
        # Add table to content
        content.append(table)
        
        # Add margins
        content.append(Spacer(1, 20))
        
        # Add footnote
        footnote_style = ParagraphStyle(
            'Footnote',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.grey,
            alignment=1  # 1 = center alignment
        )
        footnote = Paragraph(
            f"Generated on {datetime.now().strftime('%Y-%m-%d')} | For illustrative purposes only",
            footnote_style
        )
        content.append(footnote)
        
        # Build PDF
        doc.build(content)
        print(f"Income statement saved to {filepath}")

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
        
        # Shareholders' Equity components
        retained_earnings = equity * random.uniform(0.6, 0.8)
        common_stock = equity - retained_earnings
        
        # Validate calculations
        assert abs((total_current_assets + total_non_current_assets) - total_assets) < 1.0, "Assets don't balance"
        assert abs((total_current_liabilities + long_term_debt + common_stock + retained_earnings) - total_assets) < 1.0, "Balance sheet doesn't balance"
        
        return {
            # Assets
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
            # Liabilities
            'accounts_payable': accounts_payable,
            'short_term_debt': short_term_debt,
            'other_current_liabilities': other_current_liabilities,
            'total_current_liabilities': total_current_liabilities,
            'long_term_debt': long_term_debt,
            'total_liabilities': total_liabilities,
            # Equity
            'common_stock': common_stock,
            'retained_earnings': retained_earnings,
            'total_equity': equity,
            # Key Metrics
            'current_ratio': current_ratio,
            'debt_to_equity': debt_to_equity,
            'asset_turnover': asset_turnover
        }
    
    def generate(self,
                periods: int = 4,
                period_type: ReportingPeriod = ReportingPeriod.QUARTERLY,
                start_date: Optional[Union[str, date]] = None) -> pd.DataFrame:
        """
        Generate a balance sheet time series.
        
        Args:
            periods: Number of periods to generate
            period_type: Type of reporting period (quarterly/annual)
            start_date: Starting date for the time series
            
        Returns:
            DataFrame with balance sheet data
        """
        # Store generation parameters
        self.last_params = {
            'periods': periods,
            'period_type': period_type,
            'start_date': start_date
        }
        
        # Handle start date
        if start_date is None:
            start_date = date.today()
        elif isinstance(start_date, str):
            start_date = pd.to_datetime(start_date).date()
        
        # Generate revenue for all periods (needed for calculations)
        revenues = self._generate_revenue(periods, period_type)
        
        # Generate full balance sheets
        data = []
        for i in range(periods):
            metrics = self._calculate_metrics(revenues[i])
            
            # Add period information
            if period_type == ReportingPeriod.QUARTERLY:
                period_date = pd.Timestamp(start_date) - pd.DateOffset(months=3*i)
                period_label = f"Q{(period_date.month-1)//3 + 1} {period_date.year}"
            else:
                period_date = pd.Timestamp(start_date) - pd.DateOffset(years=i)
                period_label = str(period_date.year)
            
            metrics['period'] = period_label
            metrics['date'] = period_date
            data.append(metrics)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Sort by date
        df = df.sort_values('date')
        
        # Store the generated data
        self.data = df
        
        return df
    
    def to_pdf(self, filename: str, directory: str = "output") -> None:
        """Save the balance sheet to a professionally formatted PDF."""
        if self.data is None:
            raise ValueError("No data has been generated yet. Call generate() first.")
        
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter, landscape
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.enums import TA_RIGHT, TA_CENTER
        import os
        
        # Create the output directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, f"{filename}.pdf")
        
        # Create document in landscape orientation
        doc = SimpleDocTemplate(
            filepath,
            pagesize=landscape(letter),
            rightMargin=inch/4,
            leftMargin=inch/4,
            topMargin=inch/2,
            bottomMargin=inch/2
        )
        
        # Styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        # Create content
        content = []
        
        # Title
        period_type = self.last_params.get('period_type', ReportingPeriod.QUARTERLY)
        title = f"Balance Sheet ({period_type.value.title()})"
        content.append(Paragraph(title, title_style))
        
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

        def style_table(table, table_data, sections, include_section_lines=True):
            """Helper function to style a table."""
            style = [
                # Headers
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 8),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 4),
                ('TOPPADDING', (0, 0), (-1, 0), 4),
                ('WORDWRAP', (0, 0), (-1, 0), True),
                # Data
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 7),
                ('ALIGN', (1 if include_section_lines else 0, 1), (-1, -1), 'RIGHT'),
                ('ALIGN', (0, 0), (0, -1), 'LEFT'),
                ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                # Grid
                ('GRID', (0, 0), (-1, -1), 0.25, colors.black),
                ('LINEBELOW', (0, 0), (-1, 0), 0.5, colors.black),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                # Alternating row colors
                *[('BACKGROUND', (0, i), (-1, i), colors.Color(0.93, 0.93, 0.93)) 
                  for i in range(2, len(table_data), 2)],
                # Subtotal lines
                ('LINEBELOW', (0, -1), (-1, -1), 0.5, colors.black),
            ]
            
            if include_section_lines:
                # Add section separators
                current_col = 1
                for section_fields in sections.values():
                    style.append(
                        ('LINEAFTER', (current_col + len(section_fields) - 1, 0),
                         (current_col + len(section_fields) - 1, -1), 0.5, colors.black)
                    )
                    current_col += len(section_fields)
            
            table.setStyle(TableStyle(style))
            return table

        # Create Assets table
        assets_data = create_table_data(assets_sections)
        assets_col_widths = [0.8*inch] + [0.9*inch] * (len(assets_data[0]) - 1)
        assets_table = Table(assets_data, repeatRows=1, colWidths=assets_col_widths)
        assets_table = style_table(assets_table, assets_data, assets_sections)
        
        # Create Liabilities & Equity table
        liab_equity_data = create_table_data(liab_equity_sections)
        liab_equity_col_widths = [0.8*inch] + [0.9*inch] * (len(liab_equity_data[0]) - 1)
        liab_equity_table = Table(liab_equity_data, repeatRows=1, colWidths=liab_equity_col_widths)
        liab_equity_table = style_table(liab_equity_table, liab_equity_data, liab_equity_sections)
        
        # Add tables to content
        content.append(assets_table)
        content.append(Spacer(1, 20))  # Add space between tables
        content.append(liab_equity_table)
        
        # Add margins
        content.append(Spacer(1, 20))
        
        # Add footnote
        footnote_style = ParagraphStyle(
            'Footnote',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.grey,
            alignment=1  # 1 = center alignment
        )
        footnote = Paragraph(
            f"Generated on {datetime.now().strftime('%Y-%m-%d')} | For illustrative purposes only",
            footnote_style
        )
        content.append(footnote)
        
        # Build PDF
        doc.build(content)
        print(f"Balance sheet saved to {filepath}")
