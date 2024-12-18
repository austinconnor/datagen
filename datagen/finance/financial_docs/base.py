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
                'operating_margin': (0.05, 0.15),
                # Balance Sheet Metrics
                'current_ratio': (1.0, 2.0),
                'debt_to_equity': (0.3, 0.8),
                'inventory_turnover': (4, 8),
                'asset_turnover': (1.5, 2.5),
                # Cash Flow Metrics
                'operating_cash_flow_margin': (0.05, 0.15),
                'capex_to_revenue': (0.02, 0.08),
                'fcf_margin': (0.03, 0.10)
            }
            # Add more industries as needed
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
                         period_type: ReportingPeriod) -> pd.Series:
        """Generate revenue with growth and seasonality."""
        # Base annual revenue
        annual_revenue = self.base_revenue
        
        # Generate quarterly seasonality factors
        seasonality = np.array([1.0, 1.1, 0.9, 1.2])  # Example seasonality pattern
        
        # Generate revenue for each period
        revenue = []
        for i in range(periods):
            if period_type == ReportingPeriod.QUARTERLY:
                # Add seasonality and some random variation for quarters
                quarter_factor = seasonality[i % 4]
                period_revenue = (annual_revenue / 4) * quarter_factor
                # Add random noise
                period_revenue *= (1 + np.random.normal(0, self.volatility))
            else:
                # Annual revenue with growth
                period_revenue = annual_revenue
            
            revenue.append(period_revenue)
            # Update annual revenue for next period
            annual_revenue *= (1 + self.growth_rate)
        
        return pd.Series(revenue)
    
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
