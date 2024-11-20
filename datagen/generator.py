import pandas as pd
import numpy as np
from faker import Faker
import re
from typing import Dict, Optional, Union

class Generator:
    """Base class for all data generators."""
    
    def __init__(self):
        """Initialize the generator."""
        self.data = None
        self.last_params = {}
    
    def generate(self, *args, **kwargs):
        """Generate data with the given parameters."""
        raise NotImplementedError("Subclasses must implement generate()")
    
    def save(self, filename: str, directory: str = "output", **kwargs) -> None:
        """Save the most recently generated data to a file.
        
        Args:
            filename: Name of the file (without extension)
            directory: Directory to save the file in (default: "output")
            **kwargs: Additional arguments for specific file formats
        
        Raises:
            ValueError: If no data has been generated yet
        """
        if self.data is None:
            raise ValueError("No data has been generated yet. Call generate() first.")
        
        # Create the output directory if it doesn't exist
        import os
        os.makedirs(directory, exist_ok=True)
        
        # Let subclasses implement their specific save logic
        self._save_impl(filename, directory, **kwargs)
    
    def _save_impl(self, filename: str, directory: str, **kwargs) -> None:
        """Implementation of save logic for specific data types.
        
        Args:
            filename: Name of the file (without extension)
            directory: Directory to save the file in
            **kwargs: Additional arguments for specific file formats
        """
        raise NotImplementedError("Subclasses must implement _save_impl()")

class DataGenerator(Generator):
    """A class for generating synthetic tabular data."""
    
    def __init__(self, example_data: Optional[pd.DataFrame] = None):
        """
        Initialize the DataGenerator.
        
        Args:
            example_data: Optional DataFrame to use as a template for generating similar data
        """
        super().__init__()
        self.faker = Faker()
        self.example_data = example_data
        
    def _generate_column(self, dtype: str, size: int) -> pd.Series:
        """Generate a single column of data based on the specified type."""
        if dtype.startswith('integer'):
            # Parse range if specified, e.g., 'integer[0:100]'
            range_match = re.match(r'integer\[(\d+):(\d+)\]', dtype)
            if range_match:
                min_val, max_val = map(int, range_match.groups())
                return pd.Series(np.random.randint(min_val, max_val + 1, size=size))
            return pd.Series(np.random.randint(0, 1000, size=size))
            
        elif dtype == 'name':
            return pd.Series([self.faker.name() for _ in range(size)])
            
        elif dtype == 'email':
            return pd.Series([self.faker.email() for _ in range(size)])
            
        elif dtype.startswith('float'):
            # Parse range if specified, e.g., 'float[0:1]'
            range_match = re.match(r'float\[(-?\d+\.?\d*):(-?\d+\.?\d*)\]', dtype)
            if range_match:
                min_val, max_val = map(float, range_match.groups())
                return pd.Series(np.random.uniform(min_val, max_val, size=size))
            return pd.Series(np.random.randn(size))
            
        elif dtype == 'date':
            return pd.Series([self.faker.date() for _ in range(size)])
            
        else:
            raise ValueError(f"Unsupported data type: {dtype}")

    def generate(self, schema: Optional[Dict[str, str]] = None, rows: int = 100) -> pd.DataFrame:
        """
        Generate synthetic tabular data.
        
        Args:
            schema: Dictionary mapping column names to their data types
                   If None and example_data exists, will generate similar data
            rows: Number of rows to generate
            
        Returns:
            DataFrame containing the generated data
        """
        if schema is None and self.example_data is None:
            raise ValueError("Either schema or example_data must be provided")
            
        if schema:
            # Generate data from scratch based on schema
            data = {}
            for col_name, dtype in schema.items():
                data[col_name] = self._generate_column(dtype, rows)
            self.data = pd.DataFrame(data)
            return self.data
            
        else:
            # Generate data similar to example_data
            # This is a simple implementation that could be enhanced
            data = {}
            for col in self.example_data.columns:
                if pd.api.types.is_numeric_dtype(self.example_data[col]):
                    mean = self.example_data[col].mean()
                    std = self.example_data[col].std()
                    data[col] = pd.Series(np.random.normal(mean, std, size=rows))
                else:
                    # For non-numeric columns, sample from existing values
                    data[col] = pd.Series(np.random.choice(self.example_data[col], size=rows))
            self.data = pd.DataFrame(data)
            return self.data

    def _save_impl(self, filename: str, directory: str, **kwargs) -> None:
        """Implementation of save logic for DataFrames."""
        self.data.to_csv(f"{directory}/{filename}.csv", index=False)
