# DataGen

A Python utility for generating tabular data, either from scratch or based on example data.

## Features
- Generate synthetic tabular data with customizable schemas
- Create data that follows patterns from example datasets
- Support for various data types including numerical, categorical, and datetime

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```python
from datagen import DataGenerator

# Generate data from scratch
generator = DataGenerator()
df = generator.generate(
    schema={
        'name': 'name',
        'age': 'integer[18:80]',
        'email': 'email'
    },
    rows=100
)

# Or generate based on example data
example_df = pd.read_csv('example.csv')
generator = DataGenerator(example_df)
synthetic_df = generator.generate(rows=100)
```
