# DataGen

A Python utility for generating synthetic data for various domains including financial, healthcare, and text data. The library provides specialized generators with built-in domain knowledge and patterns.

## Features
- Multiple specialized data generators:
  - Financial data (OHLCV market data)
  - Healthcare data (patient records, medical events)
  - Text data (structured text content)
  - Generic tabular data with customizable schemas
- Built-in save functionality for all generators
- Support for various data types including numerical, categorical, datetime, and text
- Pattern-based generation from example datasets

## Installation
```bash
pip install -r requirements.txt
```

## Usage

### Financial Data Generation
```python
from datagen.financial import OHLCVGenerator

# Create a financial data generator
generator = OHLCVGenerator()

# Generate OHLCV market data
data = generator.generate(
    start_date="2023-01-01",
    end_date="2023-12-31",
    initial_price=100,
    volatility=0.2
)

# Save the generated data to a CSV file
generator.save_to_csv("market_data.csv")
```

### Healthcare Data Generation
```python
from datagen.healthcare import HealthcareDataGenerator

# Create a healthcare data generator
generator = HealthcareDataGenerator()

# Generate patient records
data = generator.generate(num_patients=100)

# Save the generated data
generator.save("patient_records.csv")
```

### Text Data Generation
```python
from datagen.text import TextGenerator

# Create a text data generator
generator = TextGenerator()

# Generate structured text data
data = generator.generate(rows=50)

# Save the generated data
generator.save("text_data.csv")
```

### Generic Data Generation
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

# Save generated data
generator.save("generated_data.csv")
```

## Data Types and Schemas
The generic data generator supports various data types that can be specified in the schema:
- Basic types: integer, float, string, boolean
- Ranges: integer[min:max], float[min:max]
- Special types: name, email, date, datetime
- Categories: category[value1,value2,...]

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
