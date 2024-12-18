from datagen import DataGenerator
import pandas as pd

def main():
    # Example 1: Generate data from scratch
    generator = DataGenerator()
    df = generator.generate(
        schema={
            'name': 'name',
            'age': 'integer[18:80]',
            'email': 'email',
            'score': 'float[0:100]'
        },
        rows=10
    )
    print("\nGenerated data from schema:")
    print(df)
    
    # Example 2: Generate data based on example
    example_data = pd.DataFrame({
        'temperature': [20.5, 21.2, 19.8, 22.1, 20.9],
        'humidity': [45, 48, 42, 50, 46],
        'condition': ['Sunny', 'Cloudy', 'Rainy', 'Sunny', 'Cloudy']
    })
    
    generator = DataGenerator(example_data)
    similar_df = generator.generate(rows=10)
    print("\nGenerated data similar to example:")
    print(similar_df)

if __name__ == '__main__':
    main()
