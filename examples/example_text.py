from datagen import TextDataGenerator
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 120)

def main():
    # Initialize generator with seed for reproducibility
    generator = TextDataGenerator(seed=42)
    
    # Generate user profiles
    print("\n=== User Profiles ===")
    users_df = generator.generate_user_profiles(count=5)
    print(users_df[['username', 'email', 'occupation', 'preferences']].to_string())
    
    # Generate company data
    print("\n=== Company Data ===")
    companies_df = generator.generate_company_data(count=3)
    print(companies_df[['name', 'industry', 'description', 'employee_count', 'revenue_range']].to_string())
    
    # Generate articles
    print("\n=== Articles ===")
    articles_df = generator.generate_articles(count=2)
    print(articles_df[['title', 'category', 'author', 'read_time', 'tags']].to_string())
    
    # Show a sample article content
    print("\nSample Article Content:")
    print("------------------------")
    print(articles_df.iloc[0]['content'])
    
    # Generate log entries
    print("\n=== Log Entries ===")
    logs_df = generator.generate_log_entries(count=5)
    print(logs_df[['timestamp', 'level', 'service', 'message', 'status_code']].to_string())
    
    # Example: Save to different formats
    users_df.to_csv('example_users.csv', index=False)
    companies_df.to_json('example_companies.json', orient='records', lines=True)
    articles_df.to_pickle('example_articles.pkl')

if __name__ == '__main__':
    main()
