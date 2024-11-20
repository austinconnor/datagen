import pandas as pd
from faker import Faker
from typing import Optional, List, Dict, Union
import random
from datetime import datetime, timedelta
import uuid
import json

class TextDataGenerator:
    """Generator for various types of text data."""
    
    def __init__(self, locale: str = 'en_US', seed: Optional[int] = None):
        """
        Initialize the text data generator.
        
        Args:
            locale: Locale for generating region-specific data
            seed: Random seed for reproducibility
        """
        self.faker = Faker(locale)
        if seed is not None:
            Faker.seed(seed)
            random.seed(seed)
            
    def generate_user_profiles(self, count: int = 100) -> pd.DataFrame:
        """Generate realistic user profile data."""
        data = []
        for _ in range(count):
            profile = {
                'user_id': str(uuid.uuid4()),
                'username': self.faker.user_name(),
                'email': self.faker.email(),
                'full_name': self.faker.name(),
                'birth_date': self.faker.date_of_birth(minimum_age=18, maximum_age=90),
                'address': self.faker.address().replace('\n', ', '),
                'phone': self.faker.phone_number(),
                'occupation': self.faker.job(),
                'registration_date': self.faker.date_time_between(
                    start_date='-5y',
                    end_date='now'
                ),
                'preferences': {
                    'language': self.faker.language_name(),
                    'theme': random.choice(['light', 'dark', 'system']),
                    'notifications': random.choice(['all', 'important', 'none'])
                }
            }
            data.append(profile)
        
        df = pd.DataFrame(data)
        df['preferences'] = df['preferences'].apply(json.dumps)
        return df
        
    def generate_company_data(self, count: int = 50) -> pd.DataFrame:
        """Generate business/company data."""
        data = []
        industries = ['Technology', 'Healthcare', 'Finance', 'Retail', 
                     'Manufacturing', 'Energy', 'Education', 'Entertainment']
        
        for _ in range(count):
            company = {
                'company_id': str(uuid.uuid4()),
                'name': self.faker.company(),
                'industry': random.choice(industries),
                'description': self.faker.catch_phrase(),
                'founded_date': self.faker.date_between(
                    start_date='-30y',
                    end_date='-1y'
                ),
                'website': self.faker.url(),
                'email': self.faker.company_email(),
                'phone': self.faker.phone_number(),
                'address': self.faker.address().replace('\n', ', '),
                'employee_count': random.randint(5, 10000),
                'revenue_range': random.choice([
                    '<1M', '1M-10M', '10M-50M', '50M-100M', '100M-500M', '>500M'
                ]),
                'is_public': random.choice([True, False])
            }
            data.append(company)
            
        return pd.DataFrame(data)
        
    def generate_articles(self, count: int = 20) -> pd.DataFrame:
        """Generate article/content data."""
        data = []
        categories = ['Technology', 'Business', 'Science', 'Health', 
                     'Politics', 'Entertainment', 'Sports', 'Travel']
        
        for _ in range(count):
            paragraphs = self.faker.paragraphs(nb=random.randint(3, 7))
            article = {
                'article_id': str(uuid.uuid4()),
                'title': self.faker.sentence(nb_words=6, variable_nb_words=True),
                'content': '\n\n'.join(paragraphs),
                'category': random.choice(categories),
                'author': self.faker.name(),
                'publication_date': self.faker.date_time_between(
                    start_date='-1y',
                    end_date='now'
                ),
                'tags': [self.faker.word() for _ in range(random.randint(2, 5))],
                'read_time': random.randint(3, 15),
                'likes': random.randint(0, 1000),
                'comments': random.randint(0, 100)
            }
            data.append(article)
            
        df = pd.DataFrame(data)
        df['tags'] = df['tags'].apply(json.dumps)
        return df
        
    def generate_log_entries(self, 
                           count: int = 1000,
                           start_date: Optional[datetime] = None,
                           include_errors: bool = True) -> pd.DataFrame:
        """Generate system log entries."""
        if start_date is None:
            start_date = datetime.now() - timedelta(days=7)
            
        log_levels = ['INFO', 'DEBUG', 'WARN'] + (['ERROR', 'CRITICAL'] if include_errors else [])
        services = ['web-server', 'auth-service', 'database', 'cache', 'api-gateway']
        endpoints = ['/api/users', '/api/products', '/auth/login', 
                    '/api/orders', '/api/search']
        
        data = []
        for _ in range(count):
            timestamp = self.faker.date_time_between(
                start_date=start_date,
                end_date='now'
            )
            
            level = random.choice(log_levels)
            service = random.choice(services)
            
            if level in ['ERROR', 'CRITICAL']:
                message = f"Failed to {random.choice(['connect to', 'process', 'validate'])} " \
                         f"{random.choice(['database', 'request', 'response', 'user input'])}"
                status_code = random.choice([400, 401, 403, 404, 500, 502, 503])
            else:
                message = f"Successfully {random.choice(['processed', 'handled', 'completed'])} " \
                         f"{random.choice(['request', 'operation', 'transaction'])}"
                status_code = random.choice([200, 201, 204])
            
            log_entry = {
                'timestamp': timestamp,
                'level': level,
                'service': service,
                'endpoint': random.choice(endpoints),
                'message': message,
                'status_code': status_code,
                'response_time': round(random.uniform(0.1, 2.0), 3),
                'user_id': str(uuid.uuid4()) if random.random() > 0.3 else None,
                'ip_address': self.faker.ipv4(),
                'user_agent': self.faker.user_agent()
            }
            data.append(log_entry)
            
        return pd.DataFrame(data).sort_values('timestamp')

    def generate(self, num_samples: int = 100, text_type: str = 'article') -> pd.DataFrame:
        """Generate synthetic text data."""
        
        # Store generation parameters
        self.last_params = {
            'num_samples': num_samples,
            'text_type': text_type
        }
        
        # Generate data
        if text_type == 'user_profiles':
            data = self.generate_user_profiles(num_samples)
        elif text_type == 'company_data':
            data = self.generate_company_data(num_samples)
        elif text_type == 'articles':
            data = self.generate_articles(num_samples)
        elif text_type == 'log_entries':
            data = self.generate_log_entries(num_samples)
        else:
            raise ValueError("Invalid text type")
        
        # Store the generated data
        self.data = data
        
        return data

    def save(self, filename: str, directory: str) -> None:
        """Save text data to CSV."""
        self.data.to_csv(f"{directory}/{filename}.csv", index=False)
        print(f"Data saved to {directory}/{filename}.csv")
