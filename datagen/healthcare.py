import pandas as pd
import numpy as np
from faker import Faker
from typing import Optional, List, Dict, Union
import random
from datetime import datetime, timedelta
import uuid
import json

class HealthcareGenerator:
    """Generator for healthcare-related data."""
    
    def __init__(self, locale: str = 'en_US', seed: Optional[int] = None):
        """
        Initialize the healthcare data generator.
        
        Args:
            locale: Locale for generating region-specific data
            seed: Random seed for reproducibility
        """
        self.faker = Faker(locale)
        if seed is not None:
            Faker.seed(seed)
            random.seed(seed)
            
        # Common medical data
        self.conditions = [
            'Hypertension', 'Type 2 Diabetes', 'Asthma', 'Arthritis',
            'Depression', 'Anxiety', 'GERD', 'Migraine', 'Hypothyroidism',
            'Hyperlipidemia', 'Allergic Rhinitis', 'Osteoporosis'
        ]
        
        self.medications = [
            'Lisinopril', 'Metformin', 'Amlodipine', 'Omeprazole',
            'Sertraline', 'Levothyroxine', 'Atorvastatin', 'Albuterol',
            'Gabapentin', 'Escitalopram', 'Losartan', 'Pantoprazole'
        ]
        
        self.specialists = [
            'Cardiologist', 'Endocrinologist', 'Neurologist', 'Psychiatrist',
            'Rheumatologist', 'Pulmonologist', 'Gastroenterologist',
            'Dermatologist', 'Oncologist', 'Orthopedist'
        ]
        
        self.procedures = [
            'Blood Test', 'X-Ray', 'MRI', 'CT Scan', 'Ultrasound',
            'ECG', 'Colonoscopy', 'Endoscopy', 'Biopsy', 'Stress Test'
        ]
        
        self.vital_ranges = {
            'blood_pressure_systolic': (90, 140),
            'blood_pressure_diastolic': (60, 90),
            'heart_rate': (60, 100),
            'temperature': (97.0, 99.0),
            'respiratory_rate': (12, 20),
            'oxygen_saturation': (95, 100)
        }
        
        self.last_params = None
        self.data = None
        
    def _generate_vitals(self) -> Dict[str, float]:
        """Generate a set of vital signs."""
        vitals = {}
        for name, (min_val, max_val) in self.vital_ranges.items():
            if name == 'temperature':
                vitals[name] = round(random.uniform(min_val, max_val), 1)
            else:
                vitals[name] = round(random.uniform(min_val, max_val))
        return vitals
        
    def _generate_patient_conditions(self) -> List[Dict[str, str]]:
        """Generate a list of medical conditions for a patient."""
        num_conditions = random.randint(0, 4)
        conditions = []
        for _ in range(num_conditions):
            condition = {
                'condition': random.choice(self.conditions),
                'diagnosed_date': self.faker.date_between(
                    start_date='-10y',
                    end_date='today'
                ).strftime('%Y-%m-%d'),
                'status': random.choice(['Active', 'Managed', 'Resolved'])
            }
            conditions.append(condition)
        return conditions
        
    def generate_patients(self, count: int = 100) -> pd.DataFrame:
        """Generate patient demographic and medical history data."""
        data = []
        blood_types = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
        
        for _ in range(count):
            birth_date = self.faker.date_of_birth(minimum_age=18, maximum_age=90)
            first_name = self.faker.first_name()
            last_name = self.faker.last_name()
            patient = {
                'patient_id': f"P{str(uuid.uuid4())[:8]}",
                'first_name': first_name,
                'last_name': last_name,
                'birth_date': birth_date,
                'gender': random.choice(['M', 'F']),
                'blood_type': random.choice(blood_types),
                'weight_kg': round(random.uniform(45, 120), 1),
                'height_cm': random.randint(150, 195),
                'conditions': self._generate_patient_conditions(),
                'allergies': random.sample(
                    ['Penicillin', 'Peanuts', 'Latex', 'Shellfish', 'None'],
                    random.randint(0, 3)
                ),
                'emergency_contact': {
                    'name': self.faker.name(),
                    'relationship': random.choice(['Spouse', 'Parent', 'Child', 'Sibling']),
                    'phone': self.faker.phone_number()
                }
            }
            data.append(patient)
            
        df = pd.DataFrame(data)
        df['conditions'] = df['conditions'].apply(json.dumps)
        df['allergies'] = df['allergies'].apply(json.dumps)
        df['emergency_contact'] = df['emergency_contact'].apply(json.dumps)
        return df
        
    def generate_visits(self, 
                       patient_ids: List[str],
                       start_date: Optional[datetime] = None,
                       count: int = 200) -> pd.DataFrame:
        """Generate medical visit records."""
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365)
            
        data = []
        visit_types = ['Routine Check-up', 'Follow-up', 'Emergency', 'Specialist Consultation']
        
        for _ in range(count):
            vitals = self._generate_vitals()
            visit_date = self.faker.date_time_between(
                start_date=start_date,
                end_date='now'
            )
            
            visit = {
                'visit_id': f"V{str(uuid.uuid4())[:8]}",
                'patient_id': random.choice(patient_ids),
                'visit_date': visit_date,
                'visit_type': random.choice(visit_types),
                'provider': self.faker.name(),
                'specialty': random.choice(self.specialists),
                'chief_complaint': self.faker.sentence(),
                'diagnosis': random.choice(self.conditions),
                'procedures': random.sample(
                    self.procedures,
                    random.randint(0, 3)
                ),
                'medications_prescribed': random.sample(
                    self.medications,
                    random.randint(0, 3)
                ),
                'blood_pressure_systolic': vitals['blood_pressure_systolic'],
                'blood_pressure_diastolic': vitals['blood_pressure_diastolic'],
                'heart_rate': vitals['heart_rate'],
                'temperature': vitals['temperature'],
                'respiratory_rate': vitals['respiratory_rate'],
                'oxygen_saturation': vitals['oxygen_saturation'],
                'notes': self.faker.text(max_nb_chars=200)
            }
            data.append(visit)
            
        df = pd.DataFrame(data)
        df['procedures'] = df['procedures'].apply(json.dumps)
        df['medications_prescribed'] = df['medications_prescribed'].apply(json.dumps)
        return df.sort_values('visit_date')
        
    def generate_lab_results(self,
                           visit_ids: List[str],
                           count: int = 300) -> pd.DataFrame:
        """Generate laboratory test results."""
        data = []
        lab_tests = {
            'Complete Blood Count': {
                'WBC': ('10^3/µL', 4.5, 11.0),
                'RBC': ('10^6/µL', 4.0, 5.5),
                'Hemoglobin': ('g/dL', 12.0, 16.0),
                'Platelets': ('10^3/µL', 150, 450)
            },
            'Basic Metabolic Panel': {
                'Glucose': ('mg/dL', 70, 100),
                'Calcium': ('mg/dL', 8.5, 10.5),
                'Sodium': ('mEq/L', 135, 145),
                'Potassium': ('mEq/L', 3.5, 5.0)
            },
            'Lipid Panel': {
                'Total Cholesterol': ('mg/dL', 150, 240),
                'HDL': ('mg/dL', 40, 60),
                'LDL': ('mg/dL', 70, 130),
                'Triglycerides': ('mg/dL', 50, 150)
            }
        }
        
        for _ in range(count):
            test_panel = random.choice(list(lab_tests.keys()))
            test_components = lab_tests[test_panel]
            
            results = []
            for component, (unit, min_val, max_val) in test_components.items():
                # Sometimes generate slightly out of range values
                range_extension = 0.2 * (max_val - min_val)
                actual_min = min_val - range_extension if random.random() < 0.1 else min_val
                actual_max = max_val + range_extension if random.random() < 0.1 else max_val
                
                result = {
                    'component': component,
                    'value': round(random.uniform(actual_min, actual_max), 1),
                    'unit': unit,
                    'reference_range': f"{min_val}-{max_val}",
                    'flag': 'H' if actual_max > max_val else 'L' if actual_min < min_val else 'N'
                }
                results.append(result)
            
            lab_record = {
                'lab_id': f"L{str(uuid.uuid4())[:8]}",
                'visit_id': random.choice(visit_ids),
                'test_name': test_panel,
                'collection_date': self.faker.date_time_between(
                    start_date='-30d',
                    end_date='now'
                ),
                'results': results,
                'status': random.choice(['Final', 'Preliminary', 'Corrected']),
                'performing_lab': f"{self.faker.company()} Laboratories"
            }
            data.append(lab_record)
            
        df = pd.DataFrame(data)
        df['results'] = df['results'].apply(json.dumps)
        return df.sort_values('collection_date')
        
    def generate(self, num_records: int = 100, include_vitals: bool = True) -> pd.DataFrame:
        """Generate synthetic healthcare data."""
        
        # Store generation parameters
        self.last_params = {
            'num_records': num_records,
            'include_vitals': include_vitals
        }
        
        # Generate data
        data = self._generate_data(num_records, include_vitals)
        
        # Store the generated data
        self.data = data
        
        return data

    def _save_impl(self, filename: str, directory: str, **kwargs) -> None:
        """Save healthcare data to CSV."""
        self.data.to_csv(f"{directory}/{filename}.csv", index=False)
        print(f"Data saved to {directory}/{filename}.csv")
