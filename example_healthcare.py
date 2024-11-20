from datagen import HealthcareGenerator
import pandas as pd
import json
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 120)

def main():
    # Initialize generator with seed for reproducibility
    generator = HealthcareGenerator(seed=42)
    
    # Generate patient data
    print("\n=== Patient Data ===")
    patients_df = generator.generate_patients(count=5)
    print("\nPatient Demographics:")
    print(patients_df[['patient_id', 'first_name', 'last_name', 'birth_date', 'gender', 'blood_type']].to_string())
    
    # Show detailed information for one patient
    print("\nDetailed Patient Information:")
    patient = patients_df.iloc[0]
    print(f"\nPatient ID: {patient['patient_id']}")
    print(f"Name: {patient['first_name']} {patient['last_name']}")
    print(f"Medical Conditions: {json.loads(patient['conditions'])}")
    print(f"Allergies: {json.loads(patient['allergies'])}")
    print(f"Emergency Contact: {json.loads(patient['emergency_contact'])}")
    
    # Generate visit data
    print("\n=== Visit Records ===")
    visits_df = generator.generate_visits(
        patient_ids=patients_df['patient_id'].tolist(),
        count=10
    )
    print(visits_df[['visit_id', 'patient_id', 'visit_date', 'visit_type', 'diagnosis']].to_string())
    
    # Show detailed information for one visit
    print("\nDetailed Visit Information:")
    visit = visits_df.iloc[0]
    print(f"\nVisit ID: {visit['visit_id']}")
    print(f"Provider: {visit['provider']} ({visit['specialty']})")
    print(f"Chief Complaint: {visit['chief_complaint']}")
    print(f"Procedures: {json.loads(visit['procedures'])}")
    print(f"Medications: {json.loads(visit['medications_prescribed'])}")
    print(f"Vitals:")
    print(f"  - BP: {visit['blood_pressure_systolic']}/{visit['blood_pressure_diastolic']}")
    print(f"  - Heart Rate: {visit['heart_rate']}")
    print(f"  - Temperature: {visit['temperature']}")
    print(f"  - Respiratory Rate: {visit['respiratory_rate']}")
    print(f"  - O2 Saturation: {visit['oxygen_saturation']}%")
    
    # Generate lab results
    print("\n=== Laboratory Results ===")
    lab_results_df = generator.generate_lab_results(
        visit_ids=visits_df['visit_id'].tolist(),
        count=8
    )
    print(lab_results_df[['lab_id', 'visit_id', 'test_name', 'collection_date', 'status']].to_string())
    
    # Show detailed results for one lab test
    print("\nDetailed Lab Results:")
    lab_result = lab_results_df.iloc[0]
    print(f"\nLab ID: {lab_result['lab_id']}")
    print(f"Test: {lab_result['test_name']}")
    print(f"Performing Lab: {lab_result['performing_lab']}")
    print("\nResults:")
    for result in json.loads(lab_result['results']):
        print(f"  - {result['component']}: {result['value']} {result['unit']}")
        print(f"    Reference Range: {result['reference_range']} {result['unit']}")
        print(f"    Flag: {result['flag']}")
    
    # Save data to files
    patients_df.to_csv('example_patients.csv', index=False)
    visits_df.to_json('example_visits.json', orient='records', lines=True)
    lab_results_df.to_pickle('example_lab_results.pkl')

if __name__ == '__main__':
    main()
