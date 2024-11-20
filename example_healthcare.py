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
    patients_df = generator.generate_patients(count=100)
    
    # Clean up JSON fields before saving
    json_columns = ['conditions', 'allergies', 'emergency_contact']
    for col in json_columns:
        patients_df[col] = patients_df[col].apply(lambda x: json.loads(x))
    
    print("\nPatient Demographics:")
    print(patients_df[['patient_id', 'first_name', 'last_name', 'birth_date', 'gender', 'blood_type']].to_string())
    

    # Show detailed information for one patient
    print("\nDetailed Patient Information:")
    patient = patients_df.iloc[0]
    conditions = patient['conditions']
    conditions = [f"- {c['condition']} (Status: {c['status']})" for c in conditions]
    allergies = patient['allergies']
    allergies = [f"- {a}" for a in allergies]
    emergency_contact = patient['emergency_contact']

    print(f"\nPatient ID: {patient['patient_id']}")
    print(f"Name: {patient['first_name']} {patient['last_name']}")
    print("Medical Conditions:")
    for condition in conditions:
        print(condition)
    print("Allergies:")
    for allergy in allergies:
        print(allergy)
    print(f"Emergency Contact: {emergency_contact}")
    
    # Generate visit data
    print("\n=== Visit Records ===")
    visits_df = generator.generate_visits(
        patient_ids=patients_df['patient_id'].tolist(),
        count=10
    )
    
    # Clean up JSON fields in visits
    json_columns = ['procedures', 'medications_prescribed']
    for col in json_columns:
        visits_df[col] = visits_df[col].apply(lambda x: json.loads(x))
    
    print(visits_df[['visit_id', 'patient_id', 'visit_date', 'visit_type', 'diagnosis']].to_string())
    
    # Show detailed information for one visit
    print("\nDetailed Visit Information:")
    visit = visits_df.iloc[0]
    print(f"\nVisit ID: {visit['visit_id']}")
    print(f"Provider: {visit['provider']} ({visit['specialty']})")
    print(f"Chief Complaint: {visit['chief_complaint']}")
    print(f"Procedures: {visit['procedures']}")
    print(f"Medications: {visit['medications_prescribed']}")
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
    
    # Clean up JSON fields in lab results
    lab_results_df['results'] = lab_results_df['results'].apply(lambda x: json.loads(x))
    
    print(lab_results_df[['lab_id', 'visit_id', 'test_name', 'collection_date', 'status']].to_string())
    
    # Show detailed results for one lab test
    print("\nDetailed Lab Results:")
    lab_result = lab_results_df.iloc[0]
    print(f"\nLab ID: {lab_result['lab_id']}")
    print(f"Test: {lab_result['test_name']}")
    print(f"Performing Lab: {lab_result['performing_lab']}")
    print("\nResults:")
    for result in lab_result['results']:
        print(f"  - {result['component']}: {result['value']} {result['unit']}")
        print(f"    Reference Range: {result['reference_range']} {result['unit']}")
        print(f"    Flag: {result['flag']}")
    
    # Save data to files
    patients_df.to_csv('example_patients.csv', index=False)
    visits_df.to_json('example_visits.json', orient='records', lines=True)
    lab_results_df.to_pickle('example_lab_results.pkl')

if __name__ == '__main__':
    main()
