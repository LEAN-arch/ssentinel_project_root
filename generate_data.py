import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date as date_type
import random
import uuid # For more unique encounter IDs if needed
import io # For reading your provided CSV string
from pathlib import Path # Added for path manipulation

# --- Your Provided CSV Data (as a string) ---
csv_data_string = """encounter_id,patient_id,encounter_date,encounter_type,age,gender,zone_id,clinic_id,chw_id,min_spo2_pct,vital_signs_temperature_celsius,max_skin_temp_celsius,condition,ai_risk_score,ai_followup_priority_score,patient_reported_symptoms,test_type,test_result,referral_status,item,item_stock_agg_zone,consumption_rate_per_day,fall_detected_today,avg_daily_steps,days_task_overdue,medication_adherence_self_report,tb_contact_traced,sample_collection_date,sample_status,rejection_reason,test_turnaround_days,pregnancy_status,chronic_condition_flag,hrv_rmssd_ms,movement_activity_level,ambient_heat_index_c,ppe_compliant_flag,signs_of_fatigue_observed_flag,rapid_psychometric_distress_score,referred_to_facility_id,referral_outcome,referral_outcome_date,referral_reason,quantity_dispensed,notes,diagnosis_code_icd10,physician_id,avg_spo2,resting_heart_rate,avg_sleep_duration_hrs,sleep_score_pct,stress_level_score,hiv_viral_load_copies_ml,key_chronic_conditions_summary,patient_latitude,patient_longitude,chw_visit
SENC_001,PID_001,2023-11-15T09:00:00Z,CHW_HOME_VISIT,35,Female,ZoneA,CLINIC01,CHW001,97,37.1,37.0,Wellness Visit,25,30,none,None,N/A,N/A,ORS Packet,100,0.5,0,5600,0,Good,0,,Accepted by Lab,,0.0,0,0,55,2,28,1,0,2,FACIL_DEST_01,Attended,2023-11-16T00:00:00Z,Routine Checkup,5,Routine visit,Z00.0,DrA,97,65,7.5,85,20,,Well,-1.286,36.817,1
SENC_002,PID_002,2023-11-15T10:30:00Z,CHW_HOME_VISIT,68,Male,ZoneB,CLINIC02,CHW002,91,38.5,38.2,Pneumonia,85,90,"cough;fever;short breath",Sputum-GeneXpert,Pending,Pending,Amoxicillin,50,1.2,0,1200,1,Fair,0,2023-11-15T10:35:00Z,Pending Collection by CHW,,NaN,0,1,30,1,33,1,1,8,FACIL_DEST_02,Pending,,Urgent Clinical Review,10,Severe cough,J18.9,DrB,92,88,5.0,60,75,,Pneumonia,-1.29,36.82,1
SENC_003,PID_003,2023-11-14T11:15:00Z,CLINIC_INTAKE,45,Female,ZoneA,CLINIC01,N/A_ClinicStaff,98,36.8,36.5,Hypertension Check,40,45,headache,BP Check,140/90,N/A,Lisinopril,200,0.2,0,8000,0,Good,0,,,0.0,1,0,60,2,27,1,0,1,N/A,N/A,,Routine BP,30,Follow up,I10,DrA,98,70,6.5,75,30,,Hypertension,-1.285,36.815,0
SENC_004,PID_001,2023-11-13T09:30:00Z,CHW_HOME_VISIT,35,Female,ZoneA,CLINIC01,CHW001,96,37.0,36.8,Wellness Visit,22,25,none,None,N/A,N/A,ORS Packet,98,0.5,0,6100,0,Good,0,,,0.0,0,0,58,3,29,1,0,2,N/A,N/A,,Previous well visit,5,All good,Z00.0,DrA,96,66,7.0,80,22,,Well,-1.286,36.817,1
SENC_005,PID_004,2023-11-15T14:00:00Z,CHW_ALERT_RESPONSE,72,Male,ZoneC,HUB01,CHW003,88,39.6,39.3,TB,92,95,"fever;severe cough;weight loss",Sputum-GeneXpert,Positive,Completed,Paracetamol,30,2.1,1,800,3,Poor,1,2023-11-15T14:05:00Z,Result Entered,,0.5,0,1,25,0,42,0,1,12,FACIL_DEST_03,Treatment Started,2023-11-16T00:00:00Z,Confirm TB & Start DOTS,20,Fall today,A15.0,DrC,89,95,4.5,50,85,2000,TB;Fall,-1.295,36.825,1
SENC_006,PID_005,2023-11-12T11:00:00Z,CLINIC_INTAKE,28,Female,ZoneB,CLINIC02,N/A_ClinicStaff,99,36.5,36.3,Antenatal Care,15,20,none,HIV-Rapid,Negative,N/A,Iron-Folate,150,0.1,0,7500,0,Excellent,0,2023-11-12T11:05:00Z,Accepted by Lab,,0.1,1,0,65,3,26,1,0,1,N/A,N/A,,ANC Visit 1,60,Routine ANC,Z34.0,DrB,99,60,8.0,90,15,,,Pregnant;-1.291,36.821,0
SENC_007,PID_002,2023-11-10T15:00:00Z,CHW_SCHEDULED_DOTS,68,Male,ZoneB,CLINIC02,CHW002,93,37.5,37.3,Pneumonia,70,75,persistent cough,Sputum-AFB,Negative,Completed,Amoxicillin,45,1.2,0,1100,0,Fair,1,2023-11-09T10:35:00Z,Result Entered,,1.0,0,1,33,1,32,1,0,7,N/A,N/A,,DOTS visit,10,Improving,J18.9,DrB,93,85,5.5,65,70,,Pneumonia,-1.29,36.82,1
SENC_008,CHW001_SELF,2023-11-15T08:00:00Z,WORKER_SELF_CHECK,30,Female,ZoneA,HUB01,CHW001,98,36.7,36.5,Wellness Visit,10,15,none,None,N/A,N/A,,,,,0,9500,0,Good,0,,,0.0,0,0,70,3,28,1,0,2,N/A,N/A,,CHW Self Check,,,,68,8.0,88,10,,,CHW,-1.286,36.817,0
SENC_009,PID_006,2023-11-14T10:00:00Z,CHW_HOME_VISIT,5,Male,ZoneD,CLINIC03,CHW004,94,39.1,38.8,Malaria,80,85,"fever;chills;vomiting",RDT-Malaria,Positive,Pending,ACT,60,0.8,0,500,0,Poor,0,2023-11-14T10:05:00Z,Sample collected,,0.1,0,0,45,1,38,1,1,10,FACIL_DEST_04,Pending,,Urgent Malaria,14,Severe symptoms,B54,DrD,95,110,6.0,70,60,,Malaria,-1.28,36.83,1
SENC_010,PID_007,2023-11-15T12:00:00Z,CLINIC_INTAKE,55,Male,ZoneA,CLINIC01,N/A_ClinicStaff,97,37.0,36.8,Diabetes Check,50,55,fatigue,Blood Glucose,180mg/dL,N/A,Metformin,120,0.3,0,4000,0,Fair,0,,,0.0,0,1,50,2,29,1,0,3,N/A,N/A,,Routine DM,30,Checkup,E11.9,DrA,97,75,6.0,70,40,,Diabetes,-1.285,36.815,0
SENC_011,PID_004,2023-11-08T09:00:00Z,CHW_HOME_VISIT,72,Male,ZoneC,HUB01,CHW003,90,38.0,37.8,TB,85,90,"cough;fever",Sputum-AFB,Negative,Completed,Paracetamol,25,2.1,0,750,0,Poor,1,2023-11-07T14:05:00Z,Result Entered,,1.0,0,1,28,0,40,0,1,10,FACIL_DEST_03,Treatment Followup,2023-11-09T00:00:00Z,TB Followup,20,Ongoing care,A15.0,DrC,91,92,5.0,55,80,1500,TB,-1.295,36.825,1
SENC_012,PID_008,2023-11-15T10:15:00Z,CHW_ALERT_RESPONSE,2,Female,ZoneE,CLINIC03,CHW005,85,40.1,39.8,Severe Dehydration,98,99,"lethargy;sunken eyes;no tears",None,N/A,Pending,ORS Packet,80,1.5,0,200,0,N/A,0,,,1.0,0,0,15,0,35,1,1,14,FACIL_DEST_05,Pending,,Urgent Dehydration,10,Critical,E86,DrE,86,130,3.0,40,95,,Dehydration,-1.275,36.835,1
SENC_013,PID_009,2023-11-15T11:30:00Z,CLINIC_INTAKE,42,Male,ZoneB,CLINIC02,N/A_ClinicStaff,96,37.2,37.0,Injury - Minor,30,35,pain in arm,None,N/A,N/A,Bandages,500,5,0,7000,0,N/A,0,,,0.0,0,0,55,2,30,1,0,2,N/A,N/A,,Minor Injury,5,Sutures,S51.9,DrB,96,72,7.0,80,25,,Injury,-1.292,36.822,0
SENC_014,PID_010,2023-11-14T16:00:00Z,CHW_HOME_VISIT,60,Female,ZoneA,CLINIC01,CHW001,93,37.8,37.5,TB,82,88,"persistent cough;weight loss;night sweats",Sputum-GeneXpert,Positive,Pending,TB-Regimen,20,0.2,0,2500,2,Fair,0,2023-11-14T16:05:00Z,Pending Collection by CHW,,0.2,1,1,38,1,34,1,1,9,FACIL_DEST_01,Pending,,New TB Case,28,Start Treatment,A15.0,DrA,94,80,6.0,70,65,1800,TB;Anemia,-1.283,36.813,1
SENC_015,PID_006,2023-11-07T09:45:00Z,CHW_HOME_VISIT,5,Male,ZoneD,CLINIC03,CHW004,98,37.0,36.9,Malaria,20,25,"feeling much better",RDT-Malaria,Positive,Completed,ACT,55,0.8,0,1500,0,Good,0,2023-11-06T10:05:00Z,Result Entered,,1.0,0,0,50,2,30,1,0,3,FACIL_DEST_04,Treatment Complete,2023-11-07T00:00:00Z,Malaria Followup,14,Recovered,B54,DrD,98,90,7.5,85,20,,Malaria,-1.28,36.83,1
SENC_016,PID_011,2023-11-15T13:00:00Z,CLINIC_INTAKE,33,Male,ZoneC,HUB01,N/A_ClinicStaff,99,36.4,36.2,Wellness Visit,18,22,routine check,None,N/A,N/A,Multivitamins,300,0.1,0,10500,0,Excellent,0,,,0.0,0,0,68,3,25,1,0,1,N/A,N/A,,Annual Physical,30,Healthy,Z00.0,DrC,99,62,7.8,89,18,,,Well,-1.296,36.826,0
SENC_017,PID_012,2023-11-13T10:00:00Z,CHW_HOME_VISIT,78,Female,ZoneE,CLINIC03,CHW005,92,37.2,37.0,Hypertension,65,70,dizzy;headache,BP Check,160/95,Pending,Amlodipine,80,0.4,0,3200,1,Fair,0,,,0.0,0,1,35,1,36,1,0,7,FACIL_DEST_05,Pending,,HTN Management,28,Adjust meds,I10,DrE,93,88,6.2,72,55,,Hypertension;-1.276,36.836,1
SENC_018,PID_013,2023-11-15T16:30:00Z,CHW_ALERT_RESPONSE,8,Male,ZoneA,CLINIC01,CHW001,96,40.5,40.2,Fever Unknown Origin,90,92,"high fever;listless",None,N/A,Pending,Paracetamol,100,1.0,0,600,0,N/A,0,,,0.0,0,0,20,0,33,1,1,13,FACIL_DEST_01,Pending,,Investigate Fever,10,High fever,R50.9,DrA,97,115,4.0,50,80,,Fever,-1.282,36.812,1
SENC_019,CHW002_SELF,2023-11-14T17:00:00Z,WORKER_SELF_CHECK,42,Male,ZoneB,HUB01,CHW002,97,36.9,36.7,Wellness Visit,12,18,tired,None,N/A,N/A,,,,,0,7800,0,Fair,0,,,0.0,0,0,48,2,30,1,0,2,N/A,N/A,,CHW End Day Check,,,,70,7.0,80,35,,,CHW,-1.29,36.82,0
SENC_020,PID_014,2023-11-15T09:15:00Z,CLINIC_INTAKE,22,Female,ZoneD,CLINIC03,N/A_ClinicStaff,98,37.5,37.3,STI Screening,35,40,none,Syphilis RPR,Negative,N/A,Condoms,50,0.05,0,8800,0,N/A,0,2023-11-15T09:20:00Z,Rejected by Lab,Poor Sample Quality,0.1,1,0,62,2,29,1,0,3,N/A,N/A,,STI Screen,10,Resample needed,Z11.3,DrD,98,68,7.2,82,28,,STI Screen;-1.281,36.831,0
"""

# --- Configuration for Data Generation ---
NUM_PATIENTS_NEW = 250  # Generate data for 250 *new* unique patients
NUM_CHWS = 7
NUM_ZONES = 8 
NUM_CLINICS = 4
DAYS_OF_DATA = 180 # Generate data for the last ~6 months for better trends
AVG_ENCOUNTERS_PER_PATIENT = 7 # More encounters per patient on average
STD_DEV_ENCOUNTERS = 2.5

END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(days=DAYS_OF_DATA - 1)

ENCOUNTER_TYPES_CHW = ["CHW_HOME_VISIT", "COMMUNITY_SCREENING", "FOLLOW_UP_VISIT", 
                       "CHW_ALERT_RESPONSE", "CHW_SCHEDULED_DOTS", "NUTRITION_COUNSELING", 
                       "HEALTH_EDUCATION_SESSION", "REFERRAL_FOLLOW_UP"]
ENCOUNTER_TYPES_CLINIC = ["CLINIC_INTAKE", "CLINIC_CONSULTATION", "EMERGENCY_VISIT", 
                          "LAB_ONLY_VISIT", "IMMUNIZATION_VISIT", "PHARMACY_REFILL"]
WORKER_SELF_CHECK = ["WORKER_SELF_CHECK"]

CONDITIONS = [
    "Wellness Visit", "Malaria", "ARI (Acute Respiratory Infection)", "Pneumonia", "Diarrhea", 
    "Antenatal Care", "Postnatal Care", "TB Suspect", "TB Confirmed", "Hypertension Check", 
    "Diabetes Check", "Malnutrition Screen", "Minor Injury", "Skin Infection", "Fever Unknown Origin",
    "CHW Follow-up", "Asthma Exacerbation", "Chronic Disease Management", "Family Planning"
]
CONDITIONS_WEIGHTS = [
    0.12, 0.15, 0.10, 0.06, 0.07, 0.05, 0.03, 0.05, 0.03, 0.06, 
    0.05, 0.04, 0.03, 0.02, 0.04, 0.04, 0.02, 0.02, 0.02
]
if abs(sum(CONDITIONS_WEIGHTS) - 1.0) > 1e-6: # Normalize weights
    CONDITIONS_WEIGHTS = np.array(CONDITIONS_WEIGHTS) / np.sum(CONDITIONS_WEIGHTS)

GENDERS = ["Male", "Female"] 
GENDER_WEIGHTS = [0.49, 0.51]

SYMPTOM_LIST_EXPANDED = [
    "fever", "high fever", "persistent cough", "mild cough", "headache", "severe headache", "fatigue", "extreme fatigue",
    "chills", "body ache", "sore throat", "runny nose", "shortness of breath", "difficulty breathing",
    "diarrhea", "watery diarrhea", "vomiting", "nausea", "abdominal pain", "stomach cramps",
    "skin rash", "itchy rash", "dizziness", "lightheadedness", "joint pain", "swollen joints",
    "muscle pain", "weight loss", "night sweats", "loss of appetite", "chest pain", "confusion",
    "lethargy", "sunken eyes", "no tears", "rapid heartbeat", "swelling in legs", "blurry vision",
    "wheezing", "palpitations", "general malaise", "dehydration signs"
]

REFERRAL_STATUSES = ["Not Referred", "Pending", "Completed", "Declined", "Expired", "In Progress"]
REFERRAL_STATUS_WEIGHTS = [0.50, 0.18, 0.15, 0.05, 0.02, 0.10]

MED_ADHERENCE = ["Good", "Fair", "Poor", "Unknown", "Not Applicable"]
MED_ADHERENCE_WEIGHTS = [0.50, 0.20, 0.15, 0.10, 0.05]

ITEMS = ["ORS Packet", "Amoxicillin 250mg", "Paracetamol 500mg", "ACT Malaria Adult", "ACT Malaria Child",
         "TB Regimen Cat I", "TB Regimen Cat II", "Iron-Folate Tabs", "Zinc Syrup", "Metformin 500mg", 
         "Amlodipine 5mg", "Salbutamol Inhaler", "RDT-Malaria Kit", "Sputum Container", "Syringes 5ml", 
         "Gloves (Pair)", "Plumpy Nut", "Deworming Tablet", "Vitamin A Capsule"]

TEST_TYPES = ["RDT-Malaria", "Sputum-GeneXpert", "Sputum-AFB", "HIV-Rapid", 
              "Blood Glucose", "BP Check", "Urine Dipstick", "HbA1c", "Syphilis RPR", "Pregnancy Test", "Hemoglobin Test"]
TEST_RESULTS_COMMON = ["Positive", "Negative", "Pending", "Indeterminate", "Not Done", "Reactive", "Non-Reactive"]
TEST_RESULTS_WEIGHTS = [0.12, 0.58, 0.10, 0.05, 0.05, 0.05, 0.05] # Ensure sums to 1
if abs(sum(TEST_RESULTS_WEIGHTS) - 1.0) > 1e-6:
    TEST_RESULTS_WEIGHTS = np.array(TEST_RESULTS_WEIGHTS) / np.sum(TEST_RESULTS_WEIGHTS)

SAMPLE_STATUSES = ["Collected by CHW", "Transport to Lab", "Received at Lab", "Accepted by Lab", 
                   "Processing", "Result Entered", "Rejected", "Awaiting Collection"]
SAMPLE_REJECTION_REASONS = ["Poor Sample Quality", "Incorrect Labeling", "Insufficient Volume", 
                            "Contaminated", "Delay in Transport", "Hemolyzed Sample", "Improper Storage"]

# --- Helper Functions ---
def random_date_in_range(start_date: datetime, end_date: datetime) -> datetime:
    time_between_dates = end_date - start_date
    random_number_of_seconds = random.randint(0, int(time_between_dates.total_seconds()))
    random_date_val = start_date + timedelta(seconds=random_number_of_seconds)
    return random_date_val

def generate_realistic_symptoms(condition: str, patient_age: int, num_symptoms_avg: int = 2) -> str:
    base_symptoms = []
    # ... (symptom generation logic from previous script, can be enhanced further) ...
    if "Malaria" in condition: base_symptoms.extend(random.sample(["fever", "chills", "headache", "fatigue", "muscle pain", "vomiting"], k=random.randint(2,4)))
    if "ARI" in condition or "Pneumonia" in condition: base_symptoms.extend(random.sample(["cough", "fever", "shortness of breath", "sore throat", "runny nose", "chest pain"],k=random.randint(2,4)))
    if "Diarrhea" in condition: base_symptoms.extend(random.sample(["diarrhea", "abdominal pain", "vomiting", "dehydration signs", "nausea"],k=random.randint(2,4)))
    if "TB" in condition: base_symptoms.extend(random.sample(["persistent cough", "fever", "night sweats", "weight loss", "chest pain", "fatigue"],k=random.randint(2,4)))
    if "Hypertension" in condition and random.random() < 0.4: base_symptoms.extend(random.sample(["headache", "dizziness", "blurry vision"], k=random.randint(1,2)))
    if "Diabetes" in condition and random.random() < 0.5: base_symptoms.extend(random.sample(["fatigue", "increased thirst", "blurry vision", "frequent urination"],k=random.randint(1,2)))
    if "Antenatal" in condition and random.random() < 0.3: base_symptoms.extend(random.sample(["nausea", "fatigue", "swelling in legs"], k=random.randint(1,2)))


    num_needed_for_avg = max(0, int(random.normalvariate(num_symptoms_avg, 1)))
    num_additional_symptoms = max(0, num_needed_for_avg - len(base_symptoms))
    
    potential_additional_symptoms = [s for s in SYMPTOM_LIST_EXPANDED if s not in base_symptoms]
    if potential_additional_symptoms: # Check if list is not empty
        additional_symptoms = random.sample(
            potential_additional_symptoms, 
            min(num_additional_symptoms, len(potential_additional_symptoms))
        )
        base_symptoms.extend(additional_symptoms)
    
    all_symptoms = list(set(base_symptoms)) # Unique symptoms
    if not all_symptoms:
        if condition == "Wellness Visit" or "Check" in condition :
            return "none" if random.random() < 0.9 else random.choice(["general checkup", "routine follow-up"])
        else: # For potentially sick conditions but no specific symptoms, add a generic one
            return random.choice(["general malaise", "feeling unwell"])
            
    return "; ".join(all_symptoms)

print("Starting data generation...")

# Load existing data
try:
    existing_df = pd.read_csv(io.StringIO(csv_data_string))
    print(f"Successfully read existing {len(existing_df)} records from provided string.")
    date_cols_to_parse = ['encounter_date', 'sample_collection_date', 'referral_outcome_date']
    for col in date_cols_to_parse:
        if col in existing_df.columns:
            existing_df[col] = pd.to_datetime(existing_df[col], errors='coerce', utc=True) # Assume UTC if Z present
            if existing_df[col].dt.tz is not None: # Make timezone naive for consistency
                existing_df[col] = existing_df[col].dt.tz_localize(None)
    
    max_existing_enc_id_num = 0
    if 'encounter_id' in existing_df.columns and not existing_df['encounter_id'].empty:
        numeric_ids = existing_df['encounter_id'].astype(str).str.extract(r'(\d+)')
        if not numeric_ids.empty and numeric_ids[0].notna().any():
            max_existing_enc_id_num = int(numeric_ids[0].dropna().astype(int).max())
    encounter_counter = max_existing_enc_id_num + 1
    print(f"Starting new encounter_ids from: {encounter_counter}")

except Exception as e:
    print(f"Error reading or processing existing CSV data: {e}. Starting with an empty DataFrame for new records.")
    existing_df = pd.DataFrame()
    encounter_counter = 1

# --- Generate New Patient Profiles ---
new_patient_profiles = []
existing_patient_ids_set = set(existing_df['patient_id'].dropna().unique()) if 'patient_id' in existing_df.columns else set()

max_pid_num_existing = 0
if existing_patient_ids_set:
    for pid_str_val in existing_patient_ids_set:
        if isinstance(pid_str_val, str):
            num_part_val = ''.join(filter(str.isdigit, pid_str_val))
            if num_part_val:
                max_pid_num_existing = max(max_pid_num_existing, int(num_part_val))
start_new_pid_num = max_pid_num_existing + 1

chw_ids_list = [f"CHW_{i:03d}" for i in range(1, NUM_CHWS + 1)]
zone_ids_list = [f"ZONE_{chr(65+i)}" for i in range(NUM_ZONES)] 
clinic_ids_list = [f"CLINIC_{i:03d}" for i in range(1, NUM_CLINICS + 1)]

for i in range(NUM_PATIENTS_NEW):
    pid = f"PID_{start_new_pid_num + i:04d}"
    new_patient_profiles.append({
        "patient_id": pid, "age": random.randint(0, 90),
        "gender": random.choices(GENDERS, weights=GENDER_WEIGHTS, k=1)[0],
        "zone_id": random.choice(zone_ids_list), "clinic_id": random.choice(clinic_ids_list),
        "chronic_condition_flag": 1 if random.random() < 0.25 else 0, 
        "key_chronic_conditions_summary": random.choice(["Hypertension", "Diabetes", "Asthma", "HIV", "COPD", ""]) if random.random() < 0.25 else "",
        "patient_latitude": round(random.uniform(-1.40, -1.15), 5), 
        "patient_longitude": round(random.uniform(36.70, 37.00), 5)
    })

# Combine existing patient profiles (if any) with new ones
all_patient_profiles_for_enc_gen = []
if 'patient_id' in existing_df.columns and not existing_df.empty:
    # Get unique patient profiles from existing_df
    unique_existing_patients = existing_df.drop_duplicates(subset=['patient_id'])
    for _, row in unique_existing_patients.iterrows():
        all_patient_profiles_for_enc_gen.append({
            "patient_id": row['patient_id'], "age": int(row.get('age', random.randint(0,90))), 
            "gender": row.get('gender', random.choices(GENDERS, weights=GENDER_WEIGHTS, k=1)[0]),
            "zone_id": row.get('zone_id', random.choice(zone_ids_list)), 
            "clinic_id": row.get('clinic_id', random.choice(clinic_ids_list)),
            "chronic_condition_flag": int(row.get('chronic_condition_flag', 1 if random.random() < 0.15 else 0)),
            "key_chronic_conditions_summary": str(row.get('key_chronic_conditions_summary', "")),
            "patient_latitude": float(row.get('patient_latitude', round(random.uniform(-1.40, -1.15), 5))),
            "patient_longitude": float(row.get('patient_longitude', round(random.uniform(36.70, 37.00), 5)))
        })
all_patient_profiles_for_enc_gen.extend(new_patient_profiles)
random.shuffle(all_patient_profiles_for_enc_gen) 

all_generated_records_list = []

for patient_prof in all_patient_profiles_for_enc_gen:
    num_encounters_for_this_pat = max(1, int(random.normalvariate(AVG_ENCOUNTERS_PER_PATIENT, STD_DEV_ENCOUNTERS)))
    
    # Sort encounters by date for this patient to make stock/consumption more logical over time
    patient_encounter_dates = sorted([random_date_in_range(START_DATE, END_DATE) for _ in range(num_encounters_for_this_pat)])

    last_stock_level = {} # item: stock_level

    for i_enc, encounter_datetime_obj in enumerate(patient_encounter_dates):
        assigned_chw_id_val = random.choice(chw_ids_list) if random.random() > 0.15 else "N/A_ClinicStaff"
        
        current_encounter_type = ""
        if "CHW" in assigned_chw_id_val: current_encounter_type = random.choice(ENCOUNTER_TYPES_CHW)
        elif "N/A_ClinicStaff" in assigned_chw_id_val: current_encounter_type = random.choice(ENCOUNTER_TYPES_CLINIC)
        else: current_encounter_type = "UNKNOWN_ENCOUNTER"

        current_condition = random.choices(CONDITIONS, weights=CONDITIONS_WEIGHTS, k=1)[0]
        current_symptoms = generate_realistic_symptoms(current_condition, patient_prof["age"], 2 if current_condition != "Wellness Visit" else 0)

        base_risk_score = random.uniform(5, 25) + (25 if patient_prof["chronic_condition_flag"] else 0)
        base_fup_prio = random.uniform(5, 20) + (20 if patient_prof["chronic_condition_flag"] else 0)
        # ... (enhance risk/priority logic as in previous script if needed) ...
        if any(s in current_symptoms.lower() for s in ["fever", "cough", "shortness of breath", "difficulty breathing"]):
            base_risk_score += random.uniform(15, 40)
            base_fup_prio += random.uniform(20, 50)
        if patient_prof["age"] < 5 or patient_prof["age"] > 70:
            base_risk_score += random.uniform(10, 20)
            base_fup_prio += random.uniform(10, 20)


        min_spo2_val, temp_c_val, max_skin_temp_val, avg_spo2_sim, rest_hr_sim = None, None, None, None, None
        if random.random() < 0.85: # 85% chance vitals are taken
            min_spo2_val = random.randint(88 if base_risk_score > 50 else 93, 100)
            temp_c_val = round(random.uniform(36.0, 40.5 if base_risk_score > 60 else 38.5), 1)
            max_skin_temp_val = round(temp_c_val + random.uniform(-0.3, 0.8), 1) if temp_c_val else None
            if min_spo2_val: avg_spo2_sim = min(100, min_spo2_val + random.randint(0,2))
            rest_hr_sim = random.randint(50, 120)
        
        if min_spo2_val is not None and min_spo2_val < 92: base_fup_prio = max(base_fup_prio, 80); base_risk_score = max(base_risk_score, 75)
        if temp_c_val is not None and temp_c_val > 38.8: base_fup_prio = max(base_fup_prio, 75); base_risk_score = max(base_risk_score, 70)

        fall_detected_val = 1 if random.random() < 0.02 else 0
        if fall_detected_val: base_fup_prio = max(base_fup_prio, 90); base_risk_score = max(base_risk_score, 80)
        
        current_test_type, current_test_result, current_sample_status, current_rejection_reason, current_test_tat, current_sample_coll_date = "None", "N/A", "N/A", "", np.nan, None
        if current_condition not in ["Wellness Visit", "Antenatal Care", "Postnatal Care"] and random.random() < 0.7:
            current_test_type = random.choice(TEST_TYPES)
            current_test_result = random.choices(TEST_RESULTS_COMMON, weights=TEST_RESULTS_WEIGHTS, k=1)[0]
            current_sample_status = random.choice(SAMPLE_STATUSES)
            if current_sample_status == "Rejected": current_rejection_reason = random.choice(SAMPLE_REJECTION_REASONS)
            if current_test_result not in ["Pending", "Not Done", "N/A"]: current_test_tat = round(random.uniform(0.1, 7.0), 1)
            if current_sample_status not in ["Awaiting Collection", "N/A"]:
                current_sample_coll_date = encounter_datetime_obj - timedelta(hours=random.randint(0,4), minutes=random.randint(0,59))
        
        current_referral_status = random.choices(REFERRAL_STATUSES, weights=REFERRAL_STATUS_WEIGHTS, k=1)[0]
        current_referral_reason, current_referred_to_facility, current_referral_outcome, current_referral_outcome_date = "", "", "", None
        if current_referral_status not in ["Not Referred", "N/A"]:
            current_referral_reason = f"Further investigation for {current_condition}"
            current_referred_to_facility = f"FACIL_DEST_{random.randint(1,NUM_CLINICS+2):02d}" # Refer to other clinics/hospitals
            if current_referral_status == "Completed":
                current_referral_outcome = random.choice(["Attended - Improved", "Attended - Admitted", "Service Provided", "Attended - No Change"])
                current_referral_outcome_date = encounter_datetime_obj + timedelta(days=random.randint(1,14))
            elif current_referral_status == "Declined": current_referral_outcome = "Patient Declined Services"


        current_item, current_item_stock, current_consumption_rate, current_qty_dispensed = "None", 0, 0.0, 0
        if current_condition != "Wellness Visit" or random.random() < 0.3: # Some wellness visits might get supplements
            current_item = random.choice(ITEMS)
            # Simulate stock decreasing over time for a patient if multiple encounters for same item
            if last_stock_level.get(current_item, -1) == -1 : # First time seeing this item for patient
                current_item_stock = random.randint(20, 250)
            else:
                current_item_stock = max(0, last_stock_level[current_item] - int(random.uniform(5,30))) # Simulate consumption
            last_stock_level[current_item] = current_item_stock # Update last known stock for this item

            current_consumption_rate = round(random.uniform(0.05, 3.0), 3)
            current_qty_dispensed = random.randint(1, 60) if current_item_stock > 0 else 0
            if current_qty_dispensed > current_item_stock : current_qty_dispensed = current_item_stock # Cannot dispense more than stock
            
        record_data = {
            'encounter_id': f"SENC_{encounter_counter:05d}", 'patient_id': patient_prof["patient_id"],
            'encounter_date': encounter_datetime_obj.strftime("%Y-%m-%dT%H:%M:%SZ"), 'encounter_type': current_encounter_type,
            'age': patient_prof["age"], 'gender': patient_prof["gender"], 'zone_id': patient_prof["zone_id"],
            'clinic_id': patient_prof["clinic_id"], 'chw_id': assigned_chw_id_val,
            'min_spo2_pct': min_spo2_val, 'vital_signs_temperature_celsius': temp_c_val, 'max_skin_temp_celsius': max_skin_temp_val,
            'condition': current_condition, 'ai_risk_score': round(np.clip(base_risk_score, 0, 100),1),
            'ai_followup_priority_score': round(np.clip(base_fup_prio, 0, 100),1),
            'patient_reported_symptoms': current_symptoms, 'test_type': current_test_type, 'test_result': current_test_result,
            'referral_status': current_referral_status, 'item': current_item, 'item_stock_agg_zone': current_item_stock,
            'consumption_rate_per_day': current_consumption_rate, 'fall_detected_today': fall_detected_val,
            'avg_daily_steps': random.randint(500, 15000) if random.random() < 0.6 else None,
            'days_task_overdue': random.randint(0,7) if random.random() < 0.05 else 0,
            'medication_adherence_self_report': random.choices(MED_ADHERENCE, weights=MED_ADHERENCE_WEIGHTS, k=1)[0],
            'tb_contact_traced': 1 if "TB" in current_condition and random.random() < 0.6 else 0,
            'sample_collection_date': current_sample_coll_date.strftime("%Y-%m-%dT%H:%M:%SZ") if current_sample_coll_date else None,
            'sample_status': current_sample_status, 'rejection_reason': current_rejection_reason,
            'test_turnaround_days': current_test_tat,
            'pregnancy_status': 1 if patient_prof["gender"] == "Female" and 15 < patient_prof["age"] < 49 and ("Antenatal" in current_condition or random.random() < 0.03) else 0,
            'chronic_condition_flag': patient_prof["chronic_condition_flag"],
            'hrv_rmssd_ms': random.randint(15, 120) if random.random() < 0.2 else None,
            'movement_activity_level': random.randint(1,5) if random.random() < 0.4 else None,
            'ambient_heat_index_c': round(random.uniform(20,42),1) if random.random() < 0.15 else None,
            'ppe_compliant_flag': 1 if random.random() < 0.9 else 0,
            'signs_of_fatigue_observed_flag': 1 if base_fup_prio > 70 and random.random() < 0.4 else 0,
            'rapid_psychometric_distress_score': random.randint(0,24) if random.random() < 0.25 else None,
            'referred_to_facility_id': current_referred_to_facility, 'referral_outcome': current_referral_outcome,
            'referral_outcome_date': current_referral_outcome_date.strftime("%Y-%m-%dT%H:%M:%SZ") if current_referral_outcome_date else None,
            'referral_reason': current_referral_reason, 'quantity_dispensed': current_qty_dispensed,
            'notes': f"Encounter {i_enc+1} for {patient_prof['patient_id']}. CHW: {assigned_chw_id_val}.",
            'diagnosis_code_icd10': "R50.9" if "Fever" in current_condition else ("J18.9" if "Pneumonia" in current_condition else ("B54" if "Malaria" in current_condition else ("A15.0" if "TB" in current_condition else "Z00.0"))),
            'physician_id': f"Dr_{chr(65+random.randint(0,4))}" if "CLINIC" in current_encounter_type else "",
            'avg_spo2': avg_spo2_sim, 'resting_heart_rate': rest_hr_sim,
            'avg_sleep_duration_hrs': round(random.uniform(3.5,10.0),1) if random.random() < 0.3 else None,
            'sleep_score_pct': random.randint(30,99) if random.random() < 0.3 else None,
            'stress_level_score': random.randint(5, 95) if random.random() < 0.25 else None,
            'hiv_viral_load_copies_ml': random.randint(20, 200000) if "HIV" in patient_prof["key_chronic_conditions_summary"] and random.random() < 0.05 else np.nan,
            'key_chronic_conditions_summary': patient_prof["key_chronic_conditions_summary"],
            'patient_latitude': patient_prof["patient_latitude"], 'patient_longitude': patient_prof["patient_longitude"],
            'chw_visit': 1 if "CHW" in current_encounter_type else 0
        }
        all_generated_records_list.append(record_data)
        encounter_counter += 1

    # CHW Self-Check (one per CHW per few days, on average)
    for chw_id_val in chw_ids_list:
        if random.random() < (NUM_PATIENTS_NEW * AVG_ENCOUNTERS_PER_PATIENT) / (NUM_CHWS * DAYS_OF_DATA * 3) : # Adjust probability
            self_check_dt = random_date_in_range(START_DATE, END_DATE)
            all_generated_records_list.append({
                'encounter_id': f"SENC_{encounter_counter:05d}_SC", 'patient_id': f"{chw_id_val}_SELF",
                'encounter_date': self_check_dt.strftime("%Y-%m-%dT%H:%M:%SZ"), 'encounter_type': "WORKER_SELF_CHECK",
                'age': random.randint(25,55), 'gender': random.choice(GENDERS), 'zone_id': random.choice(zone_ids_list),
                'clinic_id': random.choice(clinic_ids_list), 'chw_id': chw_id_val,
                'min_spo2_pct': random.randint(95,99), 'vital_signs_temperature_celsius': round(random.uniform(36.1, 37.3),1),
                'condition': "Routine Self-Check", 'ai_risk_score': random.uniform(5,25),
                'ai_followup_priority_score': random.uniform(0,50), # Used as fatigue proxy
                'patient_reported_symptoms': random.choice(["none", "feeling tired", "minor headache", "good"]),
                'hrv_rmssd_ms': random.randint(40,90), 'movement_activity_level': random.randint(2,4),
                'ambient_heat_index_c': round(random.uniform(24,32),1), 'ppe_compliant_flag': 1,
                'signs_of_fatigue_observed_flag': 0, 'rapid_psychometric_distress_score': random.randint(0,15),
                'avg_daily_steps': random.randint(6000,12000),
                 # Fill all other columns with appropriate NA or default for self-check
                **{col: np.nan for col in existing_df.columns if col not in ['encounter_id', 'patient_id', 'encounter_date', 'encounter_type', 'age', 'gender', 'zone_id', 'clinic_id', 'chw_id', 'min_spo2_pct', 'vital_signs_temperature_celsius', 'condition', 'ai_risk_score', 'ai_followup_priority_score', 'patient_reported_symptoms', 'hrv_rmssd_ms', 'movement_activity_level', 'ambient_heat_index_c', 'ppe_compliant_flag', 'rapid_psychometric_distress_score', 'avg_daily_steps']}
            })
            encounter_counter += 1


generated_df_new = pd.DataFrame(all_generated_records_list)

# Concatenate with existing data, aligning columns
if not existing_df.empty:
    # Get all unique columns from both DataFrames
    all_cols = list(set(existing_df.columns.tolist() + generated_df_new.columns.tolist()))
    
    # Reindex both DataFrames to have all columns, filling missing with NaN
    existing_df_reindexed = existing_df.reindex(columns=all_cols, fill_value=np.nan)
    generated_df_new_reindexed = generated_df_new.reindex(columns=all_cols, fill_value=np.nan)
    
    combined_df = pd.concat([existing_df_reindexed, generated_df_new_reindexed], ignore_index=True)
else:
    combined_df = generated_df_new

# Ensure all original columns are present, fill with NA if created only in one df
final_expected_columns = list(io.StringIO(csv_data_string).readline().strip().split(','))
for col in final_expected_columns:
    if col not in combined_df.columns:
        combined_df[col] = np.nan
combined_df = combined_df[final_expected_columns] # Enforce original column order

# Final type and format conversions for CSV output
for col in ['encounter_date', 'sample_collection_date', 'referral_outcome_date']:
    if col in combined_df.columns:
        # Convert to datetime, then to string. NaT becomes empty string for CSV.
        combined_df[col] = pd.to_datetime(combined_df[col], errors='coerce').dt.strftime("%Y-%m-%dT%H:%M:%SZ").replace('NaT', '')

# Ensure numeric columns are indeed numeric, and strings are strings
# This is a final pass; ideally, types are correct from generation.
# For example, ensure 'age' is int where possible, scores are float, etc.
# This can be quite detailed depending on desired strictness for the CSV.
# For now, we rely on the generation logic for dtypes.

# --- Save to CSV ---
output_dir = Path("data_sources")
output_dir.mkdir(parents=True, exist_ok=True) 

# Choose output filename
output_filename_choice = "health_records_expanded.csv" # To overwrite the original
# output_filename_choice = "health_records_expanded_GENERATED_MORE.csv" # To create a new file

output_filepath = output_dir / output_filename_choice

combined_df.to_csv(output_filepath, index=False, date_format="%Y-%m-%dT%H:%M:%SZ") # date_format for any remaining datetime objects

print(f"\nGenerated {len(generated_df_new)} new records.")
print(f"Original records: {len(existing_df) if not existing_df.empty else 0}")
print(f"Total records in output: {len(combined_df)}")
print(f"Data saved to {output_filepath.resolve()}")
print(f"\nDate range of generated encounters: {START_DATE.strftime('%Y-%m-%d')} to {END_DATE.strftime('%Y-%m-%d')}")

print("\nValue counts for some key columns in the final combined data:")
if 'condition' in combined_df.columns:
    print("\nCondition distribution (Top 15):")
    print(combined_df['condition'].value_counts().nlargest(15))
if 'encounter_type' in combined_df.columns:
    print("\nEncounter type distribution:")
    print(combined_df['encounter_type'].value_counts())
if 'chw_id' in combined_df.columns:
    print("\nCHW ID distribution (Top 5):")
    print(combined_df['chw_id'].value_counts().nlargest(5))
if 'zone_id' in combined_df.columns:
    print("\nZone ID distribution:")
    print(combined_df['zone_id'].value_counts())
