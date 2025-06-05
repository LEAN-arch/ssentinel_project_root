# Sentinel Health Co-Pilot: Settings Explained (Thresholds & Acceptable Ranges)

This document provides context for key thresholds and values defined in `config/settings.py`.
"Acceptable" often means "not triggering an alert" or "within target goals." Criticality and specific actions depend on the complete context and escalation protocols.

## I. Health & Operational Thresholds (`config/settings.py`)

### A. Vital Signs

1.  **`ALERT_SPO2_CRITICAL_LOW_PCT = 90`**
    *   **Meaning**: Peripheral Capillary Oxygen Saturation (SpO₂) below 90%.
    *   **Criteria**: **CRITICAL**. Indicates severe hypoxia. Requires immediate medical attention, oxygen if available, and likely urgent referral.
    *   **Acceptable Range**: Generally > 94%. Values 90-94% are a warning.

2.  **`ALERT_SPO2_WARNING_LOW_PCT = 94`**
    *   **Meaning**: SpO₂ below 94%.
    *   **Criteria**: **WARNING**. Indicates mild to moderate hypoxia. Requires monitoring, re-checking, and potential clinical assessment, especially if persistent or associated with symptoms.
    *   **Acceptable Range**: Ideally >= 95%.

3.  **`ALERT_BODY_TEMP_FEVER_C = 38.0`**
    *   **Meaning**: Body temperature at or above 38.0° Celsius.
    *   **Criteria**: **WARNING (Fever Present)**. Indicates a fever. Requires monitoring, hydration, and investigation for cause.
    *   **Acceptable Range**: Typically < 37.5°C. (Normal range varies slightly, e.g., 36.1°C to 37.2°C).

4.  **`ALERT_BODY_TEMP_HIGH_FEVER_C = 39.5`**
    *   **Meaning**: Body temperature at or above 39.5° Celsius.
    *   **Criteria**: **CRITICAL (High Fever)**. Indicates a significant fever that can be dangerous. Requires urgent medical assessment and management.
    *   **Acceptable Range**: < 38.0°C (to avoid even moderate fever).

5.  **`ALERT_HR_TACHYCARDIA_BPM = 100`** (Adult resting)
    *   **Meaning**: Resting heart rate above 100 beats per minute for an adult.
    *   **Criteria**: **WARNING/INFO**. May indicate stress, fever, dehydration, or underlying cardiac issues. Context (activity, age, symptoms) is important.
    *   **Acceptable Range**: Typically 60-100 bpm for resting adults.

6.  **`ALERT_HR_BRADYCARDIA_BPM = 50`** (Adult resting)
    *   **Meaning**: Resting heart rate below 50 beats per minute for an adult.
    *   **Criteria**: **WARNING/INFO**. Can be normal in athletes but may indicate cardiac issues in others. Context is key.
    *   **Acceptable Range**: Typically 60-100 bpm. Below 60 bpm can be normal for some, but below 50 warrants attention if symptomatic or new.

### B. Worker/Patient Specific Environmental & Exertion

1.  **`HEAT_STRESS_RISK_BODY_TEMP_C = 37.5`**
    *   **Meaning**: Core/skin body temperature indicating potential early heat stress for an individual (e.g., a CHW).
    *   **Criteria**: **INFO/NUDGE**. Suggests taking precautions (hydrate, rest, seek shade).
    *   **Acceptable Range**: Below this value, especially if ambient conditions are hot.

2.  **`HEAT_STRESS_DANGER_BODY_TEMP_C = 38.5`**
    *   **Meaning**: Core/skin body temperature indicating dangerous heat stress.
    *   **Criteria**: **WARNING/CRITICAL**. Requires immediate cooling measures, rest, hydration, and monitoring for signs of heat exhaustion/stroke.
    *   **Acceptable Range**: Below 37.5°C.

### C. Clinic/Shared Environment (IoT Sensors)

1.  **`ALERT_AMBIENT_CO2_HIGH_PPM = 1500`**
    *   **Meaning**: Ambient CO₂ levels above 1500 parts per million.
    *   **Criteria**: **WARNING**. Suggests poor ventilation or overcrowding, increasing airborne transmission risk. Action: Improve ventilation.
    *   **Acceptable Range**: Ideally < 800-1000 ppm indoors. ASHRAE recommends <1000 ppm.

2.  **`ALERT_AMBIENT_CO2_VERY_HIGH_PPM = 2500`**
    *   **Meaning**: Ambient CO₂ levels above 2500 ppm.
    *   **Criteria**: **HIGH CONCERN/CRITICAL**. Indicates very poor ventilation. Urgent action needed.
    *   **Acceptable Range**: < 1000-1500 ppm.

3.  **`ALERT_AMBIENT_PM25_HIGH_UGM3 = 35`**
    *   **Meaning**: PM2.5 particulate matter above 35 µg/m³ (over 24-hour avg, WHO interim target 1).
    *   **Criteria**: **WARNING**. Increased health risk, especially for vulnerable individuals.
    *   **Acceptable Range**: WHO Guideline: Annual mean < 5 µg/m³, 24-hour mean < 15 µg/m³. Higher values are common, 35 is an action threshold.

4.  **`ALERT_AMBIENT_PM25_VERY_HIGH_UGM3 = 50`**
    *   **Meaning**: PM2.5 above 50 µg/m³.
    *   **Criteria**: **HIGH CONCERN**. Significant health risk.
    *   **Acceptable Range**: As low as possible, ideally < 15 µg/m³.

5.  **`ALERT_AMBIENT_NOISE_HIGH_DBA = 85`**
    *   **Meaning**: Sustained ambient noise levels above 85 dBA.
    *   **Criteria**: **WARNING**. Risk of hearing damage with prolonged exposure (e.g., 8 hours). In a clinic, can impede communication and cause stress.
    *   **Acceptable Range**: For patient care areas, ideally < 45-55 dBA. 85 dBA is an occupational hazard level.

6.  **`ALERT_AMBIENT_HEAT_INDEX_RISK_C = 32`**
    *   **Meaning**: Heat Index (feels like temperature) at or above 32°C.
    *   **Criteria**: **CAUTION/RISK**. Fatigue possible with prolonged exposure and activity. Likelihood of heat cramps/exhaustion.
    *   **Acceptable Range**: Depends on activity, but generally < 27-30°C for comfort.

7.  **`ALERT_AMBIENT_HEAT_INDEX_DANGER_C = 41`**
    *   **Meaning**: Heat Index at or above 41°C.
    *   **Criteria**: **DANGER**. Heat cramps/exhaustion likely, heat stroke possible with prolonged exposure/activity.
    *   **Acceptable Range**: < 32°C.

### D. AI & System Scores

1.  **`FATIGUE_INDEX_MODERATE_THRESHOLD = 60`**
    *   **Meaning**: Worker fatigue index score (0-100 scale) meeting moderate level.
    *   **Criteria**: **INFO/NUDGE**. Suggests worker take a break or adjust workload.
    *   **Acceptable Range**: < 60 (Low fatigue).

2.  **`FATIGUE_INDEX_HIGH_THRESHOLD = 80`**
    *   **Meaning**: Worker fatigue index score meeting high level.
    *   **Criteria**: **WARNING**. Worker should rest, may need intervention/support.
    *   **Acceptable Range**: < 60 (Low fatigue).

3.  **`STRESS_HRV_LOW_THRESHOLD_MS = 20`**
    *   **Meaning**: Heart Rate Variability (RMSSD) below 20 ms.
    *   **Criteria**: **INFO/WARNING**. Low HRV can indicate high physiological stress or poor recovery.
    *   **Acceptable Range**: Highly individual, but generally > 40-50ms is considered better for adults. Values < 20ms are consistently low.

4.  **`RISK_SCORE_MODERATE_THRESHOLD = 60`**
    *   **Meaning**: Patient AI Risk Score (0-100) at or above 60.
    *   **Criteria**: Indicates moderate overall health risk. May trigger closer monitoring or specific care pathways.
    *   **Acceptable Range**: < 40 (Low Risk). 40-59 is typically considered "Low to Moderate".

5.  **`RISK_SCORE_HIGH_THRESHOLD = 75`**
    *   **Meaning**: Patient AI Risk Score at or above 75.
    *   **Criteria**: Indicates high overall health risk. Triggers more urgent review, alerts, or specific protocols.
    *   **Acceptable Range**: < 60 (Low/Moderate Risk).

### E. Clinic Operations

1.  **`TARGET_CLINIC_WAITING_ROOM_OCCUPANCY_MAX = 10`**
    *   **Meaning**: Maximum desired number of people in a waiting room.
    *   **Criteria**: **INFO/WARNING**. Exceeding this may indicate overcrowding, increased transmission risk, or long wait times.
    *   **Acceptable Range**: <= 10 persons (or as per clinic capacity).

2.  **`TARGET_CLINIC_PATIENT_THROUGHPUT_MIN_PER_HOUR = 5`**
    *   **Meaning**: Minimum target for patient throughput (e.g., consultations completed) per hour per provider/consult room.
    *   **Criteria**: **INFO**. Falling below may indicate inefficiencies or bottlenecks.
    *   **Acceptable Range**: >= 5 patients/hour/provider (highly context-dependent).

### F. District Level (Aggregated Zonal Data)

1.  **`DISTRICT_ZONE_HIGH_RISK_AVG_SCORE = 70`**
    *   **Meaning**: Average AI patient risk score for a zone. If >= 70, the zone is flagged as "high risk".
    *   **Criteria**: **INTERVENTION PLANNING**. Zones meeting this may be prioritized for resource allocation or public health programs.
    *   **Acceptable Range**: Ideally, average zonal risk < 60 (Moderate).

2.  **`DISTRICT_INTERVENTION_FACILITY_COVERAGE_LOW_PCT = 60`**
    *   **Meaning**: Facility coverage score (e.g., access, capacity relative to population) below 60%.
    *   **Criteria**: **INTERVENTION PLANNING**. Indicates potential underservice.
    *   **Acceptable Range**: Depends on district goals, but generally > 75-80% might be a target.

3.  **`DISTRICT_INTERVENTION_TB_BURDEN_HIGH_ABS = 10`**
    *   **Meaning**: Absolute number of active TB cases in a zone. If >= 10, may trigger intervention.
    *   **Criteria**: **INTERVENTION PLANNING**. Used to identify high TB burden zones.
    *   **Acceptable Range**: As low as possible. Threshold is context-specific.

### G. Supply Chain

1.  **`CRITICAL_SUPPLY_DAYS_REMAINING = 7`**
    *   **Meaning**: Days of supply for an item is less than 7 days.
    *   **Criteria**: **CRITICAL ALERT**. Indicates an imminent stockout. Urgent re-supply needed.
    *   **Acceptable Range**: > 14 days (above Low Supply threshold).

2.  **`LOW_SUPPLY_DAYS_REMAINING = 14`**
    *   **Meaning**: Days of supply for an item is less than 14 days.
    *   **Criteria**: **WARNING ALERT**. Indicates low stock. Re-supply should be initiated.
    *   **Acceptable Range**: Ideally > 14 days.

### H. General Activity / Wellness

1.  **`TARGET_DAILY_STEPS = 8000`**
    *   **Meaning**: A general target for daily physical activity for individuals.
    *   **Criteria**: **INFO/WELLNESS GOAL**.
    *   **Acceptable Range**: Meeting or exceeding this target is generally good. Consistently low values might be a concern.

## II. Data Semantics (`config/settings.py`)

1.  **`KEY_TEST_TYPES_FOR_ANALYSIS`**
    *   **Meaning**: A dictionary defining key diagnostic tests monitored by the system. Includes `target_tat_days` (Target Turnaround Time) and a `critical` flag.
    *   **`target_tat_days` Criteria**: Tests should ideally be completed within this timeframe. Exceeding it impacts patient care and may trigger alerts.
    *   **`critical` Flag Criteria**: `True` indicates tests that are vital for urgent clinical decisions or public health response. These often have stricter TAT monitoring.

2.  **`TARGET_TEST_TURNAROUND_DAYS = 2`**
    *   **Meaning**: A *general fallback* target TAT if a specific test doesn't have one defined in `KEY_TEST_TYPES_FOR_ANALYSIS`.
    *   **Acceptable Range**: Results within this timeframe.

3.  **`TARGET_OVERALL_TESTS_MEETING_TAT_PCT_FACILITY = 85.0`**
    *   **Meaning**: Facility-level KPI. Percentage of all (or critical) tests meeting their respective TAT targets.
    *   **Acceptable Range**: >= 85.0%. Lower values indicate systemic delays.

4.  **`TARGET_SAMPLE_REJECTION_RATE_PCT_FACILITY = 5.0`**
    *   **Meaning**: Facility-level KPI. Percentage of lab samples rejected.
    *   **Acceptable Range**: < 5.0%. Higher rates indicate issues with sample collection, transport, or pre-analytical processing.

5.  **`KEY_CONDITIONS_FOR_ACTION`**
    *   **Meaning**: A list of specific diseases/conditions prioritized for focused monitoring, alert generation, and potentially specific care pathways or public health interventions (e.g., TB, Malaria).
    *   **Criteria**: Presence of these conditions in patient records often elevates risk or triggers specific workflows.

6.  **`KEY_DRUG_SUBSTRINGS_SUPPLY`**
    *   **Meaning**: A list of substrings used to identify key drugs/medical supplies for stock monitoring and forecasting.
    *   **Criteria**: Items matching these are considered essential and are subject to critical/low stock alerts.

7.  **`TARGET_MALARIA_POSITIVITY_RATE = 10.0`**
    *   **Meaning**: A target threshold for malaria RDT positivity rate in a clinic or zone.
    *   **Criteria**: **EPIDEMIOLOGICAL SIGNAL**. Rates significantly above this target (e.g., >15-20% depending on baseline) might indicate an outbreak or increased transmission, triggering investigation.
    *   **Acceptable Range**: Depends on local endemicity. The target is a benchmark for deviation.

This document is not exhaustive but covers many of the core configurable thresholds. Refer to specific dashboard components and analytics logic for how these values are used in calculations and decision-making.
