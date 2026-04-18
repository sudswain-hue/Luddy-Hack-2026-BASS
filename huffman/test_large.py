# huffman/test_large.py
from pipeline import huffman_compress, huffman_decompress

large_text = """
UNIVERSITY HOSPITAL SYSTEM — PATIENT DISCHARGE SUMMARY
=======================================================
Patient Name    : Jonathan Alexander Worthington III
Patient ID      : PAT-2024-88421-XYZ
Date of Birth   : March 14, 1978
Admission Date  : January 10, 2024
Discharge Date  : January 17, 2024
Ward            : Cardiology — Room 412B
Attending Physician : Dr. Priya Subramaniam, MD, FACC

CHIEF COMPLAINT:
Patient presented to the emergency department with acute chest pain,
shortness of breath, and diaphoresis. Pain rated 8/10, radiating to
left arm and jaw. Onset approximately 2 hours prior to admission.
No prior history of cardiac events. Family history positive for
coronary artery disease (father, age 62).

DIAGNOSIS:
Primary   : ST-Elevation Myocardial Infarction (STEMI) — anterior wall
Secondary : Hypertension (Stage 2), Hyperlipidemia, Type 2 Diabetes Mellitus

VITAL SIGNS ON ADMISSION:
Blood Pressure  : 172/98 mmHg
Heart Rate      : 102 bpm (irregular)
Respiratory Rate: 22 breaths/min
Temperature     : 37.2 C
SpO2            : 94% on room air
Weight          : 89.4 kg
Height          : 178 cm
BMI             : 28.2

LABORATORY RESULTS:
Troponin I      : 4.82 ng/mL (HIGH — normal < 0.04)
CK-MB           : 28.4 U/L   (HIGH — normal < 5.0)
BNP             : 412 pg/mL  (HIGH — normal < 100)
HbA1c           : 8.1%       (HIGH — target < 7.0)
LDL Cholesterol : 142 mg/dL  (HIGH — target < 70)
HDL Cholesterol : 38 mg/dL   (LOW  — target > 40)
Creatinine      : 1.1 mg/dL  (normal)
eGFR            : 74 mL/min  (mildly reduced)
Sodium          : 138 mEq/L  (normal)
Potassium       : 4.2 mEq/L  (normal)
WBC             : 11.2 x10^9 (mildly elevated)
Hemoglobin      : 13.8 g/dL  (normal)
Platelets       : 224 x10^9  (normal)

PROCEDURES PERFORMED:
1. Emergency Coronary Angiography (Jan 10, 2024 — 03:42 AM)
   Findings: 95% stenosis of LAD (left anterior descending artery)
             40% stenosis of RCA (right coronary artery)
             LV ejection fraction estimated at 45%

2. Percutaneous Coronary Intervention (PCI) with Drug-Eluting Stent
   Stent Type    : Xience Sierra 3.0 x 28mm
   Deployed at   : Proximal LAD
   Post-procedure TIMI flow: Grade 3 (normal)
   Fluoroscopy time: 18.4 minutes
   Contrast used : 210 mL Omnipaque 350

3. Echocardiogram (Jan 12, 2024)
   LVEF          : 48% (mildly reduced)
   Wall motion   : Hypokinesis of anterior and anteroseptal walls
   No pericardial effusion. Mild mitral regurgitation.

4. Continuous Cardiac Monitoring (Jan 10–17, 2024)
   Episodes of non-sustained VT: 2 (self-terminating, < 30 seconds)
   No atrial fibrillation detected.

MEDICATIONS PRESCRIBED AT DISCHARGE:
1.  Aspirin 81mg — once daily (indefinite)
2.  Clopidogrel (Plavix) 75mg — once daily x 12 months
3.  Atorvastatin (Lipitor) 80mg — once daily at bedtime
4.  Metoprolol Succinate 50mg — once daily
5.  Lisinopril 10mg — once daily (titrate to 20mg in 4 weeks)
6.  Nitroglycerin 0.4mg SL — PRN chest pain
7.  Metformin 1000mg — twice daily with meals
8.  Insulin Glargine 18 units — subcutaneous injection at bedtime
9.  Furosemide 20mg — once daily (for mild fluid retention)
10. Potassium Chloride 20mEq — once daily (to counteract furosemide)

FOLLOW-UP INSTRUCTIONS:
- Cardiac Rehabilitation Program: Enroll within 2 weeks of discharge
  Contact: CardioRehab Center, Building C, Room 204, Tel: 812-555-0192
- Follow-up with Dr. Subramaniam in 2 weeks (Jan 31, 2024 at 10:00 AM)
- Follow-up with Endocrinology for diabetes management in 4 weeks
- Repeat echocardiogram in 6–8 weeks to reassess LVEF
- Low-sodium diet (< 2g/day), low-saturated-fat diet
- No driving for 1 week. No heavy lifting for 4 weeks.
- Daily weight monitoring — call if weight increases > 2 lbs overnight
- Blood pressure target: < 130/80 mmHg
- Blood glucose target (fasting): 80–130 mg/dL

ALLERGIES: Penicillin (rash), Sulfa drugs (anaphylaxis)
SMOKING STATUS: Former smoker (quit 2019, 15 pack-year history)
ALCOHOL: Occasional (1–2 drinks/week)
EXERCISE: Sedentary prior to admission

DISCHARGE CONDITION: Stable. Patient ambulating independently.
Chest pain resolved. Vitals stable on current medications.
Patient and family educated on warning signs, medication compliance,
diet modifications, and when to return to the emergency department.

CODED BY    : Medical Records Dept — Coder ID 4821
ICD-10 CODES: I21.09 (STEMI), I10 (Hypertension), E11.9 (T2DM),
              E78.5 (Hyperlipidemia), Z87.891 (tobacco use history)
DRG CODE    : 247 — Percutaneous Cardiovascular Procedure w/ Drug-Eluting Stent
TOTAL LOS   : 7 days
TOTAL CHARGES: $84,291.00

END OF DISCHARGE SUMMARY — CONFIDENTIAL MEDICAL RECORD
=======================================================
""".strip()

print("Running large text compression test...")
print(f"Input size: {len(large_text)} characters\n")

# compress
result    = huffman_compress(large_text)
compressed = result["compressed_bytes"]
padding    = result["padding"]
m          = result["metrics"]

# decompress
recovered = huffman_decompress(compressed, padding)

# results
lossless = recovered == large_text

print("=" * 55)
print("  COMPRESSION RESULTS")
print("=" * 55)
print(f"  Original size       : {len(large_text)} chars")
print(f"  Compressed size     : {len(compressed)} bytes")
print(f"  Padding bits        : {padding}")
print(f"  Lossless recovery   : {lossless}")
print()
print("  --- Graduate Metrics ---")
print(f"  Compression Ratio   : {m['compression_ratio']}x")
print(f"  Shannon Entropy     : {m['entropy']} bits/symbol")
print(f"  Encoding Efficiency : {round(m['encoding_efficiency']*100, 2)}%")
print()

if lossless:
    print("  LOSSLESS CONFIRMED — first 100 chars of recovery:")
    print(f"  {recovered[:100]}...")
else:
    print("  MISMATCH DETECTED")
    # find first difference
    for i, (a, b) in enumerate(zip(large_text, recovered)):
        if a != b:
            print(f"  First diff at index {i}")
            print(f"  Expected: {large_text[i-10:i+10]!r}")
            print(f"  Got     : {recovered[i-10:i+10]!r}")
            break

print("=" * 55)