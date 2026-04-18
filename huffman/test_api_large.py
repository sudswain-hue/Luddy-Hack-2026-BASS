# huffman/test_api_large.py
"""
Large-text end-to-end test against the running microservice.
Hits /compress and /decompress endpoints with ~12,000 chars of realistic OCR text.
"""
import requests
import time

BASE = "http://localhost:8001"

# ── generate realistic ~12,000 character OCR-style text ───────────
base_doc = """
UNIVERSITY HOSPITAL SYSTEM PATIENT DISCHARGE SUMMARY
Patient Name Jonathan Alexander Worthington III
Patient ID PAT-2024-88421-XYZ Date of Birth March 14 1978
Admission Date January 10 2024 Discharge Date January 17 2024
Ward Cardiology Room 412B Attending Physician Dr Priya Subramaniam

CHIEF COMPLAINT Patient presented to the emergency department with acute
chest pain shortness of breath and diaphoresis Pain rated 8 out of 10
radiating to left arm and jaw Onset approximately 2 hours prior to admission
No prior history of cardiac events Family history positive for coronary
artery disease father age 62

DIAGNOSIS Primary ST Elevation Myocardial Infarction STEMI anterior wall
Secondary Hypertension Stage 2 Hyperlipidemia Type 2 Diabetes Mellitus

VITAL SIGNS ON ADMISSION Blood Pressure 172 over 98 mmHg Heart Rate 102 bpm
Respiratory Rate 22 breaths per minute Temperature 37.2 Celsius SpO2 94 percent
on room air Weight 89.4 kg Height 178 cm BMI 28.2

LABORATORY RESULTS Troponin I 4.82 ng per mL HIGH normal less than 0.04
CK-MB 28.4 U per L HIGH normal less than 5.0 BNP 412 pg per mL HIGH normal
less than 100 HbA1c 8.1 percent HIGH target less than 7.0 LDL Cholesterol
142 mg per dL HIGH target less than 70 HDL Cholesterol 38 mg per dL LOW
target greater than 40 Creatinine 1.1 mg per dL normal eGFR 74 mL per min
mildly reduced Sodium 138 mEq per L normal Potassium 4.2 mEq per L normal

PROCEDURES PERFORMED Emergency Coronary Angiography finding 95 percent
stenosis of LAD left anterior descending artery 40 percent stenosis of RCA
right coronary artery LV ejection fraction estimated at 45 percent
Percutaneous Coronary Intervention PCI with Drug Eluting Stent Xience
Sierra 3.0 by 28mm deployed at Proximal LAD Post procedure TIMI flow
Grade 3 normal Fluoroscopy time 18.4 minutes Contrast used 210 mL
Omnipaque 350 Echocardiogram LVEF 48 percent mildly reduced Wall motion
Hypokinesis of anterior and anteroseptal walls No pericardial effusion
Mild mitral regurgitation Continuous Cardiac Monitoring for 7 days
Episodes of non sustained VT 2 self terminating less than 30 seconds
No atrial fibrillation detected

MEDICATIONS PRESCRIBED AT DISCHARGE Aspirin 81mg once daily indefinite
Clopidogrel Plavix 75mg once daily for 12 months Atorvastatin Lipitor 80mg
once daily at bedtime Metoprolol Succinate 50mg once daily Lisinopril 10mg
once daily titrate to 20mg in 4 weeks Nitroglycerin 0.4mg sublingual as
needed for chest pain Metformin 1000mg twice daily with meals Insulin
Glargine 18 units subcutaneous injection at bedtime Furosemide 20mg once
daily for mild fluid retention Potassium Chloride 20mEq once daily to
counteract furosemide

FOLLOW UP INSTRUCTIONS Cardiac Rehabilitation Program Enroll within 2 weeks
of discharge Follow up with Dr Subramaniam in 2 weeks January 31 2024 at
10 AM Follow up with Endocrinology for diabetes management in 4 weeks
Repeat echocardiogram in 6 to 8 weeks to reassess LVEF Low sodium diet
less than 2g per day low saturated fat diet No driving for 1 week
No heavy lifting for 4 weeks Daily weight monitoring call if weight
increases more than 2 lbs overnight Blood pressure target less than 130
over 80 mmHg Blood glucose target fasting 80 to 130 mg per dL

ALLERGIES Penicillin rash Sulfa drugs anaphylaxis SMOKING STATUS Former
smoker quit 2019 15 pack year history ALCOHOL Occasional 1 to 2 drinks
per week EXERCISE Sedentary prior to admission DISCHARGE CONDITION Stable
Patient ambulating independently Chest pain resolved Vitals stable on
current medications Patient and family educated on warning signs
medication compliance diet modifications and when to return to the
emergency department
"""

# repeat to get ~12,000 characters
large_text = (base_doc * 4).strip()

print(f"Test input size: {len(large_text)} characters\n")

# ── hit /compress ─────────────────────────────────────────────────
print("=" * 60)
print("  STEP 1 — Calling POST /compress")
print("=" * 60)

start = time.perf_counter()
r = requests.post(f"{BASE}/compress", json={"text": large_text})
compress_time = (time.perf_counter() - start) * 1000

assert r.status_code == 200, f"Status {r.status_code}: {r.text}"
resp = r.json()
data = resp["data"]
m    = data["metrics"]

print(f"  Status              : {r.status_code} OK")
print(f"  Source              : {resp['source']}")
print(f"  Latency             : {compress_time:.2f} ms")
print(f"  Lossless verified   : {data['lossless_verified']}")
print(f"  Original size       : {data['original_size_chars']} chars")
print(f"  Compressed size     : {data['compressed_size_bytes']} bytes")
print(f"  Padding             : {data['padding']} bits")
print(f"\n  --- Graduate Metrics ---")
print(f"  Compression ratio   : {m['compression_ratio']}x")
print(f"  Shannon entropy     : {m['entropy']} bits/symbol")
print(f"  Encoding efficiency : {m['encoding_efficiency_percent']}%")

# ── hit /decompress ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 2 — Calling POST /decompress")
print("=" * 60)

start = time.perf_counter()
r2 = requests.post(f"{BASE}/decompress", json={
    "compressed_bytes_b64": data["compressed_bytes_b64"],
    "padding":              data["padding"]
})
decompress_time = (time.perf_counter() - start) * 1000

assert r2.status_code == 200, f"Status {r2.status_code}: {r2.text}"
recovered = r2.json()["recovered_text"]

print(f"  Status              : {r2.status_code} OK")
print(f"  Latency             : {decompress_time:.2f} ms")
print(f"  Recovered length    : {r2.json()['recovered_length']} chars")
print(f"  Exact match         : {recovered == large_text}")

# ── summary ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  END-TO-END RESULT")
print("=" * 60)

if recovered == large_text and m["compression_ratio"] > 1.0:
    print(f"  PASS — compression ratio {m['compression_ratio']}x "
          f"with lossless recovery")
    print(f"  Total round-trip latency: "
          f"{compress_time + decompress_time:.2f} ms")
else:
    print(f"  FAIL — ratio={m['compression_ratio']}, "
          f"lossless={recovered == large_text}")

print("=" * 60)