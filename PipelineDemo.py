"""
MedLink Egypt — Combined: Data Generation + Pipeline
=====================================================
Generates synthetic Egyptian patient data and processes it
into AI-ready artifacts in one script.

Install:
    pip install pandas numpy faker pyarrow tqdm

Run:
    python medlink_pipeline_combined.py --out_dir ./output
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker
from datetime import datetime, timedelta
from tqdm import tqdm

# =============================================================================
# CONFIG
# =============================================================================
SEED = 42
N_PATIENTS = 100_000
MIN_VISITS = 1
MAX_VISITS = 10
CURRENT_YEAR = 2024

random.seed(SEED)
np.random.seed(SEED)
fake = Faker("ar_EG")
Faker.seed(SEED)

# =============================================================================
# GOVERNORATES
# =============================================================================
REAL_GOVS = [
    "Cairo", "Giza", "Alexandria", "Dakahlia", "Red Sea", "Beheira", "Fayoum",
    "Gharbia", "Ismailia", "Menofia", "Minya", "Qalyubia", "New Valley",
    "Suez", "Aswan", "Assiut", "Beni Suef", "Port Said", "Damietta",
    "Sharkia", "South Sinai", "Kafr El Sheikh", "Matrouh",
    "Luxor", "Qena", "North Sinai", "Sohag"
]

SECRET_CODES = {g: f"G{500+i}" for i, g in enumerate(REAL_GOVS)}
GOV_DECODE   = {v: k for k, v in SECRET_CODES.items()}

URBAN_FACTOR = {
    "Cairo": 1.3, "Giza": 1.2, "Alexandria": 1.25,
    "Red Sea": 0.9, "South Sinai": 0.8, "New Valley": 0.75
}

RISK_LABELS = {
    (0,   3):   "Low",
    (3,   6):   "Moderate",
    (6,   10):  "High",
    (10, 999):  "Very High",
}

LAB_REFERENCE = {
    "glucose":       (70,  100,  "mg/dL"),
    "cholesterol":   (0,   200,  "mg/dL"),
    "creatinine":    (0.6, 1.2,  "mg/dL"),
    "egfr":          (60,  999,  "mL/min"),
    "bp_sys":        (90,  120,  "mmHg"),
    "bp_dia":        (60,  80,   "mmHg"),
    "hdl":           (40,  999,  "mg/dL"),
    "ldl":           (0,   100,  "mg/dL"),
    "triglycerides": (0,   150,  "mg/dL"),
    "crp":           (0,   1,    "mg/L"),
    "hemoglobin":    (12,  17.5, "g/dL"),
    "platelets":     (150, 400,  "×10³/μL"),
    "bmi":           (18.5, 24.9,"kg/m²"),
}

# =============================================================================
# GENERATION HELPERS
# =============================================================================
AGE_DIST = [(18,29,0.27),(30,44,0.22),(45,59,0.15),(60,75,0.07),(76,90,0.01)]

def sample_age():
    r = random.random()
    cumulative = 0
    for low, high, p in AGE_DIST:
        cumulative += p
        if r <= cumulative:
            return random.randint(low, high)
    return 30

def generate_fake_nid(birth_date, gender):
    century = 4 if birth_date.year < 2000 else 5
    birth_str = birth_date.strftime("%y%m%d")
    serial = random.randint(100, 999)
    gender_digit = random.choice([1,3,5,7,9]) if gender == "Male" else random.choice([0,2,4,6,8])
    return f"{century}{birth_str}{serial}{gender_digit}"

def generate_diseases(age, bmi, gov):
    ub = URBAN_FACTOR.get(gov, 1)
    obesity  = np.random.binomial(1, min(0.15 + 0.003*age + 0.01*(bmi-20) + 0.05*ub, 0.75))
    diabetes = np.random.binomial(1, min(0.05 + 0.0025*age + 0.2*obesity + 0.03*ub, 0.7))
    htn      = np.random.binomial(1, min(0.07 + 0.003*age + 0.18*obesity + 0.02*ub, 0.7))
    cardio   = np.random.binomial(1, min(0.02 + 0.15*htn + 0.12*diabetes, 0.7))
    ckd      = np.random.binomial(1, min(0.01 + 0.25*diabetes, 0.6))
    return obesity, diabetes, htn, cardio, ckd

# =============================================================================
# STAGE 1 — GENERATE SYNTHETIC DATA
# =============================================================================
def generate_data():
    print("\n[1/4] Generating synthetic patient data...")
    patients, diseases, labs, family = [], [], [], []

    for pid in tqdm(range(N_PATIENTS)):
        age = sample_age()
        birth_year = CURRENT_YEAR - age
        birth_date = datetime(birth_year, random.randint(1,12), random.randint(1,28)).date()
        gender = random.choice(["Male","Female"])
        national_id = generate_fake_nid(birth_date, gender)
        smoking = np.random.binomial(1, 0.25 if age >= 18 else 0.05)

        original_gov = random.choices(REAL_GOVS, weights=[0.15,0.12,0.1]+[0.73/24]*24)[0]
        other_govs = [g for g in REAL_GOVS if g not in [original_gov,"Cairo","Alexandria"]]
        population = [original_gov, "Cairo", "Alexandria"] + other_govs
        n_other = len(other_govs)
        current_gov = random.choices(population, weights=[0.6,0.15,0.15]+[0.1/n_other]*n_other, k=1)[0]

        height = np.random.normal(172 if gender == "Male" else 160, 6)
        base_weight = np.random.normal(70 + 0.3*age, 10)
        bmi = round(base_weight / (height/100)**2, 1)

        obesity, diabetes, htn, cardio, ckd = generate_diseases(age, bmi, current_gov)
        n_visits = random.randint(MIN_VISITS, MAX_VISITS)
        first_visit = birth_date + timedelta(days=max(18*365, random.randint(0, (age-18)*365)))

        lab_means = {k: [] for k in ["bmi","glucose","cholesterol","creatinine",
                                      "bp_sys","bp_dia","hdl","ldl",
                                      "triglycerides","crp","hemoglobin","platelets"]}

        for v in range(n_visits):
            visit_date = first_visit + timedelta(days=random.randint(30, 400))
            if visit_date.year > CURRENT_YEAR:
                visit_date = first_visit

            bmi_v        = round(bmi + 0.5*v + 0.7*obesity, 1)
            glucose      = round(np.random.normal(85 + 55*diabetes + 0.3*age, 15), 1)
            cholesterol  = round(np.random.normal(165 + 35*cardio + 15*obesity, 20), 1)
            creat        = round(np.random.normal(0.8 + 1.1*ckd + 0.05*v, 0.1), 2)
            egfr         = round(120 - creat*25, 1)
            bp_sys       = round(np.random.normal(110 + 15*htn + 0.4*age, 8), 1)
            bp_dia       = round(np.random.normal(70 + 10*htn + 0.2*age, 5), 1)
            hdl          = round(np.random.normal(50 - 5*obesity, 10), 1)
            ldl          = round(np.random.normal(100 + 20*cardio, 15), 1)
            triglycerides= round(np.random.lognormal(5 + 0.2*obesity, 0.3), 1)
            crp          = round(np.random.exponential(2) + 3*cardio, 1)
            hem          = round(np.random.normal(13, 1), 1)
            platelets    = round(np.random.normal(250, 40), 1)

            labs.append([pid, visit_date, bmi_v, height, base_weight,
                         glucose, cholesterol, creat, egfr,
                         bp_sys, bp_dia, hdl, ldl, triglycerides,
                         crp, hem, platelets])

            for k, val in zip(lab_means, [bmi_v, glucose, cholesterol, creat,
                                           bp_sys, bp_dia, hdl, ldl,
                                           triglycerides, crp, hem, platelets]):
                lab_means[k].append(val)

        risk_score = 2*diabetes + 2*htn + 3*cardio + 2*ckd + obesity + smoking + age/70
        patients.append([pid, national_id, gender, birth_date, age, smoking,
                         SECRET_CODES[original_gov], SECRET_CODES[current_gov]])
        diseases.append([pid, age, obesity, diabetes, htn, cardio, ckd,
                         *[np.mean(v) for v in lab_means.values()], risk_score])

        for _ in range(random.randint(1, 5)):
            rel_type = random.choices(["parent","sibling","child"], weights=[0.3,0.5,0.2])[0]
            if rel_type == "parent":    rel_age = age + random.randint(18,40)
            elif rel_type == "sibling": rel_age = max(0, age + random.randint(-10,10))
            else:                       rel_age = max(0, age - random.randint(0,35))
            rel_birth = datetime(CURRENT_YEAR - rel_age, random.randint(1,12), random.randint(1,28)).date()
            rel_gender = random.choice(["Male","Female"])
            rel_disease = 1 if rel_age > 15 and random.random() < 0.2 else 0
            family.append([pid, rel_birth, rel_gender, rel_disease])

    patients_df = pd.DataFrame(patients, columns=[
        "patient_id","national_id","gender","birth_date","age","smoking","original_gov","current_gov"])
    diseases_df = pd.DataFrame(diseases, columns=[
        "patient_id","age","Obesity","Diabetes","Hypertension","Cardiovascular","CKD",
        "avg_bmi","avg_glucose","avg_cholesterol","avg_creatinine",
        "avg_bp_sys","avg_bp_dia","avg_hdl","avg_ldl","avg_triglycerides",
        "avg_crp","avg_hemoglobin","avg_platelets","risk_score"])
    labs_df = pd.DataFrame(labs, columns=[
        "patient_id","visit_date","bmi","height_cm","weight_kg","glucose","cholesterol",
        "creatinine","egfr","bp_sys","bp_dia","hdl","ldl","triglycerides","crp",
        "hemoglobin","platelets"])
    family_df = pd.DataFrame(family, columns=[
        "patient_id","relative_birth_date","relative_gender","relative_disease"])

    print(f"  Patients:      {len(patients_df):,}")
    print(f"  Disease rows:  {len(diseases_df):,}")
    print(f"  Lab rows:      {len(labs_df):,}")
    print(f"  Family rows:   {len(family_df):,}")
    return patients_df, diseases_df, labs_df, family_df

# =============================================================================
# STAGE 2 — LOAD & DECODE  (works on in-memory DFs or from CSV files)
# =============================================================================
def decode_gov(code):
    return GOV_DECODE.get(str(code), code)

def load_from_disk(data_dir: Path):
    """Alternative: load from saved CSVs instead of generating in memory."""
    dfs = {}
    for name in ["patients","diseases","labs","family_history"]:
        path = data_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
        dfs[name] = pd.read_csv(path)
        print(f"  Loaded {name}.csv")
    return dfs["patients"], dfs["diseases"], dfs["labs"], dfs["family_history"]

# =============================================================================
# STAGE 3 — CLEAN & MERGE
# =============================================================================
def clean_and_merge(patients_df, diseases_df, labs_df, family_df):
    print("\n[2/4] Cleaning and merging...")
    patients_df = patients_df.copy()
    patients_df["original_gov"] = patients_df["original_gov"].apply(decode_gov)
    patients_df["current_gov"]  = patients_df["current_gov"].apply(decode_gov)

    labs_df["visit_date"] = pd.to_datetime(labs_df["visit_date"])
    latest_labs = labs_df.sort_values("visit_date").groupby("patient_id").last().reset_index()
    fam_risk = family_df.groupby("patient_id").size().reset_index(name="family_disease_count")

    df = patients_df \
        .merge(diseases_df, on="patient_id", how="left") \
        .merge(latest_labs,  on="patient_id", how="left") \
        .merge(fam_risk,     on="patient_id", how="left")

    df["family_disease_count"] = df["family_disease_count"].fillna(0)
    print(f"  Merged shape: {df.shape}")
    return df

# =============================================================================
# STAGE 4 — ENRICH: RISK LABELS + ABNORMAL FLAGS
# =============================================================================
def risk_label(score):
    for (lo, hi), label in RISK_LABELS.items():
        if lo <= score < hi:
            return label
    return "Unknown"

def flag_abnormals(row):
    flags = []
    for col, (lo, hi, unit) in LAB_REFERENCE.items():
        val = row.get(col)
        if pd.notna(val):
            if val < lo:   flags.append(f"LOW {col} ({val} {unit})")
            elif val > hi: flags.append(f"HIGH {col} ({val} {unit})")
    return flags

def enrich(df):
    print("\n[3/4] Enriching with risk labels and abnormal flags...")
    df = df.copy()
    df["risk_label"]     = df["risk_score"].apply(risk_label)
    df["abnormal_flags"] = df.apply(flag_abnormals, axis=1)
    high_risk = (df["risk_label"].isin(["High","Very High"])).sum()
    print(f"  High/Very High risk patients: {high_risk:,}")
    return df

# =============================================================================
# STAGE 5 — BUILD AI ARTIFACTS
# =============================================================================
def build_narrative(row):
    return f"""Patient ID: {row['patient_id']}
Age: {row.get('age', 'N/A')}  Gender: {row['gender']}  Location: {row['current_gov']}

Risk Score: {row['risk_score']:.1f} ({row['risk_label']})

Conditions:
- Obesity:         {int(row['Obesity'])}
- Diabetes:        {int(row['Diabetes'])}
- Hypertension:    {int(row['Hypertension'])}
- Cardiovascular:  {int(row['Cardiovascular'])}
- CKD:             {int(row['CKD'])}

Latest Labs:
- Glucose:       {row.get('glucose', 'N/A')} mg/dL
- Cholesterol:   {row.get('cholesterol', 'N/A')} mg/dL
- BP:            {row.get('bp_sys','N/A')}/{row.get('bp_dia','N/A')} mmHg
- eGFR:          {row.get('egfr','N/A')} mL/min
- BMI:           {row.get('bmi','N/A')} kg/m²
- HbA1c proxy:   {row.get('avg_glucose','N/A'):.1f} mg/dL avg

Abnormal Flags: {row['abnormal_flags']}
Family disease count: {int(row['family_disease_count'])}""".strip()

def write_artifacts(df, out_dir: Path):
    print("\n[4/4] Writing AI artifacts...")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parquet for ML
    df.to_parquet(out_dir / "medlink_patients_clean.parquet", index=False, engine="pyarrow")
    df.to_csv(out_dir / "medlink_patients_clean.csv", index=False)
    print(f"  medlink_patients_clean.parquet  ({len(df):,} rows)")

    # JSONL chunks for RAG / embeddings
    with open(out_dir / "medlink_chunks.jsonl", "w", encoding="utf-8") as f:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="  chunks"):
            f.write(json.dumps({
                "patient_id": int(row["patient_id"]),
                "text": build_narrative(row)
            }) + "\n")

    # Full context JSON for LLM prompting
    context = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="  context"):
        context.append({
            "patient_id":  int(row["patient_id"]),
            "national_id": str(row["national_id"]),
            "risk":        row["risk_label"],
            "narrative":   build_narrative(row)
        })
    with open(out_dir / "medlink_context.json", "w", encoding="utf-8") as f:
        json.dump(context, f, indent=2)

    # README
    readme = """MedLink Egypt — AI Data Package
================================
Files:
  medlink_patients_clean.parquet  → ML training / feature engineering
  medlink_patients_clean.csv      → Spreadsheet / EDA use
  medlink_context.json            → LLM context injection (one record per patient)
  medlink_chunks.jsonl            → RAG / embedding pipelines (one chunk per patient)

Usage notes:
  - Use .parquet for pandas/sklearn/XGBoost pipelines
  - Use .jsonl for vector DB ingestion (Pinecone, Weaviate, Chroma, etc.)
  - Use .json for direct LLM prompting with patient context
  - Governorate codes have been decoded back to real names
  - risk_label: Low / Moderate / High / Very High
  - abnormal_flags: list of out-of-range lab values per patient
"""
    (out_dir / "README.txt").write_text(readme, encoding="utf-8")
    print(f"\n  Output → {out_dir.resolve()}")

# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="MedLink Egypt combined pipeline")
    parser.add_argument("--out_dir",  default="./output", help="Output directory")
    parser.add_argument("--from_csv", default=None,
                        help="Skip generation, load CSVs from this directory instead")
    args = parser.parse_args()

    if args.from_csv:
        print(f"Loading CSVs from {args.from_csv}...")
        patients_df, diseases_df, labs_df, family_df = load_from_disk(Path(args.from_csv))
    else:
        patients_df, diseases_df, labs_df, family_df = generate_data()

    df = clean_and_merge(patients_df, diseases_df, labs_df, family_df)
    df = enrich(df)
    write_artifacts(df, Path(args.out_dir))
    print("\n✅ Done!")

if __name__ == "__main__":
    main()