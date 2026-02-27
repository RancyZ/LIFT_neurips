"""
generate_data.py — Creates synthetic datasets for LIFT framework testing.

Datasets produced:
  lift/data/t1d_balanced.csv     ~1 000 rows · 67 features · balanced outcome
  lift/data/t1d_imbalanced.csv   ~1 000 rows · 67 features · imbalanced outcome (80/20)
  lift/data/nhanes_2017.csv      ~2 000 rows · 38 features · NHANES-style
  lift/data/nacc_uds.csv         ~5 000 rows · 165 features · NACC-style

All datasets have:
  y    — binary outcome (0/1)
  race — protected attribute (4-group)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
OUT  = Path(__file__).parent / "lift" / "data"
OUT.mkdir(parents=True, exist_ok=True)


# ── helpers ──────────────────────────────────────────────

def add_missing(df: pd.DataFrame, rate: float, rng: np.random.RandomState) -> pd.DataFrame:
    """Randomly blank out `rate` fraction of numeric cells."""
    df = df.copy()
    num_cols = df.select_dtypes(include="number").columns.tolist()
    num_cols = [c for c in num_cols if c not in ("y",)]
    n_cells  = int(len(df) * len(num_cols) * rate)
    rows = rng.randint(0, len(df), n_cells)
    cols = rng.choice(num_cols, n_cells)
    for r, c in zip(rows, cols):
        df.at[df.index[r], c] = np.nan
    return df


def save(df: pd.DataFrame, name: str) -> None:
    path = OUT / name
    df.to_csv(path, index=False)
    print(f"  ✓  {name:30s}  {len(df):>6,} rows × {len(df.columns):>3} cols  "
          f"  y=1: {df['y'].mean()*100:.0f}%"
          f"  missing: {df.isna().mean().mean()*100:.1f}%")


# ═══════════════════════════════════════════════════════════
# 1. T1D datasets  (TEDDY-style SNP + clinical)
# ═══════════════════════════════════════════════════════════

def make_t1d(N: int, outcome_rate: float, seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)

    races  = rng.choice(["White", "Black", "Hispanic", "Asian"],
                        p=[0.55, 0.20, 0.15, 0.10], size=N)
    ages   = rng.normal(8, 3, N).clip(1, 18)           # paediatric cohort
    sexes  = rng.choice(["M", "F"], size=N)
    bmis   = rng.normal(18, 3, N).clip(10, 35)
    hba1cs = rng.normal(5.5, 0.8, N).clip(4, 14)

    # 62 SNP features  (rs-IDs, values in {0,1,2})
    snp_data = {f"rs{100000+i}": rng.choice([0, 1, 2], size=N,
                p=[0.6, 0.3, 0.1]) for i in range(62)}

    # logistic outcome driven by hba1c + 3 key SNPs + race effect
    log_odds = (
        -3.0
        + 0.8  * ((hba1cs - 5.5) / 0.8)
        + 0.5  * snp_data["rs100000"]
        + 0.4  * snp_data["rs100001"]
        + 0.3  * snp_data["rs100002"]
        + np.where(races == "Black",    0.3, 0)
        + np.where(races == "Hispanic", 0.1, 0)
    )
    prob = 1 / (1 + np.exp(-log_odds))

    # Calibrate to target prevalence
    threshold = np.quantile(prob, 1 - outcome_rate)
    y = (prob >= threshold).astype(int)

    df = pd.DataFrame({
        "age":   ages.round(1),
        "sex":   sexes,
        "bmi":   bmis.round(1),
        "hba1c": hba1cs.round(2),
        "race":  races,
        **snp_data,
        "y": y,
    })
    return df


print("\n── T1D datasets ──────────────────────────────────────")

rng = np.random.RandomState(SEED)

df_t1d_bal  = make_t1d(1000, outcome_rate=0.50, seed=SEED)
df_t1d_bal  = add_missing(df_t1d_bal,  rate=0.03, rng=rng)
save(df_t1d_bal, "t1d_balanced.csv")

df_t1d_imb  = make_t1d(1000, outcome_rate=0.20, seed=SEED + 1)
df_t1d_imb  = add_missing(df_t1d_imb,  rate=0.04, rng=rng)
save(df_t1d_imb, "t1d_imbalanced.csv")


# ═══════════════════════════════════════════════════════════
# 2. NHANES 2017  (cardiovascular risk)
# ═══════════════════════════════════════════════════════════

def make_nhanes(N: int = 2000, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.RandomState(seed)

    races = rng.choice(
        ["Non-Hispanic White", "Non-Hispanic Black", "Hispanic", "Asian/Other"],
        p=[0.40, 0.25, 0.25, 0.10], size=N,
    )
    ages     = rng.normal(48, 17, N).clip(18, 85).round(0)
    sexes    = rng.choice(["Male", "Female"], size=N)
    bmis     = rng.normal(29, 6,  N).clip(15, 60).round(1)
    sbps     = rng.normal(122, 18, N).clip(80, 200).round(0)
    dbps     = rng.normal(76, 12, N).clip(50, 130).round(0)
    totchols = rng.normal(196, 40, N).clip(100, 400).round(0)
    ldls     = rng.normal(118, 35, N).clip(40, 300).round(0)
    hdls     = rng.normal(53, 15,  N).clip(20, 120).round(0)
    tgs      = rng.normal(148, 80, N).clip(30, 800).round(0)
    glucose  = rng.normal(102, 28, N).clip(60, 400).round(0)
    hba1cs   = rng.normal(5.7, 0.9, N).clip(4, 14).round(1)
    smokings = rng.choice(["Never", "Former", "Current"], p=[0.50, 0.30, 0.20], size=N)
    alcohols = rng.choice(["None", "Moderate", "Heavy"],  p=[0.35, 0.50, 0.15], size=N)
    exercises= rng.choice(["None", "Some", "Active"],     p=[0.30, 0.40, 0.30], size=N)
    incomes  = rng.choice(["<20k", "20-50k", "50-100k", ">100k"],
                          p=[0.25, 0.35, 0.25, 0.15], size=N)
    edus     = rng.choice(["<HS", "HS", "Some College", "College+"],
                          p=[0.15, 0.25, 0.30, 0.30], size=N)

    # Continuous labs
    creatinines = rng.normal(0.9, 0.3, N).clip(0.4, 5.0).round(2)
    egfrs       = rng.normal(82, 20,  N).clip(10, 130).round(0)
    uric_acids  = rng.normal(5.5, 1.5, N).clip(2, 12).round(1)
    wbcs        = rng.normal(6.8, 1.8, N).clip(2, 20).round(1)
    hemoglobins = rng.normal(14.2, 1.8, N).clip(7, 20).round(1)
    platelets   = rng.normal(250, 70,  N).clip(80, 600).round(0)
    alt         = rng.lognormal(3.3, 0.5, N).clip(5, 300).round(0)
    ast         = rng.lognormal(3.1, 0.4, N).clip(5, 300).round(0)
    crps        = rng.lognormal(0.5, 1.0, N).clip(0.1, 100).round(2)
    pulse_rates = rng.normal(72, 12,  N).clip(40, 130).round(0)
    waist_circs = rng.normal(97, 15,  N).clip(55, 165).round(0)
    hip_circs   = rng.normal(104, 13, N).clip(65, 165).round(0)

    # Logistic CVD outcome
    log_odds = (
        -4.0
        + 0.04 * (ages - 48)
        + 0.03 * (sbps - 122)
        + 0.02 * (bmis - 29)
        + 0.01 * (totchols - 196)
        - 0.02 * (hdls - 53)
        + 0.005 * tgs
        + np.where(sexes == "Male", 0.4, 0)
        + np.where(smokings == "Current", 0.6, 0)
        + np.where(smokings == "Former",  0.2, 0)
        + np.where(races == "Non-Hispanic Black", 0.3, 0)
    )
    prob = 1 / (1 + np.exp(-log_odds))
    threshold = np.quantile(prob, 0.75)  # ~25% prevalence
    y = (prob >= threshold).astype(int)

    df = pd.DataFrame({
        "age": ages, "sex": sexes, "race": races,
        "bmi": bmis, "waist_circ": waist_circs, "hip_circ": hip_circs,
        "sbp": sbps, "dbp": dbps, "pulse_rate": pulse_rates,
        "total_chol": totchols, "ldl": ldls, "hdl": hdls, "triglycerides": tgs,
        "glucose": glucose, "hba1c": hba1cs,
        "creatinine": creatinines, "egfr": egfrs, "uric_acid": uric_acids,
        "wbc": wbcs, "hemoglobin": hemoglobins, "platelets": platelets,
        "alt": alt, "ast": ast, "crp": crps,
        "smoking": smokings, "alcohol": alcohols, "exercise": exercises,
        "income": incomes, "education": edus,
        # binary flags
        "hypertension":  (sbps > 130).astype(int),
        "diabetes":      (glucose > 125).astype(int),
        "obese":         (bmis >= 30).astype(int),
        "dyslipidemia":  ((ldls > 160) | (hdls < 40)).astype(int),
        "ckd":           (egfrs < 60).astype(int),
        "anemia":        (hemoglobins < 12).astype(int),
        "elevated_crp":  (crps > 3).astype(int),
        "elevated_liver":(alt > 56).astype(int),
        "y": y,
    })
    return add_missing(df, rate=0.05, rng=rng)


print("\n── NHANES dataset ────────────────────────────────────")
save(make_nhanes(), "nhanes_2017.csv")


# ═══════════════════════════════════════════════════════════
# 3. NACC UDS  (Alzheimer's / dementia risk)
# ═══════════════════════════════════════════════════════════

def make_nacc(N: int = 5000, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.RandomState(seed)

    races = rng.choice(
        ["White", "Black", "Hispanic", "Asian", "Other"],
        p=[0.60, 0.20, 0.10, 0.07, 0.03], size=N,
    )
    ages   = rng.normal(72, 10, N).clip(50, 95).round(0)
    sexes  = rng.choice(["Male", "Female"], p=[0.42, 0.58], size=N)
    edus   = rng.normal(15, 3, N).clip(0, 25).round(0)   # years of education
    apoe4  = rng.choice([0, 1, 2], p=[0.60, 0.32, 0.08], size=N)  # APOE-ε4 alleles

    # Neuropsychological battery (z-scores, normed)
    mmse         = rng.normal(27, 3,   N).clip(0, 30).round(0)
    moca         = rng.normal(25, 4,   N).clip(0, 30).round(0)
    cdr_global   = rng.choice([0, 0.5, 1, 2, 3], p=[0.45, 0.30, 0.15, 0.07, 0.03], size=N)
    fas          = rng.normal(40, 10,  N).clip(5, 75).round(0)    # functional activities
    gds          = rng.normal(4, 3,    N).clip(0, 15).round(0)    # geriatric depression
    npi_total    = rng.lognormal(1.5, 1.0, N).clip(0, 144).round(0)

    # Memory tests
    lm_immediate = rng.normal(10, 4,  N).clip(0, 25).round(0)
    lm_delayed   = rng.normal(8, 4,   N).clip(0, 25).round(0)
    digit_fwd    = rng.normal(7, 2,   N).clip(2, 14).round(0)
    digit_bkwd   = rng.normal(5, 2,   N).clip(2, 14).round(0)
    benson_copy  = rng.normal(15, 3,  N).clip(0, 17).round(0)
    benson_recall= rng.normal(10, 4,  N).clip(0, 17).round(0)

    # Trails / executive
    tmta_time    = rng.lognormal(3.7, 0.4, N).clip(10, 300).round(0)
    tmtb_time    = rng.lognormal(4.5, 0.5, N).clip(20, 600).round(0)
    animals      = rng.normal(17, 5,  N).clip(3, 40).round(0)    # verbal fluency
    vegetables   = rng.normal(13, 4,  N).clip(2, 35).round(0)
    boston_naming= rng.normal(27, 4,  N).clip(5, 30).round(0)

    # MRI volumetrics (cc, simulated)
    hippo_vol    = rng.normal(6800, 900, N).clip(3000, 10000).round(0)
    whole_brain  = rng.normal(1100000, 120000, N).clip(600000, 1500000).round(0)
    ventricle_vol= rng.lognormal(10.5, 0.5, N).clip(5000, 150000).round(0)
    wmh_vol      = rng.lognormal(3.0, 1.2, N).clip(0, 100000).round(0)

    # Biomarkers (CSF / plasma)
    amyloid_42   = rng.normal(900, 300, N).clip(200, 2000).round(0)
    tau_total    = rng.normal(280, 100, N).clip(50, 1000).round(0)
    tau_181      = rng.normal(22, 10,  N).clip(2, 120).round(1)
    nfl          = rng.lognormal(2.8, 0.6, N).clip(2, 500).round(1)

    # Comorbidities (binary)
    hypertension = rng.binomial(1, 0.55, N)
    diabetes     = rng.binomial(1, 0.22, N)
    depression   = rng.binomial(1, 0.20, N)
    stroke       = rng.binomial(1, 0.08, N)
    tbi          = rng.binomial(1, 0.12, N)
    heart_disease= rng.binomial(1, 0.25, N)
    afib         = rng.binomial(1, 0.10, N)
    smoking_hist = rng.binomial(1, 0.35, N)
    family_hx    = rng.binomial(1, 0.30, N)

    # Longitudinal visit count (1-8 visits)
    n_visits     = rng.choice(range(1, 9), size=N)
    years_follow = (n_visits * 1.2 + rng.uniform(0, 0.5, N)).round(1)

    # Medications (binary flags)
    on_cholinesterase = rng.binomial(1, 0.18, N)
    on_antihypertensive= rng.binomial(1, 0.50, N)
    on_antidepressant = rng.binomial(1, 0.22, N)
    on_statin         = rng.binomial(1, 0.45, N)

    # Logistic dementia outcome
    log_odds = (
        -5.0
        + 0.08 * (ages - 72)
        + 1.5  * apoe4
        - 0.15 * (mmse - 27)
        - 0.05 * (lm_delayed - 8)
        + 0.5  * cdr_global
        + 0.3  * stroke
        + 0.2  * diabetes
        - 0.04 * (hippo_vol - 6800) / 100
        + np.where(races == "Black",    0.2, 0)
        + np.where(races == "Hispanic", 0.15, 0)
    )
    prob = 1 / (1 + np.exp(-log_odds))
    threshold = np.quantile(prob, 0.70)  # ~30% prevalence
    y = (prob >= threshold).astype(int)

    # Collect all 165 columns
    data = {
        # Demographics (5)
        "age": ages, "sex": sexes, "race": races,
        "education_yrs": edus, "apoe4_alleles": apoe4,
        # Cognitive screening (6)
        "mmse": mmse, "moca": moca, "cdr_global": cdr_global,
        "fas": fas, "gds": gds, "npi_total": npi_total,
        # Memory (6)
        "lm_immediate": lm_immediate, "lm_delayed": lm_delayed,
        "digit_fwd": digit_fwd, "digit_bkwd": digit_bkwd,
        "benson_copy": benson_copy, "benson_recall": benson_recall,
        # Executive / language (5)
        "tmta_time": tmta_time, "tmtb_time": tmtb_time,
        "animals_fluency": animals, "vegetables_fluency": vegetables,
        "boston_naming": boston_naming,
        # MRI (4)
        "hippo_vol_cc": hippo_vol, "whole_brain_vol": whole_brain,
        "ventricle_vol": ventricle_vol, "wmh_vol": wmh_vol,
        # Biomarkers (4)
        "csf_abeta42": amyloid_42, "csf_tau_total": tau_total,
        "plasma_ptau181": tau_181, "plasma_nfl": nfl,
        # Comorbidities binary (9)
        "hypertension": hypertension, "diabetes": diabetes,
        "depression": depression, "stroke_hx": stroke, "tbi_hx": tbi,
        "heart_disease": heart_disease, "afib": afib,
        "smoking_hx": smoking_hist, "family_hx_dementia": family_hx,
        # Medications (4)
        "on_cholinesterase": on_cholinesterase,
        "on_antihypertensive": on_antihypertensive,
        "on_antidepressant": on_antidepressant,
        "on_statin": on_statin,
        # Longitudinal (2)
        "n_visits": n_visits, "years_follow": years_follow,
    }

    # Pad to 165 features with additional simulated cognitive sub-scores
    sub_tests = [
        "craft_immediate", "craft_delayed", "num_span_fwd", "num_span_bkwd",
        "mint_score", "rey_total", "rey_delayed", "cowa_fluency",
        "judgement_score", "orientation_score", "attention_score",
        "visuospatial_score", "praxis_score", "abstraction_score",
        "problem_solving", "processing_speed", "language_score",
        "episodic_memory_z", "semantic_memory_z", "working_memory_z",
        "exec_function_z", "global_cognition_z",
        "moca_memory", "moca_visuospatial", "moca_exec",
        "moca_language", "moca_abstraction", "moca_attention",
        "moca_orientation",
        "daily_living_score", "iadl_score", "adl_score",
        "caregiver_burden", "informant_cdr",
        "pvlt_immediate", "pvlt_delayed", "pvlt_recognition",
        "bvmt_immediate", "bvmt_delayed", "bvmt_discrimination",
        "delis_naming", "delis_fluency", "delis_learning",
        "grooved_pegboard_dom", "grooved_pegboard_nondom",
        "grip_strength_dom", "grip_strength_nondom",
        "gait_speed", "balance_score", "timed_up_go",
        "sleep_hours", "sleep_quality", "ess_score",
        "bmi", "waist_circ", "systolic_bp", "diastolic_bp",
        "heart_rate", "weight_kg", "height_cm",
        "total_chol", "ldl", "hdl", "triglycerides",
        "glucose_fasting", "hba1c", "creatinine", "egfr",
        "albumin", "hemoglobin", "wbc", "platelets",
        "alt", "ast", "crp", "homocysteine",
        "vitamin_d", "b12", "folate", "tsh",
        "insulin", "igf1", "il6", "tnf_alpha",
        "cortisol_am", "testosterone", "estradiol",
        "income_level", "employment_status", "marital_status",
        "living_alone", "social_engagement_score",
        "physical_activity_hrs", "sedentary_hrs",
        "alcohol_drinks_wk", "caffeine_cups_day",
        "visual_acuity", "hearing_loss_db",
        "pain_score", "fatigue_score", "anxiety_score",
        "subjective_memory_concern",
        "brain_age_gap", "white_matter_integrity",
        "cortical_thickness_entorhinal",
        "cortical_thickness_prefrontal",
        "amyloid_pet_suvr", "tau_pet_suvr", "fdg_pet_suvr",
        "csf_nfl", "csf_ygranin", "csf_neurogranin",
        "snp_rs429358", "snp_rs7412",   # APOE genotyping markers
        "snp_rs6656401", "snp_rs35349669", "snp_rs10948363",
        "snp_rs11136000", "snp_rs9331896", "snp_rs3865444",
        "snp_rs10498633", "snp_rs17125944",
        "snp_rs7561528", "snp_rs4147929",
        "polygenic_risk_score",
    ]

    # Only as many as we need to reach 165 total
    current_n = len(data) + 1  # +1 for y
    needed = 165 - current_n
    for col in sub_tests[:needed]:
        data[col] = rng.normal(0, 1, N).round(3)

    data["y"] = y
    df = pd.DataFrame(data)
    return add_missing(df, rate=0.06, rng=rng)


print("\n── NACC UDS dataset ──────────────────────────────────")
save(make_nacc(), "nacc_uds.csv")

print(f"\n✅  All datasets written to  {OUT}/\n")
