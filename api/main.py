from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import joblib
import pandas as pd
import os

app = FastAPI(title="Home Credit Scoring API")

BASE_DIR = os.path.dirname(__file__)
model  = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
config = joblib.load(os.path.join(BASE_DIR, "config.pkl"))
THRESHOLD = config["threshold"]

FEATURE_NAMES = [
    'NAME_CONTRACT_TYPE','CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY',
    'CNT_CHILDREN','AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY',
    'AMT_GOODS_PRICE','NAME_TYPE_SUITE','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE',
    'NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','REGION_POPULATION_RELATIVE',
    'DAYS_BIRTH','DAYS_EMPLOYED','DAYS_REGISTRATION','DAYS_ID_PUBLISH',
    'FLAG_MOBIL','FLAG_EMP_PHONE','FLAG_WORK_PHONE','FLAG_CONT_MOBILE',
    'FLAG_PHONE','FLAG_EMAIL','OCCUPATION_TYPE','CNT_FAM_MEMBERS',
    'REGION_RATING_CLIENT','REGION_RATING_CLIENT_W_CITY',
    'WEEKDAY_APPR_PROCESS_START','HOUR_APPR_PROCESS_START',
    'REG_REGION_NOT_LIVE_REGION','REG_REGION_NOT_WORK_REGION',
    'LIVE_REGION_NOT_WORK_REGION','REG_CITY_NOT_LIVE_CITY',
    'REG_CITY_NOT_WORK_CITY','LIVE_CITY_NOT_WORK_CITY','ORGANIZATION_TYPE',
    'EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','APARTMENTS_AVG',
    'BASEMENTAREA_AVG','YEARS_BEGINEXPLUATATION_AVG','ELEVATORS_AVG',
    'ENTRANCES_AVG','FLOORSMAX_AVG','LANDAREA_AVG','LIVINGAREA_AVG',
    'NONLIVINGAREA_AVG','APARTMENTS_MODE','BASEMENTAREA_MODE',
    'YEARS_BEGINEXPLUATATION_MODE','ELEVATORS_MODE','ENTRANCES_MODE',
    'FLOORSMAX_MODE','LANDAREA_MODE','LIVINGAREA_MODE','NONLIVINGAREA_MODE',
    'APARTMENTS_MEDI','BASEMENTAREA_MEDI','YEARS_BEGINEXPLUATATION_MEDI',
    'ELEVATORS_MEDI','ENTRANCES_MEDI','FLOORSMAX_MEDI','LANDAREA_MEDI',
    'LIVINGAREA_MEDI','NONLIVINGAREA_MEDI','HOUSETYPE_MODE','TOTALAREA_MODE',
    'WALLSMATERIAL_MODE','EMERGENCYSTATE_MODE','OBS_30_CNT_SOCIAL_CIRCLE',
    'DEF_30_CNT_SOCIAL_CIRCLE','OBS_60_CNT_SOCIAL_CIRCLE',
    'DEF_60_CNT_SOCIAL_CIRCLE','DAYS_LAST_PHONE_CHANGE','FLAG_DOCUMENT_2',
    'FLAG_DOCUMENT_3','FLAG_DOCUMENT_4','FLAG_DOCUMENT_5','FLAG_DOCUMENT_6',
    'FLAG_DOCUMENT_7','FLAG_DOCUMENT_8','FLAG_DOCUMENT_9','FLAG_DOCUMENT_10',
    'FLAG_DOCUMENT_11','FLAG_DOCUMENT_12','FLAG_DOCUMENT_13','FLAG_DOCUMENT_14',
    'FLAG_DOCUMENT_15','FLAG_DOCUMENT_16','FLAG_DOCUMENT_17','FLAG_DOCUMENT_18',
    'FLAG_DOCUMENT_19','FLAG_DOCUMENT_20','FLAG_DOCUMENT_21',
    'AMT_REQ_CREDIT_BUREAU_HOUR','AMT_REQ_CREDIT_BUREAU_DAY',
    'AMT_REQ_CREDIT_BUREAU_WEEK','AMT_REQ_CREDIT_BUREAU_MON',
    'AMT_REQ_CREDIT_BUREAU_QRT','AMT_REQ_CREDIT_BUREAU_YEAR',
    'nb_credits_exterieurs','montant_moyen_credit_ext','nb_credits_actifs',
    'CREDIT_INCOME_RATIO','ANNUITY_INCOME_RATIO','CREDIT_TERM',
    'GOODS_CREDIT_RATIO','AGE_YEARS','YEARS_EMPLOYED',
    'EMPLOYED_FLAG_ANOMALY','NB_DOCUMENTS_FOURNIS'
]

class LoanApplication(BaseModel):
    # Features principales (obligatoires)
    AMT_INCOME_TOTAL: float
    AMT_CREDIT: float
    AMT_ANNUITY: float
    AMT_GOODS_PRICE: float
    DAYS_BIRTH: int
    DAYS_EMPLOYED: int
    # Features importantes avec valeurs par défaut
    CODE_GENDER: int = 1
    CNT_CHILDREN: int = 0
    NAME_CONTRACT_TYPE: int = 0
    FLAG_OWN_CAR: int = 0
    FLAG_OWN_REALTY: int = 1
    NAME_EDUCATION_TYPE: int = 1
    NAME_FAMILY_STATUS: int = 0
    NAME_INCOME_TYPE: int = 0
    NAME_HOUSING_TYPE: int = 0
    NAME_TYPE_SUITE: int = 0
    OCCUPATION_TYPE: int = 0
    ORGANIZATION_TYPE: int = 0
    REGION_RATING_CLIENT: int = 2
    REGION_RATING_CLIENT_W_CITY: int = 2
    REGION_POPULATION_RELATIVE: float = 0.02
    EXT_SOURCE_1: float = 0.5
    EXT_SOURCE_2: float = 0.5
    EXT_SOURCE_3: float = 0.5
    # Features calculées automatiquement si non fournies
    CREDIT_INCOME_RATIO: Optional[float] = None
    ANNUITY_INCOME_RATIO: Optional[float] = None
    CREDIT_TERM: Optional[float] = None
    GOODS_CREDIT_RATIO: Optional[float] = None
    AGE_YEARS: Optional[int] = None
    YEARS_EMPLOYED: Optional[float] = None
    EMPLOYED_FLAG_ANOMALY: int = 0
    NB_DOCUMENTS_FOURNIS: int = 3
    # Autres features secondaires
    DAYS_REGISTRATION: float = -2000.0
    DAYS_ID_PUBLISH: float = -2000.0
    DAYS_LAST_PHONE_CHANGE: float = -500.0
    FLAG_MOBIL: int = 1
    FLAG_EMP_PHONE: int = 1
    FLAG_WORK_PHONE: int = 0
    FLAG_CONT_MOBILE: int = 1
    FLAG_PHONE: int = 0
    FLAG_EMAIL: int = 0
    CNT_FAM_MEMBERS: float = 2.0
    WEEKDAY_APPR_PROCESS_START: int = 2
    HOUR_APPR_PROCESS_START: int = 10
    REG_REGION_NOT_LIVE_REGION: int = 0
    REG_REGION_NOT_WORK_REGION: int = 0
    LIVE_REGION_NOT_WORK_REGION: int = 0
    REG_CITY_NOT_LIVE_CITY: int = 0
    REG_CITY_NOT_WORK_CITY: int = 0
    LIVE_CITY_NOT_WORK_CITY: int = 0
    APARTMENTS_AVG: float = 0.1
    BASEMENTAREA_AVG: float = 0.08
    YEARS_BEGINEXPLUATATION_AVG: float = 0.97
    ELEVATORS_AVG: float = 0.0
    ENTRANCES_AVG: float = 0.1
    FLOORSMAX_AVG: float = 0.1
    LANDAREA_AVG: float = 0.05
    LIVINGAREA_AVG: float = 0.1
    NONLIVINGAREA_AVG: float = 0.0
    APARTMENTS_MODE: float = 0.1
    BASEMENTAREA_MODE: float = 0.08
    YEARS_BEGINEXPLUATATION_MODE: float = 0.97
    ELEVATORS_MODE: float = 0.0
    ENTRANCES_MODE: float = 0.1
    FLOORSMAX_MODE: float = 0.1
    LANDAREA_MODE: float = 0.05
    LIVINGAREA_MODE: float = 0.1
    NONLIVINGAREA_MODE: float = 0.0
    APARTMENTS_MEDI: float = 0.1
    BASEMENTAREA_MEDI: float = 0.08
    YEARS_BEGINEXPLUATATION_MEDI: float = 0.97
    ELEVATORS_MEDI: float = 0.0
    ENTRANCES_MEDI: float = 0.1
    FLOORSMAX_MEDI: float = 0.1
    LANDAREA_MEDI: float = 0.05
    LIVINGAREA_MEDI: float = 0.1
    NONLIVINGAREA_MEDI: float = 0.0
    HOUSETYPE_MODE: int = 0
    TOTALAREA_MODE: float = 0.1
    WALLSMATERIAL_MODE: int = 0
    EMERGENCYSTATE_MODE: int = 0
    OBS_30_CNT_SOCIAL_CIRCLE: float = 1.0
    DEF_30_CNT_SOCIAL_CIRCLE: float = 0.0
    OBS_60_CNT_SOCIAL_CIRCLE: float = 1.0
    DEF_60_CNT_SOCIAL_CIRCLE: float = 0.0
    FLAG_DOCUMENT_2: int = 0
    FLAG_DOCUMENT_3: int = 1
    FLAG_DOCUMENT_4: int = 0
    FLAG_DOCUMENT_5: int = 0
    FLAG_DOCUMENT_6: int = 0
    FLAG_DOCUMENT_7: int = 0
    FLAG_DOCUMENT_8: int = 0
    FLAG_DOCUMENT_9: int = 0
    FLAG_DOCUMENT_10: int = 0
    FLAG_DOCUMENT_11: int = 0
    FLAG_DOCUMENT_12: int = 0
    FLAG_DOCUMENT_13: int = 0
    FLAG_DOCUMENT_14: int = 0
    FLAG_DOCUMENT_15: int = 0
    FLAG_DOCUMENT_16: int = 0
    FLAG_DOCUMENT_17: int = 0
    FLAG_DOCUMENT_18: int = 0
    FLAG_DOCUMENT_19: int = 0
    FLAG_DOCUMENT_20: int = 0
    FLAG_DOCUMENT_21: int = 0
    AMT_REQ_CREDIT_BUREAU_HOUR: float = 0.0
    AMT_REQ_CREDIT_BUREAU_DAY: float = 0.0
    AMT_REQ_CREDIT_BUREAU_WEEK: float = 0.0
    AMT_REQ_CREDIT_BUREAU_MON: float = 0.0
    AMT_REQ_CREDIT_BUREAU_QRT: float = 0.0
    AMT_REQ_CREDIT_BUREAU_YEAR: float = 1.0
    nb_credits_exterieurs: float = 3.0
    montant_moyen_credit_ext: float = 100000.0
    nb_credits_actifs: float = 1.0

@app.get("/")
def root():
    return {"status": "Home Credit Scoring API opérationnelle", "threshold": THRESHOLD}

@app.post("/predict")
def predict(data: LoanApplication):
    d = data.dict()

    # Calcul automatique des features engineered
    if d['CREDIT_INCOME_RATIO'] is None:
        d['CREDIT_INCOME_RATIO'] = d['AMT_CREDIT'] / (d['AMT_INCOME_TOTAL'] + 1)
    if d['ANNUITY_INCOME_RATIO'] is None:
        d['ANNUITY_INCOME_RATIO'] = d['AMT_ANNUITY'] / (d['AMT_INCOME_TOTAL'] + 1)
    if d['CREDIT_TERM'] is None:
        d['CREDIT_TERM'] = d['AMT_ANNUITY'] / (d['AMT_CREDIT'] + 1)
    if d['GOODS_CREDIT_RATIO'] is None:
        d['GOODS_CREDIT_RATIO'] = d['AMT_GOODS_PRICE'] / (d['AMT_CREDIT'] + 1)
    if d['AGE_YEARS'] is None:
        d['AGE_YEARS'] = int(-d['DAYS_BIRTH'] / 365)
    if d['YEARS_EMPLOYED'] is None:
        d['YEARS_EMPLOYED'] = max(0, -d['DAYS_EMPLOYED'] / 365)

    df = pd.DataFrame([[d.get(f, 0) for f in FEATURE_NAMES]], columns=FEATURE_NAMES)
    df = df.astype(float)

    proba = model.predict_proba(df)[0, 1]
    prediction = int(proba >= THRESHOLD)
    risk_level = "Élevé" if proba > 0.6 else "Modéré" if proba > 0.3 else "Faible"

    return {
        "probability_of_default": round(float(proba), 4),
        "decision": "REFUS" if prediction == 1 else "ACCORD",
        "risk_level": risk_level,
        "threshold_used": THRESHOLD
    }