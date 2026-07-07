import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_clean(path_main, path_bureau=None):
    print("Chargement de application_train.csv...")
    df = pd.read_csv(path_main)

    # Target et ID
    y = df['TARGET']
    df = df.drop(columns=['TARGET'])

    # Supprimer colonnes avec trop de valeurs manquantes (>60%)
    missing_ratio = df.isnull().mean()
    df = df.drop(columns=missing_ratio[missing_ratio > 0.6].index)
    print(f"Colonnes après suppression NaN>60% : {df.shape[1]}")

    # Remplir les valeurs manquantes
    for col in df.select_dtypes(include='number').columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Enrichissement avec bureau.csv
    if path_bureau:
        print("Chargement de bureau.csv...")
        bureau = pd.read_csv(path_bureau)
        bureau_agg = bureau.groupby('SK_ID_CURR').agg(
            nb_credits_exterieurs=('SK_ID_BUREAU', 'count'),
            montant_moyen_credit_ext=('AMT_CREDIT_SUM', 'mean'),
            nb_credits_actifs=('CREDIT_ACTIVE', lambda x: (x == 'Active').sum())
        ).reset_index()
        df = df.merge(bureau_agg, on='SK_ID_CURR', how='left')
        df[['nb_credits_exterieurs', 'montant_moyen_credit_ext', 'nb_credits_actifs']] = \
            df[['nb_credits_exterieurs', 'montant_moyen_credit_ext', 'nb_credits_actifs']].fillna(0)
        print("Bureau.csv fusionné ✅")

    return df, y

def feature_engineering(df):
    print("Feature engineering...")

    # Ratios financiers
    df['CREDIT_INCOME_RATIO']   = df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL'] + 1)
    df['ANNUITY_INCOME_RATIO']  = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + 1)
    df['CREDIT_TERM']           = df['AMT_ANNUITY'] / (df['AMT_CREDIT'] + 1)
    df['GOODS_CREDIT_RATIO']    = df['AMT_GOODS_PRICE'] / (df['AMT_CREDIT'] + 1)

    # Age et ancienneté
    df['AGE_YEARS']      = (-df['DAYS_BIRTH'] / 365).astype(int)
    df['YEARS_EMPLOYED'] = (-df['DAYS_EMPLOYED'].clip(upper=0) / 365)

    # Anomalie connue du dataset
    df['EMPLOYED_FLAG_ANOMALY'] = (df['DAYS_EMPLOYED'] == 365243).astype(int)
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, 0)

    # Nombre de documents fournis
    doc_cols = [c for c in df.columns if 'FLAG_DOCUMENT' in c]
    df['NB_DOCUMENTS_FOURNIS'] = df[doc_cols].sum(axis=1)

    print("Feature engineering terminé ✅")
    return df

def encode_and_split(df, y, test_size=0.2, random_state=42):
    df = df.drop(columns=['SK_ID_CURR'], errors='ignore')

    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    feature_names = list(df.columns)
    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"Taux de défaut : {y_train.mean():.2%}")
    return X_train, X_test, y_train, y_test, feature_names

def prepare_data(path_main, path_bureau=None):
    df, y = load_and_clean(path_main, path_bureau)
    df = feature_engineering(df)
    return encode_and_split(df, y)