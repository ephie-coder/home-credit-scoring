import sys, mlflow, mlflow.sklearn, joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
import shap, matplotlib.pyplot as plt
sys.path.append('.')
from src.preprocessing import prepare_data
from src.business_score import business_cost, business_score_normalized, find_optimal_threshold

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("home_credit_scoring")

print("Chargement des données...")
X_train, X_test, y_train, y_test, feature_names = prepare_data(
    "data/HC_application_train.csv",
    "data/HC_bureau.csv"
)

print("Application SMOTE...")
smote = SMOTE(random_state=42, sampling_strategy=0.3)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
print(f"Après SMOTE — Train: {X_train_bal.shape}")

models_config = {
    "baseline_dummy": {
        "model": DummyClassifier(strategy="most_frequent"),
        "params": {}
    },
    "logistic_regression": {
        "model": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        "params": {"C": [0.01, 0.1, 1]}
    },
    "random_forest": {
        "model": RandomForestClassifier(random_state=42, n_jobs=-1),
        "params": {"n_estimators": [100], "max_depth": [5, 10]}
    },
    "xgboost": {
        "model": XGBClassifier(random_state=42, eval_metric='logloss',
                               scale_pos_weight=10, n_jobs=-1),
        "params": {"n_estimators": [100, 200], "max_depth": [3, 5],
                   "learning_rate": [0.05, 0.1]}
    }
}

results = []
best_model_obj = None
best_biz_score = -1
best_threshold = 0.5

for name, config in models_config.items():
    print(f"\n🔄 Entraînement : {name}")
    with mlflow.start_run(run_name=name):
        if config["params"]:
            grid = GridSearchCV(config["model"], config["params"],
                                cv=3, scoring='roc_auc', n_jobs=-1, verbose=0)
            grid.fit(X_train_bal, y_train_bal)
            model = grid.best_estimator_
            mlflow.log_params(grid.best_params_)
        else:
            model = config["model"]
            model.fit(X_train_bal, y_train_bal)

        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] \
                  if hasattr(model, 'predict_proba') else y_pred.astype(float)

        auc  = roc_auc_score(y_test, y_proba)
        f1   = f1_score(y_test, y_pred)
        rec  = recall_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        cost = business_cost(y_test, y_pred)
        biz  = business_score_normalized(y_test, y_pred)
        opt_t, opt_cost = find_optimal_threshold(y_test, y_proba)

        mlflow.log_metrics({
            "auc": auc, "f1": f1, "recall": rec, "precision": prec,
            "business_cost_euros": cost,
            "business_score_normalized": biz,
            "optimal_threshold": opt_t
        })
        if "xgboost" in name:
            import mlflow.xgboost
            mlflow.xgboost.log_model(model, "model")
        else:
            mlflow.sklearn.log_model(model, "model")

        results.append({"model": name, "AUC": round(auc,3), "F1": round(f1,3),
                        "Recall": round(rec,3), "Coût (€)": cost, "Score métier": biz})

        if biz > best_biz_score:
            best_biz_score = biz
            best_model_obj = model
            best_threshold = opt_t

        print(f"  ✅ AUC={auc:.3f} | F1={f1:.3f} | Recall={rec:.3f} | Coût={cost:,}€ | Score={biz:.3f}")

# Résultats
print("\n=== Comparaison finale ===")
print(pd.DataFrame(results).sort_values("Score métier", ascending=False).to_string(index=False))

# Export du meilleur modèle
import os
os.makedirs("api", exist_ok=True)
joblib.dump(best_model_obj, "api/model.pkl")
joblib.dump({"threshold": best_threshold}, "api/config.pkl")
print(f"\n✅ Meilleur modèle exporté → api/model.pkl (seuil: {best_threshold:.2f})")

# SHAP sur le meilleur modèle
print("\nCalcul SHAP (patience)...")
sample = X_test.sample(200, random_state=42)
background = X_train_bal.sample(100, random_state=42)

# Fix pour XGBoost : utiliser tree_path_dependent
explainer = shap.TreeExplainer(
    best_model_obj,
    background,
    feature_perturbation="tree_path_dependent"
)
shap_values = explainer.shap_values(sample)

shap.summary_plot(
    shap_values,
    sample,
    feature_names=feature_names,
    show=False
)
plt.tight_layout()
plt.savefig("shap_summary.png")
print("✅ shap_summary.png sauvegardé")