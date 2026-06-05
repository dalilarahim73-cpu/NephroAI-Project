# ============================================================
#  train.py  —  ESRD Prediction  —  XGBoost Pipeline
#  Exécuter une fois pour générer esrd_pipeline.pkl
# ============================================================
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)
from xgboost import XGBClassifier

# ── 1. Chargement ─────────────────────────────────────────────
print("=" * 60)
print("  ESRD PREDICTION — TRAINING PIPELINE (XGBoost)")
print("=" * 60)

data = pd.read_csv("esrd_prediction_dataset.csv")
print(f"Dataset shape : {data.shape}")
print(f"ESRD Risk dist:\n{data['ESRD Risk'].value_counts()}\n")

# ── 2. Encodage catégoriel ────────────────────────────────────
CAT_COLS = [
    "Gender", "Smoking", "Alcohol", "Hypertension",
    "Coronary Artery Disease", "Cancer", "Chronic Liver Disease",
    "Diabetic Retinopathy", "NSAID", "Statin", "Metformin",
    "Insulin", "Dipeptidyl Peptidase-4 Inhibitor"
]
le = LabelEncoder()
for col in CAT_COLS:
    if col in data.columns:
        data[col] = le.fit_transform(data[col].astype(str))

data['class'] = (data['ESRD Risk'] == 'Yes').astype(int)

# ── 3. Split train / test (colonne existante) ─────────────────
data = data.reset_index(drop=True)
META_COLS   = ['Patient ID', 'Dataset Split', 'ESRD Risk', 'class']
feature_cols = [c for c in data.columns
                if c not in META_COLS
                and pd.api.types.is_numeric_dtype(data[c])]

train_idx = data.index[data['Dataset Split'] == 'Training'].tolist()
test_idx  = data.index[data['Dataset Split'] == 'Testing'].tolist()
print(f"Train: {len(train_idx)} rows  |  Test: {len(test_idx)} rows")

X_raw = data[feature_cols]
y     = data['class'].values

# ── 4. Imputation + Scaling (fit sur train seulement) ─────────
imputer = SimpleImputer(strategy='median')
X_tr    = imputer.fit_transform(X_raw.iloc[train_idx])
X_te    = imputer.transform(X_raw.iloc[test_idx])

scaler  = StandardScaler()
X_tr    = scaler.fit_transform(X_tr)
X_te    = scaler.transform(X_te)

y_train = y[train_idx]
y_test  = y[test_idx]

n_neg = int(np.sum(y_train == 0))
n_pos = int(np.sum(y_train == 1))
spw   = round(n_neg / n_pos)
print(f"Class balance — No: {n_neg}  Yes: {n_pos}  → scale_pos_weight={spw}\n")

# ── 5. Entraînement XGBoost ───────────────────────────────────
print("Training XGBoost...")
model = XGBClassifier(
    n_estimators      = 200,
    max_depth         = 5,
    learning_rate     = 0.1,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    scale_pos_weight  = spw,      # gère le déséquilibre des classes
    eval_metric       = 'auc',
    n_jobs            = -1,
    random_state      = 42,
    tree_method       = 'hist'
)
model.fit(X_tr, y_train,
          eval_set=[(X_te, y_test)],
          verbose=False)

# ── 6. Évaluation ─────────────────────────────────────────────
probs  = model.predict_proba(X_te)[:, 1]
y_pred = model.predict(X_te)

print("\n" + "=" * 60)
print("  RÉSULTATS SUR LE TEST SET")
print("=" * 60)
print(f"Accuracy          : {accuracy_score(y_test, y_pred):.4f}")
print(f"Balanced Accuracy : {balanced_accuracy_score(y_test, y_pred):.4f}")
print(f"F1 (weighted)     : {f1_score(y_test, y_pred, average='weighted'):.4f}")
print(f"F1 (macro)        : {f1_score(y_test, y_pred, average='macro'):.4f}")
print(f"AUC-ROC           : {roc_auc_score(y_test, probs):.4f}")
print()
print(classification_report(y_test, y_pred,
                              target_names=['No ESRD Risk', 'ESRD Risk'],
                              zero_division=0))

cm = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n{cm}\n")

# ── 7. Sauvegarde du pipeline complet ─────────────────────────
pipeline = {
    'model':        model,
    'imputer':      imputer,
    'scaler':       scaler,
    'features':     feature_cols,
    'cat_cols':     CAT_COLS,
    'threshold':    0.5,
    'label_names':  ['No ESRD Risk', 'ESRD Risk'],
    'model_name':   'XGBoost',
    'n_features':   len(feature_cols),
}
joblib.dump(pipeline, 'esrd_pipeline.pkl')
print("=" * 60)
print("  ✅ Pipeline sauvegardé : esrd_pipeline.pkl")
print("=" * 60)

# ── 8. Fonction de prédiction (pour l'interface) ──────────────
def predict_esrd(patient_data: dict) -> dict:
    """
    Prédire le risque ESRD pour un nouveau patient.

    Parameters
    ----------
    patient_data : dict
        Dictionnaire avec les valeurs des features.
        Exemple:
        {
            'Age': 65,
            'Gender': 'Male',
            'Hypertension': 'Yes',
            'Baseline Serum Creatinine (mg/dL)': 1.8,
            ...
        }

    Returns
    -------
    dict avec 'prediction', 'probability', 'risk_label'
    """
    pipeline = joblib.load('esrd_pipeline.pkl')

    df = pd.DataFrame([patient_data])

    # Encoder les colonnes catégorielles
    le_pred = LabelEncoder()
    for col in pipeline['cat_cols']:
        if col in df.columns:
            df[col] = le_pred.fit_transform(df[col].astype(str))

    # Aligner les features
    for col in pipeline['features']:
        if col not in df.columns:
            df[col] = np.nan

    X = df[pipeline['features']]
    X = pipeline['imputer'].transform(X)
    X = pipeline['scaler'].transform(X)

    prob  = pipeline['model'].predict_proba(X)[0, 1]
    pred  = int(prob >= pipeline['threshold'])
    label = pipeline['label_names'][pred]

    return {
        'prediction':  pred,
        'probability': round(float(prob), 4),
        'risk_label':  label,
        'risk_pct':    f"{prob * 100:.1f}%"
    }


# ── Exemple d'utilisation ─────────────────────────────────────
if __name__ == '__main__':
    example = {
        'Age': 65,
        'Gender': 'Male',
        'Smoking': 'Yes',
        'Alcohol': 'No',
        'Hypertension': 'Yes',
        'Coronary Artery Disease': 'No',
        'Cancer': 'No',
        'Chronic Liver Disease': 'No',
        'Diabetic Retinopathy': 'Yes',
        'Baseline Serum Creatinine (mg/dL)': 2.1,
        'Mean Serum Creatinine (mg/dL)': 1.9,
        'Cholesterol (mg/dL)': 210,
        'Triglyceride (mg/dL)': 150,
        'LDL-C (mg/dL)': 130,
        'HDL-C (mg/dL)': 45,
        'Uric Acid (mg/dL)': 7.2,
        'Calcium (mg/dL)': 9.1,
        'Phosphate (mg/dL)': 3.8,
        'Hemoglobin (g/dL)': 11.5,
        'Albumin (g/dL)': 3.9,
        'HS-CRP (mg/dL)': 0.8,
        'HbA1c (%)': 7.2,
        'Glucose (mg/dL)': 140,
        'NSAID': 'No',
        'Statin': 'Yes',
        'Metformin': 'Yes',
        'Insulin': 'No',
        'Dipeptidyl Peptidase-4 Inhibitor': 'No',
    }
    result = predict_esrd(example)
    print(f"\n📋 Exemple de prédiction:")
    print(f"   Risque ESRD  : {result['risk_label']}")
    print(f"   Probabilité  : {result['risk_pct']}")