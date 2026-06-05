# ================================================================
#  App.py — NephroAI Flask Backend v6.0 (XGBoost ESRD Pipeline)
# ================================================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import os, sqlite3, hashlib, secrets, traceback
from datetime import datetime

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    print("[WARN] joblib non disponible — fallback heuristique actif")

# ================================================================
# CONFIGURATION
# ================================================================
app = Flask(__name__)

CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    allow_headers=["Content-Type", "Authorization"],
    methods=["GET", "POST", "DELETE", "OPTIONS"],
)

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DB_PATH       = os.path.join(BASE_DIR, "nephroai.db")
PIPELINE_PATH = os.path.join(BASE_DIR, "esrd_pipeline.pkl")

# ================================================================
# CHARGEMENT DU PIPELINE XGBoost
# ================================================================
pipeline = None

def load_pipeline():
    global pipeline
    if not JOBLIB_AVAILABLE:
        print("[WARN] joblib absent — fallback heuristique")
        return
    if not os.path.exists(PIPELINE_PATH):
        print(f"[WARN] {PIPELINE_PATH} introuvable — exécuter Train.py d'abord")
        return
    try:
        pipeline = joblib.load(PIPELINE_PATH)
        print(f"[OK] Pipeline XGBoost chargé — {pipeline['n_features']} features")
        print(f"     Features : {pipeline['features'][:4]}…")
    except Exception as e:
        pipeline = None
        print(f"[ERR] Chargement pipeline : {e}")

load_pipeline()

CAT_COLS = [
    "Gender", "Smoking", "Alcohol", "Hypertension",
    "Coronary Artery Disease", "Cancer", "Chronic Liver Disease",
    "Diabetic Retinopathy", "NSAID", "Statin", "Metformin",
    "Insulin", "Dipeptidyl Peptidase-4 Inhibitor"
]

# ================================================================
# BASE DE DONNÉES
# ================================================================
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode("utf-8")).hexdigest()

def init_db():
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL UNIQUE,
        email TEXT UNIQUE,
        password TEXT NOT NULL,
        role TEXT NOT NULL CHECK(role IN ('medecin','patient')),
        nom TEXT, prenom TEXT,
        created_at TEXT)""")

    cur.execute("""CREATE TABLE IF NOT EXISTS tokens (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL REFERENCES users(id),
        token TEXT NOT NULL UNIQUE,
        created_at TEXT NOT NULL)""")

    cur.execute("""CREATE TABLE IF NOT EXISTS patients (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        doctor_id INTEGER REFERENCES users(id),
        user_id INTEGER REFERENCES users(id),
        nom TEXT NOT NULL,
        prenom TEXT NOT NULL,
        age REAL NOT NULL,
        gender TEXT NOT NULL DEFAULT 'Male',
        created_at TEXT NOT NULL)""")

    cur.execute("""CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id INTEGER NOT NULL REFERENCES patients(id),
        age REAL, gender TEXT, smoking TEXT, alcohol TEXT, hypertension TEXT,
        coronary_artery_disease TEXT, cancer TEXT, chronic_liver_disease TEXT,
        diabetic_retinopathy TEXT,
        baseline_creatinine REAL, mean_creatinine REAL, cholesterol REAL,
        triglyceride REAL, ldl_c REAL, hdl_c REAL, uric_acid REAL,
        calcium REAL, phosphate REAL, hemoglobin REAL, albumin REAL,
        hs_crp REAL, hba1c REAL, glucose REAL,
        nsaid TEXT, statin TEXT, metformin TEXT, insulin TEXT, dpp4_inhibitor TEXT,
        egfr REAL,
        probability REAL, percentage REAL, risk_level TEXT,
        esrd_risk_label TEXT, model_used TEXT, predicted_at TEXT)""")

    if cur.execute("SELECT COUNT(*) as n FROM users").fetchone()["n"] == 0:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cur.executemany(
            "INSERT INTO users (username,email,password,role,nom,prenom,created_at) VALUES (?,?,?,?,?,?,?)",
            [
                ("docteur_homme", "karim@nephroai.dz",    hash_pw("medecin123"), "medecin", "Benali",    "Karim",   now),
                ("docteur_femme", "sara@nephroai.dz",     hash_pw("medecin456"), "medecin", "Ait Yahia", "Sara",    now),
                ("patient1",      "patient1@nephroai.dz", hash_pw("patient123"), "patient", "Amrani",    "Mohamed", now),
            ]
        )

    conn.commit()
    conn.close()

    # Migration douce : ajouter egfr si elle n'existe pas encore
    conn2 = get_db()
    try:
        conn2.execute("ALTER TABLE predictions ADD COLUMN egfr REAL")
        conn2.commit()
    except Exception:
        pass  # Colonne déjà présente
    finally:
        conn2.close()

    print(f"[OK] DB prête : {DB_PATH}")

init_db()

# ================================================================
# AUTH
# ================================================================
def get_current_user():
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return None
    token = auth[7:].strip()
    if not token:
        return None
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""SELECT u.* FROM users u
                   JOIN tokens t ON t.user_id = u.id
                   WHERE t.token = ?""", (token,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None

def require_login(role=None):
    user = get_current_user()
    if not user:
        return None, (jsonify({"success": False, "error": "Non connecté", "code": "NOT_LOGGED"}), 401)
    if role and user["role"] != role:
        return None, (jsonify({"success": False, "error": "Accès refusé", "code": "FORBIDDEN"}), 403)
    return user, None

# ================================================================
# LOGIQUE DE PRÉDICTION
# ================================================================

def norm(raw_data):
    """Normalise les clés du frontend vers les noms du dataset."""
    SEX_MAP = {"0": "Female", "1": "Male", 0: "Female", 1: "Male"}
    YESNO   = {"0": "No", "1": "Yes", 0: "No", 1: "Yes"}
    cr = float(raw_data.get("creatinine") or raw_data.get("Baseline Serum Creatinine (mg/dL)") or 1.0)
    return {
        "Age":                                float(raw_data.get("age") or raw_data.get("Age") or 45),
        "Gender":                             SEX_MAP.get(raw_data.get("sex"), raw_data.get("Gender", "Male")),
        "Smoking":                            raw_data.get("Smoking", "No"),
        "Alcohol":                            raw_data.get("Alcohol", "No"),
        "Hypertension":                       raw_data.get("hypertension") or raw_data.get("Hypertension", "No"),
        "Coronary Artery Disease":            raw_data.get("Coronary Artery Disease", "No"),
        "Cancer":                             raw_data.get("Cancer", "No"),
        "Chronic Liver Disease":              raw_data.get("Chronic Liver Disease", "No"),
        "Diabetic Retinopathy":               YESNO.get(raw_data.get("diabetes"), raw_data.get("Diabetic Retinopathy", "No")),
        "Baseline Serum Creatinine (mg/dL)":  cr,
        "Mean Serum Creatinine (mg/dL)":      float(raw_data.get("mean_creatinine") or raw_data.get("Mean Serum Creatinine (mg/dL)") or cr),
        "Cholesterol (mg/dL)":                float(raw_data.get("Cholesterol (mg/dL)") or raw_data.get("cholesterol") or 180.0),
        "Triglyceride (mg/dL)":               float(raw_data.get("Triglyceride (mg/dL)") or raw_data.get("triglyceride") or 150.0),
        "LDL-C (mg/dL)":                      float(raw_data.get("LDL-C (mg/dL)") or raw_data.get("ldl") or 100.0),
        "HDL-C (mg/dL)":                      float(raw_data.get("HDL-C (mg/dL)") or raw_data.get("hdl") or 50.0),
        "Uric Acid (mg/dL)":                  float(raw_data.get("Uric Acid (mg/dL)") or raw_data.get("uric_acid") or 5.5),
        "Calcium (mg/dL)":                    float(raw_data.get("Calcium (mg/dL)") or raw_data.get("calcium") or 9.5),
        "Phosphate (mg/dL)":                  float(raw_data.get("Phosphate (mg/dL)") or raw_data.get("phosphate") or 3.5),
        "Hemoglobin (g/dL)":                  float(raw_data.get("hemoglobin") or raw_data.get("Hemoglobin (g/dL)") or 13.0),
        "Albumin (g/dL)":                     float(raw_data.get("albumin") or raw_data.get("Albumin (g/dL)") or 4.0),
        "HS-CRP (mg/dL)":                     float(raw_data.get("HS-CRP (mg/dL)") or raw_data.get("hs_crp") or 0.3),
        "HbA1c (%)":                          float(raw_data.get("HbA1c (%)") or raw_data.get("hba1c") or 5.5),
        "Glucose (mg/dL)":                    float(raw_data.get("glucose") or raw_data.get("Glucose (mg/dL)") or 90.0),
        "NSAID":                              raw_data.get("NSAID", "No"),
        "Statin":                             raw_data.get("Statin", "No"),
        "Metformin":                          raw_data.get("Metformin", "No"),
        "Insulin":                            raw_data.get("Insulin", "No"),
        "Dipeptidyl Peptidase-4 Inhibitor":   raw_data.get("Dipeptidyl Peptidase-4 Inhibitor", "No"),
    }

def encode_cat(value: str, col: str) -> int:
    mappings = {
        "Gender":                           {"Female": 0, "Male": 1},
        "Smoking":                          {"No": 0, "Yes": 1},
        "Alcohol":                          {"No": 0, "Yes": 1},
        "Hypertension":                     {"No": 0, "Yes": 1},
        "Coronary Artery Disease":          {"No": 0, "Yes": 1},
        "Cancer":                           {"No": 0, "Yes": 1},
        "Chronic Liver Disease":            {"No": 0, "Yes": 1},
        "Diabetic Retinopathy":             {"No": 0, "Yes": 1},
        "NSAID":                            {"No": 0, "Yes": 1},
        "Statin":                           {"No": 0, "Yes": 1},
        "Metformin":                        {"No": 0, "Yes": 1},
        "Insulin":                          {"No": 0, "Yes": 1},
        "Dipeptidyl Peptidase-4 Inhibitor": {"No": 0, "Yes": 1},
    }
    m = mappings.get(col, {})
    val = str(value).strip().capitalize() if value not in ("Yes", "No", "Male", "Female") else str(value)
    return m.get(val, 0)

def esrd_predict(data: dict) -> dict:
    if pipeline is None:
        return _heuristic_predict(data)

    row = {}
    for feat in pipeline['features']:
        if feat in CAT_COLS:
            row[feat] = encode_cat(data.get(feat, "No"), feat)
        else:
            row[feat] = float(data.get(feat, np.nan))

    df = pd.DataFrame([row])
    X = pipeline['imputer'].transform(df[pipeline['features']])
    X = pipeline['scaler'].transform(X)

    prob  = float(pipeline['model'].predict_proba(X)[0, 1])
    pred  = int(prob >= pipeline['threshold'])
    label = pipeline['label_names'][pred]

    return {
        "probability":     prob,
        "percentage":      round(prob * 100, 1),
        "risk_level":      classify_risk(prob),
        "esrd_risk_label": label,
        "model_used":      pipeline['model_name'],
    }

def _heuristic_predict(data: dict) -> dict:
    s  = 0.0
    cr = float(data.get("Baseline Serum Creatinine (mg/dL)", 1.0))
    hb = float(data.get("Hemoglobin (g/dL)", 13.0))
    gl = float(data.get("Glucose (mg/dL)", 90.0))
    al = float(data.get("Albumin (g/dL)", 3.8))
    ag = float(data.get("Age", 45.0))
    if cr > 1.2: s += (cr - 1.2) * 0.18
    if hb < 12:  s += (12 - hb) * 0.05
    if gl > 100: s += (gl - 100) * 0.0015
    if al < 3.5: s += (3.5 - al) * 0.10
    if data.get("Hypertension") == "Yes":         s += 0.15
    if data.get("Diabetic Retinopathy") == "Yes": s += 0.12
    if ag > 60:  s += (ag - 60) * 0.005
    prob = min(s, 0.97)
    pred = int(prob >= 0.5)
    return {
        "probability":     prob,
        "percentage":      round(prob * 100, 1),
        "risk_level":      classify_risk(prob),
        "esrd_risk_label": ["No ESRD Risk", "ESRD Risk"][pred],
        "model_used":      "heuristic_fallback",
    }

def classify_risk(p: float) -> str:
    return "low" if p < 0.30 else ("medium" if p < 0.65 else "high")


# ================================================================
# CLINICAL OVERRIDE LAYER  —  règles médicales obligatoires
# ================================================================
def clinical_override(data: dict, egfr: float, ml_result: dict) -> dict:
    """
    Applique des règles cliniques strictes (KDIGO / lignes directrices
    néphrologiques) qui peuvent élever ou modifier le niveau de risque
    renvoyé par le modèle ML.

    Retourne un dict enrichi avec :
      - risk_level     : niveau final (low/medium/high) après overrides
      - probability    : probabilité finale (inchangée ou plancher clinique)
      - percentage     : idem en %
      - clinical_flags : liste des alertes détectées
      - clinical_message : message médical dynamique
      - urgency        : normal / moderate / urgent / critical
    """
    flags   = []      # liste des red flags détectés
    min_risk = ml_result["risk_level"]   # plancher minimal = résultat ML
    min_prob = ml_result["probability"]  # plancher minimal de probabilité

    # ── Valeurs biologiques ─────────────────────────────────────
    cr   = float(data.get("Baseline Serum Creatinine (mg/dL)", 1.0))
    k    = float(data.get("potassium",  data.get("Potassium (mEq/L)", 4.0)))
    na   = float(data.get("sodium",     data.get("Sodium (mEq/L)", 140.0)))
    alb  = float(data.get("Albumin (g/dL)", 4.0))
    hb   = float(data.get("Hemoglobin (g/dL)", 13.0))
    ph   = float(data.get("Phosphate (mg/dL)", 3.5))
    ca   = float(data.get("Calcium (mg/dL)", 9.5))
    uac  = float(data.get("Uric Acid (mg/dL)", 5.5))
    crp  = float(data.get("HS-CRP (mg/dL)", 0.3))
    hba1c= float(data.get("HbA1c (%)", 5.5))
    prot = data.get("protein", 0)    # présence de protéinurie (frontend simple)

    # ── CKD Stage par eGFR (KDIGO) ─────────────────────────────
    if egfr < 15:
        flags.append({"code": "EGFR_G5",    "severity": "critical",
                      "label": f"eGFR {egfr} mL/min — Stade G5 (insuffisance rénale terminale imminente)",
                      "label_ar": f"معدل الترشيح {egfr} — المرحلة G5 (فشل كلوي وشيك)"})
        min_risk = "high"; min_prob = max(min_prob, 0.90)
    elif egfr < 30:
        flags.append({"code": "EGFR_G4",    "severity": "critical",
                      "label": f"eGFR {egfr} mL/min — Stade G4 : risque très élevé (KDIGO)",
                      "label_ar": f"معدل الترشيح {egfr} — المرحلة G4: خطر مرتفع جداً"})
        min_risk = "high"; min_prob = max(min_prob, 0.80)
    elif egfr < 45:
        flags.append({"code": "EGFR_G3B",   "severity": "high",
                      "label": f"eGFR {egfr} mL/min — Stade G3b : surveillance néphrologue",
                      "label_ar": f"معدل الترشيح {egfr} — المرحلة G3b: متابعة أخصائي الكلى"})
        if min_risk == "low": min_risk = "medium"
        min_prob = max(min_prob, 0.55)
    elif egfr < 60:
        flags.append({"code": "EGFR_G3A",   "severity": "moderate",
                      "label": f"eGFR {egfr} mL/min — Stade G3a (IRC modérée)",
                      "label_ar": f"معدل الترشيح {egfr} — المرحلة G3a (قصور كلوي معتدل)"})
        if min_risk == "low": min_risk = "medium"
        min_prob = max(min_prob, 0.40)

    # ── Hyperkaliémie / Hypokaliémie ────────────────────────────
    if k > 6.5:
        flags.append({"code": "K_CRITICAL",  "severity": "critical",
                      "label": f"Kaliémie {k} mEq/L — Hyperkaliémie critique (risque arythmie)",
                      "label_ar": f"البوتاسيوم {k} — فرط البوتاسيوم الحرج (خطر اضطراب النظم)"})
        min_risk = "high"; min_prob = max(min_prob, 0.85)
    elif k > 5.5:
        flags.append({"code": "K_HIGH",      "severity": "high",
                      "label": f"Kaliémie {k} mEq/L — Hyperkaliémie (surveillance cardiaque)",
                      "label_ar": f"البوتاسيوم {k} — فرط البوتاسيوم (مراقبة قلبية)"})
        if min_risk != "high": min_risk = "high"
        min_prob = max(min_prob, 0.70)
    elif k < 3.0:
        flags.append({"code": "K_LOW",       "severity": "high",
                      "label": f"Kaliémie {k} mEq/L — Hypokaliémie sévère",
                      "label_ar": f"البوتاسيوم {k} — نقص البوتاسيوم الشديد"})
        if min_risk != "high": min_risk = "high"
        min_prob = max(min_prob, 0.65)
    elif k < 3.5:
        flags.append({"code": "K_WARN",      "severity": "moderate",
                      "label": f"Kaliémie {k} mEq/L — Hypokaliémie légère",
                      "label_ar": f"البوتاسيوم {k} — نقص بوتاسيوم خفيف"})
        if min_risk == "low": min_risk = "medium"

    # ── Dyskaliémie liée à l'IRC ────────────────────────────────
    # Déjà couverte ci-dessus

    # ── Dysnatrémie ─────────────────────────────────────────────
    if na < 120 or na > 160:
        flags.append({"code": "NA_CRITICAL", "severity": "critical",
                      "label": f"Natrémie {na} mEq/L — Dysnatrémie critique",
                      "label_ar": f"الصوديوم {na} — اضطراب الصوديوم الحرج"})
        min_risk = "high"; min_prob = max(min_prob, 0.85)
    elif na < 130 or na > 155:
        flags.append({"code": "NA_HIGH",     "severity": "high",
                      "label": f"Natrémie {na} mEq/L — Dysnatrémie sévère",
                      "label_ar": f"الصوديوم {na} — اضطراب صوديوم حاد"})
        if min_risk != "high": min_risk = "high"
        min_prob = max(min_prob, 0.70)
    elif na < 135 or na > 148:
        flags.append({"code": "NA_WARN",     "severity": "moderate",
                      "label": f"Natrémie {na} mEq/L — Légère dysnatrémie",
                      "label_ar": f"الصوديوم {na} — اضطراب خفيف بالصوديوم"})
        if min_risk == "low": min_risk = "medium"

    # ── Hypoalbuminémie ─────────────────────────────────────────
    if alb < 2.5:
        flags.append({"code": "ALB_CRITICAL","severity": "critical",
                      "label": f"Albuminémie {alb} g/dL — Hypoalbuminémie sévère (dénutrition/syndrome néphrotique)",
                      "label_ar": f"الألبومين {alb} — نقص ألبومين حاد"})
        min_risk = "high"; min_prob = max(min_prob, 0.80)
    elif alb < 3.0:
        flags.append({"code": "ALB_LOW",     "severity": "high",
                      "label": f"Albuminémie {alb} g/dL — Hypoalbuminémie (facteur pronostique défavorable)",
                      "label_ar": f"الألبومين {alb} — نقص ألبومين (مؤشر إنذار سيئ)"})
        if min_risk == "low": min_risk = "medium"
        min_prob = max(min_prob, 0.55)
    elif alb < 3.5:
        flags.append({"code": "ALB_WARN",    "severity": "moderate",
                      "label": f"Albuminémie {alb} g/dL — Légèrement abaissée",
                      "label_ar": f"الألبومين {alb} — منخفض بشكل طفيف"})
        if min_risk == "low": min_risk = "medium"

    # ── Anémie néphrogène ────────────────────────────────────────
    if hb < 8.0:
        flags.append({"code": "HB_CRITICAL", "severity": "critical",
                      "label": f"Hémoglobine {hb} g/dL — Anémie sévère (transfusion à envisager)",
                      "label_ar": f"الهيموغلوبين {hb} — فقر دم حاد (قد يتطلب نقل دم)"})
        if min_risk != "high": min_risk = "high"
        min_prob = max(min_prob, 0.75)
    elif hb < 10.0:
        flags.append({"code": "HB_LOW",      "severity": "high",
                      "label": f"Hémoglobine {hb} g/dL — Anémie modérée néphrogène",
                      "label_ar": f"الهيموغلوبين {hb} — فقر دم معتدل كلوي المنشأ"})
        if min_risk == "low": min_risk = "medium"
        min_prob = max(min_prob, 0.45)

    # ── Hyperphosphatémie ────────────────────────────────────────
    if ph > 6.0:
        flags.append({"code": "PH_HIGH",     "severity": "high",
                      "label": f"Phosphatémie {ph} mg/dL — Hyperphosphatémie (risque calcifications vasculaires)",
                      "label_ar": f"الفوسفات {ph} — ارتفاع الفوسفات (خطر تكلسات وعائية)"})
        if min_risk == "low": min_risk = "medium"
        min_prob = max(min_prob, 0.50)
    elif ph > 4.5:
        flags.append({"code": "PH_WARN",     "severity": "moderate",
                      "label": f"Phosphatémie {ph} mg/dL — Légèrement élevée",
                      "label_ar": f"الفوسفات {ph} — مرتفع بشكل طفيف"})

    # ── Créatinine très élevée ──────────────────────────────────
    if cr > 5.0:
        flags.append({"code": "CR_CRITICAL", "severity": "critical",
                      "label": f"Créatinine {cr} mg/dL — Insuffisance rénale avancée",
                      "label_ar": f"الكرياتينين {cr} — قصور كلوي متقدم"})
        min_risk = "high"; min_prob = max(min_prob, 0.85)
    elif cr > 3.0:
        flags.append({"code": "CR_HIGH",     "severity": "high",
                      "label": f"Créatinine {cr} mg/dL — Élévation importante",
                      "label_ar": f"الكرياتينين {cr} — ارتفاع ملحوظ"})
        if min_risk != "high": min_risk = "high"
        min_prob = max(min_prob, 0.70)

    # ── Acide urique élevé ───────────────────────────────────────
    if uac > 10.0:
        flags.append({"code": "UA_HIGH",     "severity": "high",
                      "label": f"Uricémie {uac} mg/dL — Hyperuricémie sévère (néphropathie urique)",
                      "label_ar": f"حمض اليوريك {uac} — فرط حمض البول الشديد"})
        if min_risk == "low": min_risk = "medium"

    # ── CRP élevée ───────────────────────────────────────────────
    if crp > 3.0:
        flags.append({"code": "CRP_HIGH",    "severity": "moderate",
                      "label": f"HS-CRP {crp} mg/dL — Inflammation systémique élevée",
                      "label_ar": f"بروتين سي التفاعلي {crp} — التهاب جهازي مرتفع"})

    # ── HbA1c ────────────────────────────────────────────────────
    if hba1c > 10.0:
        flags.append({"code": "HBA1C_CRIT",  "severity": "high",
                      "label": f"HbA1c {hba1c}% — Diabète très déséquilibré (risque néphropathique élevé)",
                      "label_ar": f"السكر التراكمي {hba1c}% — سكري غير متوازن بشدة"})
        if min_risk == "low": min_risk = "medium"
        min_prob = max(min_prob, 0.50)

    # ── Détermination de l'urgence ──────────────────────────────
    critical_flags = [f for f in flags if f["severity"] == "critical"]
    high_flags     = [f for f in flags if f["severity"] == "high"]

    if critical_flags:
        urgency = "critical"
    elif high_flags or min_risk == "high":
        urgency = "urgent"
    elif min_risk == "medium":
        urgency = "moderate"
    else:
        urgency = "normal"

    # ── Message médical dynamique ────────────────────────────────
    if urgency == "critical":
        message = "⚠️ URGENCE NÉPHROLOGIQUE — Anomalies biologiques critiques détectées. Transfert hospitalier immédiat recommandé."
        message_ar = "⚠️ حالة طارئة كلوية — شذوذات حيوية حرجة. يُنصح بالإحالة الفورية للمستشفى."
    elif urgency == "urgent":
        message = "🔴 Anomalies sévères détectées. Consultation néphrologue urgente dans les 48 h."
        message_ar = "🔴 شذوذات بيولوجية حادة. استشارة أخصائي الكلى خلال 48 ساعة."
    elif urgency == "moderate":
        message = "🟡 Paramètres perturbés nécessitant une surveillance rapprochée. Consultation dans les 4 semaines."
        message_ar = "🟡 مؤشرات مضطربة تستدعي مراقبة دقيقة. استشارة طبية خلال 4 أسابيع."
    else:
        message = "🟢 Paramètres biologiques dans les limites normales. Suivi annuel recommandé."
        message_ar = "🟢 المؤشرات البيولوجية ضمن الحدود الطبيعية. يُنصح بالمتابعة السنوية."

    # ── Reconstruction du résultat final ───────────────────────
    final = dict(ml_result)  # copie du résultat ML
    final["risk_level"]        = min_risk
    final["probability"]       = round(min_prob, 4)
    final["percentage"]        = round(min_prob * 100, 1)
    final["esrd_risk_label"]   = "ESRD Risk" if min_risk == "high" else "No ESRD Risk"
    final["clinical_flags"]    = flags
    final["clinical_message"]  = message
    final["clinical_message_ar"] = message_ar
    final["urgency"]           = urgency
    final["n_flags"]           = len(flags)
    final["n_critical"]        = len(critical_flags)

    return final

def compute_egfr(creatinine: float, age: float, gender: str) -> float:
    """CKD-EPI 2021 (sans race)."""
    is_female = str(gender).lower() in ("female", "0")
    kappa = 0.7 if is_female else 0.9
    alpha = -0.241 if is_female else -0.302
    sex_c = 1.012 if is_female else 1.0
    ratio = creatinine / kappa
    if ratio < 1:
        egfr = 142 * (ratio ** alpha) * (0.9938 ** age) * sex_c
    else:
        egfr = 142 * (ratio ** -1.200) * (0.9938 ** age) * sex_c
    return max(1.0, round(egfr, 1))

def factor_status(data: dict) -> dict:
    cr  = float(data.get("Baseline Serum Creatinine (mg/dL)", 1.0))
    hb  = float(data.get("Hemoglobin (g/dL)", 13.0))
    gl  = float(data.get("Glucose (mg/dL)", 90.0))
    al  = float(data.get("Albumin (g/dL)", 3.8))
    uac = float(data.get("Uric Acid (mg/dL)", 5.0))
    ca  = float(data.get("Calcium (mg/dL)", 9.5))
    ph  = float(data.get("Phosphate (mg/dL)", 3.5))
    hba = float(data.get("HbA1c (%)", 5.5))
    cho = float(data.get("Cholesterol (mg/dL)", 180.0))
    return {
        "creatinine":           "ok" if cr  <= 1.2  else ("warn" if cr  <= 2.0  else "bad"),
        "hemoglobin":           "ok" if hb  >= 12.0 else ("warn" if hb  >= 10.0 else "bad"),
        "glucose":              "ok" if gl  <= 100  else ("warn" if gl  <= 126  else "bad"),
        "albumin":              "ok" if al  >= 3.5  else ("warn" if al  >= 3.0  else "bad"),
        "uric_acid":            "ok" if uac <= 7.0  else ("warn" if uac <= 9.0  else "bad"),
        "calcium":              "ok" if 8.5 <= ca <= 10.5 else "warn",
        "phosphate":            "ok" if 2.5 <= ph <= 4.5  else "warn",
        "hba1c":                "ok" if hba <= 5.7  else ("warn" if hba <= 6.4  else "bad"),
        "cholesterol":          "ok" if cho <= 200  else ("warn" if cho <= 240  else "bad"),
        "hypertension":         "warn" if data.get("Hypertension") == "Yes" else "ok",
        "diabetic_retinopathy": "warn" if data.get("Diabetic Retinopathy") == "Yes" else "ok",
        "smoking":              "warn" if data.get("Smoking") == "Yes" else "ok",
    }

# ================================================================
# ROUTES AUTH
# ================================================================
@app.route("/login", methods=["POST"])
def login():
    data       = request.get_json(force=True) or {}
    identifier = data.get("username", "").strip()
    password   = data.get("password", "").strip()
    if not identifier or not password:
        return jsonify({"success": False, "error": "Champs manquants"}), 400
    conn = get_db()
    cur  = conn.cursor()
    cur.execute("SELECT * FROM users WHERE (username=? OR email=?) AND password=?",
                (identifier, identifier, hash_pw(password)))
    user = cur.fetchone()
    if not user:
        conn.close()
        return jsonify({"success": False, "error": "Identifiants incorrects"}), 401
    token = secrets.token_hex(32)
    now   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cur.execute("INSERT INTO tokens (user_id,token,created_at) VALUES (?,?,?)",
                (user["id"], token, now))
    conn.commit()
    conn.close()
    return jsonify({"success": True, "token": token,
                    "id":      user["id"],       "role":   user["role"],
                    "username":user["username"],  "email":  user["email"]  or "",
                    "nom":     user["nom"]    or "", "prenom": user["prenom"] or ""})

@app.route("/logout", methods=["POST"])
def logout():
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        token = auth[7:].strip()
        conn  = get_db()
        conn.execute("DELETE FROM tokens WHERE token=?", (token,))
        conn.commit()
        conn.close()
    return jsonify({"success": True})

@app.route("/me", methods=["GET"])
def me():
    user = get_current_user()
    if not user:
        return jsonify({"success": False, "code": "NOT_LOGGED"}), 401
    return jsonify({"success": True, "id": user["id"], "role": user["role"],
                    "username": user["username"],
                    "nom":      user["nom"]    or "",
                    "prenom":   user["prenom"] or ""})

@app.route("/register", methods=["POST"])
def register():
    user, err = require_login(role="medecin")
    if err: return err
    data     = request.get_json(force=True) or {}
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()
    email    = data.get("email",    "").strip() or None
    nom      = data.get("nom",      "").strip()
    prenom   = data.get("prenom",   "").strip()
    if not username or not password:
        return jsonify({"success": False, "error": "Username et mot de passe requis"}), 400
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        conn = get_db()
        cur  = conn.cursor()
        cur.execute(
            "INSERT INTO users (username,email,password,role,nom,prenom,created_at) VALUES (?,?,?,?,?,?,?)",
            (username, email, hash_pw(password), "patient", nom, prenom, now))
        new_id = cur.lastrowid
        conn.commit()
        conn.close()
        return jsonify({"success": True, "user_id": new_id,
                        "username": username, "nom": nom, "prenom": prenom})
    except sqlite3.IntegrityError:
        return jsonify({"success": False, "error": "Ce nom d'utilisateur est déjà pris"}), 409

# ================================================================
# PRÉDICTION
# ================================================================
@app.route("/predict", methods=["POST"])
def predict():
    user = get_current_user()
    if not user:
        return jsonify({"success": False, "error": "Non connecté", "code": "NOT_LOGGED"}), 401

    conn = None
    try:
        raw = request.get_json(force=True) or {}

        # Normalisation des clés frontend → dataset
        d = norm(raw)

        # Prédiction ML brute
        ml_raw  = esrd_predict(d)
        factors = factor_status(d)

        # eGFR
        egfr = compute_egfr(
            d["Baseline Serum Creatinine (mg/dL)"],
            d["Age"],
            d["Gender"]
        )

        # ── Clinical override layer (règles médicales strictes) ──
        d_ext = dict(d)
        d_ext["potassium"] = float(raw.get("potassium") or raw.get("Potassium (mEq/L)") or 4.0)
        d_ext["sodium"]    = float(raw.get("sodium")    or raw.get("Sodium (mEq/L)")    or 140.0)
        result = clinical_override(d_ext, egfr, ml_raw)

        # Infos patient
        nom       = raw.get("nom", "").strip()    or user.get("nom")    or "Inconnu"
        prenom    = raw.get("prenom", "").strip() or user.get("prenom") or "Patient"
        now       = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        link_user = user["id"] if user["role"] == "patient" else None
        doctor_id = user["id"] if user["role"] == "medecin" else None

        # Sauvegarde DB
        conn = get_db()
        cur  = conn.cursor()

        # Upsert patient
        cur.execute("""
            SELECT id FROM patients
            WHERE nom=? AND prenom=?
            AND (user_id IS ? OR doctor_id IS ?)
            LIMIT 1
        """, (nom, prenom, link_user, doctor_id))
        row = cur.fetchone()

        if row:
            patient_id = row["id"]
            cur.execute("UPDATE patients SET age=?, gender=? WHERE id=?",
                        (d["Age"], d["Gender"], patient_id))
        else:
            cur.execute(
                "INSERT INTO patients (doctor_id, user_id, nom, prenom, age, gender, created_at) VALUES (?,?,?,?,?,?,?)",
                (doctor_id, link_user, nom, prenom, d["Age"], d["Gender"], now)
            )
            patient_id = cur.lastrowid

        cur.execute("""INSERT INTO predictions
            (patient_id, age, gender, smoking, alcohol, hypertension,
             coronary_artery_disease, cancer, chronic_liver_disease, diabetic_retinopathy,
             baseline_creatinine, mean_creatinine, cholesterol, triglyceride, ldl_c, hdl_c,
             uric_acid, calcium, phosphate, hemoglobin, albumin, hs_crp, hba1c, glucose,
             nsaid, statin, metformin, insulin, dpp4_inhibitor,
             egfr, probability, percentage, risk_level, esrd_risk_label, model_used, predicted_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (patient_id,
             d["Age"], d["Gender"], d["Smoking"], d["Alcohol"], d["Hypertension"],
             d["Coronary Artery Disease"], d["Cancer"], d["Chronic Liver Disease"],
             d["Diabetic Retinopathy"],
             d["Baseline Serum Creatinine (mg/dL)"], d["Mean Serum Creatinine (mg/dL)"],
             d["Cholesterol (mg/dL)"], d["Triglyceride (mg/dL)"], d["LDL-C (mg/dL)"],
             d["HDL-C (mg/dL)"], d["Uric Acid (mg/dL)"], d["Calcium (mg/dL)"],
             d["Phosphate (mg/dL)"], d["Hemoglobin (g/dL)"], d["Albumin (g/dL)"],
             d["HS-CRP (mg/dL)"], d["HbA1c (%)"], d["Glucose (mg/dL)"],
             d["NSAID"], d["Statin"], d["Metformin"], d["Insulin"],
             d["Dipeptidyl Peptidase-4 Inhibitor"],
             egfr, result["probability"], result["percentage"],
             result["risk_level"], result["esrd_risk_label"],
             result["model_used"], now))
        conn.commit()

        return jsonify({
            "success":            True,
            "probability":        result["probability"],
            "percentage":         result["percentage"],
            "risk_level":         result["risk_level"],
            "esrd_risk_label":    result["esrd_risk_label"],
            "model_used":         result["model_used"],
            "egfr":               egfr,
            "patient_id":         patient_id,
            "saved":              True,
            "factors":            factors,
            "pipeline_loaded":    pipeline is not None,
            "clinical_flags":     result.get("clinical_flags", []),
            "clinical_message":   result.get("clinical_message", ""),
            "clinical_message_ar":result.get("clinical_message_ar", ""),
            "urgency":            result.get("urgency", "normal"),
            "n_flags":            result.get("n_flags", 0),
            "n_critical":         result.get("n_critical", 0),
        })

    except ValueError as ve:
        return jsonify({"success": False, "error": str(ve)}), 400
    except Exception:
        traceback.print_exc()
        return jsonify({"success": False, "error": "Erreur interne"}), 500
    finally:
        if conn:
            conn.close()

# ================================================================
# DASHBOARD
# ================================================================
@app.route("/patients", methods=["GET"])
def get_patients():
    user, err = require_login(role="medecin")
    if err: return err
    conn = get_db()
    cur  = conn.cursor()
    cur.execute("""SELECT
        p.id, p.nom, p.prenom, p.age, p.gender, p.created_at,
        pr.baseline_creatinine AS creatinine,
        pr.hemoglobin, pr.glucose, pr.albumin, pr.hba1c, pr.cholesterol,
        pr.egfr, pr.percentage, pr.risk_level,
        pr.esrd_risk_label, pr.model_used, pr.predicted_at
        FROM patients p
        LEFT JOIN predictions pr ON pr.patient_id = p.id
        ORDER BY p.id DESC LIMIT 200""")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return jsonify({"success": True, "patients": rows, "total": len(rows)})

@app.route("/patients/<int:patient_id>", methods=["GET"])
def get_patient_detail(patient_id):
    user, err = require_login(role="medecin")
    if err: return err
    conn = get_db()
    cur  = conn.cursor()
    cur.execute("SELECT * FROM patients WHERE id=?", (patient_id,))
    p = cur.fetchone()
    if not p:
        conn.close()
        return jsonify({"success": False, "error": "Patient introuvable"}), 404
    cur.execute("SELECT * FROM predictions WHERE patient_id=? ORDER BY predicted_at DESC", (patient_id,))
    preds = [dict(r) for r in cur.fetchall()]
    conn.close()
    return jsonify({"success": True, "patient": dict(p), "predictions": preds})

@app.route("/patients/<int:patient_id>", methods=["DELETE"])
def delete_patient(patient_id):
    user, err = require_login(role="medecin")
    if err: return err
    conn = get_db()
    cur  = conn.cursor()
    cur.execute("SELECT id FROM patients WHERE id=?", (patient_id,))
    if not cur.fetchone():
        conn.close()
        return jsonify({"success": False, "error": "Patient introuvable"}), 404
    cur.execute("DELETE FROM predictions WHERE patient_id=?", (patient_id,))
    cur.execute("DELETE FROM patients WHERE id=?", (patient_id,))
    conn.commit()
    conn.close()
    return jsonify({"success": True, "deleted_id": patient_id})

@app.route("/stats", methods=["GET"])
def get_stats():
    user, err = require_login(role="medecin")
    if err: return err
    conn = get_db()
    cur  = conn.cursor()
    h        = cur.execute("SELECT COUNT(*) as n FROM predictions WHERE risk_level='high'").fetchone()["n"]
    m        = cur.execute("SELECT COUNT(*) as n FROM predictions WHERE risk_level='medium'").fetchone()["n"]
    l        = cur.execute("SELECT COUNT(*) as n FROM predictions WHERE risk_level='low'").fetchone()["n"]
    esrd_pos = cur.execute("SELECT COUNT(*) as n FROM predictions WHERE esrd_risk_label='ESRD Risk'").fetchone()["n"]
    conn.close()
    total = h + m + l
    return jsonify({
        "success": True,
        "total": total, "high": h, "medium": m, "low": l,
        "esrd_positive": esrd_pos,
        "esrd_negative": total - esrd_pos,
    })

@app.route("/my-results", methods=["GET"])
def my_results():
    user, err = require_login(role="patient")
    if err: return err
    conn = get_db()
    cur  = conn.cursor()
    cur.execute("""SELECT pr.*, p.nom, p.prenom, p.age FROM predictions pr
        JOIN patients p ON p.id = pr.patient_id
        WHERE p.user_id=? ORDER BY pr.predicted_at DESC""", (user["id"],))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return jsonify({"success": True, "results": rows})

@app.route("/accounts", methods=["GET"])
def get_accounts():
    user, err = require_login(role="medecin")
    if err: return err
    conn = get_db()
    cur  = conn.cursor()
    cur.execute("SELECT id,username,email,nom,prenom,created_at FROM users WHERE role='patient' ORDER BY id DESC")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return jsonify({"success": True, "accounts": rows})

@app.route("/model-info", methods=["GET"])
def model_info():
    p = pipeline or {}
    return jsonify({
        "pipeline_loaded": pipeline is not None,
        "model_type":      p.get("model_name", "heuristic_fallback"),
        "n_features":      p.get("n_features", 0),
        "features":        p.get("features", []),
        "threshold":       p.get("threshold", 0.5),
        "label_names":     p.get("label_names", ["No ESRD Risk", "ESRD Risk"]),
        "api_version":     "6.0.0",
    })

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "status":          "NephroAI API v6.0 — XGBoost ESRD Pipeline",
        "pipeline_loaded": pipeline is not None,
        "model_type":      (pipeline or {}).get("model_name", "heuristic_fallback"),
    })

# ================================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  NephroAI Flask API v6.0 — XGBoost ESRD Pipeline")
    print(f"  Pipeline : {'OUI — ' + str((pipeline or {}).get('n_features',0)) + ' features' if pipeline else 'NON (fallback heuristique)'}")
    print(f"  Modèle   : {(pipeline or {}).get('model_name', 'N/A')}")
    print("  Auth     : Bearer Token")
    print("  URL      : http://127.0.0.1:5000")
    print("="*60 + "\n")
    app.run(debug=True, port=5000)