# ========================================
# Pertemuan 5 — Modeling Kelulusan Mahasiswa
# ========================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)
import joblib

# === Langkah 1 — Muat Data ===
df = pd.read_csv("processed_kelulusan.csv")

X = df.drop("Lulus", axis=1)
y = df["Lulus"]

# Karena dataset kecil, stratify dihapus pada split kedua
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

print("\n=== BENTUK DATA ===")
print("Train:", X_train.shape)
print("Val  :", X_val.shape)
print("Test :", X_test.shape)

# === Langkah 2 — Baseline Model: Logistic Regression ===
num_cols = X_train.select_dtypes(include="number").columns

pre = ColumnTransformer([
    ("num", Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler())
    ]), num_cols),
], remainder="drop")

logreg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
pipe_lr = Pipeline([("pre", pre), ("clf", logreg)])

pipe_lr.fit(X_train, y_train)
y_val_pred = pipe_lr.predict(X_val)

print("\n=== BASELINE MODEL (LOGISTIC REGRESSION) ===")
print("F1 Score (val):", f1_score(y_val, y_val_pred, average="macro"))
print(classification_report(y_val, y_val_pred, digits=3))

# === Langkah 3 — Model Alternatif: Random Forest ===
rf = RandomForestClassifier(
    n_estimators=300, max_features="sqrt", class_weight="balanced", random_state=42
)
pipe_rf = Pipeline([("pre", pre), ("clf", rf)])
pipe_rf.fit(X_train, y_train)
y_val_rf = pipe_rf.predict(X_val)

print("\n=== RANDOM FOREST ===")
print("F1 Score (val):", f1_score(y_val, y_val_rf, average="macro"))
print(classification_report(y_val, y_val_rf, digits=3))

# === Langkah 4 — (Opsional) Tuning Sederhana ===
# karena data kecil, kita buat n_splits=2 biar gak error
skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
param = {
    "clf__max_depth": [None, 5, 10],
    "clf__min_samples_split": [2, 5]
}

gs = GridSearchCV(pipe_rf, param_grid=param, cv=skf,
                  scoring="f1_macro", n_jobs=-1, verbose=1)
gs.fit(X_train, y_train)

print("\n=== HASIL GRID SEARCH ===")
print("Best Params:", gs.best_params_)
print("Best CV F1 :", gs.best_score_)

best_rf = gs.best_estimator_
y_val_best = best_rf.predict(X_val)

print("\n=== BEST RANDOM FOREST (VALIDASI) ===")
print("F1(val):", f1_score(y_val, y_val_best, average="macro"))
print(classification_report(y_val, y_val_best, digits=3))

# === Langkah 5 — Evaluasi Akhir di Test Set ===
final_model = best_rf  # bisa diganti pipe_lr jika lebih baik

y_test_pred = final_model.predict(X_test)
print("\n=== EVALUASI AKHIR (TEST SET) ===")
print("F1(test):", f1_score(y_test, y_test_pred, average="macro"))
print(classification_report(y_test, y_test_pred, digits=3))
print("Confusion Matrix (test):")
print(confusion_matrix(y_test, y_test_pred))

# ROC-AUC jika model punya predict_proba
if hasattr(final_model, "predict_proba"):
    y_test_proba = final_model.predict_proba(X_test)[:, 1]
    try:
        auc = roc_auc_score(y_test, y_test_proba)
        print("ROC-AUC(test):", auc)
    except:
        auc = None

    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC={auc:.2f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.title("ROC Curve - Test Set")
    plt.tight_layout()
    plt.savefig("roc_test.png", dpi=120)
    plt.show()

# === Langkah 6 — Simpan Model ===
joblib.dump(final_model, "model.pkl")
print("\n✅ Model tersimpan sebagai 'model.pkl'")
