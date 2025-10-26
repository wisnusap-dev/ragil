# pertemuan6.py
# ========================================
# Pertemuan 6 — Random Forest untuk Klasifikasi
# ========================================

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve
)

# -------------------------
# Langkah 0 — Pastikan file processed_kelulusan.csv ada
# -------------------------
CSV = "processed_kelulusan.csv"
if not os.path.exists(CSV):
    raise FileNotFoundError(f"File '{CSV}' tidak ditemukan. Pastikan hasil Pertemuan 4 ada di folder ini.")

# -------------------------
# Langkah 1 — Muat Data & Split
# -------------------------
df = pd.read_csv(CSV)
if "Lulus" not in df.columns:
    raise KeyError("Kolom target 'Lulus' tidak ditemukan di processed_kelulusan.csv")

X = df.drop("Lulus", axis=1)
y = df["Lulus"]

# Jika dataset sangat kecil, stratify kedua split mungkin gagal.
# Kita akan coba stratify di split pertama; split kedua hanya jika aman.
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

# Cek apakah y_temp memiliki tiap kelas minimal 2 untuk stratify
vc = y_temp.value_counts()
if vc.min() >= 2:
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )
    stratify_second = True
else:
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42
    )
    stratify_second = False

print("\n=== Bentuk data ===")
print("Total:", X.shape)
print("Train:", X_train.shape)
print("Val  :", X_val.shape, "  (y_val distrib):", y_val.value_counts().to_dict())
print("Test :", X_test.shape, "  (y_test distrib):", y_test.value_counts().to_dict())
print("Stratify kedua digunakan?" , stratify_second)

# -------------------------
# Langkah 2 — Pipeline & Baseline Random Forest
# -------------------------
num_cols = X_train.select_dtypes(include=np.number).columns.tolist()

pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), num_cols),
], remainder="drop")

rf = RandomForestClassifier(
    n_estimators=300, max_features="sqrt",
    class_weight="balanced", random_state=42
)

pipe = Pipeline([("pre", pre), ("clf", rf)])

# Fit baseline
pipe.fit(X_train, y_train)
y_val_pred = pipe.predict(X_val)

print("\n=== Baseline Random Forest ===")
print("F1 (val):", f1_score(y_val, y_val_pred, average="macro"))
print(classification_report(y_val, y_val_pred, digits=4))

# -------------------------
# Langkah 3 — Validasi Silang (di train only)
# -------------------------
# Untuk dataset kecil, set n_splits kecil (min 2)
n_splits = 5
if len(y_train) < 10:
    n_splits = 2
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

try:
    scores = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="f1_macro", n_jobs=-1)
    print(f"\nCV F1-macro (train, {n_splits} folds): {scores.mean():.4f} ± {scores.std():.4f}")
except Exception as e:
    print("\nCross-val gagal:", e)

# -------------------------
# Langkah 4 — Tuning Ringkas (GridSearch)
# -------------------------
param = {
  "clf__max_depth": [None, 12, 20],
  "clf__min_samples_split": [2, 5]
}

gs = GridSearchCV(pipe, param_grid=param, cv=skf, scoring="f1_macro", n_jobs=-1, verbose=0)
gs.fit(X_train, y_train)

print("\n=== GridSearchCV Hasil ===")
print("Best params:", gs.best_params_)
print("Best CV F1:", gs.best_score_)

best_model = gs.best_estimator_
y_val_best = best_model.predict(X_val)
print("\n=== Evaluasi model terbaik pada Val ===")
print("F1 (val) best:", f1_score(y_val, y_val_best, average="macro"))
print(classification_report(y_val, y_val_best, digits=4))

# -------------------------
# Langkah 5 — Evaluasi Akhir pada Test Set
# -------------------------
final_model = best_model  # bisa disesuaikan
y_test_pred = final_model.predict(X_test)

print("\n=== Evaluasi Akhir (Test Set) ===")
print("F1 (test):", f1_score(y_test, y_test_pred, average="macro"))
print(classification_report(y_test, y_test_pred, digits=4))
print("Confusion matrix (test):")
cm = confusion_matrix(y_test, y_test_pred)
print(cm)

# ROC & PR (jika possible)
if hasattr(final_model, "predict_proba"):
    try:
        y_test_proba = final_model.predict_proba(X_test)[:, 1]
        # AUC may be undefined if only one class present in y_test
        try:
            auc = roc_auc_score(y_test, y_test_proba)
            print("ROC-AUC (test):", auc)
        except Exception as e:
            auc = None
            print("ROC-AUC tidak dapat dihitung (satu kelas di y_test).", e)

        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_test_proba, pos_label=1)
        plt.figure()
        plt.plot(fpr, tpr, marker=".")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve (AUC={auc if auc is not None else 'N/A'})")
        plt.tight_layout()
        plt.savefig("roc_test.png", dpi=120)
        plt.close()

        # Precision-Recall curve
        prec, rec, _ = precision_recall_curve(y_test, y_test_proba, pos_label=1)
        plt.figure()
        plt.plot(rec, prec, marker=".")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve (test)")
        plt.tight_layout()
        plt.savefig("pr_test.png", dpi=120)
        plt.close()
    except Exception as e:
        print("Tidak dapat membuat ROC/PR:", e)
else:
    print("Model tidak memiliki predict_proba, ROC/PR tidak tersedia.")

# -------------------------
# Langkah 6 — Pentingnya Fitur
# -------------------------
print("\n=== Feature Importances (native RF) ===")
try:
    # Dapatkan feature names yang diproses — di sini hanya num_cols
    feat_names = num_cols
    importances = final_model.named_steps["clf"].feature_importances_
    feat_imp = sorted(zip(feat_names, importances), key=lambda x: x[1], reverse=True)
    for name, val in feat_imp:
        print(f"{name}: {val:.4f}")
except Exception as e:
    print("Feature importance tidak tersedia:", e)

# Optional: permutation importance (jika ingin, uncomment)
# from sklearn.inspection import permutation_importance
# r = permutation_importance(final_model, X_val, y_val, n_repeats=10, random_state=42, n_jobs=-1)
# sorted_idx = r.importances_mean.argsort()[::-1]
# for i in sorted_idx:
#     print(f"{feat_names[i]}: mean imp {r.importances_mean[i]:.4f} std {r.importances_std[i]:.4f}")

# -------------------------
# Langkah 7 — Simpan Model
# -------------------------
MODEL_PATH = "rf_model.pkl"
joblib.dump(final_model, MODEL_PATH)
print(f"\n✅ Model disimpan sebagai '{MODEL_PATH}'")

# -------------------------
# Langkah 8 — Cek Inference Lokal (contoh)
# -------------------------
print("\n=== Contoh inference lokal ===")
sample = pd.DataFrame([{
  "IPK": 3.4,
  "Jumlah_Absensi": 4,
  "Waktu_Belajar_Jam": 7,
  # pastikan fitur turunan sama seperti di processed_kelulusan.csv
  "Rasio_Absensi": 4/14,
  "IPK_x_Study": 3.4*7
}])
try:
    pred = final_model.predict(sample)[0]
    proba = final_model.predict_proba(sample)[:,1][0] if hasattr(final_model, "predict_proba") else None
    print("Sample prediction:", int(pred), " proba:", proba)
except Exception as e:
    print("Gagal infer sample:", e)

print("\n=== Selesai ===")
