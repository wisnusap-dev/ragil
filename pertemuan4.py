# ========================================
# Analisis Kelulusan Mahasiswa (versi FIX FINAL)
# ========================================

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ---------- Pastikan file CSV ada ----------
csv_name = "kelulusan_mahasiswa.csv"
if not os.path.exists(csv_name):
    data = {
        "IPK": [3.8, 2.5, 3.4, 2.1, 3.9, 2.8, 3.2, 2.7, 3.6, 2.3],
        "Jumlah_Absensi": [3, 8, 4, 12, 2, 6, 5, 7, 4, 9],
        "Waktu_Belajar_Jam": [10, 5, 7, 2, 12, 4, 8, 3, 9, 4],
        "Lulus": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    }
    pd.DataFrame(data).to_csv(csv_name, index=False)
    print(f"File '{csv_name}' tidak ditemukan → dibuat otomatis.")

# ---------- Langkah 2 — Collection ----------
df = pd.read_csv(csv_name)
print("\n=== INFO DATA ===")
print(df.info())
print(df.head())

# ---------- Langkah 3 — Cleaning ----------
print("\n=== CEK MISSING VALUE ===")
print(df.isnull().sum())

# Hapus duplikat
before = len(df)
df = df.drop_duplicates()
after = len(df)
print(f"Duplikat dihapus: {before - after} baris")

# Visualisasi outlier IPK
plt.figure(figsize=(6,4))
sns.boxplot(x=df['IPK'])
plt.title("Boxplot IPK")
plt.tight_layout()
plt.show()

# ---------- Langkah 4 — EDA ----------
print("\n=== STATISTIK DESKRIPTIF ===")
print(df.describe())

plt.figure(figsize=(6,4))
sns.histplot(df['IPK'], bins=10, kde=True)
plt.title("Distribusi IPK")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
sns.scatterplot(x='IPK', y='Waktu_Belajar_Jam', data=df, hue='Lulus', s=100)
plt.title("IPK vs Waktu Belajar (berdasarkan Kelulusan)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Heatmap Korelasi")
plt.tight_layout()
plt.show()

# ---------- Langkah 5 — Feature Engineering ----------
df['Rasio_Absensi'] = df['Jumlah_Absensi'] / 14
df['IPK_x_Study'] = df['IPK'] * df['Waktu_Belajar_Jam']

processed_name = "processed_kelulusan.csv"
df.to_csv(processed_name, index=False)
print(f"\n✅ File '{processed_name}' berhasil dibuat!")

# ---------- Langkah 6 — Splitting Dataset ----------
X = df.drop('Lulus', axis=1)
y = df['Lulus']

# Split pertama: Train (70%) dan temp (30%)
try:
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    print("\nSplit pertama dengan stratify berhasil.")
except Exception as e:
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    print("\nSplit pertama tanpa stratify karena error:", e)

# Split kedua: Temp (30%) jadi val (15%) dan test (15%)
vc = y_temp.value_counts()
use_stratify_second = vc.min() >= 2

if use_stratify_second:
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
    print("Split kedua dilakukan DENGAN stratify.")
else:
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    print("Split kedua dilakukan TANPA stratify (dataset kecil).")

print("\n=== PEMBAGIAN DATASET ===")
print("Train :", X_train.shape, "  (class distribution:", y_train.value_counts().to_dict(), ")")
print("Validation :", X_val.shape, "  (class distribution:", y_val.value_counts().to_dict(), ")")
print("Test :", X_test.shape, "  (class distribution:", y_test.value_counts().to_dict(), ")")

# Simpan hasil split
X_train.join(y_train).to_csv("train.csv", index=False)
X_val.join(y_val).to_csv("val.csv", index=False)
X_test.join(y_test).to_csv("test.csv", index=False)
print("\n✅ File 'train.csv', 'val.csv', 'test.csv' telah disimpan.")

# ---------- Langkah 7 — Ringkasan ----------
print("\n=== RINGKASAN ===")
print("Dataset bersih disimpan di", processed_name)
print("Split disimpan di: train.csv, val.csv, test.csv")
print("Visualisasi dan analisis selesai ✅")
