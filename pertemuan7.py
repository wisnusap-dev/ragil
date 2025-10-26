# ===============================================================
# Langkah 1 — Siapkan Data
# ===============================================================
print("\n=== Langkah 1: Load dan Split Data ===")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("processed_kelulusan.csv")
X = df.drop("Lulus", axis=1)
y = df["Lulus"]

sc = StandardScaler()
Xs = sc.fit_transform(X)

X_train, X_temp, y_train, y_temp = train_test_split(
    Xs, y, test_size=0.3, stratify=y, random_state=42
)

try:
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
    print("✅ Stratified split sukses.")
except ValueError:
    print("⚠️ Stratified gagal (terlalu sedikit data), pakai random split saja.")
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
print("Distribusi label Train:", y_train.value_counts().to_dict())
print("Distribusi label Val  :", y_val.value_counts().to_dict())
print("Distribusi label Test :", y_test.value_counts().to_dict())

# ===============================================================
# Langkah 2 — Bangun Model ANN
# ===============================================================
print("\n=== Langkah 2: Bangun Model ANN ===")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.random.set_seed(42)

model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy", "AUC"]
)

model.summary()

# ===============================================================
# Langkah 3 — Training dengan Early Stopping
# ===============================================================
print("\n=== Langkah 3: Training Model ===")

es = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=4,
    callbacks=[es],
    verbose=1
)

# ===============================================================
# Langkah 4 — Evaluasi di Test Set
# ===============================================================
print("\n=== Langkah 4: Evaluasi Model di Test Set ===")

from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score

loss, acc, auc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {acc:.4f} | AUC: {auc:.4f} | Loss: {loss:.4f}")

y_proba = model.predict(X_test).ravel()
y_pred = (y_proba >= 0.5).astype(int)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=3))

# ===============================================================
# Langkah 5 — Visualisasi Learning Curve
# ===============================================================
print("\n=== Langkah 5: Visualisasi Learning Curve ===")

import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Learning Curve (Loss vs Epoch)")
plt.tight_layout()
plt.savefig("learning_curve.png", dpi=120)
plt.show()

# ===============================================================
# Langkah 6 — Eksperimen (Ubah Arsitektur, Optimizer, Regularisasi)
# ===============================================================
print("\n=== Langkah 6: Eksperimen ===")

from tensorflow.keras import regularizers

configs = [
    {"neurons": 32, "dropout": 0.3, "opt": "adam"},
    {"neurons": 64, "dropout": 0.4, "opt": "adam"},
    {"neurons": 128, "dropout": 0.5, "opt": "sgd"},
]

results = []

for cfg in configs:
    print(f"\n--- Eksperimen: neurons={cfg['neurons']} dropout={cfg['dropout']} opt={cfg['opt']} ---")

    if cfg["opt"] == "adam":
        optimizer = keras.optimizers.Adam(1e-3)
    else:
        optimizer = keras.optimizers.SGD(learning_rate=1e-2, momentum=0.9)

    model = keras.Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(cfg["neurons"], activation="relu", kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(cfg["dropout"]),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy", "AUC"])

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=4,
        verbose=0
    )

    y_proba = model.predict(X_test).ravel()
    y_pred = (y_proba >= 0.5).astype(int)

    f1 = f1_score(y_test, y_pred)
    auc_val = roc_auc_score(y_test, y_proba)

    print(f"F1={f1:.3f} | AUC={auc_val:.3f}")
    results.append((cfg["neurons"], cfg["dropout"], cfg["opt"], f1, auc_val))
    
    # ===============================================================
# Langkah 5 — Visualisasi Learning Curve
# ===============================================================
print("\n=== Langkah 5: Visualisasi Learning Curve ===")

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Train Loss", linewidth=2)
plt.plot(history.history["val_loss"], label="Val Loss", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Learning Curve (Loss vs Epoch)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("learning_curve.png", dpi=120)
plt.show()

# (Opsional) tampilkan juga akurasi
plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Train Accuracy", linewidth=2)
plt.plot(history.history["val_accuracy"], label="Val Accuracy", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Learning Curve (Accuracy vs Epoch)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("accuracy_curve.png", dpi=120)
plt.show()


# Ringkasan hasil eksperimen
print("\n=== Ringkasan Eksperimen ===")
for r in results:
    print(f"Neurons={r[0]}, Dropout={r[1]}, Opt={r[2]}, F1={r[3]:.3f}, AUC={r[4]:.3f}")
