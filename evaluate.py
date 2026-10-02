import os

import numpy as np
import pandas as pd
import joblib

from tensorflow.keras.models import load_model

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    classification_report
)


# ============================================================
# MODEL EVALUATION
# ============================================================

print("\n")
print("=" * 60)
print("             MODEL EVALUATION")
print("=" * 60)


# ============================================================
# LOAD DATA
# ============================================================

data = pd.read_csv(
    "Crop_recommendation.csv"
)


features = [

    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall"

]


X = data[features]

y = data["label"]


# ============================================================
# LOAD ENCODER
# ============================================================

label_encoder = joblib.load(

    "models/label_encoder.pkl"

)


y_encoded = label_encoder.transform(y)


# ============================================================
# SAME TRAIN/TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y_encoded,

    test_size=0.20,

    random_state=42,

    stratify=y_encoded

)


# ============================================================
# RANDOM FOREST
# ============================================================

random_forest = joblib.load(

    "models/crop_model.pkl"

)


rf_prediction = random_forest.predict(

    X_test

)


rf_accuracy = accuracy_score(

    y_test,

    rf_prediction

)


# ============================================================
# ANN
# ============================================================

scaler = joblib.load(

    "models/scaler.pkl"

)


X_test_scaled = scaler.transform(

    X_test

)


ann_model = load_model(

    "models/ann_model.keras"

)


ann_probability = ann_model.predict(

    X_test_scaled,

    verbose=0

)


ann_prediction = np.argmax(

    ann_probability,

    axis=1

)


ann_accuracy = accuracy_score(

    y_test,

    ann_prediction

)


# ============================================================
# RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("                  RESULTS")
print("=" * 60)


print(

    f"\nRandom Forest Accuracy: "
    f"{rf_accuracy * 100:.2f}%"

)


print(

    f"ANN Accuracy: "
    f"{ann_accuracy * 100:.2f}%"

)


# ============================================================
# RANDOM FOREST REPORT
# ============================================================

print("\n")
print("=" * 60)
print("       RANDOM FOREST CLASSIFICATION")
print("=" * 60)


print(

    classification_report(

        y_test,

        rf_prediction,

        target_names=label_encoder.classes_

    )

)


# ============================================================
# ANN REPORT
# ============================================================

print("\n")
print("=" * 60)
print("          ANN CLASSIFICATION")
print("=" * 60)


print(

    classification_report(

        y_test,

        ann_prediction,

        target_names=label_encoder.classes_

    )

)


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 60)
print("          EVALUATION COMPLETED")
print("=" * 60)

print("\nCheck the results folder for graphs.") 