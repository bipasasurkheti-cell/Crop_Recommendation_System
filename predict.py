import numpy as np
import pandas as pd
import joblib

from tensorflow.keras.models import load_model


# ============================================================
# LOAD MODELS
# ============================================================

random_forest = joblib.load(
    "models/crop_model.pkl"
)

scaler = joblib.load(
    "models/scaler.pkl"
)

label_encoder = joblib.load(
    "models/label_encoder.pkl"
)

ann_model = load_model(
    "models/ann_model.keras"
)


# ============================================================
# TITLE
# ============================================================

print("\n")
print("=" * 50)
print("       CROP RECOMMENDATION SYSTEM")
print("=" * 50)


# ============================================================
# USER INPUT
# ============================================================

print("\nEnter soil and environmental values:\n")


N = float(
    input("Nitrogen (N): ")
)

P = float(
    input("Phosphorus (P): ")
)

K = float(
    input("Potassium (K): ")
)

temperature = float(
    input("Temperature (°C): ")
)

humidity = float(
    input("Humidity (%): ")
)

ph = float(
    input("Soil pH: ")
)

rainfall = float(
    input("Rainfall (mm): ")
)


# ============================================================
# CREATE DATAFRAME
# ============================================================

features = [

    "N",

    "P",

    "K",

    "temperature",

    "humidity",

    "ph",

    "rainfall"

]


input_data = pd.DataFrame(

    [[

        N,

        P,

        K,

        temperature,

        humidity,

        ph,

        rainfall

    ]],

    columns=features

)


# ============================================================
# RANDOM FOREST
# ============================================================

rf_prediction = random_forest.predict(

    input_data

)


rf_crop = label_encoder.inverse_transform(

    rf_prediction

)[0]


# ============================================================
# ANN
# ============================================================

input_scaled = scaler.transform(

    input_data

)


ann_probability = ann_model.predict(

    input_scaled,

    verbose=0

)


ann_prediction = np.argmax(

    ann_probability,

    axis=1

)


ann_crop = label_encoder.inverse_transform(

    ann_prediction

)[0]


# ============================================================
# CONFIDENCE
# ============================================================

ann_confidence = (

    np.max(ann_probability)

    * 100

)


# ============================================================
# RESULT
# ============================================================

print("\n")
print("=" * 50)
print("                 RESULT")
print("=" * 50)


print(
    "\nRandom Forest Recommendation:",
    rf_crop
)


print(
    "ANN Recommendation:",
    ann_crop
)


print(
    f"ANN Confidence: "
    f"{ann_confidence:.2f}%"
)


print("\n")
print("=" * 50)