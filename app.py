import os

from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model


# ============================================================
# FLASK APP
# ============================================================

# IMPORTANT:
# Your index.html is in the same folder as app.py
app = Flask(__name__, template_folder=".")


# ============================================================
# LOAD MODELS
# ============================================================

print("=" * 60)
print("CROP RECOMMENDATION SYSTEM")
print("=" * 60)

try:
    random_forest = joblib.load(
        "models/crop_model.pkl"
    )
    print("[OK] Random Forest model loaded")

except Exception as e:
    random_forest = None
    print("[ERROR] Random Forest:", e)


try:
    scaler = joblib.load(
        "models/scaler.pkl"
    )
    print("[OK] Scaler loaded")

except Exception as e:
    scaler = None
    print("[ERROR] Scaler:", e)


try:
    label_encoder = joblib.load(
        "models/label_encoder.pkl"
    )
    print("[OK] Label encoder loaded")

except Exception as e:
    label_encoder = None
    print("[ERROR] Label encoder:", e)


try:
    ann_model = load_model(
        "models/ann_model.keras"
    )
    print("[OK] ANN model loaded")

except Exception as e:
    ann_model = None
    print("[ERROR] ANN model:", e)


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/", methods=["GET", "POST"])
def home():

    rf_crop = None
    ann_crop = None
    confidence = None
    error_message = None

    # --------------------------------------------------------
    # WHEN FORM IS SUBMITTED
    # --------------------------------------------------------

    if request.method == "POST":

        try:

            # =================================================
            # GET USER INPUT
            # =================================================

            N = float(request.form["N"])

            P = float(request.form["P"])

            K = float(request.form["K"])

            temperature = float(
                request.form["temperature"]
            )

            humidity = float(
                request.form["humidity"]
            )

            ph = float(
                request.form["ph"]
            )

            rainfall = float(
                request.form["rainfall"]
            )


            # =================================================
            # CREATE DATAFRAME
            # =================================================

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

                columns=[
                    "N",
                    "P",
                    "K",
                    "temperature",
                    "humidity",
                    "ph",
                    "rainfall"
                ]

            )


            # =================================================
            # RANDOM FOREST PREDICTION
            # =================================================

            if random_forest is not None:

                rf_prediction = random_forest.predict(
                    input_data
                )

                if label_encoder is not None:

                    rf_crop = label_encoder.inverse_transform(
                        rf_prediction
                    )[0]

                else:

                    rf_crop = str(
                        rf_prediction[0]
                    )


            # =================================================
            # ANN PREDICTION
            # =================================================

            if (
                ann_model is not None
                and scaler is not None
            ):

                scaled_input = scaler.transform(
                    input_data
                )

                probabilities = ann_model.predict(
                    scaled_input,
                    verbose=0
                )

                ann_prediction = np.argmax(
                    probabilities,
                    axis=1
                )

                if label_encoder is not None:

                    ann_crop = label_encoder.inverse_transform(
                        ann_prediction
                    )[0]

                else:

                    ann_crop = str(
                        ann_prediction[0]
                    )


                # ANN confidence

                confidence = round(
                    float(
                        np.max(probabilities) * 100
                    ),
                    2
                )


        except Exception as e:

            error_message = str(e)


    # ========================================================
    # SHOW INDEX.HTML
    # ========================================================

    return render_template(
        "index.html",
        rf_crop=rf_crop,
        ann_crop=ann_crop,
        confidence=confidence,
        error_message=error_message
    )


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("SERVER STARTING")
    print("=" * 60)
    print()
    print("Open this address in your browser:")
    print()
    print("http://127.0.0.1:5000")
    print()
    print("=" * 60)

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "5000")),
        debug=False
    )