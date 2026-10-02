import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping


# ============================================================
# CROP RECOMMENDATION SYSTEM
# MACHINE LEARNING + ARTIFICIAL NEURAL NETWORK
# ============================================================

print("\n")
print("=" * 60)
print("           CROP RECOMMENDATION SYSTEM")
print("=" * 60)


# ============================================================
# 1. LOAD DATASET
# ============================================================

DATASET_PATH = "Crop_recommendation.csv"

if not os.path.exists(DATASET_PATH):
    print("\nERROR!")
    print("Crop_recommendation.csv was not found.")
    print("Put the CSV file in the same folder as train.py")
    exit()

data = pd.read_csv(DATASET_PATH)

print("\nDataset loaded successfully.")
print("Dataset shape:", data.shape)

print("\nFirst 5 records:")
print(data.head())


# ============================================================
# 2. CHECK DATA
# ============================================================

print("\nDataset columns:")
print(data.columns.tolist())

print("\nMissing values:")
print(data.isnull().sum())


# ============================================================
# 3. DEFINE FEATURES AND TARGET
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

target = "label"


# Check required columns

for column in features + [target]:

    if column not in data.columns:

        print(
            f"\nERROR: Column '{column}' "
            "does not exist in dataset."
        )

        exit()


X = data[features]

y = data[target]


# ============================================================
# 4. LABEL ENCODING
# ============================================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)


print("\nNumber of crop classes:")

print(
    len(label_encoder.classes_)
)


print("\nCrop classes:")

for number, crop in enumerate(
    label_encoder.classes_
):

    print(
        f"{number} = {crop}"
    )


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y_encoded,

    test_size=0.20,

    random_state=42,

    stratify=y_encoded
)


print("\nTraining samples:", len(X_train))

print("Testing samples:", len(X_test))


# ============================================================
# 6. RANDOM FOREST CLASSIFIER
# ============================================================

print("\n")
print("=" * 60)
print("             RANDOM FOREST")
print("=" * 60)


random_forest = RandomForestClassifier(

    n_estimators=200,

    random_state=42

)


random_forest.fit(

    X_train,

    y_train

)


# Prediction

rf_prediction = random_forest.predict(

    X_test

)


# Accuracy

rf_accuracy = accuracy_score(

    y_test,

    rf_prediction

)


print(
    f"\nRandom Forest Accuracy: "
    f"{rf_accuracy * 100:.2f}%"
)


# ============================================================
# RANDOM FOREST CLASSIFICATION REPORT
# ============================================================

rf_report = classification_report(

    y_test,

    rf_prediction,

    target_names=label_encoder.classes_

)


print("\nRandom Forest Classification Report:")

print(rf_report)


# ============================================================
# 7. STANDARDIZATION FOR ANN
# ============================================================

scaler = StandardScaler()


X_train_scaled = scaler.fit_transform(

    X_train

)


X_test_scaled = scaler.transform(

    X_test

)


# ============================================================
# 8. CREATE ANN
# ============================================================

print("\n")
print("=" * 60)
print("       ARTIFICIAL NEURAL NETWORK")
print("=" * 60)


number_of_classes = len(

    label_encoder.classes_

)


ann_model = Sequential([

    Input(
        shape=(len(features),)
    ),

    Dense(
        128,
        activation="relu"
    ),

    Dropout(
        0.20
    ),

    Dense(
        64,
        activation="relu"
    ),

    Dropout(
        0.20
    ),

    Dense(
        32,
        activation="relu"
    ),

    Dense(
        number_of_classes,
        activation="softmax"
    )

])


# ============================================================
# 9. COMPILE ANN
# ============================================================

ann_model.compile(

    optimizer="adam",

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)


print("\nANN model created.")

ann_model.summary()


# ============================================================
# 10. EARLY STOPPING
# ============================================================

early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=10,

    restore_best_weights=True

)


# ============================================================
# 11. TRAIN ANN
# ============================================================

print("\nTraining ANN...")

history = ann_model.fit(

    X_train_scaled,

    y_train,

    validation_split=0.20,

    epochs=100,

    batch_size=32,

    callbacks=[early_stopping],

    verbose=1

)


# ============================================================
# 12. ANN EVALUATION
# ============================================================

ann_loss, ann_accuracy = ann_model.evaluate(

    X_test_scaled,

    y_test,

    verbose=0

)


print(
    f"\nANN Accuracy: "
    f"{ann_accuracy * 100:.2f}%"
)


# ============================================================
# 13. ANN PREDICTION
# ============================================================

ann_probability = ann_model.predict(

    X_test_scaled,

    verbose=0

)


ann_prediction = np.argmax(

    ann_probability,

    axis=1

)


# ============================================================
# 14. ANN CLASSIFICATION REPORT
# ============================================================

ann_report = classification_report(

    y_test,

    ann_prediction,

    target_names=label_encoder.classes_

)


print("\nANN Classification Report:")

print(ann_report)


# ============================================================
# 15. CREATE FOLDERS
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)

os.makedirs(
    "results",
    exist_ok=True
)


# ============================================================
# 16. SAVE RANDOM FOREST
# ============================================================

joblib.dump(

    random_forest,

    "models/crop_model.pkl"

)


# ============================================================
# 17. SAVE SCALER
# ============================================================

joblib.dump(

    scaler,

    "models/scaler.pkl"

)


# ============================================================
# 18. SAVE LABEL ENCODER
# ============================================================

joblib.dump(

    label_encoder,

    "models/label_encoder.pkl"

)


# ============================================================
# 19. SAVE ANN MODEL
# ============================================================

ann_model.save(

    "models/ann_model.keras"

)


# ============================================================
# 20. ANN ACCURACY GRAPH
# ============================================================

plt.figure(
    figsize=(9, 6)
)


plt.plot(

    history.history["accuracy"],

    label="Training Accuracy"

)


plt.plot(

    history.history["val_accuracy"],

    label="Validation Accuracy"

)


plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)

plt.title(
    "ANN Training and Validation Accuracy"
)

plt.legend()

plt.grid(True)

plt.tight_layout()


plt.savefig(

    "results/ann_accuracy_graph.png"

)

plt.close()


# ============================================================
# 21. ANN LOSS GRAPH
# ============================================================

plt.figure(
    figsize=(9, 6)
)


plt.plot(

    history.history["loss"],

    label="Training Loss"

)


plt.plot(

    history.history["val_loss"],

    label="Validation Loss"

)


plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "ANN Training and Validation Loss"
)

plt.legend()

plt.grid(True)

plt.tight_layout()


plt.savefig(

    "results/ann_loss_graph.png"

)

plt.close()


# ============================================================
# 22. RANDOM FOREST CONFUSION MATRIX
# ============================================================

rf_cm = confusion_matrix(

    y_test,

    rf_prediction

)


fig, ax = plt.subplots(

    figsize=(14, 12)

)


disp = ConfusionMatrixDisplay(

    confusion_matrix=rf_cm,

    display_labels=label_encoder.classes_

)


disp.plot(

    ax=ax,

    xticks_rotation=90,

    colorbar=False

)


plt.title(
    "Random Forest Confusion Matrix"
)

plt.tight_layout()


plt.savefig(

    "results/random_forest_confusion_matrix.png"

)

plt.close()


# ============================================================
# 23. ANN CONFUSION MATRIX
# ============================================================

ann_cm = confusion_matrix(

    y_test,

    ann_prediction

)


fig, ax = plt.subplots(

    figsize=(14, 12)

)


disp = ConfusionMatrixDisplay(

    confusion_matrix=ann_cm,

    display_labels=label_encoder.classes_

)


disp.plot(

    ax=ax,

    xticks_rotation=90,

    colorbar=False

)


plt.title(
    "ANN Confusion Matrix"
)

plt.tight_layout()


plt.savefig(

    "results/ann_confusion_matrix.png"

)

plt.close()


# ============================================================
# 24. MODEL ACCURACY COMPARISON
# ============================================================

model_names = [

    "Random Forest",

    "ANN"

]


model_accuracies = [

    rf_accuracy * 100,

    ann_accuracy * 100

]


plt.figure(

    figsize=(8, 6)

)


plt.bar(

    model_names,

    model_accuracies

)


plt.ylabel(
    "Accuracy (%)"
)

plt.xlabel(
    "Model"
)

plt.title(
    "Random Forest vs ANN Accuracy"
)

plt.ylim(
    0,
    100
)


for i, value in enumerate(

    model_accuracies

):

    plt.text(

        i,

        value + 1,

        f"{value:.2f}%",

        ha="center"

    )


plt.tight_layout()


plt.savefig(

    "results/model_accuracy_comparison.png"

)

plt.close()


# ============================================================
# 25. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

importance = (

    random_forest.feature_importances_

)


plt.figure(

    figsize=(9, 6)

)


plt.bar(

    features,

    importance

)


plt.xlabel(
    "Features"
)

plt.ylabel(
    "Importance"
)

plt.title(
    "Random Forest Feature Importance"
)

plt.xticks(
    rotation=45
)

plt.tight_layout()


plt.savefig(

    "results/feature_importance.png"

)

plt.close()


# ============================================================
# 26. SAVE MODEL REPORT
# ============================================================

with open(

    "results/model_report.txt",

    "w"

) as file:

    file.write(
        "CROP RECOMMENDATION SYSTEM\n"
    )

    file.write(
        "MODEL EVALUATION REPORT\n"
    )

    file.write(
        "====================================\n\n"
    )

    file.write(

        f"Dataset size: {len(data)}\n"

    )

    file.write(

        f"Training samples: {len(X_train)}\n"

    )

    file.write(

        f"Testing samples: {len(X_test)}\n\n"

    )

    file.write(

        f"Random Forest Accuracy: "
        f"{rf_accuracy * 100:.2f}%\n"

    )

    file.write(

        f"ANN Accuracy: "
        f"{ann_accuracy * 100:.2f}%\n\n"

    )

    file.write(
        "RANDOM FOREST CLASSIFICATION REPORT\n"
    )

    file.write(
        "====================================\n"
    )

    file.write(
        rf_report
    )

    file.write(
        "\n\n"
    )

    file.write(
        "ANN CLASSIFICATION REPORT\n"
    )

    file.write(
        "====================================\n"
    )

    file.write(
        ann_report
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 60)
print("             TRAINING COMPLETED")
print("=" * 60)

print(
    f"\nRandom Forest Accuracy: "
    f"{rf_accuracy * 100:.2f}%"
)

print(
    f"ANN Accuracy: "
    f"{ann_accuracy * 100:.2f}%"
)

print("\nModels saved:")

print(
    "models/crop_model.pkl"
)

print(
    "models/scaler.pkl"
)

print(
    "models/label_encoder.pkl"
)

print(
    "models/ann_model.keras"
)

print("\nResults saved:")

print(
    "results/ann_accuracy_graph.png"
)

print(
    "results/ann_loss_graph.png"
)

print(
    "results/random_forest_confusion_matrix.png"
)

print(
    "results/ann_confusion_matrix.png"
)

print(
    "results/model_accuracy_comparison.png"
)

print(
    "results/feature_importance.png"
)

print(
    "results/model_report.txt"
)

print("\nTraining finished successfully.")