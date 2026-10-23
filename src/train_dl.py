import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# =========================
# LOAD DATA
# =========================

df = pd.read_csv(
    "dataset/disaster_relief_dataset.csv"
)


features = [
    "disaster_type",
    "location",
    "population_affected",
    "disaster_severity",
    "injuried_people",
    "displaced_people",
    "duration_days",
    "temperature",
    "rainfall",
    "infrastructure_damage"
]


targets = [
    "food_required",
    "water_required",
    "medical_supplies_required",
    "rescue_teams_required",
    "shelter_required"
]


X = df[features]
y = df[targets]


# =========================
# PREPROCESSING
# =========================

categorical_features = [
    "disaster_type",
    "location"
]

numeric_features = [
    "population_affected",
    "disaster_severity",
    "injuried_people",
    "displaced_people",
    "duration_days",
    "temperature",
    "rainfall",
    "infrastructure_damage"
]


preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        ),
        (
            "numeric",
            StandardScaler(),
            numeric_features
        )
    ]
)


X_processed = preprocessor.fit_transform(X)


joblib.dump(
    preprocessor,
    "models/dl_preprocessor.pkl"
)


# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X_processed,
    y,
    test_size=0.2,
    random_state=42
)


# =========================
# ANN MODEL
# =========================

model = Sequential([

    Dense(
        128,
        activation="relu",
        input_shape=(X_train.shape[1],)
    ),

    Dropout(0.2),

    Dense(
        64,
        activation="relu"
    ),

    Dropout(0.2),

    Dense(
        32,
        activation="relu"
    ),

    Dense(
        5,
        activation="linear"
    )
])


model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)


# =========================
# EARLY STOPPING
# =========================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)


# =========================
# TRAIN
# =========================

history = model.fit(

    X_train,
    y_train,

    validation_split=0.2,

    epochs=100,

    batch_size=32,

    callbacks=[early_stopping],

    verbose=1
)


# =========================
# EVALUATION
# =========================

predictions = model.predict(
    X_test
)


mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = mean_squared_error(
    y_test,
    predictions
) ** 0.5

r2 = r2_score(
    y_test,
    predictions
)


print("\n==============================")
print("DEEP LEARNING RESULTS")
print("==============================")

print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2  :", round(r2, 4))


# =========================
# SAVE MODEL
# =========================

model.save(
    "models/disaster_relief_ann.keras"
)

print(
    "\nANN model saved successfully!"
)