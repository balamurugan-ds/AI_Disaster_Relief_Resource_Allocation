# ============================================================
# AI DISASTER RELIEF RESOURCE ALLOCATION
# MACHINE LEARNING MODEL TRAINING
# ============================================================

# 1. IMPORT REQUIRED MODULES

import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# 2. LOAD DATASET
# ============================================================

DATASET_PATH = "dataset/disaster_relief_dataset.csv"

df = pd.read_csv(DATASET_PATH)

print("\nDataset Loaded Successfully!")
print("Dataset Shape:", df.shape)

print("\nDataset Columns:")
print(df.columns.tolist())


# ============================================================
# 3. CLEAN COLUMN NAMES
# ============================================================

df.columns = df.columns.str.strip()


# ============================================================
# 4. DEFINE INPUT FEATURES
# These names match your dataset headers exactly.
# ============================================================

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


# ============================================================
# 5. DEFINE TARGET COLUMNS
# RESCUE TEAM TARGET ADDED
# ============================================================

targets = [
    "food_required",
    "water_required",
    "medical_supplies_required",
    "rescue_teams_required",
    "shelter_required"
]


# ============================================================
# 6. VALIDATE DATASET COLUMNS
# ============================================================

required_columns = features + targets

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in dataset: {missing_columns}"
    )


# ============================================================
# 7. PREPARE INPUT DATA
# ============================================================

X = df[features].copy()

categorical_features = [
    "disaster_type",
    "location"
]

numeric_features = [
    column for column in features
    if column not in categorical_features
]


# ============================================================
# 8. CREATE MODELS FOLDER
# ============================================================

os.makedirs("models", exist_ok=True)


# ============================================================
# 9. TRAIN AND SAVE EACH MODEL
# ============================================================

for target in targets:

    print("\n" + "=" * 55)
    print(f"Training Model: {target}")
    print("=" * 55)

    # Select target column
    y = pd.to_numeric(
        df[target],
        errors="coerce"
    )

    # Remove rows with missing input or target values
    valid_mask = X.notna().all(axis=1) & y.notna()

    X_target = X.loc[valid_mask].copy()
    y_target = y.loc[valid_mask].copy()

    if len(X_target) < 5:
        print(
            f"Skipping {target}: Not enough valid rows."
        )
        continue

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_target,
        y_target,
        test_size=0.2,
        random_state=42
    )

    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            )
        ],
        remainder="passthrough"
    )

    # Random Forest Pipeline
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=200,
                    max_depth=15,
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]
    )

    # Train model
    model.fit(
        X_train,
        y_train
    )

    # Predictions
    predictions = model.predict(
        X_test
    )

    # Evaluation
    mae = mean_absolute_error(
        y_test,
        predictions
    )

    mse = mean_squared_error(
        y_test,
        predictions
    )

    rmse = mse ** 0.5

    r2 = r2_score(
        y_test,
        predictions
    )

    print("Training Rows:", len(X_train))
    print("Testing Rows:", len(X_test))
    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2 Score:", round(r2, 4))

    # Save model
    filename = f"models/{target}_model.pkl"

    joblib.dump(
        model,
        filename
    )

    print("Model Saved:", filename)


# ============================================================
# 10. TRAINING COMPLETED
# ============================================================

print("\n" + "=" * 55)
print("ALL ML MODEL TRAINING COMPLETED!")
print("=" * 55)

print("\nCheck the models folder for saved model files.")