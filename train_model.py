import argparse
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Update these if your dataset uses different names
TARGET_COLUMN = "HDI Score"
FEATURE_COLUMNS = [
    "Life Expectancy",
    "Mean Years of Schooling",
    "Expected Years of Schooling",
    "GNI per Capita",
]


def infer_columns(df: pd.DataFrame):
    """Try to infer common variations of expected column names."""
    col_map = {}

    def find_col(candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    # target
    target = find_col([TARGET_COLUMN, "HDI", "Human Development Index", "hdi_score", "hdi"])
    if target is None:
        raise ValueError(
            f"Could not find target column for HDI. Tried: {TARGET_COLUMN} and common variations."
        )
    col_map["TARGET"] = target

    # features
    feature_variants = {
        "Life Expectancy": [
            "Life Expectancy",
            "life_expectancy",
            "Life expectancy",
        ],
        "Mean Years of Schooling": [
            "Mean Years of Schooling",
            "mean_years_of_schooling",
            "Mean years of schooling",
            "MYS",
        ],
        "Expected Years of Schooling": [
            "Expected Years of Schooling",
            "expected_years_of_schooling",
            "Expected years of schooling",
            "EYS",
        ],
        "GNI per Capita": [
            "GNI per Capita",
            "gni_per_capita",
            "GNI per capita",
            "GNI",
            "Gross National Income",
            "Gross National Income (GNI) per capita",
        ],
    }

    resolved_features = []
    for logical_name, candidates in feature_variants.items():
        c = find_col(candidates)
        if c is None:
            raise ValueError(
                f"Could not find feature column for '{logical_name}'. Tried: {candidates}"
            )
        resolved_features.append(c)

    return resolved_features, col_map["TARGET"]


def build_pipeline(numeric_features):
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
        ],
        remainder="drop",
    )

    model = LinearRegression()

    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
    return pipe


def score_to_tier(score: float) -> str:
    # Typical HDI tier buckets (can be adjusted)
    if score >= 0.800:
        return "Very High"
    if score >= 0.700:
        return "High"
    if score >= 0.550:
        return "Medium"
    return "Low"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to HDI dataset CSV")
    parser.add_argument(
        "--model-out", default="model.pkl", help="Output pickle model path"
    )
    args = parser.parse_args()

    data_path = args.data
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)

    # Resolve columns even if dataset uses slightly different names
    resolved_features, resolved_target = infer_columns(df)

    X = df[resolved_features]
    y = df[resolved_target]

    # Basic cleaning: ensure numeric
    X = X.apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")

    # Drop rows with missing target
    mask = ~y.isna()
    X = X.loc[mask]
    y = y.loc[mask]

    # Train/test split
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe = build_pipeline(numeric_features=resolved_features)
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print("=== Evaluation ===")
    print(f"MSE: {mse:.6f}")
    print(f"R^2: {r2:.6f}")

    # quick sanity tier distribution for predictions
    sample = preds[:10]
    print("Sample predicted tiers:")
    for s in sample:
        print(f"  {s:.4f} -> {score_to_tier(float(s))}")

    artifact = {
        "pipeline": pipe,
        "feature_columns": resolved_features,
        "target_column": resolved_target,
        "tier_buckets": [
            (0.800, "Very High"),
            (0.700, "High"),
            (0.550, "Medium"),
            (-float("inf"), "Low"),
        ],
    }

    with open(args.model_out, "wb") as f:
        pickle.dump(artifact, f)

    print(f"Saved model to: {args.model_out}")


if __name__ == "__main__":
    main()

