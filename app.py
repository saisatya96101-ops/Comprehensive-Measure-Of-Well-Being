import os
import pickle

import numpy as np
import pandas as pd
from flask import Flask, render_template, request


def score_to_tier(score: float, tier_buckets):
    for threshold, label in tier_buckets:
        if score >= threshold:
            return label
    return "Low"


app = Flask(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "model.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Trained model pickle not found at '{MODEL_PATH}'. "
        f"Run: python train_model.py --data <your_dataset.csv> --model-out {MODEL_PATH}"
    )

with open(MODEL_PATH, "rb") as f:
    artifact = pickle.load(f)

pipeline = artifact["pipeline"]
feature_columns = artifact["feature_columns"]
tier_buckets = artifact.get(
    "tier_buckets",
    [
        (0.800, "Very High"),
        (0.700, "High"),
        (0.550, "Medium"),
        (-float("inf"), "Low"),
    ],
)


@app.get("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Read values from form
    # IMPORTANT: the HTML field names must match these keys
    payload = {
        "Life Expectancy": request.form.get("life_expectancy", ""),
        "Mean Years of Schooling": request.form.get("mean_years_schooling", ""),
        "Expected Years of Schooling": request.form.get("expected_years_schooling", ""),
        "GNI per Capita": request.form.get("gni_per_capita", ""),
    }

    # Convert to float
    values = {k: float(v) for k, v in payload.items()}

    # Build a single-row DataFrame with the exact columns the pipeline expects
    # feature_columns in the artifact are resolved from training dataset
    row = {}
    # Map logical input keys to the resolved feature columns by order
    ordered_input = [
        values["Life Expectancy"],
        values["Mean Years of Schooling"],
        values["Expected Years of Schooling"],
        values["GNI per Capita"],
    ]

    for col, val in zip(feature_columns, ordered_input):
        row[col] = val

    X_new = pd.DataFrame([row], columns=feature_columns)

    pred = float(pipeline.predict(X_new)[0])
    tier = score_to_tier(pred, tier_buckets)

    return render_template(
        "result.html",
        hdi_score=pred,
        tier=tier,
    )


if __name__ == "__main__":
    app.run(debug=True)

