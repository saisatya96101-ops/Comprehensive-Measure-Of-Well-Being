import os

from flask import Flask, render_template, request

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Model inlined as a plain formula.
#
# The original scikit-learn pipeline was:
#     SimpleImputer(mean) -> StandardScaler -> LinearRegression
# All of that reduces to simple arithmetic, so we hardcode the learned
# numbers here. This removes the need for model.pkl, numpy, pandas and
# scikit-learn entirely — the app is now a single dependency (Flask).
#
# Feature order (must match the form + the lists below):
#   0: Life Expectancy
#   1: Mean Years of Schooling
#   2: Expected Years of Schooling
#   3: GNI per Capita
# ---------------------------------------------------------------------------
FEATURE_MEANS = [76.69487179487179, 10.123076923076923,
                 14.864102564102563, 32152.51282051282]   # imputer + scaler mean
FEATURE_SCALES = [7.0205175029483735, 2.990280111950884,
                  2.6065486739135726, 24465.406718280025]  # scaler std
COEFFICIENTS = [0.046433401101780125, 0.050595011575687035,
                0.03729229620842312, 0.02139065278379009]  # regression weights
INTERCEPT = 0.8102820512820512

TIER_BUCKETS = [
    (0.800, "Very High"),
    (0.700, "High"),
    (0.550, "Medium"),
    (float("-inf"), "Low"),
]


def score_to_tier(score):
    for threshold, label in TIER_BUCKETS:
        if score >= threshold:
            return label
    return "Low"


def predict_hdi(values):
    """values: list of 4 floats in the feature order above."""
    score = INTERCEPT
    for value, mean, scale, coef in zip(values, FEATURE_MEANS,
                                        FEATURE_SCALES, COEFFICIENTS):
        standardized = (value - mean) / scale
        score += coef * standardized
    return score


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/predict")
def predict():
    try:
        values = [
            float(request.form.get("life_expectancy", "")),
            float(request.form.get("mean_years_schooling", "")),
            float(request.form.get("expected_years_schooling", "")),
            float(request.form.get("gni_per_capita", "")),
        ]
    except (TypeError, ValueError):
        # Bad or missing input — send them back to the form instead of 500-ing.
        return render_template("index.html"), 400

    hdi_score = predict_hdi(values)
    tier = score_to_tier(hdi_score)

    return render_template("result.html", hdi_score=hdi_score, tier=tier)


@app.get("/favicon.ico")
def favicon():
    # No favicon file; return "no content" so it stops showing up as an error.
    return ("", 204)


if __name__ == "__main__":
    app.run(debug=True)
