"""
Week 4 Task - Predictive Modeling and Optimization in Logistics Systems
Step 1-3: Problem Definition, Model Selection/Implementation, Evaluation/Validation

Problem: forecast actual_delivery_days (a continuous target) for a shipment,
using shipment-level features known at dispatch time (distance, volume,
transport mode, region, promised delivery time, fuel surcharge).
This is a regression problem.
"""

import pandas as pd
import numpy as np
import json

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("logistics_dataset.csv", parse_dates=["date"])

# ---------------------------------------------------------------------
# Step 1: Feature preparation
# ---------------------------------------------------------------------
target = "actual_delivery_days"

feature_cols_numeric = [
    "distance_km", "shipment_volume_kg",
    "transportation_cost", "fuel_surcharge_pct",
]
# Note: promised_delivery_days is intentionally excluded. In this simulated
# dataset it was generated as a noisy function of the actual delivery time,
# so including it would leak the target into the features. In a real system,
# the SLA/promised time is set independently at booking and could be
# included safely -- but it is left out here to keep the model honest.
feature_cols_categorical = ["transport_mode", "region"]

X = pd.get_dummies(df[feature_cols_numeric + feature_cols_categorical],
                    columns=feature_cols_categorical, drop_first=True)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale numeric features for the linear model (tree-based models don't need it,
# but scaling the shared matrix does not hurt them either)
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

print("Train/test shapes:", X_train.shape, X_test.shape)
print("Features used:", list(X.columns))

# ---------------------------------------------------------------------
# Step 2: Model selection and training
# Three models of increasing complexity are compared:
#  - Linear Regression: interpretable baseline, assumes linear relationships
#  - Decision Tree: captures non-linear splits, easy to interpret, prone to overfit
#  - Random Forest: ensemble of trees, usually the best accuracy/robustness trade-off
# ---------------------------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42),
}

results = []
predictions = {}

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    use_X_train = X_train_scaled if name == "Linear Regression" else X_train
    use_X_test = X_test_scaled if name == "Linear Regression" else X_test

    model.fit(use_X_train, y_train)
    preds = model.predict(use_X_test)
    predictions[name] = preds

    rmse = mean_squared_error(y_test, preds) ** 0.5
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    cv_rmse = -cross_val_score(
        model, use_X_train, y_train, cv=kf, scoring="neg_root_mean_squared_error"
    )

    results.append({
        "model": name,
        "test_RMSE": round(rmse, 3),
        "test_MAE": round(mae, 3),
        "test_R2": round(r2, 3),
        "cv_RMSE_mean": round(cv_rmse.mean(), 3),
        "cv_RMSE_std": round(cv_rmse.std(), 3),
    })

results_df = pd.DataFrame(results)
print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)
print(results_df)
results_df.to_csv("model_comparison.csv", index=False)

# ---------------------------------------------------------------------
# Step 3: Hyperparameter tuning (Random Forest, via GridSearchCV)
# ---------------------------------------------------------------------
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [4, 6, 8, 12],
}
grid = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1,
)
grid.fit(X_train, y_train)

best_rf = grid.best_estimator_
best_preds = best_rf.predict(X_test)
tuned_rmse = mean_squared_error(y_test, best_preds) ** 0.5
tuned_mae = mean_absolute_error(y_test, best_preds)
tuned_r2 = r2_score(y_test, best_preds)

print("\n" + "=" * 60)
print("HYPERPARAMETER TUNING (Random Forest, GridSearchCV, 5-fold CV)")
print("=" * 60)
print("Best params:", grid.best_params_)
print(f"Tuned RF -> RMSE: {tuned_rmse:.3f}, MAE: {tuned_mae:.3f}, R2: {tuned_r2:.3f}")

tuning_summary = {
    "best_params": grid.best_params_,
    "tuned_RMSE": round(tuned_rmse, 3),
    "tuned_MAE": round(tuned_mae, 3),
    "tuned_R2": round(tuned_r2, 3),
}
with open("tuning_summary.json", "w") as f:
    json.dump(tuning_summary, f, indent=2)

# ---------------------------------------------------------------------
# Feature importance from the tuned Random Forest
# ---------------------------------------------------------------------
importance = pd.Series(best_rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature importances (tuned Random Forest):")
print(importance.round(3))
importance.to_csv("feature_importance.csv", header=["importance"])

# Save predictions of the best model (Random Forest, tuned) for residual/scatter plots
pred_df = pd.DataFrame({"actual": y_test.values, "predicted": best_preds})
pred_df.to_csv("best_model_predictions.csv", index=False)

# Save all model predictions for comparison plotting
all_preds_df = pd.DataFrame({"actual": y_test.values, **predictions})
all_preds_df.to_csv("all_model_predictions.csv", index=False)

print("\nDone. Artifacts saved: model_comparison.csv, tuning_summary.json, "
      "feature_importance.csv, best_model_predictions.csv, all_model_predictions.csv")
