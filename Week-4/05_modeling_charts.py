"""
Charts supporting the predictive modeling section:
 - Model comparison (RMSE/MAE by model)
 - Actual vs predicted scatter (tuned Random Forest)
 - Residual plot
 - Feature importance bar chart
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 150

results_df = pd.read_csv("model_comparison.csv")
pred_df = pd.read_csv("best_model_predictions.csv")
importance = pd.read_csv("feature_importance.csv", index_col=0).squeeze("columns")

# 8. Model comparison bar chart
plt.figure(figsize=(7, 4.5))
x = range(len(results_df))
width = 0.35
plt.bar([i - width/2 for i in x], results_df["test_RMSE"], width, label="RMSE", color="#2f6fa8")
plt.bar([i + width/2 for i in x], results_df["test_MAE"], width, label="MAE", color="#a8d0e6")
plt.xticks(list(x), results_df["model"])
plt.ylabel("Days")
plt.title("Model Comparison: Test RMSE and MAE")
plt.legend()
plt.tight_layout()
plt.savefig("chart8_model_comparison.png")
plt.close()

# 9. Actual vs predicted scatter (tuned Random Forest)
plt.figure(figsize=(6, 6))
plt.scatter(pred_df["actual"], pred_df["predicted"], alpha=0.5, color="#2f6fa8", s=20)
lims = [min(pred_df["actual"].min(), pred_df["predicted"].min()),
        max(pred_df["actual"].max(), pred_df["predicted"].max())]
plt.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
plt.xlabel("Actual Delivery Time (days)")
plt.ylabel("Predicted Delivery Time (days)")
plt.title("Actual vs. Predicted Delivery Time (Tuned Random Forest)")
plt.legend()
plt.tight_layout()
plt.savefig("chart9_actual_vs_predicted.png")
plt.close()

# 10. Residual plot
residuals = pred_df["actual"] - pred_df["predicted"]
plt.figure(figsize=(7, 4.5))
plt.scatter(pred_df["predicted"], residuals, alpha=0.5, color="#c0392b", s=20)
plt.axhline(0, color="black", linewidth=1)
plt.xlabel("Predicted Delivery Time (days)")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Residual Plot (Tuned Random Forest)")
plt.tight_layout()
plt.savefig("chart10_residuals.png")
plt.close()

# 11. Feature importance
importance_sorted = importance.sort_values(ascending=True).tail(8)
plt.figure(figsize=(7, 4.5))
plt.barh(importance_sorted.index, importance_sorted.values, color="#2f6fa8")
plt.xlabel("Importance")
plt.title("Feature Importance (Tuned Random Forest)")
plt.tight_layout()
plt.savefig("chart11_feature_importance.png")
plt.close()

print("Modeling charts generated.")
