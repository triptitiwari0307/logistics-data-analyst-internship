"""
Step 3: Visualizations
Produces charts capturing trends, distributions, and relationships
among key logistics variables using matplotlib and seaborn.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 150

df = pd.read_csv("logistics_dataset.csv", parse_dates=["date"])
df["month"] = df["date"].dt.to_period("M").astype(str)

palette = "viridis"

# 1. Distribution of delivery time (histogram + KDE)
plt.figure(figsize=(7, 4.5))
sns.histplot(df["actual_delivery_days"], bins=30, kde=True, color="#2f6fa8")
plt.title("Distribution of Actual Delivery Time")
plt.xlabel("Delivery Time (days)")
plt.ylabel("Number of Shipments")
plt.tight_layout()
plt.savefig("chart1_delivery_time_distribution.png")
plt.close()

# 2. Boxplot of transportation cost by transport mode
plt.figure(figsize=(7, 4.5))
sns.boxplot(data=df, x="transport_mode", y="transportation_cost", palette=palette)
plt.title("Transportation Cost by Transport Mode")
plt.xlabel("Transport Mode")
plt.ylabel("Transportation Cost")
plt.tight_layout()
plt.savefig("chart2_cost_by_mode_boxplot.png")
plt.close()

# 3. Monthly shipment volume trend (line chart)
monthly = df.groupby("month").agg(
    shipments=("shipment_id", "count"),
    total_volume=("shipment_volume_kg", "sum"),
).reset_index()
plt.figure(figsize=(8, 4.5))
plt.plot(monthly["month"], monthly["shipments"], marker="o", color="#2f6fa8")
plt.title("Monthly Shipment Volume Trend (Number of Shipments)")
plt.xlabel("Month")
plt.ylabel("Number of Shipments")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("chart3_monthly_trend.png")
plt.close()

# 4. Correlation heatmap
numeric_cols = [
    "distance_km", "shipment_volume_kg", "actual_delivery_days",
    "delay_days", "transportation_cost", "fuel_surcharge_pct",
    "customer_satisfaction",
]
corr = df[numeric_cols].corr()
plt.figure(figsize=(7.5, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
plt.title("Correlation Matrix of Key Logistics Metrics")
plt.tight_layout()
plt.savefig("chart4_correlation_heatmap.png")
plt.close()

# 5. Scatter: distance vs delivery time, colored by mode
plt.figure(figsize=(7.5, 5))
sns.scatterplot(
    data=df, x="distance_km", y="actual_delivery_days",
    hue="transport_mode", alpha=0.6, palette=palette,
)
plt.title("Distance vs. Delivery Time by Transport Mode")
plt.xlabel("Distance (km)")
plt.ylabel("Delivery Time (days)")
plt.legend(title="Mode", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig("chart5_distance_vs_delivery.png")
plt.close()

# 6. Bar chart: % delayed shipments by region
region_delay = df.groupby("region")["is_delayed"].mean().mul(100).sort_values(ascending=False)
plt.figure(figsize=(7, 4.5))
sns.barplot(x=region_delay.index, y=region_delay.values, palette=palette)
plt.title("Percentage of Delayed Shipments by Region")
plt.xlabel("Region")
plt.ylabel("% Shipments Delayed")
plt.tight_layout()
plt.savefig("chart6_delay_pct_by_region.png")
plt.close()

# 7. Customer satisfaction vs delay days (bar of mean satisfaction per delay bucket)
df["delay_bucket"] = pd.cut(
    df["delay_days"], bins=[-0.01, 0, 1, 2, 4, 100],
    labels=["On time", "0-1d", "1-2d", "2-4d", "4d+"],
)
sat_by_delay = df.groupby("delay_bucket", observed=True)["customer_satisfaction"].mean()
plt.figure(figsize=(7, 4.5))
sns.barplot(x=sat_by_delay.index, y=sat_by_delay.values, palette="rocket")
plt.title("Average Customer Satisfaction by Delay Bucket")
plt.xlabel("Delay Bucket")
plt.ylabel("Avg. Customer Satisfaction (1-5)")
plt.tight_layout()
plt.savefig("chart7_satisfaction_by_delay.png")
plt.close()

print("All 7 charts generated successfully.")
