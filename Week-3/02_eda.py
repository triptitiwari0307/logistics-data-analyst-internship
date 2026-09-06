"""
Step 2: Exploratory Data Analysis (EDA)
Central tendencies, distributions, correlations for the logistics dataset.
"""

import pandas as pd
import numpy as np

df = pd.read_csv("logistics_dataset.csv", parse_dates=["date"])

numeric_cols = [
    "distance_km", "shipment_volume_kg", "actual_delivery_days",
    "delay_days", "transportation_cost", "fuel_surcharge_pct",
    "customer_satisfaction",
]

print("=" * 60)
print("CENTRAL TENDENCY & SPREAD (numeric variables)")
print("=" * 60)
summary = df[numeric_cols].describe().T
summary["median"] = df[numeric_cols].median()
summary["skew"] = df[numeric_cols].skew()
print(summary.round(2))
summary.round(2).to_csv("eda_summary_stats.csv")

print("\n" + "=" * 60)
print("DELAY RATE BY TRANSPORT MODE")
print("=" * 60)
delay_by_mode = df.groupby("transport_mode").agg(
    shipments=("shipment_id", "count"),
    avg_delay_days=("delay_days", "mean"),
    pct_delayed=("is_delayed", "mean"),
    avg_cost=("transportation_cost", "mean"),
).round(2)
delay_by_mode["pct_delayed"] = (delay_by_mode["pct_delayed"] * 100).round(1)
print(delay_by_mode)
delay_by_mode.to_csv("eda_delay_by_mode.csv")

print("\n" + "=" * 60)
print("COST & DELAY BY REGION")
print("=" * 60)
by_region = df.groupby("region").agg(
    shipments=("shipment_id", "count"),
    avg_cost=("transportation_cost", "mean"),
    avg_delivery_days=("actual_delivery_days", "mean"),
    pct_delayed=("is_delayed", "mean"),
    damage_rate=("damaged_or_lost", "mean"),
).round(2)
print(by_region)
by_region.to_csv("eda_by_region.csv")

print("\n" + "=" * 60)
print("CORRELATION MATRIX")
print("=" * 60)
corr = df[numeric_cols].corr().round(2)
print(corr)
corr.to_csv("eda_correlation_matrix.csv")

print("\n" + "=" * 60)
print("OVERALL KPIs")
print("=" * 60)
kpis = {
    "total_shipments": len(df),
    "pct_delayed": round(df["is_delayed"].mean() * 100, 1),
    "avg_delay_days_when_delayed": round(df.loc[df["is_delayed"], "delay_days"].mean(), 2),
    "avg_transportation_cost": round(df["transportation_cost"].mean(), 2),
    "total_transportation_cost": round(df["transportation_cost"].sum(), 2),
    "avg_customer_satisfaction": round(df["customer_satisfaction"].mean(), 2),
    "damage_loss_rate_pct": round(df["damaged_or_lost"].mean() * 100, 2),
}
for k, v in kpis.items():
    print(f"{k}: {v}")
pd.Series(kpis).to_csv("eda_kpis.csv")
