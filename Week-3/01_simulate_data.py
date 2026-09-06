"""
Week 3 Task - Advanced Data Analysis and Visualization in Logistics
Step 1: Data Simulation

Defines a hypothetical logistics dataset representing shipments handled by a
regional logistics company over one year. Each row represents one shipment.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 1200  # number of shipments

regions = ["North", "South", "East", "West", "Central"]
transport_modes = ["Road", "Rail", "Air", "Sea"]
warehouses = ["WH-Delhi", "WH-Mumbai", "WH-Chennai", "WH-Kolkata", "WH-Bhopal"]

# Base date range across one year
start_date = pd.Timestamp("2025-01-01")
dates = start_date + pd.to_timedelta(np.random.randint(0, 365, N), unit="D")

region = np.random.choice(regions, N, p=[0.25, 0.2, 0.2, 0.2, 0.15])
mode = np.random.choice(transport_modes, N, p=[0.55, 0.2, 0.1, 0.15])
warehouse = np.random.choice(warehouses, N)

# Shipment volume (in kg) - varies by mode
base_volume = np.random.gamma(shape=2.0, scale=150, size=N)
mode_volume_factor = pd.Series(mode).map({"Road": 1.0, "Rail": 1.8, "Air": 0.4, "Sea": 2.5}).values
shipment_volume_kg = np.round(base_volume * mode_volume_factor, 1)

# Distance in km
distance_km = np.round(np.random.uniform(50, 3000, N), 1)

# Delivery time (days) - depends on mode + distance + some noise
mode_speed_factor = pd.Series(mode).map({"Road": 0.012, "Rail": 0.008, "Air": 0.003, "Sea": 0.006}).values
base_delivery_time = distance_km * mode_speed_factor
noise = np.random.normal(0, 0.8, N)
delivery_time_days = np.clip(base_delivery_time + noise + np.random.uniform(0.5, 2, N), 0.5, None)
delivery_time_days = np.round(delivery_time_days, 2)

# Promised delivery time (SLA) - slightly less than typical actual, to create delays
promised_time_days = np.round(delivery_time_days * np.random.uniform(0.75, 1.05, N), 2)
delay_days = np.round(delivery_time_days - promised_time_days, 2)
is_delayed = delay_days > 0

# Transportation cost - depends on distance, volume, mode
mode_cost_factor = pd.Series(mode).map({"Road": 8, "Rail": 5, "Air": 25, "Sea": 3}).values
transportation_cost = np.round(
    (distance_km * 0.5 + shipment_volume_kg * 0.8) * (mode_cost_factor / 8)
    + np.random.normal(0, 50, N),
    2,
)
transportation_cost = np.clip(transportation_cost, 20, None)

# Fuel surcharge (%) applied
fuel_surcharge_pct = np.round(np.random.uniform(2, 12, N), 1)

# Customer satisfaction score (1-5), inversely related to delay
base_score = 4.6 - (delay_days.clip(min=0) * 0.35) + np.random.normal(0, 0.3, N)
customer_satisfaction = np.clip(np.round(base_score), 1, 5).astype(int)

# Damaged/lost flag - rare event, more likely for longer distance + Road
damage_prob = 0.02 + (distance_km / 3000) * 0.05 + (pd.Series(mode) == "Road").astype(int) * 0.01
damaged_or_lost = np.random.binomial(1, np.clip(damage_prob, 0, 0.3))

df = pd.DataFrame({
    "shipment_id": [f"SHP-{i:05d}" for i in range(1, N + 1)],
    "date": dates,
    "region": region,
    "warehouse": warehouse,
    "transport_mode": mode,
    "distance_km": distance_km,
    "shipment_volume_kg": shipment_volume_kg,
    "promised_delivery_days": promised_time_days,
    "actual_delivery_days": delivery_time_days,
    "delay_days": delay_days,
    "is_delayed": is_delayed,
    "transportation_cost": transportation_cost,
    "fuel_surcharge_pct": fuel_surcharge_pct,
    "customer_satisfaction": customer_satisfaction,
    "damaged_or_lost": damaged_or_lost,
})

df = df.sort_values("date").reset_index(drop=True)
df.to_csv("logistics_dataset.csv", index=False)

print("Dataset simulated:", df.shape)
print(df.head())
print("\nColumn dtypes:\n", df.dtypes)
