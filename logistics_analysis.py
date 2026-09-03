# Logistics Data Analyst Internship Project
# Author: Tripti Tiwari
# Internship Period: 30 July 2026 – 27 August 2026

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans

# ---------------------------------------------------------------------------------------------------------------------------------#

# 1. Load and Clean Data
# -----------------------------
df = pd.read_csv("DataCoSupplyChainDataset.csv")
df = df.drop_duplicates()

df["Days for shipping (real)"] = pd.to_numeric(df["Days for shipping (real)"], errors="coerce")
df["Days for shipment (scheduled)"] = pd.to_numeric(df["Days for shipment (scheduled)"], errors="coerce")

df["delay_days"] = df["Days for shipping (real)"] - df["Days for shipment (scheduled)"]
df["late_delivery"] = (df["delay_days"] > 0).astype(int)

# ---------------------------------------------------------------------------------------------------------------------------------#

# 2. KPI Calculation
# -----------------------------
kpi = {
    "on_time_delivery_rate": (1 - df["late_delivery"].mean()) * 100,
    "average_delay_days": df["delay_days"].mean()
}
print("KPI Results:", kpi)

df.groupby("Shipping Mode")["late_delivery"].mean().sort_values().plot(
    kind="bar", title="Late Delivery Rate by Shipping Mode"
)
plt.ylabel("Late Delivery Rate")
plt.show()

# ---------------------------------------------------------------------------------------------------------------------------------#

# 3. Predictive Modeling – Late Delivery Risk
# -----------------------------
features = ["Days for shipment (scheduled)", "Days for shipping (real)", "Sales per customer"]
model_df = df[features + ["late_delivery"]].dropna()

X = model_df[features]
y = model_df["late_delivery"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, predictions))

#----------------------------------------------------------------------------------------------------------------------------------#

# 4. Clustering – Customer Segmentation
# -----------------------------
cluster_features = ["Sales per customer", "delay_days"]
cluster_df = df[cluster_features].dropna()

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(cluster_df)

cluster_df["Cluster"] = kmeans.labels_
print("Cluster Centers:\n", kmeans.cluster_centers_)
print("Cluster Assignments:\n", cluster_df.head())
