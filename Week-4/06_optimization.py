"""
Step 4: Optimization Strategies

Using insights from the predictive model (distance and transport mode are
the dominant drivers of delivery time and cost), this step formulates a
warehouse-to-region shipment allocation problem as a linear program: assign
shipment volume from warehouses to regions to MINIMIZE total transportation
cost, subject to each warehouse's capacity and each region's demand being
met. This represents a concrete "route planning / resource allocation"
optimization built on top of the model's findings.
"""

import numpy as np
import pandas as pd
from scipy.optimize import linprog

df = pd.read_csv("logistics_dataset.csv", parse_dates=["date"])

warehouses = sorted(df["warehouse"].unique())
regions = sorted(df["region"].unique())

# --- Build a realistic warehouse-region cost matrix -----------------------
# Cost per kg-equivalent-unit is estimated from the historical average
# transportation cost observed for each (warehouse, region) pair. Where a
# pair has no historical shipments, the regional average is used instead.
pair_cost = df.groupby(["warehouse", "region"])["transportation_cost"].mean()
region_avg_cost = df.groupby("region")["transportation_cost"].mean()

cost_matrix = pd.DataFrame(index=warehouses, columns=regions, dtype=float)
for w in warehouses:
    for r in regions:
        if (w, r) in pair_cost.index:
            cost_matrix.loc[w, r] = pair_cost.loc[(w, r)]
        else:
            cost_matrix.loc[w, r] = region_avg_cost.loc[r]

print("Cost matrix (avg transportation cost per shipment, warehouse x region):")
print(cost_matrix.round(1))
cost_matrix.to_csv("optimization_cost_matrix.csv")

# --- Supply (warehouse capacity) and demand (region requirement) ----------
# Capacity: each warehouse can dispatch up to 1.3x its current historical
# shipment count next period (reflects available operational capacity).
# Demand: each region requires shipments equal to its current historical
# shipment count next period (forecast demand held flat for this exercise).
warehouse_supply = (df.groupby("warehouse")["shipment_id"].count() * 1.3).round().astype(int)
region_demand = df.groupby("region")["shipment_id"].count().astype(int)

# Balance supply and demand (LP equality constraints require total supply >= total demand)
if warehouse_supply.sum() < region_demand.sum():
    scale = region_demand.sum() / warehouse_supply.sum() * 1.05
    warehouse_supply = (warehouse_supply * scale).round().astype(int)

print("\nWarehouse supply (capacity):")
print(warehouse_supply)
print("\nRegion demand:")
print(region_demand)

# --- Formulate the LP -------------------------------------------------
# Decision variables: x[w, r] = number of shipments routed from warehouse w to region r
# Objective: minimize sum(cost[w,r] * x[w,r])
# Constraints: sum_r x[w,r] <= supply[w]  (warehouse capacity)
#              sum_w x[w,r] >= demand[r]  (region demand met)

n_w, n_r = len(warehouses), len(regions)
c = cost_matrix.loc[warehouses, regions].values.flatten()  # length n_w * n_r

# Inequality constraints: A_ub @ x <= b_ub
A_ub = []
b_ub = []
# Warehouse capacity constraints
for i in range(n_w):
    row = np.zeros(n_w * n_r)
    row[i * n_r:(i + 1) * n_r] = 1
    A_ub.append(row)
    b_ub.append(warehouse_supply[warehouses[i]])
# Region demand constraints (as >= demand  ->  -x <= -demand)
for j in range(n_r):
    row = np.zeros(n_w * n_r)
    for i in range(n_w):
        row[i * n_r + j] = -1
    A_ub.append(row)
    b_ub.append(-region_demand[regions[j]])

result = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=(0, None), method="highs")

print("\nOptimization status:", result.message)
x_opt = result.x.reshape(n_w, n_r)
alloc_df = pd.DataFrame(x_opt.round(0), index=warehouses, columns=regions)
alloc_df.to_csv("optimization_allocation.csv")
print("\nOptimal allocation (shipments, warehouse x region):")
print(alloc_df)

optimized_total_cost = result.fun
print(f"\nOptimized total transportation cost: {optimized_total_cost:,.2f}")

# --- Baseline for comparison: proportional allocation (current practice) --
# Approximates "business as usual" by routing each region's demand from
# warehouses in proportion to each warehouse's share of total historical
# shipments (i.e., no cost-aware routing).
warehouse_share = warehouse_supply / warehouse_supply.sum()
baseline_alloc = pd.DataFrame(
    {r: (warehouse_share * region_demand[r]) for r in regions},
    index=warehouses,
)
baseline_cost = float((baseline_alloc * cost_matrix.loc[warehouses, regions]).values.sum())

print(f"Baseline (proportional) total transportation cost: {baseline_cost:,.2f}")
savings = baseline_cost - optimized_total_cost
savings_pct = savings / baseline_cost * 100
print(f"Estimated savings from optimization: {savings:,.2f} ({savings_pct:.1f}%)")

summary = {
    "baseline_total_cost": round(baseline_cost, 2),
    "optimized_total_cost": round(optimized_total_cost, 2),
    "estimated_savings": round(savings, 2),
    "estimated_savings_pct": round(savings_pct, 2),
}
pd.Series(summary).to_csv("optimization_summary.csv")
print("\nSaved: optimization_cost_matrix.csv, optimization_allocation.csv, optimization_summary.csv")
