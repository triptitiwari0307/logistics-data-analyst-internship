import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 150

summary = pd.read_csv("optimization_summary.csv", index_col=0).squeeze("columns")

plt.figure(figsize=(6, 4.5))
bars = plt.bar(
    ["Baseline\n(proportional routing)", "Optimized\n(cost-minimizing LP)"],
    [summary["baseline_total_cost"], summary["optimized_total_cost"]],
    color=["#a8d0e6", "#2f6fa8"],
)
for b in bars:
    h = b.get_height()
    plt.text(b.get_x() + b.get_width() / 2, h + 5000, f"{h:,.0f}", ha="center", fontsize=9)
plt.ylabel("Total Transportation Cost")
plt.title("Baseline vs. Optimized Warehouse-to-Region Allocation Cost")
plt.tight_layout()
plt.savefig("chart12_optimization_savings.png")
plt.close()

print("Optimization chart generated.")
