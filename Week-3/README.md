# Advanced Data Analysis and Visualization in Logistics — Week 3 Internship Task

## Description

This project was completed as part of a Week 3 internship task focused on advanced data analysis and visualization techniques applied to the logistics domain. Since a real-world dataset was not available, a realistic hypothetical logistics dataset of 1,200 shipments was simulated in Python using NumPy and pandas, modeling key operational variables such as delivery time, shipment volume, transportation cost, transport mode, region, delay days, and customer satisfaction across a full year of operations.

The workflow begins with structured data simulation designed to reflect genuine logistics dynamics — for instance, Air freight is faster but more expensive, Road transport is the most frequently used mode but carries higher delay risk, and delivery delays measurably reduce customer satisfaction. From there, exploratory data analysis (EDA) was conducted to compute central tendencies, distributions, and correlations across all numeric metrics, followed by segmented performance analysis by transport mode and region.

Seven visualizations were built using matplotlib and seaborn — including a delivery-time distribution histogram, a cost boxplot by mode, a monthly shipment trend line chart, a correlation heatmap, a distance-vs-delivery scatter plot, and delay/satisfaction bar charts — each chosen deliberately for the story it best communicates. The findings were compiled into a fully formatted Word/PDF report containing methodology, code excerpts, embedded charts, interpretations, and actionable recommendations for reducing delays and controlling costs.

**Tools used:** Python, pandas, NumPy, matplotlib, seaborn

## Repository Contents

| File | Description |
|---|---|
| `01_simulate_data.py` | Simulates the hypothetical logistics dataset |
| `02_eda.py` | Performs exploratory data analysis (stats, correlations, groupings) |
| `03_visualizations.py` | Generates all 7 charts using matplotlib/seaborn |
| `logistics_dataset.csv` | The simulated dataset (1,200 shipments) |
| `chart1_*.png` – `chart7_*.png` | Generated visualization images |
| `Week3_Logistics_Data_Analysis_Report.docx` | Full Word report |
| `Week3_Logistics_Data_Analysis_Report.pdf` | Full PDF report |

## Key Insight

Delivery delay — not distance or cost — is the strongest driver of customer dissatisfaction (correlation ≈ -0.82), highlighting the operational importance of on-time performance over pure cost optimization.
