
## Week 4: Predictive Modeling and Optimization in Logistics Systems

### Description

This project extends the Week 3 exploratory analysis into predictive modeling and optimization for logistics operations. Building on the same simulated dataset of 1,200 shipments, the goal was to forecast actual delivery time using only information available at the time a shipment is booked, and then translate the model's insights into a concrete cost-optimization strategy.

Three regression models of increasing complexity — Linear Regression, Decision Tree, and Random Forest — were trained and compared using RMSE, MAE, R-squared, and 5-fold cross-validation. A deliberate design choice was made to exclude the promised-delivery-time field once it was identified as a source of data leakage, ensuring the model reflects a realistic, honest prediction scenario rather than an inflated one. The Random Forest model was then tuned via GridSearchCV, achieving strong accuracy (R² ≈ 0.99) with distance and transport mode emerging as the dominant predictors.

Those feature-importance findings directly motivated an optimization stage: a linear program (built with SciPy's `linprog`) that reallocates warehouse-to-region shipment routing to minimize total transportation cost under capacity and demand constraints. Compared against a proportional "business as usual" baseline, the optimized routing achieved roughly 7.5% in cost savings. The full workflow — model training, evaluation, tuning, and optimization — is documented in a formatted Word/PDF report with code excerpts, evaluation charts, and actionable recommendations.

**Tools used:** Python, pandas, NumPy, scikit-learn, SciPy, matplotlib, seaborn

### Repository Contents (Week 4)

| File | Description |
|---|---|
| `04_predictive_modeling.py` | Trains, evaluates, and tunes the regression models |
| `05_modeling_charts.py` | Generates model comparison, prediction, residual, and importance charts |
| `06_optimization.py` | Formulates and solves the warehouse-to-region routing LP |
| `07_optimization_chart.py` | Generates the baseline-vs-optimized cost chart |
| `model_comparison.csv`, `feature_importance.csv` | Model evaluation results |
| `optimization_summary.csv`, `optimization_allocation.csv` | Optimization results |
| `chart8_*.png` – `chart12_*.png` | Generated modeling/optimization charts |
| `Week4_Logistics_Predictive_Modeling_Report.docx` | Full report |

### Key Insight (Week 4)

Distance and transport mode account for over 90% of the model's predictive power for delivery time, and routing shipments through a cost-minimizing linear program instead of proportional allocation cuts total transportation cost by roughly 7.5% with no change in delivery volume.
