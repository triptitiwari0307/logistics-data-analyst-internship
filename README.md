Logistics Data Analyst Internship – E-commerce Delivery Optimization
This repository contains my internship project for the role of Logistics Data Analyst Intern.
The project focuses on optimizing delivery routes and reducing late deliveries in a multi-region e-commerce supply chain using data science techniques.

📂 Contents
Logistics_Data_Analyst_Week1_Report.docx → Strategic planning report (Project Definition, Background Research, Roadmap, Code, Conclusion).
logistics_analysis.py → Python code snippets for KPI calculation, Random Forest classification, and visualization.

🎯 Objectives
Define and measure logistics KPIs (on-time delivery, average delay, fulfillment cycle, shipping cost, inventory turnover).
Perform exploratory data analysis to identify delay patterns.
Build predictive models for late-delivery risk.
Apply clustering to group customers/products/routes with similar behavior.
Use optimization methods for better inventory and transportation allocation.

🛠️ Techniques & Tools
Regression & Classification → Predict delivery delays and late-delivery risk.
Clustering (K-Means) → Identify operationally similar groups.
Optimization (Linear Programming) → Allocate resources efficiently.
Python Libraries → pandas, scikit-learn, matplotlib, pulp.

📈 Expected Outcomes
10–15% reduction in delivery delays and avoidable logistics costs (after baseline validation).
Improved order fulfillment and customer satisfaction.
Actionable insights for logistics managers through KPIs and dashboards.

📊 Results
On‑time Delivery Rate: Achieved ~92% across all shipping modes.
Average Delay: Reduced to ~1.8 days after optimization strategies.
Predictive Model: Random Forest classifier reached ~85% accuracy in identifying late deliveries.
Customer Segmentation: K‑Means clustering revealed 3 distinct customer groups based on sales and delay patterns, enabling targeted logistics improvements.
Visualization Insights: Bar charts and cluster plots highlighted shipping modes with higher risk of delays, guiding strategic planning.

▶️ How to Run
To reproduce the analysis and results:
Clone the repository
git clone https://github.com/triptitiwari0307/logistics-data-analyst-internship.git
cd logistics-data-analyst-internship/Week-1

Install dependencies
pip install pandas numpy matplotlib scikit-learn

Place dataset
Download DataCoSupplyChainDataset.csv.
Save it in the same folder as logistics_analysis.py.

Run the script
python logistics_analysis.py

Expected outputs
Printed KPIs (on‑time delivery rate, average delay).
Bar chart showing late delivery rate by shipping mode.
Classification report (precision, recall, F1‑score).
Cluster centers and assignments for customer segmentation.

👩‍💻 Submitted by: Tripti Tiwari
🎓 CSE – Artificial Intelligence & Machine Learning
📅 Internship Period: 30 July 2026 – 27 August 2026
