# E-Commerce Customer Retention & Behavioral Segmentation Engine

An enterprise-grade data science and analytics pipeline that transforms raw, messy e-commerce transactional data into actionable marketing intelligence. This project implements and contrasts two complementary customer profiling methodologies, a **Rule-Based Heuristic Approach** using statistical RFM quantiles and an **Advanced Machine Learning Approach** using log-transformed $K$-Means clustering optimized via the Elbow Method and evaluated through a custom multi-metric **Fitness Score**.

---

## 💡 What It Is & How It Is Useful

In production e-commerce environments, blasting generic marketing campaigns to an entire customer base results in wasted ad spend and high unsubscribe rates. This system solves that problem by segmenting users based on their core transactional behaviors: **Recency** (how recently they bought), **Frequency** (how often they buy), and **Monetary** (how much they spend).

### Real-World Business Value:

- **Maximized Marketing ROI:** Marketers can extract targeted audience manifests directly from the pipeline and drop them into customer relationship management (CRM) tools (e.g., Klaviyo, HubSpot, or Mailchimp) for highly specific automated email flows.
- **Proactive Churn Mitigation:** The engine automatically flags premium historical accounts shifting into higher recency windows (long periods without buying), triggering immediate win-back discount flows before the customer relationship expires.
- **Lifecycle Optimization:** It isolates high-potential new acquisitions who display strong initial intent but low order frequencies, driving targeted onboarding and cross-sell campaigns.

---

## 🛠️ Technology Stack Used

This system is built entirely using pythonic data engineering and machine learning principles:

- **Core Language:** Python 3
- **Data Engineering & ETL:** `Pandas`, `NumPy`
- **Machine Learning Engine:** `Scikit-Learn` (`StandardScaler`, `KMeans`)
- **Data Ingestion Layer:** `Kaggle API`
- **Data Visualization Dashboards:** `Matplotlib`, `Seaborn`
- **Storage & Relational Queries:** Dedicated `SQL` extraction layer

---

## 🚀 Execution Workflow & Commands

To execute the entire end-to-end data pipeline from a fresh repository setup down to final machine learning campaign manifests, run the scripts sequentially in your terminal:

```bash
# 1. Clone the repository and install environment dependencies
pip install -r requirements.txt

# 2. Ingest raw transaction logs from Kaggle API
python scripts/download_data.py

# 3. Run ETL layer to remove missing entries, filter cancellations, and calculate Total Price
python scripts/clean_data.py

# 4. Engineer baseline Recency, Frequency, and Monetary (RFM) metrics per customer
python scripts/build_rfm.py

# 5. Map statistical quantiles (1 to 5) across engineered customer dimensions
python scripts/score_rfm.py

# 6. Route customers into 8 rigid rule-based operational segments
python scripts/segments_customers.py

# 7. Generate executive summary charts for the manual groupings
python scripts/create_charts.py

# 8. Train the Unsupervised K-Means algorithm and export cluster profiles
python scripts/cluster_kmeans.py

# 9. Segment and export targeted activation tables for downstream marketing CRM engines
python scripts/create_marketing_tables.py
```

---

## 🔬 Core Methodologies & Analytical Insights

### 1. Heuristic Rule-Based Profiling (Manual Quantiles)

Utilizes rigid, explicit human-written conditional boundaries mapped directly to $R$, $F$, and $M$ quantile integers ($1$ to $5$). It groups customers into 8 discrete operational categories (e.g., _Champions_, _At Risk_, _Loyal Customers_, _Hibernating_) to provide baseline operational rules.

### 2. Unsupervised Machine Learning ($K$-Means Clustering)

To capture naturally occurring latent customer behaviors without human bias, the ML pipeline applies an automated multi-step geometric optimization:

- **Logarithmic Transformation ($np.log1p$):** Standardizes heavily skewed transaction volumes and extreme "whale" spend distributions.
- **Standard Scaling:** Normalizes features to a mean of 0 and variance of 1 to ensure spatial distance metrics are not biased by dollar magnitude.
- **Elbow Selection ($K=4$):** Evaluates Within-Cluster Sum of Squares (WCSS) to confirm the mathematically optimal cluster structure.
- **Dynamic Persona Mapping (The Fitness Score):** Maps arbitrary cluster numbers dynamically without hardcoding using a customized business valuation equation:

$$\text{Fitness Score} = \frac{\text{Avg Frequency} \times \text{Avg Monetary}}{\text{Avg Recency}}$$

This ranks and isolates the 4 core AI-discovered personas: **VIPs**, **New Customers**, **Needs Attention**, and **Hibernating**.

### 3. Methodology Cross-Reference Analysis

Implement a cross-tabulation matrix to audit our human-defined business rules against the multidimensional geometry seen by the AI. This helps us spot exactly where strict, time-based rules oversimplify customer behavior compared to the machine learning model.

## 🎯 Analytical Insights & Project Conclusion

By cross-tabulating our heuristic segments against the AI-discovered personas via the **Methodology Transition Matrix**, several critical operational realities are revealed:

- **Core Model Validation:** A definitive majority of manual `Champions` ($680$ customers) map directly to the machine learning `VIPs` cluster. This tight spatial alignment proves that our multi-metric **Fitness Score** equation successfully automates the identity mapping of elite accounts without human bias.
- **Exposing Heuristic Weaknesses (The "At Risk" Resolution):** Manual, quantile-based rules categorize accounts as `At Risk` or `Cannot Lose Them` purely because their recency window drops past a strict day-count threshold. However, the $K$-Means algorithm uncovers a major hidden division: $323$ of those customers actually belong in the `Needs Attention` bucket, while $315$ are classified as `Hibernating`. The machine learning model prevents the marketing team from wasting premium retention ad spend on dead accounts (`Hibernating`), redirecting attention toward historical high-value spenders who are still saving (`Needs Attention`).
- **Uncovering Hidden Intent:** Among manual `New Customers` and `Need Attention` pools, $K$-Means isolates a significant subset of buyers ($294$ and $333$ customers respectively) who demonstrate strong baseline interactions and low recency timelines. This groups them into the `New Customers` ML persona, flagging them as the highest-priority cross-sell targets for accelerated lifetime value cultivation.

### Final Summary

This engine demonstrates that while rule-based heuristics provide excellent foundational baselines for daily database reporting, **unsupervised machine learning is required to capture multi-dimensional customer behaviors.** Transitioning from manual thresholds to dynamic geometric clustering eliminates organizational bias, optimizes promotional budget allocations, and constructs a robust, automated customer-centric ecosystem.
