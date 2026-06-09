from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load the data columns from your processed directory
manual_path = Path("data/processed/rfm_segments.csv")
kmeans_path = Path("data/processed/kmeans_segments.csv")

if not manual_path.exists() or not kmeans_path.exists():
    raise FileNotFoundError(
        "Make sure both segments_customers.py and cluster_kmeans.py have run first!")

df_manual = pd.read_csv(manual_path)[["CustomerID", "Segment"]]
df_kmeans = pd.read_csv(kmeans_path)[["CustomerID", "KMeans_Persona"]]

# 2. Merge on CustomerID to align the transitions
merged_df = pd.merge(df_manual, df_kmeans, on="CustomerID")

# 3. Create the cross-tabulation transition matrix
# Rows = Manual Rules, Columns = ML Personas
transition_matrix = pd.crosstab(
    index=merged_df["Segment"],
    columns=merged_df["KMeans_Persona"],
    margins=False
)

# # 4. Sort the rows and columns logically so the table flows from high-value to low-value
# manual_order = ["Champions", "Loyal Customers", "New Customers",
#                 "Need Attention", "Big Spenders", "At Risk", "Cannot Lose Them", "Hibernating"]
# kmeans_order = ["Champions / VIPs", "Loyal / Need Attention",
#                 "New / Promising Customers", "Hibernating / Lost"]

# # Reindex safely based on what exists in your current dataset
# manual_order = [r for r in manual_order if r in transition_matrix.index]
# kmeans_order = [c for c in kmeans_order if c in transition_matrix.columns]
# transition_matrix = transition_matrix.reindex(
#     index=manual_order, columns=kmeans_order)

# 5. Plot the Transition Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(
    transition_matrix,
    annot=True,          # Displays the exact customer transition counts
    fmt="d",             # Normal integer formatting
    cmap="Blues",        # Professional blue gradient mapping density
    linewidths=0.8,
    edgecolor="lightgray",
    cbar_kws={'label': 'Customer Density Count'}
)

plt.title("Methodology Transition Matrix: Heuristic Rules vs. K-Means Personas",
          fontsize=14, pad=20, weight='bold')
plt.xlabel("AI-Discovered Persona (K-Means Clustering)",
           fontsize=12, labelpad=10)
plt.ylabel("Operational Segment (Manual Heuristic Rules)",
           fontsize=12, labelpad=10)
plt.xticks(rotation=30, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
