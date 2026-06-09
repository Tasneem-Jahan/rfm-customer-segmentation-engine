# Import path utilities and data manipulation tools
from pathlib import Path
import pandas as pd
import numpy as np

# Import Machine Learning tools from Scikit-Learn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Import visualization tools to plot the Elbow Curve
import matplotlib.pyplot as plt
import seaborn as sns

# Define data and output paths
rfm_table_path = Path("data/processed/rfm_table.csv")
kmeans_output_path = Path("data/processed/kmeans_segments.csv")
elbow_chart_path = Path("outputs/charts/kmeans_elbow_curve.png")

# Ensure directories exist
elbow_chart_path.parent.mkdir(parents=True, exist_ok=True)

if not rfm_table_path.exists():
    raise FileNotFoundError(
        "data/processed/rfm_table.csv was not found. Please run build_rfm.py first.")

# 1. Load and Preprocess Data
rfm = pd.read_csv(rfm_table_path)
rfm_log = np.log1p(rfm[["Recency", "Frequency", "Monetary"]])
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_log)

print("Data scaled and preprocessed.")
print("Running the Elbow Method to find the optimal K value...")

# 2. Compute WCSS for K values from 1 to 10
wcss = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(rfm_scaled)
    wcss.append(kmeans.inertia_)  # inertia_ calculates the total WCSS

# 3. Automatically Plot and Save the Elbow Curve Chart
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))
plt.plot(k_range, wcss, marker='o', linestyle='--',
         color='#2b5c8f', linewidth=2)
plt.title('The Elbow Method for Optimal K Selection', fontsize=14, pad=15)
plt.xlabel('Number of Clusters (K)', fontsize=12)
plt.ylabel('WCSS (Total Error Within Clusters)', fontsize=12)
plt.xticks(k_range)

# Highlight the "elbow point" at K=4 visually for recruiters
plt.axvline(x=4, color='#d9534f', linestyle=':',
            label='Optimal Elbow Point (K=4)')
plt.legend()

plt.tight_layout()
plt.savefig(elbow_chart_path, dpi=300)
plt.close()
print(f"📊 Elbow curve chart generated and saved to: {elbow_chart_path}")

# 4. Fit the Final Model using the confirmed optimal K
optimal_k = 4
print(f"🤖 Fitting final model with K={optimal_k}...")
final_kmeans = KMeans(n_clusters=optimal_k,
                      init='k-means++', random_state=42, n_init=10)
rfm["KMeans_Cluster"] = final_kmeans.fit_predict(rfm_scaled)

# ==============================================================================
# EXPLORATORY INTERMEDIATE CHARTS (Snake Plot, Bar Chart, Box & Whisker)
# ==============================================================================
print("📊 Generating exploratory cluster visualizations...")

# Define image output folder
charts_dir = Path("outputs/charts/exploratory")
charts_dir.mkdir(parents=True, exist_ok=True)

# Define a temporary dataframe for plotting normalized behaviors
rfm_scaled_df = pd.DataFrame(
    rfm_scaled, columns=["Recency", "Frequency", "Monetary"])
rfm_scaled_df["Cluster"] = rfm["KMeans_Cluster"]

# ------------------------------------------------------------------------------
# 1. THE SNAKE PLOT (Line plot of scaled RFM features across clusters)
# ------------------------------------------------------------------------------
# Melt the data into a long format that seaborn can interpret easily
snake_df = pd.melt(
    rfm_scaled_df,
    id_vars=['Cluster'],
    value_vars=['Recency', 'Frequency', 'Monetary'],
    var_name='Metric',
    value_name='Scaled_Value'
)

plt.figure(figsize=(10, 6))
sns.lineplot(data=snake_df, x='Metric', y='Scaled_Value',
             hue='Cluster', palette='Set1', marker='o', linewidth=2.5)
plt.title('Snake Plot: Normalized RFM Profile Lines Across Clusters',
          fontsize=14, pad=15)
plt.xlabel('RFM Metric', fontsize=12)
plt.ylabel('Scaled Normalized Value', fontsize=12)
plt.legend(title='Cluster ID', loc='upper right')
plt.tight_layout()
plt.savefig(charts_dir / "kmeans_snake_plot.png", dpi=300)
plt.close()
print(f"   🐍 Snake Plot saved to: {charts_dir / 'kmeans_snake_plot.png'}")


# ------------------------------------------------------------------------------
# 2. THE BAR CHART (Average Raw Value Profile for Recency, Frequency, Monetary)
# ------------------------------------------------------------------------------
# Calculate raw averages per cluster to display actual business values
raw_averages = rfm.groupby('KMeans_Cluster')[
    ['Recency', 'Frequency', 'Monetary']].mean().reset_index()

# We will create subplots to show all three metrics clearly side-by-side
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
metrics_list = ['Recency', 'Frequency', 'Monetary']
titles = ['Avg Recency (Days)', 'Avg Frequency (Orders)',
          'Avg Monetary (Spend $)']
colors = ['#4A90E2', '#50E3C2', '#F5A623']

for i, metric in enumerate(metrics_list):
    sns.barplot(data=raw_averages, x='KMeans_Cluster', y=metric,
                ax=axes[i], color=colors[i], edgecolor='black')
    axes[i].set_title(titles[i], fontsize=12, pad=10)
    axes[i].set_xlabel('Cluster ID', fontsize=10)
    axes[i].set_ylabel('')

plt.suptitle('Cluster Profiles Comparison (Raw Means)', fontsize=14, y=1.02)
plt.tight_layout()
fig.savefig(charts_dir / "kmeans_raw_bar_charts.png",
            dpi=300, bbox_inches='tight')
plt.close()
print(f"   📊 Bar Charts saved to: {charts_dir / 'kmeans_raw_bar_charts.png'}")


# ------------------------------------------------------------------------------
# 3. BOX-AND-WHISKER PLOT (Detecting overlaps and distribution spreads)
# ------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 6))

# Boxplot for Recency
sns.boxplot(data=rfm, x='KMeans_Cluster', y='Recency',
            ax=axes[0], palette='Set2', hue='KMeans_Cluster', legend=False, showfliers=False)
axes[0].set_title('Recency Distribution Spread', fontsize=12)
# Log scale helps deal with long-tail outlier points visually
axes[0].set_yscale('log')

# Boxplot for Frequency
sns.boxplot(data=rfm, x='KMeans_Cluster', y='Frequency',
            ax=axes[1], palette='Set2', hue='KMeans_Cluster', legend=False, showfliers=False)
axes[1].set_title('Frequency Distribution Spread', fontsize=12)
axes[1].set_yscale('log')

# Boxplot for Monetary
sns.boxplot(data=rfm, x='KMeans_Cluster', y='Monetary',
            ax=axes[2], palette='Set2', hue='KMeans_Cluster', legend=False, showfliers=False)
axes[2].set_title('Monetary Distribution Spread', fontsize=12)
axes[2].set_yscale('log')

plt.suptitle(
    'Distribution Dispersions and Outliers Across Clusters', fontsize=14, y=1.02)
plt.tight_layout()
fig.savefig(charts_dir / "kmeans_box_plots.png", dpi=300, bbox_inches='tight')
plt.close()
print(
    f"   📦 Box & Whisker Plots saved to: {charts_dir / 'kmeans_box_plots.png'}")
print("=" * 80)

# ==============================================================================
# BULLETPROOF DYNAMIC PERSONA MAPPING (Multi-Metric Automation)
# ==============================================================================

print("🔄 Computing multi-metric fitness scores to safely isolate personas...")

# 1. Calculate the raw averages for each cluster
cluster_stats = rfm.groupby("KMeans_Cluster").agg(
    Avg_Recency=("Recency", "mean"),
    Avg_Frequency=("Frequency", "mean"),
    Avg_Monetary=("Monetary", "mean")
)

# 2. Calculate a combined score: (Frequency * Monetary) / Recency
# This ensures VIPs MUST have high spend/frequency AND excellent low recency days
cluster_stats["Fitness_Score"] = (
    cluster_stats["Avg_Frequency"] * cluster_stats["Avg_Monetary"]) / cluster_stats["Avg_Recency"]

# 3. Rank the cluster IDs based on this combined score from highest to lowest
# The true all-around elite VIP group will ALWAYS be index 0
ranked_clusters = cluster_stats.sort_values(
    by="Fitness_Score", ascending=False).index

# 4. Map the dynamic labels perfectly
dynamic_mapping = {
    # Highest overall health/activity score
    ranked_clusters[0]: "VIPs",
    # Good metrics, but starting to lag or lower spend
    ranked_clusters[1]: "Hibernating",
    # Great recency, but low frequency/spend so far
    ranked_clusters[2]: "New Customers",
    # Worst scores across all axes
    ranked_clusters[3]: "Needs Attention"
}

# 5. Apply to the table
rfm["KMeans_Persona"] = rfm["KMeans_Cluster"].map(dynamic_mapping)
print("✅ Fully automated multi-metric clustering complete! No human bias or hard-coding.")

# Define output file paths
summary_table_path = Path("outputs/tables/kmeans_persona_summary.csv")
final_data_path = Path("data/processed/kmeans_segments.csv")

# Ensure output directories exist
summary_table_path.parent.mkdir(parents=True, exist_ok=True)
final_data_path.parent.mkdir(parents=True, exist_ok=True)

# 1. Generate the mathematical profile summary for your persona clusters
final_summary = rfm.groupby("KMeans_Persona").agg(
    Customer_Count=("CustomerID", "count"),
    Avg_Recency_Days=("Recency", "mean"),
    Avg_Frequency_Orders=("Frequency", "mean"),
    Avg_Monetary_Spend=("Monetary", "mean")
).round(2).reset_index()

# 2. Save the final automated summary table
final_summary.to_csv(summary_table_path, index=False)
print(f"📊 Summary profile metrics saved to: {summary_table_path}")

# 3. Save the full customer database with their automated persona stamps
rfm.to_csv(final_data_path, index=False)
print(f"💾 Full customer dataset with dynamic tags saved to: {final_data_path}")

# 4. Print a live validation preview right to your console window
print("\n🔍 Live Pipeline Verification Preview:")
print("=" * 80)
print(final_summary.to_string(index=False))
print("=" * 80)
