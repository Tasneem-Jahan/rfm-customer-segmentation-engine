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

# 5. Map Clusters to Personas
persona_mapping = {
    0: "Champions / VIPs",          # Lowest Recency, Highest Frequency & Monetary
    1: "Hibernating / Lost",        # Highest Recency, Lowest Frequency & Monetary
    # Low Recency (Recent), but low Frequency/Monetary yet
    2: "New / Promising Customers",
    3: "Loyal / Need Attention"     # Moderate stats across the board
}
rfm["KMeans_Persona"] = rfm["KMeans_Cluster"].map(persona_mapping)

# Save the final file
rfm.to_csv(kmeans_output_path, index=False)
print(f"🚀 K-Means clustering complete! Saved to: {kmeans_output_path}")

# ... (Your scaling and K-Means training math goes here) ...

# 1. The computer calculates the clusters right here in memory
rfm["KMeans_Cluster"] = final_kmeans.fit_predict(rfm_scaled)
rfm["KMeans_Persona"] = rfm["KMeans_Cluster"].map(persona_mapping)

# 2. GENERATE THE GRAPH NOW (Before saving the file to your hard drive!)

plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=rfm,
    x="Recency",
    y="Monetary",
    hue="KMeans_Persona",  # ◄── The computer uses the in-memory column to color the dots
    palette="viridis"
)
plt.title("Customer Clusters")
plt.savefig("outputs/charts/customer_clusters.png", dpi=300)
plt.close()

print("📊 Cluster Graph successfully created and saved to outputs/charts/!")

# 3. NOW FORM AND SAVE THE FINAL TABLE
rfm.to_csv(kmeans_output_path, index=False)
print("💾 Final CSV Table formed and saved!")
