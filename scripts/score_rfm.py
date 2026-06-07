# Import Path so we can work with file and folder paths.
from pathlib import Path

# Import pandas so we can score the RFM table.
import pandas as pd


# Store the path to the RFM table created in Step 8.
rfm_table_path = Path("data/processed/rfm_table.csv")

# Store the path where the scored RFM table will be saved.
rfm_scores_path = Path("data/processed/rfm_scores.csv")

# Check if the RFM table exists.
if not rfm_table_path.exists():
    raise FileNotFoundError(
        "data/processed/rfm_table.csv was not found. Run scripts/build_rfm.py first.")

# Load the RFM table.
rfm = pd.read_csv(rfm_table_path)

# Score the RFM table by assigning a score of 1 to 5 for each R, F, and M column.
# For Recency, a score of 5 is assigned to the lowest value (most recent), and a score of 1 is assigned to the highest value (least recent).
rfm["R_Score"] = pd.qcut(rfm["Recency"], q=5, labels=[5, 4, 3, 2, 1])

# For Frequency, higher is better so labels go from 1 to 5
rfm["F_Score"] = pd.qcut(
    rfm["Frequency"].rank(method="first"),
    q=5,
    labels=[1, 2, 3, 4, 5]
)

# For Monetary, higher is better so labels go from 1 to 5
rfm["M_Score"] = pd.qcut(
    rfm["Monetary"],
    q=5,
    labels=[1, 2, 3, 4, 5]
)

# Convert score columns from category to integer.
rfm["R_Score"] = rfm["R_Score"].astype(int)
rfm["F_Score"] = rfm["F_Score"].astype(int)
rfm["M_Score"] = rfm["M_Score"].astype(int)

# Create one combined RFM Score as tect, like 555 or 314
rfm["RFM_Score"] = rfm["R_Score"].astype(
    str) + rfm["F_Score"].astype(str) + rfm["M_Score"].astype(str)

# Create a simple total score by adding R + F + M
rfm["RFM_Total"] = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]

# Save the scored RFM table
rfm.to_csv(rfm_scores_path, index=False)

# Print a summary
print("RFM Scores Created")
print("=" * 40)

# Print the number of customers scored
print(f"Customers scored: {len(rfm)}")

# Print the first 10 rows
print("\nTop 10 scored customers")
print(rfm.head(10))

# print score distribution
print("\nRFM total score distribution")
print(rfm["RFM_Total"].value_counts().sort_index())

# Print where the scored table was saved
print(f"\nScored RFM table saved to: {rfm_scores_path}")
