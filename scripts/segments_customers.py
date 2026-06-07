# Import Path so we can work with file paths.
from pathlib import Path

# Import pandas so we can load and update the RFM scores table.
import pandas as pd


# Store the path to the scored RFM table.
rfm_scores_path = Path("data/processed/rfm_scores.csv")

# Store the path where the segmented customer table will be saved.
segments_output_path = Path("data/processed/rfm_segments.csv")

# Check if the RFM scores file exists.
if not rfm_scores_path.exists():
    raise FileNotFoundError(
        "data/processed/rfm_scores.csv was not found. Run scripts/score_rfm.py first.")

# Load the RFM scores table.
rfm = pd.read_csv(rfm_scores_path)

# Define a function to assign customer segments based on RFM scores.


def assign_segment(row):
    r = row["R_Score"]
    f = row["F_Score"]
    m = row["M_Score"]

    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"

    elif r >= 4 and f <= 2:
        return "New Customers"

    elif r >= 3 and f >= 4:
        return "Loyal Customers"

    elif r <= 2 and f >= 4 and m >= 4:
        return "Cannot Lose Them"

    elif r <= 2 and f >= 3:
        return "At Risk"

    elif m >= 4 and f <= 3:
        return "Big Spenders"

    elif r <= 2 and f <= 2:
        return "Hibernating"

    else:
        return "Need Attention"


def assign_segment(row):
    r = row["R_Score"]
    f = row["F_Score"]
    m = row["M_Score"]

    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"

    elif r >= 4 and f <= 2:
        return "New Customers"

    elif r >= 3 and f >= 4:
        return "Loyal Customers"

    elif r <= 2 and f >= 4 and m >= 4:
        return "Cannot Lose Them"

    elif r <= 2 and f >= 3:
        return "At Risk"

    elif m >= 4 and f <= 3:
        return "Big Spenders"

    elif r <= 2 and f <= 2:
        return "Hibernating"

    else:
        return "Need Attention"


# Apply the segment function to every customer row.
rfm["Segment"] = rfm.apply(assign_segment, axis=1)

# Save the final segmented customer table.
rfm.to_csv(segments_output_path, index=False)

# Print a completion title.
print("CUSTOMER SEGMENTS CREATED")
print("=" * 40)

# Print how many customers were segmented.
print(f"Customers segmented: {len(rfm)}")

# Print how many customers are in each segment.
print("\nSegment counts:")
print(rfm["Segment"].value_counts())

# Print average RFM values by segment.
print("\nSegment summary:")
print(rfm.groupby("Segment")[
      ["Recency", "Frequency", "Monetary"]].mean().round(2))

# Print where the final segmented table was saved.
print(f"\nSegmented customer table saved to: {segments_output_path}")
