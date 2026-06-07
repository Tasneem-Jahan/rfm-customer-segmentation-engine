# Import Path so we can work with file and folder paths.
from pathlib import Path

# Import pandas so we can create summary tables.
import pandas as pd


# Store the path to the customer segmentation file.
segments_path = Path("data/processed/rfm_segments.csv")

# Store the folder where business output tables will be saved.
output_tables_dir = Path("outputs/tables")

# Create the output tables folder if it does not already exist.
output_tables_dir.mkdir(parents=True, exist_ok=True)

# Check that the segmentation file exists.
if not segments_path.exists():
    raise FileNotFoundError(
        "data/processed/rfm_segments.csv was not found. Run scripts/segment_customers.py first.")

# Load the customer segmentation table.
rfm = pd.read_csv(segments_path)

# Count total customers.
total_customers = len(rfm)

# Calculate total revenue/spend.
total_revenue = rfm["Monetary"].sum()

# Create a summary table grouped by Segment.
segment_summary = rfm.groupby("Segment").agg(
    # Count how many customers are in each segment.
    Customer_Count=("CustomerID", "count"),

    # Calculate average recency for each segment.
    Avg_Recency=("Recency", "mean"),

    # Calculate average frequency for each segment.
    Avg_Frequency=("Frequency", "mean"),

    # Calculate average monetary value for each segment.
    Avg_Monetary=("Monetary", "mean"),

    # Calculate total monetary value for each segment.
    Total_Monetary=("Monetary", "sum"),
)

# Move Segment from the index back into a normal column.
segment_summary = segment_summary.reset_index()

# Calculate what percentage of customers are in each segment.
segment_summary["Customer_Share"] = segment_summary["Customer_Count"] / total_customers

# Calculate what percentage of revenue each segment represents.
segment_summary["Revenue_Share"] = segment_summary["Total_Monetary"] / total_revenue

# Round numeric columns to make the table easier to read.
segment_summary["Avg_Recency"] = segment_summary["Avg_Recency"].round(2)
segment_summary["Avg_Frequency"] = segment_summary["Avg_Frequency"].round(2)
segment_summary["Avg_Monetary"] = segment_summary["Avg_Monetary"].round(2)
segment_summary["Total_Monetary"] = segment_summary["Total_Monetary"].round(2)
segment_summary["Customer_Share"] = segment_summary["Customer_Share"].round(4)
segment_summary["Revenue_Share"] = segment_summary["Revenue_Share"].round(4)

# Sort the summary table by total monetary value from highest to lowest.
segment_summary = segment_summary.sort_values(by="Total_Monetary", ascending=False)

# Create a marketing strategy table manually.
marketing_strategy = pd.DataFrame([
    {
        "Segment": "Champions",
        "Priority": "High",
        "Recommended_Action": "Reward with VIP offers, loyalty perks, and early access.",
        "Reason": "These are the best customers: recent, frequent, and high spending.",
    },
    {
        "Segment": "Cannot Lose Them",
        "Priority": "Very High",
        "Recommended_Action": "Send urgent retention offers or personalized win-back campaigns.",
        "Reason": "These customers are valuable but have not purchased recently.",
    },
    {
        "Segment": "At Risk",
        "Priority": "High",
        "Recommended_Action": "Send reactivation discounts and reminders.",
        "Reason": "These customers used to buy but are becoming inactive.",
    },
    {
        "Segment": "Loyal Customers",
        "Priority": "Medium High",
        "Recommended_Action": "Offer loyalty rewards, bundles, and repeat-purchase incentives.",
        "Reason": "These customers buy repeatedly and can be grown further.",
    },
    {
        "Segment": "Big Spenders",
        "Priority": "Medium High",
        "Recommended_Action": "Recommend premium products and personalized offers.",
        "Reason": "These customers spend a lot but do not buy very often.",
    },
    {
        "Segment": "New Customers",
        "Priority": "Medium",
        "Recommended_Action": "Send welcome campaigns and second-purchase offers.",
        "Reason": "These customers are recent but have limited purchase history.",
    },
    {
        "Segment": "Need Attention",
        "Priority": "Medium",
        "Recommended_Action": "Use targeted promotions to increase repeat purchases.",
        "Reason": "These customers show some potential but are not loyal yet.",
    },
    {
        "Segment": "Hibernating",
        "Priority": "Low",
        "Recommended_Action": "Use low-cost win-back campaigns only.",
        "Reason": "These customers are inactive and low value.",
    },
])

# Select customers that should receive priority marketing attention.
high_priority_customers = rfm[
    rfm["Segment"].isin(["Cannot Lose Them", "At Risk", "Champions"])
].copy()

# Sort high priority customers by segment and monetary value.
high_priority_customers = high_priority_customers.sort_values(
    by=["Segment", "Monetary"],
    ascending=[True, False],
)

# Save the segment summary table.
segment_summary.to_csv(output_tables_dir / "segment_summary.csv", index=False)

# Save the marketing strategy table.
marketing_strategy.to_csv(output_tables_dir / "marketing_strategy.csv", index=False)

# Save the high priority customer table.
high_priority_customers.to_csv(output_tables_dir / "high_priority_customers.csv", index=False)

# Print a completion message.
print("MARKETING TABLES CREATED")
print("=" * 40)

# Print the segment summary.
print("\nSegment summary:")
print(segment_summary)

# Print how many high priority customers were selected.
print(f"\nHigh priority customers: {len(high_priority_customers)}")

# Print where the files were saved.
print("\nFiles saved:")
print(output_tables_dir / "segment_summary.csv")
print(output_tables_dir / "marketing_strategy.csv")
print(output_tables_dir / "high_priority_customers.csv")