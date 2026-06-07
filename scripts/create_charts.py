# Import Path so we can work with file and folder paths.
from pathlib import Path

# Import pandas so we can load the segment data.
import pandas as pd

# Import matplotlib so we can create charts.
import matplotlib.pyplot as plt

# Import seaborn so the charts look cleaner.
import seaborn as sns


# Store the path to the segmented customer data.
segments_path = Path("data/processed/rfm_segments.csv")

# Store the path to the segment summary table.
summary_path = Path("outputs/tables/segment_summary.csv")

# Store the folder where charts will be saved.
charts_dir = Path("outputs/charts")

# Create the charts folder if it does not already exist.
charts_dir.mkdir(parents=True, exist_ok=True)

# Check that the segmented customer file exists.
if not segments_path.exists():
    raise FileNotFoundError(
        "data/processed/rfm_segments.csv was not found. Run scripts/segment_customers.py first.")

# Check that the segment summary file exists.
if not summary_path.exists():
    raise FileNotFoundError(
        "outputs/tables/segment_summary.csv was not found. Run scripts/create_marketing_tables.py first.")

# Load the customer segment data.
rfm = pd.read_csv(segments_path)

# Load the segment summary data.
segment_summary = pd.read_csv(summary_path)

# Set a clean chart style.
sns.set_theme(style="whitegrid")

# Set the chart size style.
plt.rcParams["figure.figsize"] = (12, 7)

# -----------------------------
# Chart 1: Customer count by segment
# -----------------------------

# Sort segments by customer count.
count_data = segment_summary.sort_values(by="Customer_Count", ascending=False)

# Create a new chart figure.
plt.figure()

# Create a bar chart.
sns.barplot(
    data=count_data,
    x="Customer_Count",
    y="Segment",
    hue="Segment",
    palette="viridis",
    legend=False,
)

# Add a chart title.
plt.title("Customer Count by Segment")

# Add x-axis label.
plt.xlabel("Number of Customers")

# Add y-axis label.
plt.ylabel("Segment")

# Make the chart layout fit nicely.
plt.tight_layout()

# Save the chart as a PNG file.
plt.savefig(charts_dir / "segment_customer_count.png", dpi=300)

# Close the chart so it does not overlap with the next chart.
plt.close()

# -----------------------------
# Chart 2: Revenue share by segment
# -----------------------------

# Sort segments by revenue share.
revenue_data = segment_summary.sort_values(by="Revenue_Share", ascending=False)

# Create a new chart figure.
plt.figure()

# Create a bar chart.
sns.barplot(
    data=revenue_data,
    x="Revenue_Share",
    y="Segment",
    hue="Segment",
    palette="magma",
    legend=False,
)

# Add a chart title.
plt.title("Revenue Share by Segment")

# Add x-axis label.
plt.xlabel("Revenue Share")

# Add y-axis label.
plt.ylabel("Segment")

# Format the x-axis as percentages.
plt.gca().xaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")

# Make the chart layout fit nicely.
plt.tight_layout()

# Save the chart as a PNG file.
plt.savefig(charts_dir / "segment_revenue_share.png", dpi=300)

# Close the chart.
plt.close()

# -----------------------------
# Chart 3: Recency vs Frequency scatter plot
# -----------------------------

# Create a new chart figure.
plt.figure()

# Create a scatter plot.
sns.scatterplot(
    data=rfm,
    x="Recency",
    y="Frequency",
    hue="Segment",
    size="Monetary",
    sizes=(20, 300),
    alpha=0.7,
)

# Add a chart title.
plt.title("Recency vs Frequency by Segment")

# Add x-axis label.
plt.xlabel("Recency: Days Since Last Purchase")

# Add y-axis label.
plt.ylabel("Frequency: Number of Orders")

# Make the legend easier to read.
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

# Make the chart layout fit nicely.
plt.tight_layout()

# Save the chart as a PNG file.
plt.savefig(charts_dir / "recency_frequency_scatter.png", dpi=300)

# Close the chart.
plt.close()

# -----------------------------
# Chart 4: Average monetary value by segment
# -----------------------------

# Sort segments by average monetary value.
monetary_data = segment_summary.sort_values(by="Avg_Monetary", ascending=False)

# Create a new chart figure.
plt.figure()

# Create a bar chart.
sns.barplot(
    data=monetary_data,
    x="Avg_Monetary",
    y="Segment",
    hue="Segment",
    palette="crest",
    legend=False,
)

# Add a chart title.
plt.title("Average Monetary Value by Segment")

# Add x-axis label.
plt.xlabel("Average Customer Spend")

# Add y-axis label.
plt.ylabel("Segment")

# Make the chart layout fit nicely.
plt.tight_layout()

# Save the chart as a PNG file.
plt.savefig(charts_dir / "monetary_by_segment.png", dpi=300)

# Close the chart.
plt.close()


# Print a completion message.
print("CHARTS CREATED")
print("=" * 40)

# Print where the charts were saved.
print(f"Charts saved to: {charts_dir}")

# Print chart file names.
print("Created files:")
print("segment_customer_count.png")
print("segment_revenue_share.png")
print("recency_frequency_scatter.png")
print("monetary_by_segment.png")
