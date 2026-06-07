# Import Path so we can work with file and folder paths.
from pathlib import Path

# Import pandas so we can calculate the RFM table.
import pandas as pd

# Store the path to the cleaned transaction data
clean_data_path = Path("data/interim/clean_transactions.csv")

# Store the folder where final processed data will be saved
processed_data_dir = Path("data/processed")

# Create the processsed data folder if doesnt exist
processed_data_dir.mkdir(parents=True, exist_ok=True)

# Store the path to the RFM output file
rfm_output_path = processed_data_dir / "rfm_table.csv"

# Check if the cleaned data file exists
if not clean_data_path.exists():
    raise FileNotFoundError(
        f"Cleaned data file not found at {clean_data_path}")

# Load the cleaned transaction data
transactions = pd.read_csv(clean_data_path)

# Convert InvoiceDate to datetime
transactions["InvoiceDate"] = pd.to_datetime(transactions["InvoiceDate"])

# Find the latest invoice date in the dataset
latest_invoice_date = transactions["InvoiceDate"].max()

# Create the analysis date as one day after the latest invoice date
analysis_date = latest_invoice_date + pd.Timedelta(days=1)

# Group the data by CustomerID and calculate RFM metrics
rfm = transactions.groupby("CustomerID").agg(
    # Recency: Calculate the number of days since the last purchase
    Recency=("InvoiceDate", lambda dates: (analysis_date - dates.max()).days),

    # Frequency: Count the number of unique invoices for each customer
    Frequency=("InvoiceNo", "nunique"),

    # Monetary: Calculate the total revenue for each customer
    Monetary=("TotalPrice", "sum")
)

# Move CustomerID from index to a column
rfm = rfm.reset_index()

# Round the Monetary value to 2 decimal places
rfm["Monetary"] = rfm["Monetary"].round(2)

# Sort customers by Monetary value in descending order
rfm = rfm.sort_values(by="Monetary", ascending=False)

# Save the RFM table to a CSV file
rfm.to_csv(rfm_output_path, index=False)

# Print a summary
print(
    f"RFM table created with {len(rfm)} customers and saved to {rfm_output_path}")
print("=" * 40)

# Print the analysis date
print(f"Analysis Date: {analysis_date}")

# Print the number of customers in the RFM table
print(f"Customers in RFM table: {len(rfm)}")

# Print the first 10 rows of the RFM table
print("\nTop 20 customers by Monetary Value: ")
print(rfm.head(10))

# Print where the rfm table was saved
print(f"\nRFM table saved to: {rfm_output_path}")
