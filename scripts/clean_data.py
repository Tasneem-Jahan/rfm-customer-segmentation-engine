# Import Path to work with file and folder paths
from pathlib import Path

# Import pandas to load, clean, and save the data
import pandas as pd


# Store the path to the original raw dataset
raw_data_path = Path("data/raw/data.csv")

# Store the folder path to save the cleaned dataset
interim_data_dir = Path("data/interim")

# Create the interim data directory if it doesn't exist
interim_data_dir.mkdir(parents=True, exist_ok=True)

# Store the path to save the cleaned dataset
clean_data_path = interim_data_dir / "clean_transactions.csv"

# Check that the raw data file exists before trying to load it
if not raw_data_path.exists():
    raise FileNotFoundError(f"Raw data file not found at {raw_data_path}")

# Load the raw dataset into a pandas DataFrame
df = pd.read_csv(raw_data_path, encoding="latin1")

# Store the original number of rows in the dataset to compare before and after cleaning
original_rows = len(df)

# Convert InvoiceNo to text because invoice numbers are IDs not math Numbers
df["InvoiceNo"] = df["InvoiceNo"].astype(str)

# Convert StockCode to text because stock codes are IDs not math Numbers
df["StockCode"] = df["StockCode"].astype(str)

# Convert InvoiceDate to datetime format for easier manipulation
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format="%m/%d/%Y %H:%M")

# Remove rows with missing CustomerID values since they are essential for analysis
df = df.dropna(subset=["CustomerID"])

# Convert CustoomerID from decimal-looking numbers into a clean text
df["CustomerID"] = df["CustomerID"].astype(int).astype(str)

# Remove cancelled invoices by filtering out rows where InvoiceNo starts with 'C'
df = df[~df["InvoiceNo"].str.startswith("C")]

# Remove rows where Quantity is zero or negative
df = df[df["Quantity"] > 0]

# Remove rows where UnitPrice is zero or negative
df = df[df["UnitPrice"] > 0]

# Replace missing product descriptions woth a simple placeholder text
df["Description"] = df["Description"].fillna("No description")

# Remove extra spaces from country names to ensure consistency
df["Country"] = df["Country"].str.strip()

# Create a new column for total price by multiplying Quantity and UnitPrice
df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

# Save the cleaned data to data/interim/clean_transactions.csv without the index column
df.to_csv(clean_data_path, index=False)

# Printing a cleaning summary
print("Data Cleaning Complete")
print("=" * 40)

# Print the original and cleaned number of rows to show the impact of cleaning
print(f"Original rows: {original_rows}")

print(f"Cleaned rows: {len(df)}")

# Print how many rows were removed
print(f"Rows removed: {original_rows - len(df)}")

# Print how many unique customers are in the cleaned dataset
print(f"Unique customers: {df['CustomerID'].nunique()}")

# Print the earliest and latest invoice dates in the cleaned dataset
print(f"Earliest invoice date: {df['InvoiceDate'].min()}")
print(f"Latest invoice date: {df['InvoiceDate'].max()}")

# Print where the cleaned data was saved
print(f"Cleaned data saved to: {clean_data_path}")
