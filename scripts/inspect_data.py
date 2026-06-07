# Import Path so we can work with file paths
from pathlib import Path

# Import pandas so we can read and inspect our data
import pandas as pd


# Store the path to the raw dataset file
data_path = Path("data/raw/data.csv")


# Check if the dataset file exists
if not data_path.exists():
    raise FileNotFoundError(f"The dataset file was not found at {data_path}")

# Read the dataset into a pandas DataFrame
df = pd.read_csv(data_path, encoding="latin-1")


# Print a title for the dataset inspection
print("Dataset Inspection")
print("=" * 40)

# Print the number of rows and columns in the dataset
print("\nShape")
print(df.shape)

# print all columns names
print("\nColumns")
print(df.columns.tolist())

# Print the first 5 rows of the dataset
print("\nFirst 5 Rows")
print(df.head())

# Print data types of each column
print("\nData Types")
print(df.dtypes)

# Print the number of missing values in each column
print("\nMissing Values")
print(df.isnull().sum())

# Print the number of unique customers
print("\nUnique Customers")
print(df["CustomerID"].nunique())

# Print the number of missing customer IDs
print("\nMissing Customer IDs")
print(df["CustomerID"].isnull().sum())

# Count invoices that start with 'C' (canceled orders)
cancelled_orders = df["InvoiceNo"].astype(str).str.startswith("C").sum()

print("\nCancelled Orders")
print(cancelled_orders)

# Print summary statistics for the Quantity and UnitPrice columns
print("\nSummary Statistics")
print(df[["Quantity", "UnitPrice"]].describe())

# Convert InvoiceDate into real datetime format
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

# Print the earlest and latest invoice dates
print("\nEarliest Invoice Date")
print(df["InvoiceDate"].min())

# Print the latest invoice date
print("\nLatest Invoice Date")
print(df["InvoiceDate"].max())

# Print the top 10 countries by row count
print("\nTop 10 Countries")
print(df["Country"].value_counts().head(10))
