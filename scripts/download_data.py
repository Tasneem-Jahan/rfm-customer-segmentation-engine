# Import os to read the environment variables from the project
import os

# Import path to work woth file and folder paths safely
from pathlib import Path

# Import subprocess so that Python can run the kaggle command for us
import subprocess

# Import load_dotenv so that Python can read values from tje .env file
from dotenv import load_dotenv

# Load the variables written inside the .env file
load_dotenv()

# Read the Kaggle API token from the .env file
kaggle_token = os.getenv("KAGGLE_API_TOKEN")

# Read the Kaggle dataset name from the .env file
dataset_name = os.getenv("KAGGLE_DATASET")

# Read the raw data folder path from the .env file
raw_data_dir = os.getenv("RAW_DATA_DIR", "data/raw")

# Convert the raw data directory path to a Path object for easier handling
raw_data_path = Path(raw_data_dir)

# Create the raw data directory if it doesn't exist
raw_data_path.mkdir(parents=True, exist_ok=True)

# Check if the Kaggle API token is missing
if not kaggle_token:
    raise ValueError("KAGGLE_API_TOKEN is missing in the .env file.")

# Check if the Kaggle dataset name is missing
if not dataset_name:
    raise ValueError("KAGGLE_DATASET is missing in the .env file.")

# Create a copy of the current environment variables
env = os.environ.copy()

# Add the Kaggle API token to the environment variables so that the kaggle command can access it
env["KAGGLE_API_TOKEN"] = kaggle_token

# Print a message to indicate that the download is starting
print(f"Downloading dataset '{dataset_name}' from Kaggle...")

# Print a message to indicate where the dataset will be saved
print(f"Saving dataset to '{raw_data_path}'...")

# Build the kaggle command that downloads and unzips the dataset
command = [
    "kaggle",
    "datasets",
    "download",
    dataset_name,
    "--path",
    str(raw_data_path),
    "--unzip",
]

# Run the command
result = subprocess.run(command, env=env)

# Check if the command failed
if result.returncode != 0:
    raise RuntimeError(
        f"Failed to download dataset '{dataset_name}' from Kaggle.")

# Create the expected file path of the downloaded dataset
csv_file = raw_data_path / "data.csv"

# Check if the expected file exists
if not csv_file.exists():
    raise FileNotFoundError(
        f"Expected file '{csv_file}' not found in data/raw")

# Print a message to indicate that the download was successful
print(
    f"Dataset '{dataset_name}' downloaded successfully and saved to '{csv_file}'.")
