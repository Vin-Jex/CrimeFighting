import os
import zipfile
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import plotly.express as px

# Step 1: Define dataset paths
DATASET_PATH = "crime-data.zip"
EXTRACTED_FOLDER = "crime_data"

# Step 2: Check if dataset is already downloaded
if not os.path.exists(DATASET_PATH) and not os.path.exists(EXTRACTED_FOLDER):
    print("Downloading dataset from Kaggle...")
    os.system("kaggle datasets download -d ishajangir/crime-data -p .")
else:
    print("Dataset already exists. Skipping download.")

# Step 3: Extract the dataset if not already extracted
if not os.path.exists(EXTRACTED_FOLDER):
    if os.path.exists(DATASET_PATH):
        print("Extracting dataset...")
        with zipfile.ZipFile(DATASET_PATH, 'r') as zip_ref:
            zip_ref.extractall(EXTRACTED_FOLDER)
    else:
        raise FileNotFoundError(f"ZIP file '{DATASET_PATH}' not found, and no extracted data found.")

# Step 4: Load the dataset
csv_files = [f for f in os.listdir(EXTRACTED_FOLDER) if f.endswith(".csv")]
if not csv_files:
    raise FileNotFoundError("No CSV file found in the dataset folder!")

file_path = os.path.join(EXTRACTED_FOLDER, csv_files[0])
print(f"Loading dataset: {file_path}")

# Check available columns
available_cols = pd.read_csv(file_path, nrows=5).columns.tolist()
print("Available columns in dataset:", available_cols)

# Define columns to use
use_cols = ["AREA", "DATE OCC", "TIME OCC", "Crm Cd Desc", "AREA NAME", "Premis Desc", "Status"]
missing_cols = [col for col in use_cols if col not in available_cols]

if missing_cols:
    raise ValueError(f"Missing columns in dataset: {missing_cols}")

crime_df = pd.read_csv(file_path, usecols=use_cols, low_memory=False)
crime_df = crime_df.sample(frac=0.3, random_state=42)  # Take 30% of the data


# Convert DATE OCC safely
crime_df["DATE OCC"] = pd.to_datetime(crime_df["DATE OCC"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")

# Extract Year, Month, and Day
crime_df["Year"] = crime_df["DATE OCC"].dt.year.astype("int16")
crime_df["Month"] = crime_df["DATE OCC"].dt.month.astype("int8")
crime_df["Day"] = crime_df["DATE OCC"].dt.day.astype("int8")

# Convert TIME OCC
crime_df["TIME OCC"] = crime_df["TIME OCC"].astype("int16")


# Extract years
crime_df["Year"] = crime_df["DATE OCC"].dt.year


# Print unique years before any filtering
print("Unique years in raw dataset:", crime_df["Year"].unique())

# Step 5: Convert DATE OCC column to datetime
print("Before date conversion:")
print(crime_df["DATE OCC"].head(10))  # Print sample dates for debugging

# Automatically infer date format
crime_df["DATE OCC"] = pd.to_datetime(crime_df["DATE OCC"], errors="coerce")

print("Unique years after date conversion:", crime_df["DATE OCC"].dt.year.unique())

# Print results after conversion
print("\nAfter date conversion:")
print(crime_df["DATE OCC"].head(10))

# Step 6: Drop rows with missing dates
print("Rows before dropping NaNs:", crime_df.shape)
crime_df.dropna(subset=["DATE OCC"], inplace=True)
print("Rows after dropping NaNs:", crime_df.shape)

# Verify unique years after dropping NaNs
print("Unique years after dropping NaNs:", crime_df["DATE OCC"].dt.year.unique())


# Debug: Check if dataset is still empty
if crime_df.empty:
    raise ValueError("Dataset is still empty after date conversion! Check date parsing.")


# Step 6: Drop rows with missing dates
print("Before dropping NaNs:", crime_df.shape)
crime_df.dropna(subset=["DATE OCC"], inplace=True)
print("After dropping NaNs:", crime_df.shape)

# Step 7: Extract Year, Month, Day
crime_df["Year"] = crime_df["DATE OCC"].dt.year.astype("int16")
crime_df["Month"] = crime_df["DATE OCC"].dt.month.astype("int8")
crime_df["Day"] = crime_df["DATE OCC"].dt.day.astype("int8")

# Step 8: Drop the original date column
crime_df.drop(columns=["DATE OCC"], inplace=True)

# Step 9: Convert categorical columns to category type
categorical_cols = ["Crm Cd Desc", "AREA NAME", "Premis Desc", "Status"]
for col in categorical_cols:
    crime_df[col] = crime_df[col].astype("category")

# Step 10: Debug: Check unique values in categorical columns
print("Unique values before encoding:")
for col in categorical_cols:
    print(f"{col}: {crime_df[col].nunique()} unique values")

# Ensure dataset is not empty before encoding
if crime_df.empty:
    raise ValueError("Dataset is empty before encoding! Check previous steps.")

# Step 11: Encode categorical columns
encoders = {}  # Dictionary to store mappings
for col in categorical_cols:
    if crime_df[col].nunique() > 0:
        encoder = LabelEncoder()
        crime_df[col] = encoder.fit_transform(crime_df[col].astype(str))
        encoders[col] = encoder  # Store the encoder for decoding later
    else:
        print(f"Skipping encoding for {col}, it has no data!")

# Step 12: Select features and target
features = ["AREA", "Year", "Month", "Day", "TIME OCC", "AREA NAME", "Premis Desc", "Status"]
target = "Crm Cd Desc"

# Debug: Check dataset size before splitting
print("Dataset size before splitting:", crime_df.shape)

X = crime_df[features]
y = crime_df[target]

# Ensure X and y are not empty
if len(X) == 0 or len(y) == 0:
    raise ValueError("X or y is empty before splitting! Debug preprocessing steps.")

# Step 13: Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 14: Handle class imbalance using SMOTE
# Remove classes with fewer than 2 samples
class_counts = y_train.value_counts()
small_classes = class_counts[class_counts < 2].index

print("Dropping the following classes due to insufficient samples:")
print(small_classes)

# Keep only classes with more than `n_neighbors` samples
valid_classes = class_counts[class_counts > 6].index
mask = y_train.isin(valid_classes)
X_train = X_train[mask]
y_train = y_train[mask]

smote = SMOTE(sampling_strategy="auto", k_neighbors=1, random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Step 15: Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 16: Train a Random Forest model
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

# Step 17: Evaluate the model
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=1))
print(confusion_matrix(y_test, y_pred))

# Step 18: Visualize crime trends
# Aggregate crimes by full date (Year, Month, Day, Hour)
# Extract hour from TIME OCC (e.g., 1530 -> 15)
crime_df["Hour"] = crime_df["TIME OCC"] // 100

# Create DateTime column
crime_df["DateTime"] = pd.to_datetime(crime_df[["Year", "Month", "Day"]]) + pd.to_timedelta(crime_df["Hour"], unit="h")

# Aggregate crimes by hour
crime_hourly = crime_df.groupby("Hour").size().reset_index(name="Crime Count")

# Create bar chart
fig = px.bar(crime_hourly, x="Hour", y="Crime Count", title="Crime Count by Hour of the Day",
labels={"Hour": "Hour of the Day", "Crime Count": "Total Crimes"}, color="Crime Count")

fig.show()


# Step 19: Save the transformed dataset to a new CSV file
for col, encoder in encoders.items():
    crime_df[col] = encoder.inverse_transform(crime_df[col])

crime_df.sort_values(by=["Year", "Month", "Day", "TIME OCC"], inplace=True)

# Save the transformed dataset
transformed_file_path = "crime_data_transformed.csv"
crime_df.to_csv(transformed_file_path, index=False)
print(f"Transformed dataset saved to {transformed_file_path}")
