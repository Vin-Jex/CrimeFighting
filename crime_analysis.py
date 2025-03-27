import os
import zipfile
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import plotly.express as px

# Step 1: Define Dataset Paths - Set paths for the dataset ZIP file and extracted folder.
DATASET_PATH = "crime-statistics-for-south-africa.zip"
EXTRACTED_FOLDER = "crime_data"

# Step 2: Check if Dataset is Already Available - If the dataset isn't found, attempt to download it from Kaggle.
if not os.path.exists(DATASET_PATH) and not os.path.exists(EXTRACTED_FOLDER):
    print("Downloading dataset from Kaggle...")
    os.system(
        "kaggle datasets download -d slwessels/crime-statistics-for-south-africa -p ."
    )
else:
    print("Dataset already exists. Skipping download.")

# Step 3: Extract the Dataset if Not Already Extracted - Unzip the dataset only if the extracted folder doesn’t exist.
if not os.path.exists(EXTRACTED_FOLDER):
    if os.path.exists(DATASET_PATH):
        print("Extracting dataset...")
        with zipfile.ZipFile(DATASET_PATH, "r") as zip_ref:
            zip_ref.extractall(EXTRACTED_FOLDER)
    else:
        raise FileNotFoundError(
            f"ZIP file '{DATASET_PATH}' not found, and no extracted data found."
        )

# Step 4: Load the dataset - Identify and validate the correct CSV file and display available columns and select relevant ones.
csv_files = [f for f in os.listdir(EXTRACTED_FOLDER) if f.endswith(".csv")]
if not csv_files:
    raise FileNotFoundError("No CSV file found in the dataset folder!")

correct_file = "SouthAfricaCrimeStats_v2.csv"
file_path = os.path.join(EXTRACTED_FOLDER, correct_file)

if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"Expected file {correct_file} not found in {EXTRACTED_FOLDER}."
    )

print(f"Loading dataset: {file_path}")

# Check available columns
available_cols = pd.read_csv(file_path, nrows=5).columns.tolist()
print("Available columns in dataset:", available_cols)

# Define columns to use - updated for SA dataset
use_cols = ["Province", "Station", "Category"] + [
    col for col in available_cols if "-" in col
]  # Include year columns
missing_cols = [col for col in use_cols if col not in available_cols]

if missing_cols:
    raise ValueError(f"Missing columns in dataset: {missing_cols}")

crime_df = pd.read_csv(file_path, usecols=use_cols, low_memory=False)
crime_df = crime_df.sample(n=2000, random_state=42)  # Sample 2000 records

# Step 5: Additional SA-Specific Preprocessing - Convert year columns to numeric format and create a new feature Total_Crimes summing yearly crime counts.
# Convert year columns to numeric
year_cols = [col for col in crime_df.columns if "-" in col]
crime_df[year_cols] = crime_df[year_cols].apply(pd.to_numeric, errors="coerce")

# Add total crimes feature
crime_df["Total_Crimes"] = crime_df[year_cols].sum(axis=1)

# Step 6: Data Overview - Display dataset information (e.g., data types, missing values).
print("\nBasic dataset info:")
print(crime_df.info())

# Step 7: Convert Categorical Columns - Convert 'Province', 'Station', and 'Category' into categorical types.
categorical_cols = ['Province', 'Station', 'Category']
for col in categorical_cols:
    if col in crime_df.columns:
        crime_df[col] = crime_df[col].astype('category')

# Step 8: Check unique values in categorical columns - Display unique values for categorical features before encoding.
print("\nUnique values in categorical columns:")
for col in categorical_cols:
    if col in crime_df.columns:
        print(f"{col}: {crime_df[col].nunique()} unique values")

# Step 9: Debug: Check unique values in categorical columns
print("\nUnique values before encoding:")
for col in categorical_cols:
    print(f"{col}: {crime_df[col].nunique()} unique values")

# Step 10: Encode categorical columns - Apply LabelEncoder to categorical columns.
encoders = {}
for col in categorical_cols:
    if crime_df[col].nunique() > 0:
        encoder = LabelEncoder()
        crime_df[col] = encoder.fit_transform(crime_df[col].astype(str))
        encoders[col] = encoder

# Step 11: Select Features and Target - Use year-based columns as features and assign 'Category' as the target variable.
features = year_cols
target = 'Category' if 'Category' in crime_df.columns else None

if not target:
    raise ValueError("Target column 'Category' not found in dataset")

# Step 12: Split the dataset - Divide into training (80%) and testing (20%) sets.
X = crime_df[features]
y = crime_df[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 13: Handle Class Imbalance using SMOTE - Drop underrepresented classes with fewer than 7 samples and apply SMOTE to oversample the training data.
class_counts = y_train.value_counts()
small_classes = class_counts[class_counts < 2].index
print("\nDropping classes with insufficient samples:", small_classes.tolist())

valid_classes = class_counts[class_counts > 6].index
mask = y_train.isin(valid_classes)
X_train = X_train[mask]
y_train = y_train[mask]

smote = SMOTE(k_neighbors=3, random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Step 14: Standardize Features - Normalize the dataset using StandardScaler.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 15: Train a Random Forest model - Fit a RandomForestClassifier to the training data.
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

# Step 16: Evaluate the model - Generate a classification report and confusion matrix.
y_pred = clf.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=1))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# Step 17: Visualize crime trends - Crime Categories – Bar plot showing total crimes by category, crime Trends Over the Years – Bar plot highlighting peak and lowest crime years, time Series Analysis – Line chart of average crime rates over time, category-Year Heatmap – Heatmap displaying crime trends per category, feature Importance – Bar chart ranking feature importance in classification, interactive Plotly Visualization – Interactive crime category analysis.
plt.figure(figsize=(14, 7))

# 1. Crime Categories Visualization
crime_by_category = (
    crime_df.groupby("Category")["Total_Crimes"].sum().sort_values(ascending=False)
)
sns.barplot(x=crime_by_category.index, y=crime_by_category.values, palette="rocket")
plt.xticks(rotation=90)
plt.title("South Africa: Total Crimes by Category")
plt.xlabel("Crime Category")
plt.ylabel("Total Crimes (2005-2016)")
plt.tight_layout()
plt.show()

# 2. Crime Trends Over Years
yearly_totals = crime_df[year_cols].sum().reset_index()
yearly_totals.columns = ["Year", "Total Crimes"]

max_year = yearly_totals.loc[yearly_totals["Total Crimes"].idxmax()]
min_year = yearly_totals.loc[yearly_totals["Total Crimes"].idxmin()]

plt.figure(figsize=(14, 7))
sns.barplot(
    x="Year",
    y="Total Crimes",
    data=yearly_totals,
    palette=[
        "red" if x == max_year["Year"] else "blue" if x == min_year["Year"] else "gray"
        for x in yearly_totals["Year"]
    ],
)
plt.title("South Africa: Crime Trends by Year")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print(f"\nPeak crime year: {max_year['Year']} ({max_year['Total Crimes']} crimes)")
print(f"Lowest crime year: {min_year['Year']} ({min_year['Total Crimes']} crimes)")

# 3. Enhanced Time Series Analysis
plt.figure(figsize=(14, 7))
crime_df[year_cols].mean().plot(
    kind="line", marker="o", linestyle="--", color="green", linewidth=2
)
plt.title("South Africa: Average Crime Rate by Year")
plt.xlabel("Year")
plt.ylabel("Average Crime Count")
plt.grid(True)
plt.tight_layout()
plt.show()

# 4. Category-Year Heatmap
plt.figure(figsize=(12, 8))
category_year = crime_df.groupby("Category")[year_cols].sum()
sns.heatmap(category_year, cmap="YlOrRd", annot=True, fmt=".0f")
plt.title("South Africa: Crime Categories by Year")
plt.tight_layout()
plt.show()

# 5. Feature Importance
plt.figure(figsize=(10, 6))
feat_importances = pd.Series(clf.feature_importances_, index=features)
feat_importances.nlargest(10).plot(kind="barh")
plt.title("Feature Importance for Crime Category Prediction")
plt.tight_layout()
plt.show()

# 6. Interactive Plotly Visualization
fig = px.bar(
    crime_df,
    x="Category",
    y="Total_Crimes",
    color="Category",
    title="Interactive View of Crimes by Category",
    hover_data=year_cols,
)
fig.show()

# Step 18: Save the Transformed Dataset - Convert encoded labels back to original values and Sort the dataset and save it as "sa_crime_data_transformed.csv".
for col, encoder in encoders.items():
    crime_df[col] = encoder.inverse_transform(crime_df[col])

crime_df.sort_values(by=year_cols, inplace=True)
transformed_file_path = "sa_crime_data_transformed.csv"
crime_df.to_csv(transformed_file_path, index=False)
print(f"\nTransformed dataset saved to {transformed_file_path}")
