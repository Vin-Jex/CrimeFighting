# CrimeFighting: Data Cleaning and Crime Analytics Dashboard

This project focuses on crime data cleaning and analysis. It cleans and processes crime data, trains a model to predict crime types, and visualizes crime trends over time. The data comes from a secondary source and is analyzed using Python, with the results displayed in a dashboard built with Streamlit.

**Group 10 – Topic: Crime Fighting**

---

## Prerequisites

Ensure you have Python 3.7+ installed on your system. You can download it from the official [Python website](https://www.python.org/downloads/).

### Required Python Libraries

Before running the project, install the required dependencies from `requirements.txt` using `pip`:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file contains the following libraries:

- pandas
- seaborn
- matplotlib
- scikit-learn
- imbalanced-learn
- plotly
- streamlit
- kaggle

---

## Project Structure

```
CrimeFighting/
│
├── crime_analysis.py       # Data cleaning, model training, and crime trend visualization
├── crime_dashboard.py      # Streamlit dashboard for visualizing crime data
├── crime_data.zip          # Original crime data (ZIP file)
├── crime_data_transformed.csv # Transformed dataset after cleaning
├── requirements.txt        # Required Python libraries
└── README.md               # Project documentation
```

---

## Steps to Run the Project

### 1. Clone the Repository

Start by cloning the repository to your local machine.

```bash
git clone https://github.com/Vin-Jex/CrimeFighting.git
cd CrimeFighting
```

### 2. Install Dependencies

Install the required libraries from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 3. Download and Extract Crime Data

The `crime_analysis.py` script will automatically check if the dataset is already downloaded. If not, it will download the dataset from Kaggle. The script will also extract the data from a ZIP file if needed.

Ensure that you have a valid Kaggle API key for the dataset download. To set up your Kaggle API key, follow the instructions [here](https://github.com/Kaggle/kaggle-api#api-credentials).

### 4. Run Data Cleaning and Analysis

Run the `crime_analysis.py` script to clean the dataset, preprocess the data, and train a RandomForest model. This script also generates crime trends visualizations.

```bash
python crime_analysis.py
```

The cleaned and transformed data will be saved in `crime_data_transformed.csv`.

### 5. Run the Crime Data Dashboard

After transforming the data, you can use the `crime_dashboard.py` script to visualize the crime trends using a Streamlit dashboard.

Run the Streamlit app:

```bash
streamlit run crime_dashboard.py
```

The app will open in your default browser and allow you to explore the crime trends based on different time views: Year, Month, Day, or All.

---

## Features

1. **Data Cleaning:**
   - Download and extract crime data from Kaggle.
   - Parse the "DATE OCC" column into a `datetime` format.
   - Handle missing values and remove unnecessary columns.
   - Encode categorical columns using `LabelEncoder`.
   
2. **Crime Prediction Model:**
   - Prepare data and handle class imbalance using SMOTE.
   - Train a RandomForest Classifier to predict crime types.
   - Evaluate the model performance with metrics like classification report and confusion matrix.

3. **Crime Trends Visualization:**
   - Visualize crime trends over time using Plotly.
   - Display crime data trends by Year, Month, Day, or overall with Streamlit.

---

## Notes

- **Data Source:** The data used in this project is from Kaggle (Crime Data by Isha Jangir). Make sure to follow the Kaggle API instructions for data downloading if you don't already have the data.
- **SMOTE:** The project handles class imbalance using SMOTE (Synthetic Minority Over-sampling Technique).
- **Streamlit Dashboard:** The dashboard provides an interactive visualization of crime data trends.

---

## License

This project is not open-source and is intended for educational purposes as part of Group 10's data science project on crime fighting.

---

## Acknowledgments

- Crime dataset by [Isha Jangir](https://www.kaggle.com/ishajangir).
- Thanks to the contributors of the libraries used in this project: Pandas, Scikit-Learn, Plotly, Seaborn, Streamlit, etc.