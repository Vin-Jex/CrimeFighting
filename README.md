# Crime Data Analysis Across Africa - *Group 10*

## Project Overview – Understanding Crime Patterns Across Africa

This project analyzes crime data across African countries to identify patterns, trends, and predictive factors. We use data science techniques to clean, visualize, and model crime data, helping uncover insights that can support security policy, resource planning, and law enforcement strategy.

---

## What We Did

We analyzed data from the **Africa Organised Crime Index - 2023** to answer:

1. What are the most common and severe crimes across Africa?
2. How does crime relate to national resilience?
3. Can we predict crime levels based on socio-economic indicators?

![Top 10 Countries by Crime Index](images/top_10_countries_crime_index.png)
*Figure 1: African countries with the highest crime index*

---

## Our Step-by-Step Process

### 1. Data Collection

* **Source:** [Africa Organised Crime Index - 2023](https://ocindex.net/)
* Downloaded as a structured CSV from the official site
* Included metrics like:

  * **Criminality** (overall crime index)
  * **Resilience** (ability to resist/mitigate crime)
  * Sub-scores for state-embedded actors, criminal networks, etc.

---

### 2. Data Cleaning & Preprocessing

* Removed missing values
* Normalized and encoded categorical fields
* Engineered features like:

  * Total Crime Score
  * Crime Level Labels
  * Regional Groupings

---

### 3. Exploratory Data Analysis (EDA)

#### A. Crime Index by Country

![Crime Index by Country](images/crime_index_countries.png)
*Figure 2: Crime index across African nations*

#### B. Correlation Heatmap

![Correlation Heatmap](images/correlation_heatmap.png)
*Figure 3: Correlations between crime, resilience, and related indicators*

#### C. Crime vs Resilience

![Crime vs Resilience](images/crime_vs_resilience.png)
*Figure 4: Inverse relationship between national resilience and crime index*

#### D. Feature Pair Analysis

![Pairplot of Key Features](images/pairplot_features.png)
*Figure 5: Pairwise relationships among crime-related features*

#### E. Drug Trafficking Analysis

![Drug Trafficking Trends](images/drug_trafficking.png)
*Figure 6: Drug trafficking scores across the continent*

#### F. Criminality Score Distribution

![Criminality Score Distribution](images/criminality_score_distribution.png)
*Figure 7: Distribution of total criminality scores*

#### G. Crime Level by Resilience Category

![Resilience vs Crime Level](images/resilience_score_by_crime_level.png)
*Figure 8: Countries with lower resilience often show higher crime levels*

---

### 4. Predictive Modeling

We used a **Random Forest Classifier** to predict crime levels (e.g., low, medium, high) from socio-political indicators.

* **SMOTE** was used to balance underrepresented classes
* **StandardScaler** applied to normalize features
* Model achieved strong performance on unseen data

![Crime Level Confusion Matrix](images/crime_level_confusion_matrix.png)
*Figure 9: Model accuracy shown via confusion matrix*

![Feature Importance](images/feature_importance.png)
*Figure 10: Which features were most useful for predicting crime levels*

---

### 5. Future Predictions

Using the dataset from **2019**, **2021**, and **2023**, we applied machine learning models to **predict crime trends in Africa** from 2024 to 2027. By training on historical data and incorporating socio-economic and resilience indicators, we forecast crime levels and trends across countries.

* **Predicted Outcomes:**

  * We anticipate an increase in organized crime in certain regions due to rising political instability and economic downturns.
  * Countries with lower resilience are projected to face greater challenges in crime control, influencing future crime index rankings.
* **Modeling Approach:**

  * We used **time series forecasting** techniques to estimate future crime levels.
  * Models include **ARIMA** and **LSTM** (Long Short-Term Memory) networks for capturing trends in historical data and making future predictions.

![Comprehensive Regional Forecast](images/comprehensive_regional_forecast.png)
*Figure 11: Predicted crime trends for selected African countries (2024-2027)*

---

### 6. Bonus: Interactive Visualizations

Explore multi-dimensional relationships using these interactive tools:

1. [📁 interactive\_3d\_plot.html](images/interactive_3d_plot.html) — Explore crime data in 3D.
2. [📁 interactive\_regional\_forecast.html](images/interactive_regional_forecast.html) — Explore regional crime predictions for 2024-2027.

*Open these files in any browser for an interactive experience.*

---

## Key Insights

* Crime is not randomly distributed — it's tightly linked to governance and resilience
* Resilient countries like Botswana and Namibia score lower in crime despite geographic proximity to higher-risk regions
* Drug trafficking and criminal networks are major contributors to overall crime scores
* Our model predicts crime levels with **\[insert accuracy]%** accuracy
* The future crime prediction models suggest varying trends across the continent, with some regions expected to experience worsening crime conditions due to political and economic instability.

---

## Why This Matters

These insights can help:

* Governments prioritize reforms in high-risk zones
* NGOs allocate resources to support vulnerable populations
* Researchers uncover links between governance, resilience, and criminal behavior
* Policymakers prepare for emerging crime trends over the next few years

---

## For Non-Technical Readers

No need to understand the code — focus on:

* Visuals showing crime and resilience across countries
* Predictions of which countries are at risk
* Impacts these findings could have on safety and stability

---

## Technical Summary

### Data Pipeline

1. **Source:** Africa Organised Crime Index 2023
2. **Preprocessing:** Cleaning, normalization, feature engineering
3. **EDA:** Visualizations and feature analysis
4. **Modeling:** Random Forest with SMOTE + scaling
5. **Future Predictions:** Time series forecasting (ARIMA, LSTM)
6. **Evaluation:** Confusion matrix and accuracy reports

---

## How to Reproduce This Project

Follow these steps to set up and run the project on your local machine:

### 1. Clone the Repository

```bash
git clone https://github.com/Vin-Jex/CrimeFighting.git
cd CrimeFighting
```

### 2. Set Up a Virtual Environment

It's best to isolate dependencies using `venv`.

```bash
# Create a virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Requirements

Make sure you have `pip` installed and then run:

```bash
pip install -r requirements.txt
```

### 4. Run the Analysis

```bash
python crime_analysis.py
```

### 5. View the Outputs

* 📂 `images/` – for all generated charts and graphs
* 📄 [interactive\_3d\_plot.html](images/interactive_3d_plot.html) — open in any browser for a 3D visualization
* 📄 [interactive\_regional\_forecast.html](images/interactive_regional_forecast.html) — open in any browser for regional crime predictions
* 📁 `africa_crime_data_transformed.csv` – cleaned dataset used for modeling

---

## Future Work

* Integrate geographic visualization using Folium or GeoPandas
* Add more recent data and time series projections
* Analyze intervention success rates over time
* Improve forecasting models for better accuracy and reliability

---

## Team Members

* Okereke Ifeanyi Vincent
* \[Name 2]
* \[Name 3]
* \[Name 4]
