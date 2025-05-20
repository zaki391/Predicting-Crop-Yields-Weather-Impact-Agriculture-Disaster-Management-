# 🌦️ Weather Data Analysis & Risk Prediction

This project performs **exploratory data analysis (EDA)** on Indian weather data and implements a **risk prediction model** to classify areas into drought, flood, or normal conditions based on rainfall, temperature, and humidity.

---

## 📁 Dataset

The dataset used is `WeatherDataset.csv` and includes the following features:

- `location` – State or region
- `rainfall` – Rainfall in mm
- `temp` – Temperature in °C
- `humidity` – Relative humidity %
- *(Other columns if applicable)*

---

## 📊 Exploratory Data Analysis (EDA)

Performed using **Pandas**, **Seaborn**, and **Matplotlib**:

- Summary statistics (`.describe()`, `.info()`)
- Missing value inspection
- Distributions of rainfall, temperature, and humidity
- Correlation heatmap
- Top 10 regions by average rainfall

---

## ⚠️ Risk Classification

The risk is classified into three categories:

| Risk Type | Condition                         |
|-----------|-----------------------------------|
| Flood     | Rainfall > 200 mm                 |
| Drought   | Rainfall < 50 mm                  |
| Normal    | Rainfall between 50 mm and 200 mm |

A numeric column `risk_code` is generated for model training:
- 0 = Normal
- 1 = Flood
- 2 = Drought

---

## 🤖 Machine Learning Model

We use a **Gradient Boosting Classifier** to predict the risk based on weather features.

```python
from sklearn.ensemble import GradientBoostingClassifier
Features Used:
rainfall

temp

humidity

Evaluation:
Model accuracy score printed on test set
