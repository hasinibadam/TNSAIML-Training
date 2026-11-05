# 🌍 AeroSense – Air Quality Index (AQI) Prediction System

## 📌 Project Overview

**AeroSense** is a Machine Learning-based web application designed to predict the Air Quality Index (AQI) using environmental pollutant data.

The system analyzes pollutant concentrations such as PM2.5, PM10, NO₂, CO, SO₂, and O₃ to estimate AQI. It also provides interactive visualizations, city-wise pollution analysis, and insights into air quality.

The application is developed using Python and Streamlit, with multiple Machine Learning algorithms evaluated to identify the best-performing model.

## 🎯 Objectives

- Predict AQI based on environmental pollutant concentrations.
- Compare multiple Machine Learning regression algorithms.
- Identify the best-performing prediction model.
- Visualize air quality trends and pollution patterns.
- Explore city-wise AQI information.
- Provide basic health recommendations based on predicted AQI.

## ✨ Key Features

### 1. Overview Dashboard
- Displays average AQI, maximum AQI, number of cities, and total records.
- Shows AQI distribution and category-wise analysis.
- Displays recent air quality records.

### 2. AQI Prediction
- Accepts pollutant concentration values as input.
- Predicts AQI using the trained Machine Learning model.
- Displays the predicted AQI and corresponding category.
- Provides basic health recommendations.

### 3. City Explorer
- Explores historical AQI data for different cities.
- Displays city-wise AQI trends.
- Supports comparison of air quality across cities.

### 4. Pollution Insights
- Visualizes pollutant distributions.
- Displays relationships between different pollutants using a correlation heatmap.

### 5. Model Lab
- Compares the performance of different Machine Learning algorithms.
- Displays evaluation metrics.
- Visualizes model performance and feature importance.

## 🛠️ Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Frontend / Web Framework | Streamlit |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Visualization | Plotly |
| Model Storage | Joblib / Pickle |

## 🤖 Machine Learning Algorithms

The following regression algorithms are implemented and compared:

1. Random Forest Regressor
2. Gradient Boosting Regressor
3. Extra Trees Regressor

**Best Performing Model:** Extra Trees Regressor

## 📊 Model Performance

The models were evaluated using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² Score.

| Algorithm | MAE | RMSE | R² Score |
|---|---:|---:|---:|
| Random Forest | 17.985 | 29.112 | 0.9182 |
| Gradient Boosting | 19.218 | 29.584 | 0.9155 |
| Extra Trees | 17.618 | 28.464 | 0.9218 |

Extra Trees Regressor achieved the best performance among the evaluated models.

## 📂 Project Structure

```text
AeroSense/
│
├── data/
│   └── city_day.csv
│
├── models/
│   ├── aqi_model.pkl
│   └── model_metrics.pkl
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 📁 Dataset

The project uses the **Air Quality Data in India** dataset by Rohan Rao, available on Kaggle.

Dataset file used: `city_day.csv`

It contains air quality information for multiple Indian cities, including pollutant concentrations and AQI values.

Important columns include:

- City
- Date
- PM2.5
- PM10
- NO
- NO2
- NOx
- NH3
- CO
- SO2
- O3
- AQI
- AQI_Bucket

## ⚙️ Installation and Setup

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### Step 2: Navigate to the Project Folder

```bash
cd AeroSense
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

### Model Training (Optional)

To train the models again, run:

```bash
python train_model.py
```

The trained model and evaluation metrics will be saved in the `models/` directory.

## 📈 AQI Categories

The application uses the following AQI category ranges:

| AQI Range | Category |
|---|---|
| 0–50 | Good |
| 51–100 | Satisfactory |
| 101–200 | Moderate |
| 201–300 | Poor |
| 301–400 | Very Poor |
| Above 400 | Severe |

## 🔮 Future Enhancements

- Integration with real-time air quality data sources.
- AQI forecasting using time-series Machine Learning models.
- Interactive geographical pollution maps.
- Personalized health recommendations.
- Mobile application integration.

## ⚠️ Limitations

- Predictions depend on the pollutant values provided as input.
- The model is trained on historical dataset records.
- The application does not currently provide real-time AQI monitoring.
- Predictions are estimates and should not be treated as official government AQI readings.

## 👩‍💻 Project Information

**Project Name:** AeroSense – Air Quality Index Prediction System

**Domain:** Artificial Intelligence and Machine Learning

**Application Type:** Machine Learning-based Web Application

---

*Developed as an academic project using Python, Machine Learning, and Streamlit.*
