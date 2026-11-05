
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

print("Loading dataset...")

df = pd.read_csv("data/city_day.csv")

features = [
    "PM2.5", "PM10", "NO", "NO2", "NOx",
    "NH3", "CO", "SO2", "O3", "Benzene", "Toluene"
]

print("Cleaning dataset...")

for col in features + ["AQI"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.drop_duplicates()
df = df.dropna(subset=["AQI"])

# Keep valid AQI values
df = df[df["AQI"].between(0, 500)]

X = df[features]
y = df["AQI"]

print("Total valid records:", len(df))

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Machine Learning algorithms
algorithms = {
    "Random Forest": RandomForestRegressor(
        n_estimators=150,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=150,
        random_state=42
    ),

    "Extra Trees": ExtraTreesRegressor(
        n_estimators=150,
        random_state=42,
        n_jobs=-1
    )
}

results = []
trained_models = {}

print("\nTraining models...\n")

for name, algorithm in algorithms.items():

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", algorithm)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    rmse = mean_squared_error(
        y_test, predictions
    ) ** 0.5

    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "R2 Score": round(r2, 4)
    })

    trained_models[name] = pipeline

    print("Model:", name)
    print("MAE:", round(mae, 3))
    print("RMSE:", round(rmse, 3))
    print("R2 Score:", round(r2, 4))
    print("-" * 40)

# Select best model
best_name = max(
    results,
    key=lambda x: x["R2 Score"]
)["Model"]

best_model = trained_models[best_name]

# Save trained model
os.makedirs("models", exist_ok=True)

joblib.dump({
    "model": best_model,
    "features": features,
    "best_name": best_name,
    "medians": X_train.median().to_dict()
}, "models/aqi_model.pkl")

# Save evaluation results
metrics = pd.DataFrame(results)

joblib.dump(
    metrics,
    "models/model_metrics.pkl"
)

print("\n================================")
print("BEST MODEL:", best_name)
print("MODEL TRAINED SUCCESSFULLY!")
print("================================")
