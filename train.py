
from pathlib import Path
import json
import re
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

RAINFALL_FILE = DATA_DIR / "rainfall_statewise_daily_imd.csv"
FLOOD_FILE = DATA_DIR / "India_Flood_Inventory_v3.csv"

STATE_ALIASES = {
    "orissa": "Odisha",
    "uttaranchal": "Uttarakhand",
    "pondicherry": "Puducherry (UT)",
    "nct of delhi": "Delhi",
    "delhi": "Delhi",
    "jammu & kashmir": "Jammu and Kashmir",
    "jammu and kashmir": "Jammu and Kashmir",
    "andaman and nicobar islands": "Andaman and Nicobar Islands",
    "andaman & nicobar islands": "Andaman and Nicobar Islands",
    "dadra and nagar haveli and daman and diu":
        "Dadra and Nagar Haveli and Daman and Diu"
}


def clean_state(value):
    if pd.isna(value):
        return None

    state = re.sub(r"\s+", " ", str(value)).strip()
    if not state:
        return None

    return STATE_ALIASES.get(state.lower(), state.title())


def find_column(df, candidates):
    lookup = {str(c).strip().lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    raise ValueError(
        f"Could not find any of these columns: {candidates}. "
        f"Available columns: {list(df.columns)}"
    )


def load_rainfall_data():
    if not RAINFALL_FILE.exists():
        raise FileNotFoundError(
            f"Rainfall CSV not found: {RAINFALL_FILE}"
        )

    df = pd.read_csv(RAINFALL_FILE)

    state_col = find_column(df, ["State", "State/UT"])
    date_col = find_column(df, ["Date"])
    actual_col = find_column(
        df, ["Daily Actual", "Actual Rainfall", "Rainfall"]
    )
    normal_col = find_column(
        df, ["Daily Normal", "Normal Rainfall"]
    )

    df = df.rename(columns={
        state_col: "State",
        date_col: "Date",
        actual_col: "Daily Actual",
        normal_col: "Daily Normal"
    })

    df["State"] = df["State"].apply(clean_state)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Daily Actual"] = pd.to_numeric(
        df["Daily Actual"], errors="coerce"
    )
    df["Daily Normal"] = pd.to_numeric(
        df["Daily Normal"], errors="coerce"
    )

    df = df.dropna(
        subset=["State", "Date", "Daily Actual"]
    )

    df = df[
        (df["Daily Actual"] >= 0) &
        (df["Daily Actual"] < 2000)
    ]

    # Remove national or aggregate rows if present.
    aggregate_names = {
        "india", "all india", "total", "average"
    }
    df = df[
        ~df["State"].str.lower().isin(aggregate_names)
    ]

    df["Daily Normal"] = df["Daily Normal"].fillna(
        df.groupby(["State", df["Date"].dt.month])[
            "Daily Normal"
        ].transform("median")
    )

    df["Daily Normal"] = df["Daily Normal"].fillna(
        df["Daily Normal"].median()
    )

    df = df.dropna(subset=["Daily Normal"])
    df = df.drop_duplicates(
        subset=["State", "Date"], keep="last"
    )

    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["Month_Sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_Cos"] = np.cos(2 * np.pi * df["Month"] / 12)
    df["Day_Sin"] = np.sin(2 * np.pi * df["Day"] / 31)
    df["Day_Cos"] = np.cos(2 * np.pi * df["Day"] / 31)

    return df.sort_values("Date").reset_index(drop=True)


def build_pipeline(model):
    categorical_features = ["State"]
    numeric_features = [
        "Month_Sin", "Month_Cos",
        "Day_Sin", "Day_Cos",
        "Normal_MM"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "state",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            ),
            ("numeric", "passthrough", numeric_features)
        ]
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])


def train_rainfall_model(df):
    feature_columns = [
        "State", "Month_Sin", "Month_Cos",
        "Day_Sin", "Day_Cos", "Normal_MM"
    ]

    model_data = df.copy()
    model_data["Normal_MM"] = model_data["Daily Normal"]

    X = model_data[feature_columns]
    y = model_data["Daily Actual"]

    # Split by date, so observations from the same date
    # are not divided between training and testing.
    unique_dates = np.sort(model_data["Date"].unique())

    if len(unique_dates) < 5:
        raise ValueError(
            "Not enough unique dates to train and evaluate the model."
        )

    cutoff_index = max(1, int(len(unique_dates) * 0.8))
    cutoff_index = min(cutoff_index, len(unique_dates) - 1)
    cutoff_date = unique_dates[cutoff_index]

    train_mask = model_data["Date"] < cutoff_date
    test_mask = ~train_mask

    X_train, X_test = X.loc[train_mask], X.loc[test_mask]
    y_train, y_test = y.loc[train_mask], y.loc[test_mask]

    candidates = {
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=250,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=250,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    }

    results = []
    best_name = None
    best_mae = float("inf")

    for name, estimator in candidates.items():
        pipeline = build_pipeline(estimator)
        pipeline.fit(X_train, y_train)

        predictions = np.maximum(
            0, pipeline.predict(X_test)
        )

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        results.append({
            "model": name,
            "mae": round(float(mae), 3),
            "rmse": round(float(rmse), 3),
            "r2": round(float(r2), 3)
        })

        if mae < best_mae:
            best_mae = mae
            best_name = name

    # Refit the selected algorithm on all available records.
    final_model = build_pipeline(candidates[best_name])
    final_model.fit(X, y)

    joblib.dump(
        final_model,
        MODEL_DIR / "rainfall_model.joblib"
    )

    metrics = {
        "selected_model": best_name,
        "training_rows": int(len(model_data)),
        "testing_rows": int(test_mask.sum()),
        "test_start_date": str(pd.Timestamp(cutoff_date).date()),
        "results": results,
        "target": "Daily Actual rainfall in millimetres",
        "features": feature_columns
    }

    with open(
        MODEL_DIR / "rainfall_metrics.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(metrics, file, indent=4)

    return metrics


def save_rainfall_summaries(df):
    # Store cleaned historical data.
    df.to_csv(
        MODEL_DIR / "rainfall_history.csv",
        index=False
    )

    # Median normal rainfall for each state and month.
    climatology = (
        df.groupby(["State", "Month"], as_index=False)
        .agg(
            normal_mm=("Daily Normal", "median"),
            historical_mean_mm=("Daily Actual", "mean"),
            historical_median_mm=("Daily Actual", "median"),
            observation_count=("Daily Actual", "count")
        )
    )

    climatology.to_csv(
        MODEL_DIR / "rainfall_climatology.csv",
        index=False
    )


def build_flood_summary(rainfall_states):
    if not FLOOD_FILE.exists():
        print("Flood inventory not found. Skipping flood summary.")
        return

    df = pd.read_csv(FLOOD_FILE)

    date_col = find_column(
        df, ["Start Date", "Date", "Event Date"]
    )
    state_col = find_column(
        df, ["State", "States", "State/UT"]
    )

    df = df.rename(columns={
        date_col: "Start Date",
        state_col: "State"
    })

    df["Start Date"] = pd.to_datetime(
        df["Start Date"],
        errors="coerce",
        dayfirst=True
    )

    df = df.dropna(subset=["Start Date", "State"])

    # Some inventory rows contain multiple states.
    df["State"] = df["State"].astype(str).str.split(
        r"\s*,\s*"
    )
    df = df.explode("State")
    df["State"] = df["State"].apply(clean_state)
    df = df.dropna(subset=["State"])

    df["Year"] = df["Start Date"].dt.year
    df["Month"] = df["Start Date"].dt.month

    # Keep the flood inventory's recorded state names.
    # It is not a complete record of all flood/no-flood days.
    df = df[df["State"].isin(rainfall_states)]

    if df.empty:
        print("No matching flood state records were found.")
        return

    monthly = (
        df.groupby(["State", "Month"])
        .size()
        .reset_index(name="event_count")
    )

    annual = (
        df.groupby("State")
        .agg(
            total_events=("Start Date", "count"),
            first_year=("Year", "min"),
            last_year=("Year", "max")
        )
        .reset_index()
    )

    annual["record_year_span"] = (
        annual["last_year"] - annual["first_year"] + 1
    ).clip(lower=1)

    annual["events_per_record_year"] = (
        annual["total_events"] / annual["record_year_span"]
    )

    monthly.to_csv(
        MODEL_DIR / "flood_monthly_summary.csv",
        index=False
    )

    annual.to_csv(
        MODEL_DIR / "flood_state_summary.csv",
        index=False
    )


def main():
    print("\nAquaPredict - Model Training")
    print("-" * 45)

    df = load_rainfall_data()
    print(f"Clean rainfall records: {len(df)}")
    print(f"States found: {df['State'].nunique()}")
    print(
        f"Date range: {df['Date'].min().date()} "
        f"to {df['Date'].max().date()}"
    )

    metrics = train_rainfall_model(df)
    save_rainfall_summaries(df)
    build_flood_summary(sorted(df["State"].unique()))

    print("\nModel comparison:")
    for result in metrics["results"]:
        print(result)

    print("\nSelected model:", metrics["selected_model"])
    print("\nTraining completed successfully.")
    print("Generated files are saved in the models folder.")


if __name__ == "__main__":
    main()
