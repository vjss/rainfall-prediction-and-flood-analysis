
from pathlib import Path
from datetime import date, timedelta
import json

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

MODEL_FILE = MODEL_DIR / "rainfall_model.joblib"
CLIMATE_FILE = MODEL_DIR / "rainfall_climatology.csv"
HISTORY_FILE = MODEL_DIR / "rainfall_history.csv"
FLOOD_MONTH_FILE = MODEL_DIR / "flood_monthly_summary.csv"
FLOOD_STATE_FILE = MODEL_DIR / "flood_state_summary.csv"
METRICS_FILE = MODEL_DIR / "rainfall_metrics.json"

MONTHS = [
    "January", "February", "March", "April",
    "May", "June", "July", "August",
    "September", "October", "November", "December"
]

RAINFALL_CATEGORIES = [
    (2.5, "Very low / No rain"),
    (15.5, "Light rainfall"),
    (64.5, "Moderate rainfall"),
    (115.5, "Heavy rainfall"),
    (float("inf"), "Very heavy rainfall")
]


def load_csv(path):
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_json(path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def get_rainfall_category(value):
    for upper_limit, label in RAINFALL_CATEGORIES:
        if value < upper_limit:
            return label
    return "Very heavy rainfall"


def get_rainfall_level(value):
    if value < 2.5:
        return "very-low"
    if value < 15.5:
        return "light"
    if value < 64.5:
        return "moderate"
    if value < 115.5:
        return "heavy"
    return "very-heavy"


def get_states():
    history = load_csv(HISTORY_FILE)
    if history.empty or "State" not in history:
        return []
    return sorted(history["State"].dropna().unique().tolist())


def get_normal_rainfall(state, month):
    climate = load_csv(CLIMATE_FILE)

    if climate.empty:
        return 0.0

    match = climate[
        (climate["State"] == state) &
        (climate["Month"] == month)
    ]

    if not match.empty:
        return float(match.iloc[0]["normal_mm"])

    state_data = climate[climate["State"] == state]

    if not state_data.empty:
        return float(state_data["normal_mm"].median())

    return float(climate["normal_mm"].median())


def predict_seven_days(state, start_date):
    model = joblib.load(MODEL_FILE)
    results = []

    for offset in range(7):
        current_date = start_date + timedelta(days=offset)
        month = current_date.month
        day = current_date.day

        normal_mm = get_normal_rainfall(state, month)

        features = pd.DataFrame([{
            "State": state,
            "Month_Sin": np.sin(2 * np.pi * month / 12),
            "Month_Cos": np.cos(2 * np.pi * month / 12),
            "Day_Sin": np.sin(2 * np.pi * day / 31),
            "Day_Cos": np.cos(2 * np.pi * day / 31),
            "Normal_MM": normal_mm
        }])

        estimate = max(0.0, float(model.predict(features)[0]))

        results.append({
            "date": current_date.strftime("%d %b %Y"),
            "weekday": current_date.strftime("%A"),
            "rainfall": round(estimate, 2),
            "normal": round(normal_mm, 2),
            "category": get_rainfall_category(estimate),
            "level": get_rainfall_level(estimate)
        })

    return results


@app.route("/")
def index():
    history = load_csv(HISTORY_FILE)
    flood = load_csv(FLOOD_STATE_FILE)
    metrics = load_json(METRICS_FILE)

    states_count = history["State"].nunique() if not history.empty else 0
    records_count = len(history)

    date_range = "Not available"
    if not history.empty and "Date" in history:
        dates = pd.to_datetime(history["Date"], errors="coerce").dropna()
        if not dates.empty:
            date_range = (
                f"{dates.min().strftime('%d %b %Y')} – "
                f"{dates.max().strftime('%d %b %Y')}"
            )

    flood_events = (
        int(flood["total_events"].sum())
        if not flood.empty else 0
    )

    return render_template(
        "index.html",
        states_count=states_count,
        records_count=records_count,
        flood_events=flood_events,
        date_range=date_range,
        selected_model=metrics.get("selected_model", "Not trained")
    )


@app.route("/rainfall", methods=["GET", "POST"])
def rainfall():
    states = get_states()
    predictions = None
    selected_state = ""
    selected_date = date.today().isoformat()
    error = None

    if request.method == "POST":
        selected_state = request.form.get("state", "").strip()
        selected_date = request.form.get("start_date", "").strip()

        try:
            if selected_state not in states:
                raise ValueError("Please select a valid state.")

            start = date.fromisoformat(selected_date)

            if start < date.today():
                raise ValueError("Please select today or a future date.")

            if start > date.today() + timedelta(days=30):
                raise ValueError(
                    "Please choose a date within the next 30 days."
                )

            if not MODEL_FILE.exists():
                raise ValueError(
                    "Model files are missing. Run python train.py first."
                )

            predictions = predict_seven_days(selected_state, start)

        except Exception as exc:
            error = str(exc)

    return render_template(
        "rainfall.html",
        states=states,
        predictions=predictions,
        selected_state=selected_state,
        selected_date=selected_date,
        error=error,
        today=date.today().isoformat(),
        max_date=(date.today() + timedelta(days=30)).isoformat()
    )


@app.route("/flood", methods=["GET", "POST"])
def flood():
    monthly = load_csv(FLOOD_MONTH_FILE)
    state_summary = load_csv(FLOOD_STATE_FILE)

    states = (
        sorted(state_summary["State"].dropna().unique().tolist())
        if not state_summary.empty else []
    )

    selected_state = request.values.get("state", "")
    selected_month = request.values.get("month", "all")

    if selected_state not in states and states:
        selected_state = states[0]

    state_info = {}
    monthly_counts = [0] * 12
    total_events = 0

    if selected_state and not state_summary.empty:
        match = state_summary[
            state_summary["State"] == selected_state
        ]

        if not match.empty:
            row = match.iloc[0]
            state_info = row.to_dict()
            total_events = int(row["total_events"])

    if selected_state and not monthly.empty:
        state_months = monthly[
            monthly["State"] == selected_state
        ]

        for _, row in state_months.iterrows():
            month_index = int(row["Month"]) - 1
            if 0 <= month_index < 12:
                monthly_counts[month_index] = int(row["event_count"])

    selected_month_count = None

    if selected_month != "all":
        try:
            month_number = int(selected_month)
            if 1 <= month_number <= 12:
                selected_month_count = monthly_counts[month_number - 1]
        except ValueError:
            selected_month = "all"

    return render_template(
        "flood.html",
        states=states,
        selected_state=selected_state,
        selected_month=selected_month,
        months=MONTHS,
        monthly_counts=monthly_counts,
        total_events=total_events,
        state_info=state_info,
        selected_month_count=selected_month_count,
        max_month_count=max(monthly_counts) if monthly_counts else 0
    )


@app.route("/data")
def data():
    history = load_csv(HISTORY_FILE)
    climate = load_csv(CLIMATE_FILE)
    metrics = load_json(METRICS_FILE)

    return render_template(
        "data.html",
        history_rows=len(history),
        climate_rows=len(climate),
        states_count=history["State"].nunique()
        if not history.empty else 0,
        metrics=metrics
    )


@app.errorhandler(404)
def page_not_found(error):
    return render_template(
        "base.html",
        error_message="The requested page was not found."
    ), 404


if __name__ == "__main__":
    app.run(debug=True)
