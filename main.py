from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np

app = FastAPI(title="AgroBridge Price Prediction API")

with open('agrobridge_price_model_v2.pkl', 'rb') as f:
    model = pickle.load(f)

with open('agrobridge_label_encoder_v2.pkl', 'rb') as f:
    le = pickle.load(f)

with open('agrobridge_features_v2.pkl', 'rb') as f:
    feature_columns = pickle.load(f)

# Load historical statistics
with open('agrobridge_historical_stats.pkl', 'rb') as f:
    historical_stats = pickle.load(f)


class PredictionInput(BaseModel):
    commodity: str
    state: str
    month: int
    year: int
    lag1: float
    lag3: float
    rolling_mean3: float
    min_dist_s: float = 160.25
    mean_dist_s: float = 659.08


class PredictionResponse(BaseModel):
    commodity: str
    state: str
    unit: str
    current_price: float
    previous_price: float
    highest_price_recorded: float
    lowest_price_recorded: float
    direction: str
    advice: str


@app.get("/")
def root():
    return {"message": "AgroBridge Price Prediction API is running 🌾"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: PredictionInput):
    input_data = pd.DataFrame(columns=feature_columns)
    input_data.loc[0] = 0

    input_data['MONTH'] = data.month
    input_data['YEAR'] = data.year
    input_data['LAG_1'] = data.lag1
    input_data['LAG_3'] = data.lag3
    input_data['ROLLING_MEAN_3'] = data.rolling_mean3
    input_data['MIN_DIST_S'] = data.min_dist_s
    input_data['MEAN_DIST_S'] = data.mean_dist_s

    commodity_col = f'COMMODITY_{data.commodity.capitalize()}'
    state_col = f'STATES_{data.state}'

    if commodity_col in input_data.columns:
        input_data[commodity_col] = 1

    if state_col in input_data.columns:
        input_data[state_col] = 1

    prediction = model.predict(input_data)
    direction = le.inverse_transform(prediction)[0]

    if direction == 'Rising':
        advice = "Good time to sell NOW"
    elif direction == 'Falling':
        advice = "Wait before selling"
    else:
        advice = "Prices are steady"

    commodity_stats = historical_stats.get(data.commodity.capitalize(), {})
    highest = commodity_stats.get("HIGHEST_PRICE", "N/A")
    lowest = commodity_stats.get("LOWEST_PRICE", "N/A")

    return {
        "commodity": data.commodity,
        "state": data.state,
        "unit": "per kg",
        "current_price": data.lag1,
        "previous_price": data.lag3,
        "highest_price_recorded": highest,
        "lowest_price_recorded": lowest,
        "direction": direction,
        "advice": advice
    }


@app.get("/best-time/{commodity}")
def best_time(commodity: str):
    best_months = {
        "Tomatoes": "September",
        "Yam": "June",
        "Maize": "July",
        "Onions": "November",
        "Rice (local)": "September",
        "Beans (red)": "June",
        "Groundnuts": "September",
        "Millet": "August",
        "Sorghum": "September"
    }

    month = best_months.get(commodity, "Data not available")

    return {
        "commodity": commodity,
        "best_month_to_sell": month
    }
