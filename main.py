from dotenv import load_dotenv
from FlightWeather.src.FlightWeather import FlightWeather
from fastapi import FastAPI, Query, Depends, HTTPException
import joblib
import os

from features.src.feature_engineering_depDelay import FeatureEngineeringDepDelay
from schemas.flightDate import FlightQuery
from models.src.DelayClassifier import DelayClassifier
from openai_summarizer.src.openai_summarizer import summarize_delay_prediction, generate_delay_risk_message
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://rdulaijan.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


load_dotenv()

MODEL_PATH = "./models/delay_classifier_final.pkl"

model = None


def get_model():
    global model
    if model is not None:
        return model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Deploy the .pkl alongside the app or update MODEL_PATH."
        )
    model = joblib.load(MODEL_PATH)
    return model

def predict_delay_by_flight(flight_data):
    loaded_model = get_model()
    prediction = loaded_model.predict(flight_data)
    probability = loaded_model.predict_proba(flight_data)
    proba_row = probability[0]

    prob_scalar = None
    try:
        if hasattr(loaded_model, "classes_"):
            classes = list(getattr(loaded_model, "classes_"))
            if 1 in classes:
                prob_scalar = float(proba_row[classes.index(1)])
            else:
                prob_scalar = float(max(proba_row))
        else:
            # Assume binary classification: second column is typically the positive class.
            if hasattr(proba_row, "__len__") and len(proba_row) > 1:
                prob_scalar = float(proba_row[1])
            else:
                prob_scalar = float(proba_row)
    except Exception:
        # Best-effort: keep the raw value; the caller can handle None.
        prob_scalar = None

    print(f"Prediction: {prediction}, Probability: {probability}")
    return prediction[0], prob_scalar

@app.get("/")
def read_root():
    return {"message": "Hello FastAPI 🚀"}


@app.get("/flightsWeather")
def get_flights_weather(
    flight_prefix: str = Query(..., min_length=2, max_length=2, description="Flight number prefix, e.g., 'UA' for United Airlines"),
    flight_number: str = Query(..., min_length=1, max_length=4, description="Flight number, e.g., '2012'"),
    flight_date: str = Query(..., regex=r"^\d{4}-\d{2}-\d{2}$", description="Flight date in YYYY-MM-DD format"),
):
    flightWeather = FlightWeather(flight_num=f"{flight_prefix} {flight_number}", flight_date=flight_date)
    flight_df = flightWeather.get_full_data()
    return {"success": True,
             "data": flight_df.to_dict(orient="records"),
             "metadata": {
                 "num_records": len(flight_df)}}


@app.get("/predictDelay", status_code=200)
def predict_delay(
    flight_prefix: str = Query(..., min_length=2, max_length=2, description="Flight number prefix, e.g., 'UA' for United Airlines"),
    flight_number: str = Query(..., min_length=1, max_length=4, description="Flight number, e.g., '2012'"),
    flight_date: FlightQuery = Depends(),
    summarize: bool = Query(
        True,
        description="If true, include an OpenAI-generated summary of the prediction and key flight/weather inputs (requires OPENAI_API_KEY).",
    ),
    summarize_debug: bool = Query(
        False,
        description="If true, append the underlying OpenAI error message when the summary falls back.",
    ),
):
    flight_date_str = flight_date.flight_date.isoformat()
    # print(
    #     f"Fetching flight and weather data for Flight Number {flight_prefix} {flight_number} and date {flight_date_str}"
    # )

    try:
        flight_weather = FlightWeather(
            flight_num=f"{flight_prefix} {flight_number}",
            flight_date=flight_date_str,
        )
        flight_df = flight_weather.get_full_data()
    except RuntimeError as e:
        # Upstream API error (Aeromarket/OpenMeteo/etc.)
        message = str(e)
        # Aerodatabox proxy sometimes returns 204 when a flight isn't found for that date.
        if "status=204" in message:
            raise HTTPException(
                status_code=404,
                detail="No flight data returned for that flight/date (upstream returned 204 No Content).",
            ) from e

        raise HTTPException(status_code=502, detail=message) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to fetch flight/weather data") from e

    try:
        flight_df_engineered = FeatureEngineeringDepDelay(flight_df, type="new").engineer_features()
        prediction, probability = predict_delay_by_flight(flight_df_engineered)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to run delay prediction") from e

    response = {
        "success": True,
        "data": {
            "prediction": int(prediction),
            "probability": float(probability) if probability is not None else None,
            "flight_info": flight_df.to_dict(orient="records"),
        },
    }

    if summarize:
        response["data"]["summary"] = summarize_delay_prediction(
            prediction=int(prediction),
            probability=probability,
            flight_info=response["data"]["flight_info"],
            flight_number=f"{flight_prefix} {flight_number}",
            flight_date=flight_date_str,
        )

    if int(prediction) == 1:
        response["data"]["risk_message"] = generate_delay_risk_message(
            prediction=int(prediction),
            probability=probability,
            flight_info=response["data"]["flight_info"],
            flight_number=f"{flight_prefix} {flight_number}",
            flight_date=flight_date_str,
        )

    return response