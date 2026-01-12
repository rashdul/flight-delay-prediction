import os
from typing import Any, Dict, Iterable, Optional
from openai import OpenAI


def _as_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _safe_first_record(flight_info: Any) -> Dict[str, Any]:
    if isinstance(flight_info, dict):
        return flight_info
    if isinstance(flight_info, list) and flight_info:
        first = flight_info[0]
        if isinstance(first, dict):
            return first
    return {}


def _select_keys(record: Dict[str, Any], keys: Iterable[str]) -> Dict[str, Any]:
    return {k: record.get(k) for k in keys if k in record}


def _select_weather(record: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    wanted_suffixes = [
        "temperature_2m",
        "apparent_temperature",
        "precipitation",
        "rain",
        "snowfall",
        "relative_humidity_2m",
        "cloudcover",
        "visibility",
        "windspeed_10m",
        "windgusts_10m",
        "winddirection_10m",
        "pressure_msl",
    ]

    out: Dict[str, Any] = {}
    for suffix in wanted_suffixes:
        key = f"{prefix}{suffix}"
        if key in record:
            out[key] = record.get(key)
    return out


def _fallback_summary(
    *,
    flight_number: str,
    flight_date: str,
    origin: Optional[str],
    dest: Optional[str],
    prediction: int,
    probability: Optional[float],
) -> str:
    label = "likely delayed" if int(prediction) == 1 else "likely on time"
    if probability is None:
        prob_txt = ""
    else:
        prob_txt = f" (model confidence {probability:.0%})"

    route = ""
    if origin and dest:
        route = f" from {origin} to {dest}"

    return (
        f"For flight {flight_number}{route} on {flight_date}, the model predicts it is {label}{prob_txt}. "
        "Enable `OPENAI_API_KEY` for a richer, data-driven narrative summary."
    )


def _fallback_risk_message(
    *,
    flight_number: str,
    flight_date: str,
    origin: Optional[str],
    dest: Optional[str],
    probability: Optional[float],
) -> str:
    route = ""
    if origin and dest:
        route = f" ({origin}→{dest})"

    prob_txt = ""
    if probability is not None:
        prob_txt = f" (confidence {probability:.0%})"

    return (
        f"Delay risk flagged for flight {flight_number}{route} on {flight_date}{prob_txt}. "
        "Consider checking for updates and allowing extra buffer time."
    )


def generate_delay_risk_message(
    *,
    prediction: int,
    probability: Any,
    flight_info: Any,
    flight_number: str,
    flight_date: str,
) -> Optional[str]:
    """Generate a short, user-facing 'risk message' when prediction == 1.

    If `OPENAI_API_KEY` is not set (or OpenAI call fails), returns a deterministic fallback.
    If prediction != 1, returns None.

    Env vars:
      - OPENAI_API_KEY: required to call OpenAI
      - OPENAI_RISK_MODEL: optional (default: gpt-4o-mini)
      - OPENAI_BASE_URL: optional (for proxies/compatible providers)
      - OPENAI_TIMEOUT_SECONDS: optional (default: 20)
    """

    if int(prediction) != 1:
        return None

    record = _safe_first_record(flight_info)
    origin = record.get("Origin")
    dest = record.get("Dest")
    p = _as_float(probability)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _fallback_risk_message(
            flight_number=flight_number,
            flight_date=flight_date,
            origin=origin,
            dest=dest,
            probability=p,
        )

    model = os.getenv("OPENAI_RISK_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    base_url = os.getenv("OPENAI_BASE_URL")
    timeout_s = _as_float(os.getenv("OPENAI_TIMEOUT_SECONDS", "20")) or 20.0

    core = _select_keys(
        record,
        keys=[
            "Origin",
            "Dest",
            "Actual_Departure_Time",
            "Actual_Arrival_Time",
            "dep_scheduled_congestion",
            "arr_scheduled_congestion",
        ],
    )

    payload = {
        "flight_number": flight_number,
        "flight_date": flight_date,
        "probability": p,
        "flight_core": core,
    }

    system = (
        "You are a backend component that writes a short travel risk note when a delay risk is flagged.\n"
        "Rules:\n"
        "- Output 1–2 sentences, plain text only.\n"
        "- Be cautious and non-alarmist.\n"
        "- You may suggest generic actions (check status, allow buffer time).\n"
        "- Only reference details present in the input data.\n"
        "- Do not mention models, thresholds, or internal logic.\n"
        "- No emojis or markdown."
    )

    user = f"Write the risk message based on this data: {payload}"

    try:
        client_kwargs: Dict[str, Any] = {"api_key": api_key, "timeout": timeout_s}
        if base_url:
            client_kwargs["base_url"] = base_url

        client = OpenAI(**client_kwargs)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            max_tokens=80,
        )

        text = (resp.choices[0].message.content or "").strip()
        if not text:
            raise RuntimeError("Empty risk message returned")
        return text
    except Exception:
        return _fallback_risk_message(
            flight_number=flight_number,
            flight_date=flight_date,
            origin=origin,
            dest=dest,
            probability=p,
        )


def summarize_delay_prediction(
    *,
    prediction: int,
    probability: Any,
    flight_info: Any,
    flight_number: str,
    flight_date: str,
) -> str:
    """Summarize the model prediction + key flight/weather inputs.

    If `OPENAI_API_KEY` is not set (or OpenAI call fails), returns a deterministic fallback summary.

    Env vars:
      - OPENAI_API_KEY: required to call OpenAI
      - OPENAI_MODEL: optional (default: gpt-4o-mini)
      - OPENAI_BASE_URL: optional (for proxies/compatible providers)
      - OPENAI_TIMEOUT_SECONDS: optional (default: 20)
    """

    record = _safe_first_record(flight_info)

    core = _select_keys(
        record,
        keys=[
            "Origin",
            "Dest",
            "Distance",
            "CRSElapsedTime",
            "dep_scheduled_congestion",
            "arr_scheduled_congestion",
            "Actual_Departure_Time",
            "Actual_Arrival_Time",
        ],
    )

    origin = record.get("Origin")
    dest = record.get("Dest")

    p = _as_float(probability)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _fallback_summary(
            flight_number=flight_number,
            flight_date=flight_date,
            origin=origin,
            dest=dest,
            prediction=int(prediction),
            probability=p,
        )

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    base_url = os.getenv("OPENAI_BASE_URL")
    timeout_s = _as_float(os.getenv("OPENAI_TIMEOUT_SECONDS", "20")) or 20.0

    dep_weather = _select_weather(record, prefix="dep_")
    arr_weather = _select_weather(record, prefix="arr_")

    payload = {
        "flight_number": flight_number,
        "flight_date": flight_date,
        "prediction": int(prediction),
        "probability": p,
        "flight_core": core,
        "departure_weather": dep_weather,
        "arrival_weather": arr_weather,
    }

    system = (
    """
    You are a factual, conservative summarization component used inside a backend API.

    Your task is to produce a short, clear summary of the provided structured data for display to non-technical users.

    Rules:
    - Only state information that is explicitly present in the input.
    - state model prediction as the is a 1 hour risk of delay or is not at risk of delay.
    - Do not infer causes, impacts, or outcomes.
    - Do not apply external domain knowledge or general assumptions.
    - Do not describe conditions qualitatively (e.g., favorable, mild, severe, smooth) unless explicitly stated in the input.
    - Use simple, non-technical language.
    - Generate 1 paragraph.
    - Avoid emojis, markdown, or stylistic formatting.
    - Do not mention models, thresholds, probabilities, predictions, or internal logic unless explicitly included in the input.
    - If information is missing or uncertain, state that clearly without guessing.
    - The summary must be understandable to non-technical users.

    Congestion handling (explicit rules):
    - Do not state numeric congestion values.
    - If a scheduled congestion value is 0–30, describe the airport as "not busy".
    - If a scheduled congestion value is 31–70, describe the airport as "moderately busy".
    - If a scheduled congestion value is above 70, describe the airport as "busy".
    - These descriptions must be based only on the numeric values provided.
    - Do not suggest impact, delays, or outcomes.

    Output:
    - Return a single, neutral paragraph suitable for direct UI display.
    """.strip()
    )


    user = (
        "Write a short summary explaining what the model predicts and why, "
        "based on the structured flight + weather data below. "
        "If probability is missing, omit it.\n\n"
        f"DATA: {payload}"
    )

    try:

        client_kwargs: Dict[str, Any] = {"api_key": api_key, "timeout": timeout_s}
        if base_url:
            client_kwargs["base_url"] = base_url

        client = OpenAI(**client_kwargs)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            max_tokens=200,
        )

        text = (resp.choices[0].message.content or "").strip()
        if not text:
            raise RuntimeError("Empty summary returned")
        return text
    except Exception:
        return _fallback_summary(
            flight_number=flight_number,
            flight_date=flight_date,
            origin=origin,
            dest=dest,
            prediction=int(prediction),
            probability=p,
        )
