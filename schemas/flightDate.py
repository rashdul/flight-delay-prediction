from datetime import date, timedelta
from pydantic import BaseModel, field_validator
from fastapi import Depends

class FlightQuery(BaseModel):
    flight_date: date  # auto-parses YYYY-MM-DD

    @field_validator("flight_date")
    @classmethod
    def validate_flight_date(cls, v: date):
        today = date.today()
        max_date = today + timedelta(days=14)

        if v < today:
            raise ValueError("flight_date cannot be in the past")

        if v > max_date:
            raise ValueError("flight_date must be within the next 14 days")

        return v
