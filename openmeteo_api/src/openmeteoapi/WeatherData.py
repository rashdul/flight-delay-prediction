"""
Module for fetching and processing weather data from the Open-Meteo API.

Author: Rashed Aldulijan
Created: 2025-12-27

"""
import pandas as pd
from typing import List, Optional, Tuple, Dict, Any
import airportsdata
import math

import os
from dotenv import load_dotenv


load_dotenv()



class Weather:
    
    @staticmethod
    def _haversine_distance_km(
        lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        r = 6371.0088
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)

        a = (
            math.sin(dphi / 2) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        )
        return 2 * r * math.asin(math.sqrt(a))

    @staticmethod
    def find_nearest_airport(
        latitude: float,
        longitude: float,
        code_type: str = "iata",
        max_distance_km: Optional[float] = None,
    ) -> Tuple[str, Dict[str, Any], float]:
        """
        Find the nearest airport in `airportsdata` to the given coordinates.

        Returns:
            (airport_code, airport_dict, distance_km)
        """
        code_type_upper = code_type.strip().upper()
        if code_type_upper not in {"IATA", "ICAO"}:
            raise ValueError("code_type must be 'iata' or 'icao'.")

        airports = airportsdata.load(code_type_upper)
        best_code: Optional[str] = None
        best_airport: Optional[Dict[str, Any]] = None
        best_distance: float = float("inf")

        for airport_code, airport in airports.items():
            dist = Weather._haversine_distance_km(
                latitude, longitude, float(airport["lat"]), float(airport["lon"])
            )
            if dist < best_distance:
                best_distance = dist
                best_code = airport_code
                best_airport = airport

        if best_code is None or best_airport is None:
            raise ValueError("No airports available in airportsdata dataset.")

        if max_distance_km is not None and best_distance > max_distance_km:
            raise ValueError(
                f"No airport within {max_distance_km} km (nearest is {best_distance:.2f} km: {best_code})."
            )

        return best_code, best_airport, best_distance

    def __init__(
        self,
        api_caller,
        airport_code: str,
        start_date: str,
        end_date: str,
        code_type: str = "iata",
        timezone: str = "Local",
        hourly_vars: List[str] | None = None,
        fallback_latitude: Optional[float] = None,
        fallback_longitude: Optional[float] = None,
        nearest_max_distance_km: Optional[float] = None,
    ):
        self.api_caller = api_caller
        self.airport_code = airport_code
        self.resolved_airport_code = airport_code

        code_type_lower = code_type.strip().lower()
        if code_type_lower == "iata":
            airport = airportsdata.load("IATA").get(airport_code)
        elif code_type_lower == "icao":
            airport = airportsdata.load("ICAO").get(airport_code)
        else:
            raise ValueError("code_type must be 'iata' or 'icao'.")

        if airport is None:
            if fallback_latitude is None or fallback_longitude is None:
                raise ValueError(
                    f"Airport with {code_type_lower} code {airport_code} not found in airportsdata. "
                    "Provide fallback_latitude/fallback_longitude to use the nearest airport instead."
                )

            nearest_code, nearest_airport, _ = Weather.find_nearest_airport(
                latitude=float(fallback_latitude),
                longitude=float(fallback_longitude),
                code_type=code_type_lower,
                max_distance_km=nearest_max_distance_km,
            )
            self.resolved_airport_code = nearest_code
            airport = nearest_airport

        self.latitude = airport["lat"]
        self.longitude = airport["lon"]
        self.start_date = start_date
        self.end_date = end_date
        if timezone == "GMT":
            self.timezone = timezone
        else:
            self.timezone = airport["tz"]
        self.hourly_vars = hourly_vars or [
            "snowfall",
            "rain",
            "precipitation",
            "wind_speed_10m",
            "wind_gusts_10m",
            "cloud_cover_low",
            "cloud_cover",
            "temperature_2m",
            "apparent_temperature",
            "surface_pressure",
            "relative_humidity_2m",
            "pressure_msl",
        ]

    def fetch(self):
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "hourly": self.hourly_vars,
            "timezone": self.timezone,
        }
        ARCHIVE_URL = os.getenv("ARCHIVE_URL")

        responses = self.api_caller.get(ARCHIVE_URL, params)
        return responses[0]

    def to_hourly_dataframe(self) -> pd.DataFrame:
        response = self.fetch()
        hourly = response.Hourly()

        # time_index = pd.date_range(
        #     start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        #     end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        #     freq=pd.Timedelta(seconds=hourly.Interval()),
        #     inclusive="left",
        # )
        if self.timezone == "GMT":
            time_index = pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left",
            )
        else:
            time_index = pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True).tz_convert(self.timezone),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True).tz_convert(self.timezone),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left",
            )

        data = {"date": time_index}

        for idx, var_name in enumerate(self.hourly_vars):
            data[var_name] = hourly.Variables(idx).ValuesAsNumpy()
        df = pd.DataFrame(data)
        df['queried_airport_code'] = self.airport_code
        df['resolved_airport_code'] = self.resolved_airport_code
        # df['flight_date'] = pd.to_datetime(df['date']).dt.tz_convert(self.timezone)
        # print(self.timezone)

        return df
    
    def to_daily_dataframe(self) -> pd.DataFrame:
        response = self.fetch()
        daily = response.Daily()

        time_index = pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s", utc=True),
            end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
            freq="D",
            inclusive="left",
        )

        data = {"date": time_index}

        for idx, var_name in enumerate(self.hourly_vars):
            data[var_name] = daily.Variables(idx).ValuesAsNumpy()

        return pd.DataFrame(data)

