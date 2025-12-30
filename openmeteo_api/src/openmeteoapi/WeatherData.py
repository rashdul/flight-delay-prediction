"""
Module for fetching and processing weather data from the Open-Meteo API.

Author: Rashed Aldulijan
Created: 2025-12-27

"""
import pandas as pd
from typing import List, Sequence
from datetime import timedelta, timezone as dt_timezone

import requests
import airportsdata

import os
from dotenv import load_dotenv



load_dotenv()



class Weather:
    

    def __init__(
        self,
        api_caller,
        airport_code: str | Sequence[str],
        start_date: str,
        end_date: str,
        code_type: str = "iata",
        timezone: str = "auto",
        hourly_vars: List[str] | None = None,
    ):
        self.api_caller = api_caller
        if isinstance(airport_code, str):
            airport_codes = [c.strip().upper() for c in airport_code.split(",") if c.strip()]
        else:
            airport_codes = [str(c).strip().upper() for c in airport_code if str(c).strip()]

        if not airport_codes:
            raise ValueError("airport_code must contain at least one airport code")

        self.airport_codes = airport_codes
        self.airport_code = ",".join(airport_codes)

        normalized_code_type = code_type.strip().lower()
        if normalized_code_type not in ["iata", "icao"]:
            raise ValueError("code_type must be 'iata' or 'icao'")

        self.latitude, self.longitude = self._initialize_locations(normalized_code_type)
        self.start_date = start_date
        self.end_date = end_date
        if timezone not in ["auto", "GMT"]:
            raise ValueError("timezone must be 'auto' or 'GMT'")
        self.timezone = timezone
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

    def _initialize_locations(self, code_type: str):
        lats = []
        lons = []
        for code in self.airport_codes:
            if code_type == "iata":
                airport = airportsdata.load("IATA").get(code)
            else:
                airport = airportsdata.load("ICAO").get(code)
            if airport is None:
                lat, lon = self._handle_missing_lat_lon(code)
                if lat is None or lon is None:
                    raise ValueError(f"Airport with {code_type} code {code} not found.")
                lats.append(lat)
                lons.append(lon)
            else:
                lats.append(airport["lat"])
                lons.append(airport["lon"])
        return lats, lons
    

    def _handle_missing_lat_lon(self, code):
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            "q": f"{code} airport",
            "format": "json"
        }

        headers = {
            "User-Agent": "flight-delay-research/1.0 (rashidulaijan@gmail.com)"
        }

        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()  

        data = resp.json()

        lat = float(data[0]["lat"])
        lon = float(data[0]["lon"])

        return lat, lon

    def _fetch_all(self):
        lats = ",".join(map(str, self.latitude))
        lons = ",".join(map(str, self.longitude))
        params = {
            "latitude": lats,
            "longitude": lons,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "hourly": self.hourly_vars,
            "timezone": self.timezone,
        }
        ARCHIVE_URL = os.getenv("ARCHIVE_URL")

        return self.api_caller.get(ARCHIVE_URL, params)

    def fetch(self):
        responses = self._fetch_all()
        return responses

    def _hourly_response_to_dataframe(self, response, queried_airport_code: str) -> pd.DataFrame:
        hourly = response.Hourly()

        start_utc = pd.to_datetime(hourly.Time(), unit="s", utc=True)
        end_utc = pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True)
        interval = pd.Timedelta(seconds=hourly.Interval())

        time_index = pd.date_range(
            start=start_utc,
            end=end_utc,
            freq=interval,
            inclusive="left",
        )

        if self.timezone == "auto":
            tz_name = response.Timezone() if hasattr(response, "Timezone") else None
            if isinstance(tz_name, (bytes, bytearray)):
                tz_name = tz_name.decode("utf-8", errors="ignore")
            if isinstance(tz_name, str):
                tz_name = tz_name.strip().strip("\x00")

            if tz_name:
                try:
                    time_index = time_index.tz_convert(tz_name)
                except Exception:
                    tz_name = None

            if not tz_name:
                utc_offset_seconds = (
                    response.UtcOffsetSeconds()
                    if hasattr(response, "UtcOffsetSeconds")
                    else 0
                )
                fixed_tz = dt_timezone(timedelta(seconds=int(utc_offset_seconds)))
                time_index = time_index.tz_convert(fixed_tz)

        data = {"date": time_index}

        variables_len = hourly.VariablesLength() if hasattr(hourly, "VariablesLength") else 0
        for idx, var_name in enumerate(self.hourly_vars):
            if idx < variables_len:
                values = hourly.Variables(idx).ValuesAsNumpy()
                data[var_name] = (
                    pd.Series(values).reindex(range(len(time_index))).to_numpy()
                )
            else:
                data[var_name] = pd.Series(index=range(len(time_index)), dtype="float64").to_numpy()

        df = pd.DataFrame(data)
        df["queried_airport_code"] = queried_airport_code
        return df

    def to_hourly_dataframe(self) -> pd.DataFrame:
        responses = self._fetch_all()
        if len(responses) <= 1:
            return self._hourly_response_to_dataframe(responses[0], self.airport_codes[0])

        dfs = []
        for idx, response in enumerate(responses):
            code = self.airport_codes[idx] if idx < len(self.airport_codes) else self.airport_code
            dfs.append(self._hourly_response_to_dataframe(response, code))
        return pd.concat(dfs, ignore_index=True)
    
    def to_daily_dataframe(self) -> pd.DataFrame:
        responses = self._fetch_all()
        if len(responses) <= 1:
            response = responses[0]
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

            df = pd.DataFrame(data)
            df["queried_airport_code"] = self.airport_codes[0]
            return df

        dfs = []
        for idx, response in enumerate(responses):
            daily = response.Daily()
            time_index = pd.date_range(
                start=pd.to_datetime(daily.Time(), unit="s", utc=True),
                end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
                freq="D",
                inclusive="left",
            )

            data = {"date": time_index}
            for j, var_name in enumerate(self.hourly_vars):
                data[var_name] = daily.Variables(j).ValuesAsNumpy()
            df = pd.DataFrame(data)
            code = self.airport_codes[idx] if idx < len(self.airport_codes) else self.airport_code
            df["queried_airport_code"] = code
            dfs.append(df)

        return pd.concat(dfs, ignore_index=True)
