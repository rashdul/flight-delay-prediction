"""
Module for fetching and processing weather data from the Open-Meteo API.

Author: Rashed Aldulijan
Created: 2025-12-27

"""
import pandas as pd
from typing import List
import airportsdata

import os
from dotenv import load_dotenv


load_dotenv()



class Weather:
    

    def __init__(
        self,
        api_caller,
        airport_code: str,
        start_date: str,
        end_date: str,
        code_type: str = "iata",
        timezone: str = "Local",
        hourly_vars: List[str] | None = None,
    ):
        self.api_caller = api_caller
        self.airport_code = airport_code
        if code_type == "iata":
            airport = airportsdata.load("IATA").get(airport_code)
        else:
            airport = airportsdata.load("ICAO").get(airport_code)
        if airport is None:
            raise ValueError(f"Airport with ICAO code {airport_code} not found.")
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


