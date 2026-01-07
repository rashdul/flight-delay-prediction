

import os

from openmeteo_api.src.openmeteoapi.APICaller import APICaller as OpenMeteoAPICaller
from openmeteo_api.src.openmeteoapi.WeatherData import Weather
import pandas as pd
from aeromarket_api.src.aeroapi_market.Flights import Flights
from aeromarket_api.src.aeroapi_market.APICaller import APICaller
from dotenv import load_dotenv
from datetime import date, datetime, timedelta

load_dotenv()
class FlightWeather:

  def __init__(self, flight_num: str, flight_date: str):
    if not flight_num:
      raise ValueError("flight_num must be provided")
    if not flight_date:
      raise ValueError("flight_date must be provided")
    if isinstance(flight_num, str) is False or isinstance(flight_date, str) is False:
      raise TypeError("flight_num and flight_date must be strings")
    self.flight_num = flight_num
    self.flight_date = flight_date
    self.flight_data = None
    self.weather_data = None
    is_today_or_future = date.fromisoformat(flight_date) >= date.today()
    if is_today_or_future:
      self.time = "future"
    else:
      self.time = "past"


  def _getFlight(self):
    if self.flight_data is not None:
      return self.flight_data
    API_KEY = os.getenv("AERODATABOX_API_KEY")
    BASE_URL = os.getenv("AERODATABOX_BASE_URL")
    api_caller = APICaller(BASE_URL, API_KEY)
    flight_bt = Flights(api_caller, 
        from_local=self.flight_date,
        flight_number=self.flight_num,
        )
    flight_bt.get_airport_flights(code_type="iata")
    self.flight_data = flight_bt.build_final_flight_response()
    return self.flight_data

  def _getWeather(self):
    api_caller_weather = OpenMeteoAPICaller()
    airport_codes = [self.flight_data.iloc[0]['Origin'], self.flight_data.iloc[0]['Dest']]
    start_date = self.flight_data.iloc[0]['dep_date_local']
    end_date = (start_date + timedelta(days=3))
    weather_bt = Weather(
        api_caller=api_caller_weather,
        airport_code=airport_codes,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        code_type="IATA",
        time=self.time,
    )
    self.weather_data = weather_bt.to_hourly_dataframe()['data']
    return self.weather_data

  def get_full_data(self):
    self._getFlight()
    self._getWeather()
    prefix_dep = "dep_"
    prefix_arr = "arr_"
    weather_dep = self.weather_data[self.weather_data['queried_airport_code'] == self.flight_data.iloc[0]['Origin']].copy()
    weather_dep = weather_dep.add_prefix(prefix_dep)
    weather_arr = self.weather_data[self.weather_data['queried_airport_code'] == self.flight_data.iloc[0]['Dest']].copy()
    weather_arr = weather_arr.add_prefix(prefix_arr)

    # print("flight_data columns:", self.flight_data.columns)


    self.flight_data['arr_datetime'] = (
        self.flight_data['arr_datetime']
        .dt.tz_localize(None)
    )
    self.flight_data['dep_date_local'] = (
        self.flight_data['dep_date_local']
        .dt.tz_localize(None)
    )

    weather_arr['arr_date'] = pd.to_datetime(weather_arr['arr_date'], errors='coerce')
    weather_dep['dep_date'] = pd.to_datetime(weather_dep['dep_date'], errors='coerce')

    weather_arr['arr_date'] = weather_arr['arr_date'].dt.tz_localize(None)
    weather_dep['dep_date'] = weather_dep['dep_date'].dt.tz_localize(None)


    merged_df = self.flight_data.merge(weather_dep, left_on=['Origin', 'dep_date_local'], right_on=['dep_queried_airport_code', 'dep_date'], how='left')
    merged_df = merged_df.merge(weather_arr, left_on=['Dest', 'arr_datetime'], right_on=['arr_queried_airport_code', 'arr_date'], how='left')
    
    columns_to_drop = [
        'dep_queried_airport_code', 'arr_queried_airport_code', 'dep_date', 'arr_date', ]
    merged_df = merged_df.drop(columns=columns_to_drop)
    return merged_df
