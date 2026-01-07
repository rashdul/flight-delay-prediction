


import pandas as pd
from datetime import date
import pickle
import numpy as np

# -*- coding: utf-8 -*-


class FeatureEngineeringDepDelay:

  def __init__(self, df: pd.DataFrame, classification: bool = False, type: str = "old") -> None:
      self.df = df.copy()
      self.classification = classification
      if type not in ["old", "new"]:
          raise ValueError("type must be either 'old' or 'new'")
      self.type = type

  def _is_peak_hour(self, hour: int) -> int:
      # Define peak hours (e.g., 6-9 AM and 4-7 PM)
      if (6 <= hour <= 9) or (16 <= hour <= 19):
          return 1
      return 0

  def _seperate_time_features(self) -> pd.DataFrame:

      # convert departure time to separate features
      self.df['dep_date_local'] = pd.to_datetime(self.df['dep_date_local'])
      self.df['departure_hour'] = self.df['dep_date_local'].dt.hour
      self.df['departure_day'] = self.df['dep_date_local'].dt.day
      self.df['departure_month'] = self.df['dep_date_local'].dt.month
      self.df['departure_weekday'] = self.df['dep_date_local'].dt.weekday
      self.df['departure_is_weekend'] = self.df['departure_weekday'].apply(lambda x: 1 if x >=5 else 0)
      self.df['departure_is_peak_hour'] = self.df['departure_hour'].apply(self._is_peak_hour)

      # convert arrival time to separate features
      self.df['arr_datetime'] = pd.to_datetime(self.df['arr_datetime'])
      self.df['arrival_hour'] = self.df['arr_datetime'].dt.hour
      self.df['arrival_day'] = self.df['arr_datetime'].dt.day
      self.df['arrival_month'] = self.df['arr_datetime'].dt.month
      self.df['arrival_weekday'] = self.df['arr_datetime'].dt.weekday
      self.df['arrival_is_weekend'] = self.df['arrival_weekday'].apply(lambda x: 1 if x >=5 else 0)
      self.df['arrival_is_peak_hour'] = self.df['arrival_hour'].apply(self._is_peak_hour)

      self.df.drop(columns=['dep_date_local'], inplace=True)
      self.df.drop(columns=['arr_datetime'], inplace=True)

      return self.df

  def _add_old_stats_features(self) -> pd.DataFrame:
      """
      Create features based on airline delay statistics.
      """
      df = self.df
      airline_stats = self._open_pickle("./data/airline_stats.pkl")

      df = df.merge(
          airline_stats,
          how="left",
          left_on="IATA_Code_Operating_Airline",
          right_index=True,
      )

      df["on_time_rate"] = df["on_time_rate"].fillna(airline_stats.loc["__UNKNOWN__", "on_time_rate"])
      df["average_delay"] = df["average_delay"].fillna(airline_stats.loc["__UNKNOWN__", "average_delay"])
      df["delay_stddev"] = df["delay_stddev"].fillna(airline_stats.loc["__UNKNOWN__", "delay_stddev"])

      self.df = df
      return self.df

  def _add_stats_features(self) -> pd.DataFrame:
      """
      Create features based on airline delay statistics.
      """
      if self.type == "new":
        return self._add_old_stats_features()
      df = self.df
      airline_stats = (
          df
          .groupby("IATA_Code_Operating_Airline")
          .agg(
              total_flights=("DepDelayMinutes", "count"),
              on_time_flights=("DepDelayMinutes", lambda x: (x <= 10).sum()),
              average_delay=("DepDelayMinutes", "mean"),
              delay_stddev=("DepDelayMinutes", "std"),
              )
              )
      airline_stats["on_time_rate"] = (
          airline_stats["on_time_flights"] / airline_stats["total_flights"]
          )
      global_stats = {
          "on_time_rate": (df["DepDelayMinutes"] <= 10).mean(),
          "average_delay": df["DepDelayMinutes"].mean(),
          "delay_stddev": df["DepDelayMinutes"].std(),
          }
      airline_stats = airline_stats.drop(columns=["total_flights", "on_time_flights"])

      airline_stats.loc["__UNKNOWN__", :] = pd.Series(global_stats)

      self._save_pickle(airline_stats, "./data/airline_stats.pkl")

      df = df.merge(
          airline_stats,
            how="left",
            left_on="IATA_Code_Operating_Airline",
            right_index=True,
      )
      self.df = df
      return self.df


  def _add_old_congestion_features(self) -> pd.DataFrame:
      """
      Create features based on airport congestion levels.
      """
      df = self.df
      congestion_quantiles = self._open_pickle("./data/congestion_quantiles.pkl")
      dep_quantiles = congestion_quantiles['dep_quantiles']
      arr_quantiles = congestion_quantiles['arr_quantiles']

      df['is_high_dep_congestion'] = df['dep_scheduled_congestion'].apply(lambda x: 1 if x >= dep_quantiles[2] else 0)
      df['is_high_arr_congestion'] = df['arr_scheduled_congestion'].apply(lambda x: 1 if x >= arr_quantiles[2] else 0)

      df['dep_congestion_bucket'] = df['dep_scheduled_congestion'].apply(lambda x: self._define_congestion_bucket(x, dep_quantiles))
      df['arr_congestion_bucket'] = df['arr_scheduled_congestion'].apply(lambda x: self._define_congestion_bucket(x, arr_quantiles))
      self.df = df
      return self.df

  def _add_congestion_features(self) -> pd.DataFrame:
      """
      Create features based on airport congestion levels.

      """
      if self.type == "new":
        return self._add_old_congestion_features()
      dep_quantiles = self.df['dep_scheduled_congestion'].quantile([0.25, 0.75, 0.9]).to_list()
      arr_quantiles = self.df['arr_scheduled_congestion'].quantile([0.25, 0.75, 0.9]).to_list()
      _save_path = "./data/congestion_quantiles.pkl"
      self._save_pickle({'dep_quantiles': dep_quantiles, 'arr_quantiles': arr_quantiles}, _save_path)
      self.df['is_high_dep_congestion'] = self.df['dep_scheduled_congestion'].apply(lambda x: 1 if x >= dep_quantiles[2] else 0)
      self.df['is_high_arr_congestion'] = self.df['arr_scheduled_congestion'].apply(lambda x: 1 if x >= arr_quantiles[2] else 0)

      self.df['dep_congestion_bucket'] = self.df['dep_scheduled_congestion'].apply(lambda x: self._define_congestion_bucket(x, dep_quantiles))
      self.df['arr_congestion_bucket'] = self.df['arr_scheduled_congestion'].apply(lambda x: self._define_congestion_bucket(x, arr_quantiles))
      return self.df
    
    
  def _add_old_route_features(self) -> pd.DataFrame:
      """
      Create features based on route statistics.
      """
      df = self.df
      route_stats = self._open_pickle("./data/route_stats.pkl")

      df["route"] = df["Origin"] + "-" + df["Dest"]
      df = df.merge(
          route_stats,
          how="left",
          left_on="route",
          right_index=True,
      )

      df["average_route_delay"] = df["average_route_delay"].fillna(route_stats.loc["__UNKNOWN__", "average_route_delay"])
      df["delay_route_stddev"] = df["delay_route_stddev"].fillna(route_stats.loc["__UNKNOWN__", "delay_route_stddev"])

      self.df = df
      return self.df


  def _add_route_features(self) -> pd.DataFrame:
      """
      Create features based on route statistics.
      """
      if self.type == "new":
        return self._add_old_route_features()
      df = self.df
      df['route'] = df['Origin'] + '-' + df['Dest']
      airline_stats = (
          df
          .groupby("route")
          .agg(
              average_route_delay=("DepDelayMinutes", "mean"),
              delay_route_stddev=("DepDelayMinutes", "std"),
              )
              .fillna(0)
              )
      global_stats = {
          "average_route_delay": df["DepDelayMinutes"].mean(),
          "delay_route_stddev": df["DepDelayMinutes"].std(),
          }

      airline_stats.loc["__UNKNOWN__", :] = pd.Series(global_stats)

      self._save_pickle(airline_stats, "./data/route_stats.pkl")

      self.df["route"] = self.df["Origin"] + "-" + self.df["Dest"]
      self.df = self.df.merge(airline_stats, how='left', left_on=['route'], right_on=['route'])
      return self.df

  def _add_log1p_target(self) -> pd.DataFrame:
      """
      Apply log1p transformation to the target variable (departure delay).
      """
      self.df['log1p_DepDelayMinutes'] = self.df['DepDelayMinutes'].apply(lambda x: np.log1p(x) if x > 0 else 0)
      return self.df

  def _add_interaction_features(self) -> pd.DataFrame:
      """Add interaction features between distance, congestion, and weather.

      This method is designed to be safe to call even if some columns are missing.
      It only adds new columns when the required source columns exist.
      """
      df = self.df

      def _num(col: str) -> pd.Series:
          return pd.to_numeric(df[col], errors="coerce").fillna(0)

      # Base numeric series used in multiple interactions
      distance = _num("Distance") if "Distance" in df.columns else None
      crs_elapsed = _num("CRSElapsedTime") if "CRSElapsedTime" in df.columns else None

      # Congestion x weather interactions (departure + arrival)
      interaction_pairs = [
          ("dep_scheduled_congestion", "dep_precipitation", "dep_congestion_x_precipitation"),
          ("dep_scheduled_congestion", "dep_wind_gusts_10m", "dep_congestion_x_wind_gusts"),
          ("dep_scheduled_congestion", "dep_snowfall", "dep_congestion_x_snowfall"),
          ("arr_scheduled_congestion", "arr_precipitation", "arr_congestion_x_precipitation"),
          ("arr_scheduled_congestion", "arr_wind_gusts_10m", "arr_congestion_x_wind_gusts"),
          ("arr_scheduled_congestion", "arr_snowfall", "arr_congestion_x_snowfall"),
      ]

      for a, b, out_col in interaction_pairs:
          if a in df.columns and b in df.columns:
              df[out_col] = _num(a) * _num(b)

    # Distance x weather interactions (helps capture "long flights + bad weather" effects)
      if distance is not None:
          dist_weather_cols = [
              "dep_precipitation",
              "dep_wind_gusts_10m",
              "dep_snowfall",
              "arr_precipitation",
              "arr_wind_gusts_10m",
              "arr_snowfall",
          ]
          for w_col in dist_weather_cols:
              if w_col in df.columns:
                  df[f"distance_x_{w_col}"] = distance * _num(w_col)

    # Departure/arrival weather deltas (captures "weather mismatch" along the route)
      delta_pairs = [
          ("dep_temperature_2m", "arr_temperature_2m", "temperature_delta"),
          ("dep_apparent_temperature", "arr_apparent_temperature", "apparent_temperature_delta"),
          ("dep_relative_humidity_2m", "arr_relative_humidity_2m", "humidity_delta"),
          ("dep_pressure_msl", "arr_pressure_msl", "pressure_delta"),
          ("dep_cloud_cover", "arr_cloud_cover", "cloud_cover_delta"),
      ]
      for dep_col, arr_col, base_name in delta_pairs:
          if dep_col in df.columns and arr_col in df.columns:
              delta = _num(dep_col) - _num(arr_col)
              df[base_name] = delta
              df[f"abs_{base_name}"] = delta.abs()

      # Gust factor (gustiness relative to mean wind speed)
      for prefix in ["dep", "arr"]:
          ws = f"{prefix}_wind_speed_10m"
          gust = f"{prefix}_wind_gusts_10m"
          if ws in df.columns and gust in df.columns:
              ws_num = pd.to_numeric(df[ws], errors="coerce")
              gust_num = pd.to_numeric(df[gust], errors="coerce")
              df[f"{prefix}_gust_factor"] = (
                  gust_num / ws_num.replace(0, np.nan)
              ).replace([np.inf, -np.inf], np.nan).fillna(0)

      # Joint congestion interactions
      if "dep_scheduled_congestion" in df.columns and "arr_scheduled_congestion" in df.columns:
          dep_cong = _num("dep_scheduled_congestion")
          arr_cong = _num("arr_scheduled_congestion")
          df["congestion_delta"] = dep_cong - arr_cong

      self.df = df
      return self.df

  def _open_pickle(self, file_path: str) -> pd.DataFrame:
      with open(file_path, "rb") as f:
          data = pickle.load(f)
      return data

  def _save_pickle(self, data: pd.DataFrame, file_path: str) -> None:
      with open(file_path, "wb") as f:
          pickle.dump(data, f)

  def _define_congestion_bucket(self, x: float, quantiles: list) -> int:
      if x <= quantiles[0]:
          return 0
      elif x <= quantiles[1]:
          return 1
      elif x <= quantiles[2]:
          return 2
      else:
          return 3


  def test_method(self) -> pd.DataFrame:
      self.df = self._add_stats_features()
      return self.df
  
  def _classification_schema(self, x) -> pd.DataFrame:
      if x <= 60:
        return 0
      else:
        return 1

  def engineer_features(self, is_test=False) -> pd.DataFrame:
      cols_to_drop = ['Origin', 'Dest', 'dep_scheduled_congestion', 'arr_scheduled_congestion', 'Origin', 'Dest', 'DepDelayMinutes', 'arr_date']
      if self.type == "new":
        cols_to_drop.remove('arr_date')
        cols_to_drop.remove('DepDelayMinutes')
      if (self.classification and self.type == "old") or is_test:
          self.df['DepDelayCategory'] = self.df['DepDelayMinutes'].apply(self._classification_schema)
          cols_to_drop.append('log1p_DepDelayMinutes')
      self.df = self._seperate_time_features()
      self.df = self._add_stats_features()
      self.df = self._add_congestion_features()
      self.df = self._add_route_features()
      if self.type == "old" or is_test:
        self.df = self._add_log1p_target()
      self.df = self._add_interaction_features()
      self.df.drop(columns=cols_to_drop, inplace=True)
      return self.df



  # def _decode_dep_arr_code(self, origin_code: str, destination_code: str) -> (str, str):

  # def arr_dep_code_decode(self) -> pd.DataFrame:
  #     # Decode IATA codes to full airport names

  #     self
  #     return self.df