""""
A class for interacting with the AeroDataBox flights endpoints via API.Market for an easy to retrieve flight data.

Author: Rashed Aldulijan
Created: 2025-12-27

"""


from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from aeroapi_market.APICaller import APICaller
from dotenv import load_dotenv
import pandas as pd
import requests
import airportsdata
from datetime import datetime, timedelta

from geopy.distance import geodesic
import numpy as np


class Flights:
    """
    A class for interacting with the AeroDataBox flights endpoints via API.Market.

    Attributes:
        api_caller (APICaller): An instance of the `APICaller` class.
        endpoint (str): The API endpoint for flights.

    Methods:
        __init__(self, api_caller: APICaller) -> None:
            Initializes a `Flights` instance.

        get_airport_flights(
            airport_code: str,
            from_local: str,
            to_local: str,
            code_type: str = "icao",
            direction: str = "Both",
        ):
            Retrieves flights for a specific airport within a date range.
    """

    def __init__(self, api_caller: APICaller, from_local: str, flight_number: Optional[str] = None, code_type: str = "iata") -> None:
        """
        Initializes a `Flights` instance.

        Args:
            api_caller (APICaller): An instance of the `APICaller` class.
            from_local (str): Start date/time in local time (YYYY-MM-DD).
            flight_number (str, optional): The flight number to look up. Defaults to None.
        Raises:
            ValueError: If the date difference between `from_local` and `to_local` is not exactly 1 hour.
        """
        fmt = "%Y-%m-%d"
        wanted_fmt = "%Y-%m-%dT%H:%M"

        self.dt_from = datetime.strptime(from_local, fmt)
        
        self.api_caller = api_caller
        self.code_type = code_type
        self.endpoint = "flights"
        self.flight_number = flight_number
        self._flight_df: Optional[pd.DataFrame] = None
        self.response_flight_response = None
        base_dt = self._get_date_local_time()
        self.from_local = base_dt.strftime(wanted_fmt)
        self.to_local = (base_dt + timedelta(hours=1)).strftime(wanted_fmt)
        # self._last_airport_flights_request: Optional[Tuple[str, str, str, str, str]] = None



    
    def _migrate_legacy_state(self) -> None:
        """
        Migrates legacy double-underscore attributes to single-underscore ones.
        """
        if not hasattr(self, "_flight_df") and hasattr(self, "_Flights__flight_df"):
            self._flight_df = getattr(self, "_Flights__flight_df")

        if not hasattr(self, "_last_airport_flights_request") and hasattr(
            self, "_Flights__last_airport_flights_request"
        ):
            self._last_airport_flights_request = getattr(
                self, "_Flights__last_airport_flights_request"
            )

    def _flight_to_df(self, data: Dict[str, Any]) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []

        for flight in data.get("arrivals", []):
            rows.append(self._extract_flight_data(flight, "Arrival")) 

        for flight in data.get("departures", []):
            rows.append(self._extract_flight_data(flight, "Departure")) 

        df = pd.DataFrame(rows)

        # Convert timestamps
        for col in [
            "scheduled_local",
            "actual_local",
            "scheduled_utc",
            "actual_utc",
        ]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # df["flight_date"] = df["scheduled_local"].dt.tz_localize(None).dt.normalize()
        if self.code_type == "iata":
            airport = airportsdata.load("IATA").get(self.airport_code)
        else:
            airport = airportsdata.load("ICAO").get(self.airport_code)

        df["date"] = df["scheduled_local"].dt.floor("h")

        if self.code_type == "iata":
            airport = airportsdata.load("IATA").get(self.airport_code)
            df['queried_airport_iata'] = self.airport_code
        else:
            airport = airportsdata.load("ICAO").get(self.airport_code)
            df['queried_airport_icao'] = self.airport_code
        df['date'] = pd.to_datetime(df['date']).dt.tz_convert(airport["tz"])

        self._flight_df = df
        return df

    def get_airport_flights(
        self,
        code_type: str = "iata",
        direction: str = "Both",
    ) -> Dict[str, Any]:
        """
        Retrieves flights for a specific airport within a date range.

        Args:
            airport_code (str): The ICAO or IATA code of the airport.
            from_local (str): Start date/time in local time (YYYY-MM-DDTHH:MM).
            to_local (str): End date/time in local time (YYYY-MM-DDTHH:MM).
            code_type (str): The type of airport code ('icao' or 'iata').
            direction (str): The flight direction ('Departure', 'Arrival', or 'Both').
        Returns:
            Dict[str, Any]: The API response containing flight data.
        """
        self.code_type = code_type
        self._migrate_legacy_state()
        self.airport_code = self._get_airport_code()
        endpoint = self._get_airport_flights_endpoint(
            airport_code=self.airport_code,
            from_local=self.from_local,
            to_local=self.to_local,
            code_type=code_type,
        )
        params = self.get_airport_flights_params(direction=direction)
        response = self.api_caller.get(endpoint=endpoint, params=params, headers=None)

        if not isinstance(response, dict):
            raise TypeError(f"Expected a JSON object response, got: {type(response)}")

        self._last_airport_flights_request = (
            self.airport_code,
            self.from_local,
            self.to_local,
            code_type,
            direction,
        )
        self._flight_to_df(response)
        return response

    def _extract_flight_data(self, flight: dict, direction: str) -> Dict[str, Any]:
        movement = flight.get("movement", {})
        airport = movement.get("airport", {})
        scheduled = movement.get("scheduledTime", {})
        revised = movement.get("revisedTime", {})
        aircraft = flight.get("aircraft", {})
        airline = flight.get("airline", {})
            

        return {
            "direction": direction,
            "flight_number": flight.get("number"),
            "callsign": flight.get("callSign"),
            "status": flight.get("status"),
            "codeshare_status": flight.get("codeshareStatus"),
            "airline": airline.get("name"),
            "airline_iata": airline.get("iata"),
            "aircraft_model": aircraft.get("model"),
            "aircraft_reg": aircraft.get("reg"),
            "airport_icao": airport.get("icao"),
            "airport_iata": airport.get("iata"),
            "airport_name": airport.get("name"),
            "timezone": airport.get("timeZone"),
            "scheduled_utc": scheduled.get("utc"),
            "scheduled_local": scheduled.get("local"),
            "actual_utc": revised.get("utc"),
            "actual_local": revised.get("local"),
            "terminal": movement.get("terminal"),
            "runway": movement.get("runway"),
            "is_cargo": flight.get("isCargo"),
            "quality": ", ".join(movement.get("quality", [])),
        }

    

    def get_count_airport_flights(
        self,
    ) -> int:
        """
        Retrieves and counts flights for a specific airport within the default date range.

        Args:
            airport_code (str): The ICAO or IATA code of the airport.
            from_local (str): Start date/time in local time (YYYY-MM-DDTHH:MM).
            to_local (str): End date/time in local time (YYYY-MM-DDTHH:MM).
            code_type (str): The type of airport code ('icao' or 'iata').
            direction (str): The flight direction ('Departure', 'Arrival', or 'Both').
        Returns:
            int: The count of flights for the specified airport and date range.
        """
        
        if self._flight_df is None:
            return self.get_airport_flights_df().shape[0]

        return int(self._flight_df.shape[0])
    
    def _get_lat_lon(self, airport_code):
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            "q": f"{airport_code} airport",
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
    
    def _get_distance(self, origin_code, destination_code):
        origin = self._get_lat_lon(origin_code)      # JFK
        destination = self._get_lat_lon(destination_code)  # LAX

        distance_miles = np.floor(geodesic(origin, destination).miles)

        return distance_miles

    def _look_up_flights(self, flight_number):
        if self._flight_df is None:
            return pd.DataFrame()
        
        df_flight = self._flight_df[self._flight_df['flight_number'] == flight_number]
        return df_flight
    

    def _get_approx_flight_time_mins(self, origin_code, destination_code):
        end_point = self._get_airport_endpoint(origin_code, destination_code, code_type="iata")
        response = self.api_caller.get(endpoint=end_point, params=None, headers=None)['approxFlightTime']
        ## convert to minutes
        
        hours, minutes, seconds = map(int, response.split(":"))

        total_minutes = hours * 60 + minutes + seconds / 60
        return total_minutes

    def _get_airport_endpoint(
        self,
        airport_from: str,
        airport_to: str,
        code_type: str = "iata",
    ) -> str:
        return f"airports/{code_type}/{airport_from}/distance-time/{airport_to}"
    
    def _get_arrival_datetime(self):
        if self.response_flight_response is not None:
            return self.response_flight_response[0]['arrival']['scheduledTime']['local']
        end_point = self._get_flights_endpoint(search_by="number")
        params = {
            "dateLocalRole": "Departure"
        }
        response = self.api_caller.get(endpoint=end_point, params=params, headers=None)
        self.response_flight_response = response
        return response[0]['arrival']['scheduledTime']['local']
    
    def _get_airport_code(self):
        if self.response_flight_response is not None:
            return self.response_flight_response[0]['departure']['airport']['iata']
        end_point = self._get_flights_endpoint(search_by="number")
        params = {
            "dateLocalRole": "Departure"
        }
        response = self.api_caller.get(endpoint=end_point, params=params, headers=None)
        self.response_flight_response = response
        return response[0]['departure']['airport']['iata']
    
    def _get_date_local_time(self):
        if self.response_flight_response is not None:
            response =  self.response_flight_response[0]['departure']['scheduledTime']['local']
            dt = datetime.fromisoformat(response)
            floored_dt = dt.replace(minute=0, second=0, microsecond=0, tzinfo=None)
            # print( floored_dt)
            return floored_dt
        end_point = self._get_flights_endpoint(search_by="number")
        params = {
            "dateLocalRole": "Departure"
        }
        response = self.api_caller.get(endpoint=end_point, params=params, headers=None)
        self.response_flight_response = response
        response = response[0]['departure']['scheduledTime']['local']
        dt = datetime.fromisoformat(response)
        floored_dt = dt.replace(minute=0, second=0, microsecond=0, tzinfo=None)
        # print( floored_dt)

        return floored_dt
    
    def build_final_flight_response(
        self,
    ) -> pd.DataFrame:
        dict_df = {}
        df_flight = self._look_up_flights(self.flight_number)
        if df_flight.empty or df_flight['direction'].values[0] != 'Departure':
            raise ValueError(f"Flight number {self.flight_number} not found in the data.")
        dest = df_flight['airport_iata'].values[0]
        origin = self.airport_code
        approx_flight_time_min = self._get_approx_flight_time_mins(origin, dest)
        arrival_datetime = self._get_arrival_datetime()
        dt = datetime.fromisoformat(arrival_datetime)
        floored_dt = dt.replace(minute=0, second=0, microsecond=0, tzinfo=None)
        
        
        
        dict_df['Origin'] = self.airport_code 
        dict_df['Dest'] = df_flight['airport_iata'].values[0]
        dict_df['IATA_Code_Operating_Airline'] = self.flight_number[:2]
        dict_df['Distance'] = self._get_distance(origin, dest)
        dict_df['scheduled_congestion'] = self.get_count_airport_flights() 
        dict_df['CRSElapsedTime'] = approx_flight_time_min
        dict_df['arr_datetime'] = floored_dt
        dict_df['date_local'] = df_flight['date'].values[0]
        final_df = pd.DataFrame([dict_df])
        return final_df

        
    def get_hourly_airport_flights_count(
        self,
    ) -> pd.DataFrame:
        if self._flight_df is None:
            return pd.DataFrame()

        df_count = (
            self._flight_df.groupby(["date"])
            .size()
            .reset_index(name="flight_count")
        )
        df_count = df_count.sort_values(by=["date"]).reset_index(drop=True)

        return df_count

    def get_airport_flights_df(
        self,
    ) -> pd.DataFrame:

        if self._flight_df is None:
            response = self.get_airport_flights()
            self._flight_to_df(response)


        return self._flight_df.copy()

    def get_last_airport_flights_df(self) -> pd.DataFrame:
        self._migrate_legacy_state()
        if self._flight_df is None:
            return pd.DataFrame()
        return self._flight_df.copy()

    def get_last_airport_flights_request(self) -> Optional[Tuple[str, str, str, str, str]]:
        self._migrate_legacy_state()
        return self._last_airport_flights_request

    def get_airport_flights_params(
        self,
        direction: str = "Both",
        with_leg: bool = False,
        with_cancelled: bool = True,
        with_codeshared: bool = True,
        with_cargo: bool = True,
        with_private: bool = True,
        with_location: bool = False,
        fmt: str = "json",
    ) -> Dict[str, Any]:
        return {
            "direction": direction,
            "withLeg": with_leg,
            "withCancelled": with_cancelled,
            "withCodeshared": with_codeshared,
            "withCargo": with_cargo,
            "withPrivate": with_private,
            "withLocation": with_location,
            "format": fmt,
        }

    def get_airport_flights_custom(
        self,
        airport_code: str,
        from_local: str,
        to_local: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        self._migrate_legacy_state()
        path = (
            f"{self.endpoint}/airports/"
            f"{self.code_type}/{airport_code}/{from_local}/{to_local}"
        )

        response = self.api_caller.get(
            endpoint=path,
            params=params,
            headers=None,
        )

        if not isinstance(response, dict):
            raise TypeError(f"Expected a JSON object response, got: {type(response)}")

        direction = str(params.get("direction", "Both"))
        self._last_airport_flights_request = (
            airport_code,
            from_local,
            to_local,
            self.code_type,
            direction,
        )
        self._flight_to_df(response)
        return response

    def _get_airport_flights_endpoint(
        self,
        airport_code: str,
        from_local: str,
        to_local: str,
        code_type: str = "icao",
    ) -> str:
        return (
            f"{self.endpoint}/airports/"
            f"{code_type}/{airport_code}/{from_local}/{to_local}"
        )
    

    def _get_flights_endpoint(
        self,
        search_by: str = "number",
    ) -> str:
        date = self.dt_from.strftime("%Y-%m-%d")
        flight_number = self.flight_number.replace(" ", "")
        return (
            f"/flights/{search_by}/"
            f"{flight_number}/{date}"
        )
    
#     def test_method(self):
#         return self._get_distance("JFK", "LAX")



# # for testing purposes

# if __name__ == "__main__":
#     load_dotenv()
#     API_KEY = os.getenv("AERODATABOX_API_KEY")
#     BASE_URL = os.getenv("AERODATABOX_BASE_URL")
#     api_caller = APICaller(api_key=API_KEY, base_url=BASE_URL)

#     flights = Flights(api_caller=api_caller,
#         airport_code="ATL",
#         from_local="2025-10-26T00:00",
#         to_local="2025-10-26T01:00",
#         flight_number="F9 1241",
#         )
#     print(flights.test_method())
