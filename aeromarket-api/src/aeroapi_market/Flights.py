""""
A class for interacting with the AeroDataBox flights endpoints via API.Market for an easy to retrieve flight data.

Author: Rashed Aldulijan
Created: 2025-12-27

"""


from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from aeroapi_market.APICaller import APICaller
import pandas as pd
import airportsdata


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

    def __init__(self, api_caller: APICaller, airport_code: str, from_local: str, to_local: str, keep_cols: bool = False) -> None:
        """
        Initializes a `Flights` instance.

        Args:
            api_caller (APICaller): An instance of the `APICaller` class.
        """
        self.api_caller = api_caller
        self.endpoint = "flights"
        self.airport_code = airport_code
        self.from_local = from_local
        self.to_local = to_local
        self._flight_df: Optional[pd.DataFrame] = None
        # self._last_airport_flights_request: Optional[Tuple[str, str, str, str, str]] = None
        self.keep_cols = keep_cols


    
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

        df["date"] = df["scheduled_local"].dt.floor("H")

        if self.code_type == "iata":
            airport = airportsdata.load("IATA").get(self.airport_code)
            df['queried_airport_iata'] = self.airport_code
        else:
            airport = airportsdata.load("ICAO").get(self.airport_code)
            df['queried_airport_icao'] = self.airport_code
        df['date'] = pd.to_datetime(df['date']).dt.tz_convert(airport["tz"])
        if not self.keep_cols:
            if self.code_type == "iata":
                cols_to_keep = [
                    "direction",
                    "date",
                    "queried_airport_iata",
                ]
            else:
                cols_to_keep = [
                    "direction",
                    "date",
                    "queried_airport_icao",
                ]
            df = df[cols_to_keep]

        self._flight_df = df
        return df

    def get_airport_flights(
        self,
        code_type: str = "icao",
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
            return pd.DataFrame()


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
