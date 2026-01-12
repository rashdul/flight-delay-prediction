import logging
from typing import Any, Dict, Optional
from urllib.parse import urlencode

import requests

""""
A module for making API calls to the AeroMarket API.

Author: Rashed Aldulijan
Created: 2025-12-27
"""

class APICaller:
    """
    A class for making API calls.

    Attributes:
        base_url (str): The base URL for the API.
        session (requests.Session): The session object for making requests.

    Methods:
        _send_request(method: str, endpoint: str, payload: Optional[Dict[str, Any]]
         = None, headers: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
            Sends a request to the API.

        get(endpoint: str, headers: Optional[Dict[str, Any]] = None) ->
        Optional[Dict[str, Any]]:
            Sends a GET request to the API.

        post(endpoint: str, payload: Dict[str, Any], headers: Optional[Dict[str, Any]]
          = None) -> Optional[Dict[str, Any]]:
            Sends a POST request to the API.

        _build_path(endpoint: str, sub_path: Optional[str]
          = None, query: Optional[Dict[str, Any]] = None) -> str:
            Builds a URL path for an API request.
    """

    def __init__(self, base_url: str, api_key: str) -> None:
        """
        Initializes the APICaller class.

        Args:
            base_url (str): The base URL for the API.
            api_key (str): The API key to use for authentication.
        """
        self.base_url = base_url
        self.__API_KEY = api_key
        self.session = requests.Session()
        self.timeout_s: float = 30.0

    def _build_url(self, endpoint: str) -> str:
        return f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"

    def _send_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        payload: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Sends a request to the API.

        Args:
            method (str): The HTTP method to use for the request.
            endpoint (str): The API endpoint (path).
            payload (dict): Optional, the data to send in the request body.
            headers (dict): Optional, headers to include in the request.

        Returns:
            Any: The parsed JSON response.
        """
        url = self._build_url(endpoint)
        merged_headers: Dict[str, Any] = dict(headers) if headers else {}
        merged_headers["x-api-market-key"] = self.__API_KEY
        try:
            r = self.session.request(
                method,
                url,
                headers=merged_headers,
                params=params,
                json=payload,
                timeout=self.timeout_s,
            )
            r.raise_for_status()
            if not r.text.strip():
                # Some upstreams return 204 No Content (or an empty 200) on "not found".
                # Give enough context to debug without leaking the API key.
                raise RuntimeError(
                    f"Empty response body from upstream (status={r.status_code}) for {method} {r.url}"
                )
            return r.json()
        except requests.exceptions.RequestException as e:
            logging.error("HTTP request failed: %s", e)
            if getattr(e, "response", None) is not None:
                body_preview = (e.response.text or "")[:500]
                raise RuntimeError(f"HTTP request failed ({e.response.status_code}): {body_preview}") from e
            raise RuntimeError(f"HTTP request failed: {e}") from e
        except ValueError as e:
            logging.error("Failed to parse JSON response: %s", e)
            raise RuntimeError(f"Failed to parse JSON response: {e}") from e

    def get(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Sends a GET request to the API.

        Args:
            endpoint (str): The API endpoint (path).
            headers (dict): Optional, headers to include in the request.

        Returns:
            Any: The parsed JSON response.
        """
        return self._send_request("GET", endpoint, params=params, headers=headers)

    def post(
        self,
        endpoint: str,
        payload: Dict[str, Any],
        headers: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Sends a POST request to the API.

        Args:
            endpoint (str): The API endpoint (path).
            payload (dict): The data to send in the request body.
            headers (dict): Optional, headers to include in the request.

        Returns:
            Any: The parsed JSON response.
        """
        return self._send_request("POST", endpoint, payload=payload, headers=headers)

    def _build_path(
        self,
        endpoint: str,
        sub_path: Optional[str] = None,
        query: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Builds a URL path for an API request, including optional sub-path and query
        parameters.

        Args:
            endpoint (str): The endpoint of the API request.
            sub_path (str): Optional, a sub-path to append to the endpoint.
            query (dict): Optional, a dictionary of query parameters to include in the
            URL.

        Returns:
            str: The complete URL path for the API request.
        """
        path = endpoint.lstrip("/")
        if sub_path is not None:
            path += f"/{sub_path.lstrip('/')}"
        if query:
            filtered_query = {k: v for k, v in query.items() if v is not None}
            query_string = urlencode(filtered_query)
            path += f"?{query_string}"
        return path
