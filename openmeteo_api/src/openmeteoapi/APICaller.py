import openmeteo_requests
import requests_cache
from retry_requests import retry


"""
    API client wrapper for Open-Meteo.

    Author: Rashed Aldulijan
    Created: 2025-12-27
"""


class APICaller:
    def __init__(
        self,
        cache_dir: str = ".cache",
        expire_after: int = -1,
        retries: int = 5,
        backoff_factor: float = 0.2,
    ):
        cache_session = requests_cache.CachedSession(
            cache_dir, expire_after=expire_after
        )
        retry_session = retry(
            cache_session, retries=retries, backoff_factor=backoff_factor
        )
        self.client = openmeteo_requests.Client(session=retry_session)

    def get(self, url: str, params: dict):
        return self.client.weather_api(url, params=params)


class OpenMeteoAPICaller(APICaller):
    def weather_api(self, url: str, params: dict):
        return self.get(url, params)
