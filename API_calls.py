from urllib.parse import quote
from functools import lru_cache, wraps
from datetime import datetime, timedelta
from icecream import ic
import requests


def timed_lru_cache(seconds: int, maxsize: int = 128):
    def wrapper_cache(func):
        func = lru_cache(maxsize=maxsize)(func)
        func.lifetime = timedelta(seconds=seconds)
        func.expiration = datetime.utcnow() + func.lifetime

        @wraps(func)
        def wrapped_func(*args, **kwargs):
            if datetime.utcnow() >= func.expiration:
                func.cache_clear()
                func.expiration = datetime.utcnow() + func.lifetime

            return func(*args, **kwargs)

        return wrapped_func

    return wrapper_cache


@timed_lru_cache(600)
def get_currentWeatherReports(city):
    base_url = 'http://api.weatherapi.com/v1'
    api_method = '/current.json'
    api_key = 'b4de7d54fab44efbb39204156231310'
    city = quote(city)
    url= f'{base_url}{api_method}?key={api_key}&q={city}'
    response = requests.get(url)
    data = response.json()
    ic(response)
    ic(response.json())
    return data