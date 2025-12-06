from langchain_core.tools import tool
from src.services.weather_service import WeatherService

weather_service = WeatherService()

@tool
def weather_guess(cityName: str) -> list:
    """Returns 7-day weather forecast (temperature, rain, wind) for a city name in India."""
    return weather_service.get_weather_forecast(cityName)
