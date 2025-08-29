# weather_tool.py
from langchain_core.tools import tool
from collections import defaultdict
from geopy.geocoders import Nominatim
import requests

@tool
def weather_guess(cityName: str) -> list:
    """Returns 7-day weather forecast (temperature, rain, wind) for a city name in India."""
    if cityName == "":
        return {"message": "Please provide city name"}

    geolocator = Nominatim(user_agent="my_weather_bot")
    location = geolocator.geocode(cityName)

    if location:
        lat = location.latitude
        lon = location.longitude
        api_url = f"https://mosdac.gov.in/apiweather1/weather?lon={lon}&lat={lat}"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(api_url, headers=headers, timeout=10)

        if response.status_code == 200:
            data = response.json().get("data", [])
            daily = defaultdict(list)
            for entry in data:
                date = entry["time"].split(" ")[0]
                daily[date].append({
                    "temperature": float(entry["t2"]),
                    "rain": float(entry["rainc"]),
                    "wind_speed": float(entry["ws10"]),
                })

            results = []
            for date, entries in daily.items():
                temps = [e["temperature"] for e in entries]
                rains = [e["rain"] for e in entries]
                winds = [e["wind_speed"] for e in entries]
                results.append({
                    "date": date,
                    "avg_temperature_c": round(sum(temps) / len(temps), 1),
                    "max_temperature_c": round(max(temps), 1),
                    "min_temperature_c": round(min(temps), 1),
                    "total_rain_mm": round(sum(rains), 1),
                    "rain_probability_percent": round((sum(1 for r in rains if r > 0.1) / len(rains)) * 100),
                    "avg_wind_speed_mps": round(sum(winds) / len(winds), 1),
                    "max_wind_speed_mps": round(max(winds), 1),
                    "data_points": len(entries)
                })
            return results[:7]
        else:
            return {"error": "Try after some time"}
    else:
        return {"error": f"Could not find {cityName}"}
