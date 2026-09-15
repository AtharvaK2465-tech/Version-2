"""
Fetches recent rainfall/temperature for a point using Open-Meteo's free
archive API — no API key required, so this works in both mock and live
modes for the rest of the pipeline.
"""
import requests
from datetime import date, timedelta


def get_recent_weather(lat: float, lon: float, end: date, days: int = 14) -> dict:
    start = end - timedelta(days=days)
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start.isoformat()}&end_date={end.isoformat()}"
        "&daily=precipitation_sum,temperature_2m_max,temperature_2m_min"
        "&timezone=Asia%2FKolkata"
    )
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        daily = data.get("daily", {})
        precip = daily.get("precipitation_sum", [])
        return {
            "total_rainfall_mm": round(sum(p for p in precip if p is not None), 1),
            "days": days,
            "source": "open-meteo",
        }
    except Exception as e:
        # Weather fusion is an enhancement, not a hard dependency — degrade
        # gracefully so the pipeline still completes without internet access.
        return {"total_rainfall_mm": None, "days": days, "source": f"unavailable ({e})"}
