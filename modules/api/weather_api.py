from __future__ import annotations

from collections import Counter
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
import os
import re
from threading import Lock
import time

import requests
from dotenv import load_dotenv

from modules.api.time_api import get_now


load_dotenv()
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
CURRENT_URL = "https://api.openweathermap.org/data/2.5/weather"
FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"
GEOCODING_URL = "https://api.openweathermap.org/geo/1.0/direct"
NOMINATIM_URL = os.getenv(
    "NOMINATIM_URL", "https://nominatim.openstreetmap.org/search"
)
GEOCODER_USER_AGENT = os.getenv(
    "GEOCODER_USER_AGENT", "RAG_CHATBOT/1.0 (local weather lookup)"
)
_nominatim_lock = Lock()
_last_nominatim_request = 0.0

STOPWORDS = [
    "thành phố", "tp", "thủ đô", "city", "hôm nay", "hôm qua", "ngày mai",
    "ngày kia", "thế nào", "ra sao", "như thế nào", "dự báo",
    "tỉnh", "huyện", "quận", "thị xã", "thị trấn", "xã", "phường",
]

CITY_MAP = {
    "hà nội": "Hanoi",
    "đà nẵng": "Da Nang",
    "hồ chí minh": "Ho Chi Minh",
    "tp hcm": "Ho Chi Minh",
    "hcm": "Ho Chi Minh",
    "sài gòn": "Ho Chi Minh",
    "saigon": "Ho Chi Minh",
}


def normalize_city_name(city: str) -> str:
    city = (city or "").lower()
    city = re.sub(r"ngày\s+\d{1,2}[/-]\d{1,2}(?:[/-]\d{4})?", " ", city)
    city = re.sub(
        r"ngày\s+\d{1,2}\s+tháng\s+\d{1,2}(?:\s+năm\s+\d{4})?",
        " ",
        city,
    )
    city = re.sub(r"\d+\s+ngày\s+(?:tới|sau|nữa)", " ", city)
    city = re.sub(r"\b(?:sáng|trưa|chiều|tối|đêm)\b", " ", city)
    city = re.sub(r"[^\w\sÀ-ỹ]", " ", city)
    city = re.sub(r"\s+", " ", city).strip()

    for stopword in STOPWORDS:
        city = city.replace(stopword, " ")
    city = re.sub(r"^(?:ở|tại)\s+", "", city.strip())
    city = re.sub(r"\s+", " ", city).strip()

    return CITY_MAP.get(city, city.title() or "Hanoi")


def resolve_weather_date(query: str, now: datetime | None = None) -> date:
    """Resolve Vietnamese relative/absolute dates for weather questions."""
    now = now or get_now()
    q = (query or "").lower()

    if "hôm qua" in q:
        return now.date() - timedelta(days=1)
    if "ngày kia" in q:
        return now.date() + timedelta(days=2)
    if "ngày mai" in q or re.search(r"\bmai\b", q):
        return now.date() + timedelta(days=1)

    relative = re.search(r"(\d+)\s+ngày\s+(?:tới|sau|nữa)", q)
    if relative:
        return now.date() + timedelta(days=int(relative.group(1)))

    absolute = re.search(r"ngày\s+(\d{1,2})[/-](\d{1,2})(?:[/-](\d{4}))?", q)
    if absolute:
        day, month, year = absolute.groups()
        return date(int(year or now.year), int(month), int(day))

    words = re.search(
        r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})(?:\s+năm\s+(\d{4}))?",
        q,
    )
    if words:
        day, month, year = words.groups()
        return date(int(year or now.year), int(month), int(day))

    return now.date()


def _request_json(url: str, params: dict) -> dict | list:
    try:
        response = requests.get(url, params=params, timeout=10)
    except requests.RequestException as exc:
        return {"error": f"Lỗi kết nối thời tiết: {exc}"}

    try:
        data = response.json()
    except ValueError:
        return {"error": f"OpenWeather trả dữ liệu không hợp lệ (HTTP {response.status_code})."}

    if response.status_code != 200:
        detail = data.get("message") if isinstance(data, dict) else None
        return {"error": f"OpenWeather HTTP {response.status_code}: {detail or 'không rõ lỗi'}"}
    return data


def _geocode_vietnam(location: str) -> dict:
    """Resolve a Vietnamese locality and reject results outside Viet Nam."""
    query = normalize_city_name(location)
    data = _request_json(
        GEOCODING_URL,
        {
            "q": f"{query},VN",
            "limit": 5,
            "appid": OPENWEATHER_API_KEY,
        },
    )
    if isinstance(data, dict) and "error" in data:
        return data
    if not isinstance(data, list):
        return {"error": "OpenWeather trả kết quả địa lý không hợp lệ."}

    candidates = [item for item in data if item.get("country") == "VN"]
    if not candidates:
        return _geocode_vietnam_osm(location)

    item = candidates[0]
    local_names = item.get("local_names") or {}
    name = local_names.get("vi") or item.get("name") or query
    state = item.get("state")
    display_name = ", ".join(part for part in (name, state, "Việt Nam") if part)
    return {
        "lat": float(item["lat"]),
        "lon": float(item["lon"]),
        "name": display_name,
        "geocoder_source": "OpenWeather",
    }


def _nominatim_query_text(location: str) -> str:
    """Keep the most specific locality when the user provides a hierarchy."""
    raw = (location or "").strip()
    raw = re.sub(r"\b(?:hôm nay|hôm qua|ngày mai|ngày kia)\b", " ", raw, flags=re.I)
    raw = re.sub(r"\d+\s+ngày\s+(?:tới|sau|nữa)", " ", raw, flags=re.I)
    raw = re.sub(r"ngày\s+\d{1,2}[/-]\d{1,2}(?:[/-]\d{4})?", " ", raw, flags=re.I)
    raw = re.sub(r"\s+", " ", raw).strip(" ,")
    primary = raw.split(",", 1)[0].strip()
    return primary or normalize_city_name(location)


@lru_cache(maxsize=512)
def _geocode_vietnam_osm(location: str) -> dict:
    """Fallback for Vietnamese district/commune names missing in OpenWeather."""
    global _last_nominatim_request

    query = _nominatim_query_text(location)
    with _nominatim_lock:
        wait_seconds = 1.0 - (time.monotonic() - _last_nominatim_request)
        if wait_seconds > 0:
            time.sleep(wait_seconds)
        try:
            response = requests.get(
                NOMINATIM_URL,
                params={
                    "q": query,
                    "format": "jsonv2",
                    "limit": 5,
                    "countrycodes": "vn",
                    "addressdetails": 1,
                },
                headers={"User-Agent": GEOCODER_USER_AGENT},
                timeout=10,
            )
            _last_nominatim_request = time.monotonic()
            response.raise_for_status()
            items = response.json()
        except (requests.RequestException, ValueError) as exc:
            return {"error": f"Không thể tra cứu địa danh chi tiết: {exc}"}

    vietnam_items = [
        item
        for item in items
        if (item.get("address") or {}).get("country_code") == "vn"
    ]
    if not vietnam_items:
        return {"error": f"Không tìm thấy địa điểm '{normalize_city_name(location)}' tại Việt Nam."}

    administrative = [
        item
        for item in vietnam_items
        if item.get("type") in {"administrative", "city", "town", "village"}
        or item.get("category") == "boundary"
    ]
    item = (administrative or vietnam_items)[0]
    return {
        "lat": float(item["lat"]),
        "lon": float(item["lon"]),
        "name": item.get("display_name") or item.get("name") or query,
        "geocoder_source": "OpenStreetMap",
    }


def get_weather(city: str = "Hanoi", country: str = "VN") -> dict:
    if not OPENWEATHER_API_KEY:
        return {"error": "Thiếu OPENWEATHER_API_KEY"}

    place = _geocode_vietnam(city)
    if "error" in place:
        return place
    data = _request_json(
        CURRENT_URL,
        {
            "lat": place["lat"],
            "lon": place["lon"],
            "appid": OPENWEATHER_API_KEY,
            "units": "metric",
            "lang": "vi",
        },
    )
    if "error" in data:
        return data

    return {
        "location": place["name"],
        "temp": data["main"]["temp"],
        "feels_like": data["main"].get("feels_like"),
        "humidity": data["main"].get("humidity"),
        "desc": data["weather"][0]["description"],
        "kind": "current",
        "geocoder_source": place.get("geocoder_source"),
    }


def get_weather_forecast(
    city: str,
    target_date: date,
    country: str = "VN",
) -> dict:
    if not OPENWEATHER_API_KEY:
        return {"error": "Thiếu OPENWEATHER_API_KEY"}

    today = get_now().date()
    days_ahead = (target_date - today).days
    if days_ahead < 0:
        return {"error": "Gói dự báo miễn phí không cung cấp thời tiết quá khứ."}
    if days_ahead > 5:
        return {"error": "Dự báo miễn phí chỉ hỗ trợ tối đa khoảng 5 ngày tới."}

    place = _geocode_vietnam(city)
    if "error" in place:
        return place
    data = _request_json(
        FORECAST_URL,
        {
            "lat": place["lat"],
            "lon": place["lon"],
            "appid": OPENWEATHER_API_KEY,
            "units": "metric",
            "lang": "vi",
        },
    )
    if "error" in data:
        return data

    offset = int((data.get("city") or {}).get("timezone", 0))
    local_tz = timezone(timedelta(seconds=offset))
    slots = []
    for item in data.get("list", []):
        local_dt = datetime.fromtimestamp(int(item["dt"]), tz=timezone.utc).astimezone(local_tz)
        if local_dt.date() == target_date:
            slots.append((local_dt, item))

    if not slots:
        return {"error": "OpenWeather chưa có dữ liệu cho ngày được yêu cầu."}

    temps_min = [float(item["main"]["temp_min"]) for _, item in slots]
    temps_max = [float(item["main"]["temp_max"]) for _, item in slots]
    descriptions = [item["weather"][0]["description"] for _, item in slots]
    representative_dt, representative = min(
        slots, key=lambda pair: abs(pair[0].hour - 12)
    )

    return {
        "location": place["name"],
        "date": target_date.strftime("%d/%m/%Y"),
        "temp": representative["main"]["temp"],
        "temp_min": min(temps_min),
        "temp_max": max(temps_max),
        "humidity": representative["main"].get("humidity"),
        "desc": Counter(descriptions).most_common(1)[0][0],
        "rain_probability": round(max(float(item.get("pop", 0)) for _, item in slots) * 100),
        "representative_time": representative_dt.strftime("%H:%M"),
        "kind": "forecast",
        "geocoder_source": place.get("geocoder_source"),
    }
