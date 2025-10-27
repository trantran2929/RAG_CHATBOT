import requests
import os, re
from dotenv import load_dotenv

load_dotenv(dotenv_path="C:/Users/admin/Desktop/TT/rag_note/.env")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

STOPWORDS = ["thành phố", "tp", "thủ đô", "city", "hôm nay",
             "ngày mai", "thế nào", "ra sao", "như thế nào", "dự báo"]

CITY_MAP = {
    "hồ chí minh": "Ho Chi Minh",
    "tp hcm": "Ho Chi Minh",
    "hcm": "Ho Chi Minh",
    "sài gòn": "Ho Chi Minh",
    "saigon": "Ho Chi Minh"
}


def normalize_city_name(city: str) -> str:
    city = re.sub(r"[^\w\sÀ-ỹ]", " ", city)
    city = re.sub(r"\s+", " ", city).strip().lower()

    for sw in STOPWORDS:
        city = city.replace(sw, "").strip()
    
    if city in CITY_MAP:
        return CITY_MAP[city]
    
    return city.title()

def get_weather(city: str = "Hanoi", country: str = "Việt Nam"):
    if not OPENWEATHER_API_KEY:
        return {"error": "Thiếu OPENWEATHER_API_KEY"}

    city = normalize_city_name(city)
    url = (
        "http://api.openweathermap.org/data/2.5/weather"
        f"?q={city},{country}&appid={OPENWEATHER_API_KEY}"
        "&units=metric&lang=vi"
    )

    try:
        r = requests.get(url, timeout=10)
    except Exception as e:
        return {"error": f"Lỗi kết nối thời tiết: {e}"}

    if r.status_code == 200:
        data = r.json()
        return {
            "location": f"{data['name']}, {country}",
            "temp": data["main"]["temp"],
            "desc": data["weather"][0]["description"]
        }
    return {"error": "Không lấy được dữ liệu thời tiết."}