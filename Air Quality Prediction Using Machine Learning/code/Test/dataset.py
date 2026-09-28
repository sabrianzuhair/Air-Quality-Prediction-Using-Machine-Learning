import requests
import csv
import os
from datetime import datetime

AQICN_TOKEN = "bb7bf66677d85e1ca4a9d21a7e4db09539447ca7"
OPENWEATHER_API_KEY = "c3c265d7c469d367bc53bf55bfa12f43"

CITIES = [
    {"city": "Jakarta", "lat": -6.2088, "lon": 106.8456},
    {"city": "Bandung", "lat": -6.9175, "lon": 107.6191},
    {"city": "Surabaya", "lat": -7.2575, "lon": 112.7521},
    {"city": "Medan", "lat": 3.5952, "lon": 98.6722},
    {"city": "Makassar", "lat": -5.1477, "lon": 119.4327},
    {"city": "Semarang", "lat": -6.9667, "lon": 110.4167},
    {"city": "Palembang", "lat": -2.9911, "lon": 104.7567},
    {"city": "Denpasar", "lat": -8.6705, "lon": 115.2126},
    {"city": "Yogyakarta", "lat": -7.7956, "lon": 110.3695},
    {"city": "Malang", "lat": -7.9797, "lon": 112.6304}
]

OUTPUT_CSV = "C:\\Air Quality Prediction Using Machine Learning\\code\\Test\\dataset_cuaca_aqi.csv"

FIELDNAMES = [
    "datetime",
    "city",
    "temperature",
    "humidity",
    "wind_speed",
    "pm25",
    "pm10",
    "aqi"
]

#Fungsi dari function ini adalah mengambil data cuaca real time dari openweathermap berdasarkan koordinat lintang dan bujur yang diberikan. 
#prosessnya adalah
#1. mengirim request ke open weather 
#2. setelah itu mengambil data temperature, humidity, dan wind speed 
def get_weather(lat, lon): 
    url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        "lat": lat,
        "lon": lon,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric" #menggunakan satuan metrik untuk celcius
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200: #menangani error jika request gagal
            return None
        
        data = response.json()
        
        return {
            "temperature": data.get("main", {}).get("temp"),
            "humidity": data.get("main", {}).get("humidity"),
            "wind_speed": data.get("wind", {}).get("speed")
        }
        
    except requests.exceptions.RequestException:
        return None
    except (KeyError, ValueError):
        return None
    #outputnya adalah 
    #     {
    #   "temperature": float,
    #   "humidity": int,
    #   "wind_speed": float
    # }
    

#fungsi dari function ini adalah mengambil data kualitas udara dari aqicn berdasarkan lokasi
def get_aqi(lat, lon):
    url = "https://api.waqi.info/feed/geo:{};{}/?token={}".format(lat, lon, AQICN_TOKEN)
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return None
        
        data = response.json()
        if data.get("status") != "ok":
            return None
        
        iaqi = data.get("data", {}).get("iaqi", {})
        
        return {
            "aqi": data.get("data", {}).get("aqi"),
            "pm25": iaqi.get("pm25", {}).get("v") if iaqi.get("pm25") else None,
            "pm10": iaqi.get("pm10", {}).get("v") if iaqi.get("pm10") else None
        }
        
    except requests.exceptions.RequestException:
        return None
    except (KeyError, ValueError):
        return None

#fungsi dari function ini adalah untuk mencegah dupplikat data dari dataset csv atau output kita
def load_existing_keys():
    existing_keys = set()
    if os.path.exists(OUTPUT_CSV):
        try:
            with open(OUTPUT_CSV, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    dt = row.get("datetime", "").strip()
                    city = row.get("city", "").strip()
                    if dt and city:
                        existing_keys.add((dt, city))
        except (IOError, csv.Error):
            pass
    return existing_keys

#ini merupakan function untuk menggabungkan cuaca  dan aqi dari api menjadi dataset agar bisa dipakai untuk machine learning
def generate_dataset():
    now = datetime.now()
    current_time = now.replace(minute=0, second=0, microsecond=0).strftime("%Y-%m-%d %H:00:00")
    
    existing_keys = load_existing_keys()
    file_exists = os.path.exists(OUTPUT_CSV)
    
    rows_to_write = []
    
    for city_info in CITIES:
        city_name = city_info["city"]
        lat = city_info["lat"]
        lon = city_info["lon"]
        
        key = (current_time, city_name)
        if key in existing_keys:
            print(f"City skipped: {city_name} (duplicate: {current_time})")
            continue
        
        weather_data = get_weather(lat, lon)
        if weather_data is None:
            print(f"API failure: Weather API failed for {city_name}")
            continue
        
        aqi_data = get_aqi(lat, lon)
        if aqi_data is None:
            print(f"API failure: AQI API failed for {city_name}")
            continue
        
        temperature = weather_data.get("temperature")
        humidity = weather_data.get("humidity")
        wind_speed = weather_data.get("wind_speed")
        aqi = aqi_data.get("aqi")
        
        if temperature is None or humidity is None or wind_speed is None or aqi is None:
            print(f"City skipped: {city_name} (missing required fields)")
            continue
        
        pm25 = aqi_data.get("pm25")
        pm10 = aqi_data.get("pm10")
        
        record = {
            "datetime": current_time,
            "city": city_name,
            "temperature": temperature,
            "humidity": humidity,
            "wind_speed": wind_speed,
            "pm25": pm25 if pm25 is not None else "",
            "pm10": pm10 if pm10 is not None else "",
            "aqi": aqi
        }
        
        rows_to_write.append(record)
        print(f"Row written: {city_name} at {current_time}")
    
    if rows_to_write:
        with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            if not file_exists:
                writer.writeheader()
            for record in rows_to_write:
                writer.writerow(record)

#langsung jalanin scriptnya
if __name__ == "__main__":
    generate_dataset()

