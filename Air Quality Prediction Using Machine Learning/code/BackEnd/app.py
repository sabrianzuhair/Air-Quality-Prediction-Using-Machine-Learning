from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
import joblib
import json
import os
import pandas as pd
import numpy as np
from datetime import datetime

app = Flask(__name__)
CORS(app)


AQICN_TOKEN = "bb7bf66677d85e1ca4a9d21a7e4db09539447ca7"
OPENWEATHER_API_KEY = "c3c265d7c469d367bc53bf55bfa12f43"


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH_RF = os.path.join(PROJECT_ROOT, "C:\\Air Quality Prediction Using Machine Learning\\code\\Test\\random_forest_aqi.joblib")
MODEL_PATH_LR = os.path.join(PROJECT_ROOT, "C:\\Air Quality Prediction Using Machine Learning\\code\\Test\\linear_regression_aqi.joblib")
METADATA_PATH = os.path.join(PROJECT_ROOT, "C:\\Air Quality Prediction Using Machine Learning\\code\\Test\\model_metadata.json")
METADATA_PATH_LR = os.path.join(PROJECT_ROOT, "C:\\Air Quality Prediction Using Machine Learning\\code\\Test\\linear_regression_metadata.json")
CITIES_DATA = {
    "Jakarta": {"lat": -6.2088, "lon": 106.8456},
    "Bandung": {"lat": -6.9175, "lon": 107.6191},
    "Surabaya": {"lat": -7.2575, "lon": 112.7521},
    "Medan": {"lat": 3.5952, "lon": 98.6722},
    "Makassar": {"lat": -5.1477, "lon": 119.4327},
    "Semarang": {"lat": -6.9667, "lon": 110.4167},
    "Palembang": {"lat": -2.9911, "lon": 104.7567},
    "Denpasar": {"lat": -8.6705, "lon": 115.2126},
    "Yogyakarta": {"lat": -7.7956, "lon": 110.3695},
    "Malang": {"lat": -7.9797, "lon": 112.6304}
}


model = None  
model_lr = None  
available_cities = []
model_metadata = None  
model_metadata_lr = None  

#membuat dua model ai yaitu random forest dan linear regression beserta metadata saat server flsk dijalankan
def load_model_and_metadata():
    
    global model, model_lr, available_cities, model_metadata, model_metadata_lr
    
    try:
       
        if not os.path.exists(MODEL_PATH_RF):
            raise FileNotFoundError(f"Random Forest model file not found: {MODEL_PATH_RF}")
        
        app.logger.info(f"Loading Random Forest model from: {MODEL_PATH_RF}")
        model = joblib.load(MODEL_PATH_RF)
        app.logger.info(f"✅ Random Forest model loaded successfully")
        
        
        if not hasattr(model, 'named_steps'):
            app.logger.warning("⚠️  Random Forest model may not be a Pipeline - ensure it includes preprocessing")
        
       
        if not os.path.exists(MODEL_PATH_LR):
            app.logger.warning(f"⚠️  Linear Regression model file not found: {MODEL_PATH_LR}")
            app.logger.warning("   API will continue with Random Forest only")
            model_lr = None
        else:
            app.logger.info(f"Loading Linear Regression model from: {MODEL_PATH_LR}")
            model_lr = joblib.load(MODEL_PATH_LR)
            app.logger.info(f"✅ Linear Regression model loaded successfully")
        
        
        if not os.path.exists(METADATA_PATH):
            app.logger.warning(f"Random Forest metadata not found at {METADATA_PATH}, using default city list")
            available_cities = sorted(CITIES_DATA.keys())
            model_metadata = None
        else:
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                model_metadata = json.load(f)
                available_cities = model_metadata.get("cities", sorted(CITIES_DATA.keys()))
                app.logger.info(f"✅ Random Forest metadata loaded: {len(available_cities)} cities available")
        
        
        if os.path.exists(METADATA_PATH_LR):
            with open(METADATA_PATH_LR, "r", encoding="utf-8") as f:
                model_metadata_lr = json.load(f)
                app.logger.info(f"✅ Linear Regression metadata loaded")
        else:
            model_metadata_lr = None
            app.logger.warning(f"Linear Regression metadata not found at {METADATA_PATH_LR}")
        
        
        metadata_cities = set(available_cities)
        cities_data_keys = set(CITIES_DATA.keys())
        
        if not metadata_cities.issubset(cities_data_keys):
            missing = metadata_cities - cities_data_keys
            app.logger.warning(f"⚠️  Cities in metadata not in CITIES_DATA: {missing}")
        
        app.logger.info(f"✅ Models and metadata loaded successfully")
        app.logger.info(f"   Random Forest: {'✅' if model else '❌'}")
        app.logger.info(f"   Linear Regression: {'✅' if model_lr else '❌'}")
        app.logger.info(f"   Available cities: {available_cities}")
        
    except Exception as e:
        app.logger.error(f"❌ Failed to load models: {e}", exc_info=True)
        raise

#mengambil data cuaca dari API openweather
def get_weather_data(lat: float, lon: float) -> dict:

    url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        "lat": lat,
        "lon": lon,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric"
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200:
            app.logger.warning(f"Weather API returned status {response.status_code}")
            return None
        
        data = response.json()
        return {
            "temperature": data.get("main", {}).get("temp"),
            "humidity": data.get("main", {}).get("humidity"),
            "wind_speed": data.get("wind", {}).get("speed")
        }
    except Exception as e:
        app.logger.error(f"Weather API error: {e}")
        return None

#mengambil data kualitas udara dari aqicn berdasarkan lokasi
def get_aqi_data(lat: float, lon: float) -> dict:
    
    url = f"https://api.waqi.info/feed/geo:{lat};{lon}/?token={AQICN_TOKEN}"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            app.logger.warning(f"AQI API returned status {response.status_code}")
            return None
        
        data = response.json()
        if data.get("status") != "ok":
            app.logger.warning(f"AQI API returned non-ok status: {data.get('status')}")
            return None
        
        iaqi = data.get("data", {}).get("iaqi", {})
        return {
            "aqi": data.get("data", {}).get("aqi"),
            "pm25": iaqi.get("pm25", {}).get("v") if iaqi.get("pm25") else None,
            "pm10": iaqi.get("pm10", {}).get("v") if iaqi.get("pm10") else None
        }
    except Exception as e:
        app.logger.error(f"AQI API error: {e}")
        return None

#Mengubah nilai AQI menjadi kategori kualitas udara
def aqi_to_category(aqi_value: float) -> str:
   
    if aqi_value is None or (isinstance(aqi_value, float) and np.isnan(aqi_value)):
        return "Unknown"
    
    aqi = float(aqi_value)
    if 0 <= aqi <= 50:
        return "Good"
    elif 51 <= aqi <= 100:
        return "Moderate"
    elif 101 <= aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif 151 <= aqi <= 200:
        return "Unhealthy"
    elif 201 <= aqi <= 300:
        return "Very Unhealthy"
    elif 301 <= aqi <= 500:
        return "Hazardous"
    else:
        return "Out of Range"

#melakukan prediksi AQI menggunakan model random forest
def predict_rf(city: str, temperature: float, humidity: float, 
                wind_speed: float, pm25: float = None, pm10: float = None) -> float:
    
    if model is None:
        raise RuntimeError("Random Forest model not loaded")
    
    
    if city not in available_cities:
        raise ValueError(f"City '{city}' not in available cities: {available_cities}")
    
    
    if temperature is None or humidity is None or wind_speed is None:
        raise ValueError("Missing required weather data: temperature, humidity, wind_speed")
    
    
    input_data = pd.DataFrame({
        "temperature": [float(temperature)],
        "humidity": [float(humidity)],
        "wind_speed": [float(wind_speed)],
        "pm25": [float(pm25) if pm25 is not None else np.nan],
        "pm10": [float(pm10) if pm10 is not None else np.nan],
        "city": [str(city)]  
    })
    
    try:
        prediction = model.predict(input_data)[0]
        
        result = round(float(prediction), 2)
        app.logger.debug(f"Random Forest prediction for {city}: {result}")
        return result
    except Exception as e:
        app.logger.error(f"Random Forest prediction failed for city={city}: {e}", exc_info=True)
        raise RuntimeError(f"Random Forest prediction failed: {str(e)}")

#melakukan prediksi AQI menggunakan model linear regression
def predict_lr(city: str, temperature: float, humidity: float, 
                wind_speed: float, pm25: float = None, pm10: float = None) -> float:
   
    if model_lr is None:
        raise RuntimeError("Linear Regression model not loaded")
    
    
    if city not in available_cities:
        raise ValueError(f"City '{city}' not in available cities: {available_cities}")
    
    
    if temperature is None or humidity is None or wind_speed is None:
        raise ValueError("Missing required weather data: temperature, humidity, wind_speed")
    
    
    from sklearn.preprocessing import LabelEncoder
    
    
    cities_from_metadata = model_metadata_lr.get("cities", available_cities) if model_metadata_lr else available_cities
    
    
    le = LabelEncoder()
    le.fit(cities_from_metadata)
    city_encoded = le.transform([city])[0]
    
    features = np.array([[
        float(temperature),
        float(humidity),
        float(wind_speed),
        float(pm25) if pm25 is not None else np.nan,
        float(pm10) if pm10 is not None else np.nan,
        int(city_encoded)
    ]])
    
    try:
        prediction = model_lr.predict(features)[0]
        
        result = round(float(prediction), 2)
        app.logger.debug(f"Linear Regression prediction for {city}: {result}")
        return result
    except Exception as e:
        app.logger.error(f"Linear Regression prediction failed for city={city}: {e}", exc_info=True)
        raise RuntimeError(f"Linear Regression prediction failed: {str(e)}")

#mengecek kesehatan server API
@app.route("/health", methods=["GET"])
def health_check():
    
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "model_lr_loaded": model_lr is not None,  # NEW: Check Linear Regression model
        "cities_available": len(available_cities),
        "model_type": model_metadata.get("model_type", "Unknown") if model_metadata else "Unknown"
    }), 200

#mengirim daftar kota yang tersedia
@app.route("/api/cities", methods=["GET"])
def get_cities():
    """Get list of available cities."""
    return jsonify({
        "cities": available_cities,
        "count": len(available_cities)
    }), 200

#endpoint utama untuk prediksi AQI berdasarkan nama kota
@app.route("/api/aqi", methods=["GET"])
def get_aqi_prediction():
    
    city = request.args.get("city")
    
    if not city:
        return jsonify({
            "error": "City parameter is required",
            "available_cities": available_cities
        }), 400
    
   
    city = city.strip().title()
    
   
    if city not in available_cities:
        app.logger.warning(f"City validation failed: '{city}' not in available cities")
        return jsonify({
            "error": "City not supported",
            "available_cities": available_cities
        }), 400
    
    
    if model_metadata and city not in model_metadata.get("cities", []):
        app.logger.warning(f"City '{city}' not found in model metadata")
        
   
    city_info = CITIES_DATA.get(city)
    if not city_info:
        app.logger.error(f"City coordinates not found for {city}")
        return jsonify({
            "error": f"City coordinates not found for {city}",
            "available_cities": available_cities
        }), 500
    
    lat = city_info["lat"]
    lon = city_info["lon"]
    
    
    weather_data = get_weather_data(lat, lon)
    if weather_data is None:
        app.logger.error(f"Failed to fetch weather data for {city}")
        return jsonify({
            "error": "Failed to fetch weather data from OpenWeather API",
            "city": city
        }), 503
    
    
    temperature = weather_data.get("temperature")
    humidity = weather_data.get("humidity")
    wind_speed = weather_data.get("wind_speed")
    
    
    if temperature is None or humidity is None or wind_speed is None:
        app.logger.error(f"Incomplete weather data for {city}: {weather_data}")
        return jsonify({
            "error": "Incomplete weather data received",
            "message": "Missing required weather fields: temperature, humidity, or wind_speed",
            "city": city
        }), 400
    
    
    try:
        temperature = float(temperature)
        humidity = float(humidity)
        wind_speed = float(wind_speed)
    except (ValueError, TypeError) as e:
        app.logger.error(f"Invalid weather data types for {city}: {e}")
        return jsonify({
            "error": "Invalid weather data format",
            "message": "Weather data must be numeric values",
            "city": city
        }), 400
    
    
    aqi_data = get_aqi_data(lat, lon)
    aqi_actual = None
    pm25 = None
    pm10 = None
    
    if aqi_data:
        aqi_actual = aqi_data.get("aqi")
        pm25 = aqi_data.get("pm25")
        pm10 = aqi_data.get("pm10")
    
    
    aqi_predicted_rf = None
    aqi_predicted_lr = None
    
    
    try:
        if model is not None:
            aqi_predicted_rf = predict_rf(
                city=city,
                temperature=temperature,
                humidity=humidity,
                wind_speed=wind_speed,
                pm25=pm25,
                pm10=pm10
            )
        else:
            app.logger.warning("Random Forest model not available")
    except ValueError as e:
        
        app.logger.error(f"City validation error: {e}")
        return jsonify({
            "error": str(e),
            "available_cities": available_cities
        }), 400
    except Exception as e:
        app.logger.error(f"Random Forest prediction error for {city}: {e}", exc_info=True)
        
    
    
    try:
        if model_lr is not None:
            aqi_predicted_lr = predict_lr(
                city=city,
                temperature=temperature,
                humidity=humidity,
                wind_speed=wind_speed,
                pm25=pm25,
                pm10=pm10
            )
        else:
            app.logger.warning("Linear Regression model not available")
    except Exception as e:
        app.logger.error(f"Linear Regression prediction error for {city}: {e}", exc_info=True)
        
    
    
    if aqi_predicted_rf is None and aqi_predicted_lr is None:
        app.logger.error(f"Both models failed for {city}")
        return jsonify({
            "error": "Failed to generate predictions from any model",
            "city": city,
            "available_cities": available_cities
        }), 500
    
    
    if aqi_predicted_rf is None:
        aqi_predicted_rf = float(aqi_actual) if aqi_actual is not None else (float(aqi_predicted_lr) if aqi_predicted_lr is not None else 0.0)
        app.logger.warning(f"Using fallback value for Random Forest prediction: {aqi_predicted_rf}")
    
    if aqi_predicted_lr is None:
        aqi_predicted_lr = float(aqi_actual) if aqi_actual is not None else (float(aqi_predicted_rf) if aqi_predicted_rf is not None else 0.0)
        app.logger.warning(f"Using fallback value for Linear Regression prediction: {aqi_predicted_lr}")
    
    
    aqi_predicted_rf = round(float(aqi_predicted_rf), 2)
    aqi_predicted_lr = round(float(aqi_predicted_lr), 2)
    
    
    aqi_for_category = aqi_actual if aqi_actual is not None else ((aqi_predicted_rf + aqi_predicted_lr) / 2.0)
    category = aqi_to_category(aqi_for_category)
    
    
    aqi_actual_value = round(float(aqi_actual), 2) if aqi_actual is not None else 0.0
    
    
    response = {
        "city": str(city),
        "aqi_actual": aqi_actual_value,  
        "aqi_predicted_lr": aqi_predicted_lr,  
        "aqi_predicted_rf": aqi_predicted_rf,  
        "category": str(category),
        "weather": {
            "temperature": round(float(temperature), 2),
            "humidity": int(float(humidity)),
            "wind_speed": round(float(wind_speed), 2)
        },
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")  # ISO format
    }
    
    return jsonify(response), 200


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        "error": "Endpoint not found",
        "available_endpoints": [
            "GET /health",
            "GET /api/cities",
            "GET /api/aqi?city=<city_name>"
        ]
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    app.logger.error("Internal server error", exc_info=True)
    return jsonify({
        "error": "Internal server error",
        "message": "An unexpected error occurred"
    }), 500


if __name__ == "__main__":
    
    try:
        load_model_and_metadata()
        app.logger.info("🚀 Flask API started successfully")
        app.logger.info(f"   Models: Random Forest {'✅' if model else '❌'}, Linear Regression {'✅' if model_lr else '❌'}")
        app.logger.info(f"   Endpoints: /health, /api/cities, /api/aqi?city=<name>")
        app.run(debug=True, host="0.0.0.0", port=5000)
    except Exception as e:
        app.logger.error(f"❌ Failed to start API: {e}", exc_info=True)
        raise
