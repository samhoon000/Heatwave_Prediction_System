from flask import Flask, request, render_template
import joblib
import pandas as pd
import requests
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv('WEATHER_API_KEY')

app = Flask(__name__, template_folder='templates')

# Load model and encoder
model = joblib.load("model/heatwave_alert_model.pkl")
city_encoder = joblib.load("model/city_encoder.pkl")

# Weather API config
API_KEY = '083b4b5c19f0468da3653839250108'
API_URL = 'https://api.weatherapi.com/v1/current.json'

@app.route('/')
def index():
    return render_template('test.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        city = request.form['location']
        encoded_city = city_encoder.transform([city])[0]

        # Fetch real-time weather data
        params = {'key': API_KEY, 'q': city}
        response = requests.get(API_URL, params=params)
        weather = response.json()

        # Extract weather features
        temperature_c = weather['current']['temp_c']  # for display only
        humidity = weather['current']['humidity']
        wind_kph = weather['current']['wind_kph']
        wind_mps = wind_kph / 3.6  # Convert kph to m/s

        # Extract time features
        now = datetime.now()
        hour = now.hour
        month = now.month
        weekday = now.weekday()

        # Create input DataFrame with only expected features
        input_data = pd.DataFrame([{
            'city': encoded_city,
            'humidity': humidity,
            'wind_speed': wind_mps,
            'hour': hour,
            'month': month,
            'weekday': weekday
        }])

        # Predict
        prediction = model.predict(input_data)[0]
        result = "⚠️ Heatwave Predicted!" if prediction == 1 else "✅ No Heatwave Predicted."

        return render_template(
            'test.html',
            city=city,
            temperature=round(temperature_c, 2),
            humidity=humidity,
            wind_speed=round(wind_kph, 2),
            result=result
        )

    except Exception as e:
        return render_template('test.html', error=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
