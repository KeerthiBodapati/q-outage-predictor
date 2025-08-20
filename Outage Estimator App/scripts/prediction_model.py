import argparse
import pandas as pd
import json
import joblib

# Parse CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument('--timestamp', required=True)
parser.add_argument('--location', required=True)
parser.add_argument('--weather', required=True)
args = parser.parse_args()

# Load duration model and artifacts
duration_model = joblib.load('./artifacts/model.joblib')
with open('./artifacts/feature_config.json', 'r') as f:
    features = json.load(f)
with open('./artifacts/loc_map.json', 'r') as f:
    loc_map = json.load(f)

# Parse timestamp
timestamp = pd.to_datetime(args.timestamp)
hour = timestamp.hour
dow = timestamp.dayofweek
month = timestamp.month
is_weekend = int(dow >= 5)

# Get location code
loc_code = 0
for code, location in loc_map.items():
    if location.lower() == args.location.lower():
        loc_code = int(code)
        break

# Weather features
weather_lower = args.weather.lower()
w_rain = int('rain' in weather_lower)
w_snow = int('snow' in weather_lower)
w_windy = int('wind' in weather_lower)
w_clear = int('clear' in weather_lower or 'sunny' in weather_lower)
w_storm = int('storm' in weather_lower or 'thunder' in weather_lower)

# Weather severity
weather_severity_map = {
    'lightning': 8, 'hurricane': 12, 'thunderstorm': 10, 'equipment failure': 15,
    'tree fall': 9, 'monsoon': 11, 'snowstorm': 7, 'heavy rain': 6,
    'overload': 5, 'heatwave': 3, 'fog': 2, 'clear': 1, 'sunny': 1
}
weather_severity = max([score for keyword, score in weather_severity_map.items() if keyword in weather_lower] + [4])

# Weather-time interactions
severe_night = int((hour < 6) or (hour > 20)) * weather_severity
severe_weekend = is_weekend * weather_severity

# Load historical data
train = pd.read_csv('./data/train.csv')
validation = pd.read_csv('./data/validation.csv')
all_data = pd.concat([train, validation]).reset_index(drop=True)

# Get historical data for this location
loc_data = all_data[all_data['loc_code'] == loc_code].copy()
if len(loc_data) > 0:
    duration_lag1 = loc_data['duration_mins'].iloc[-1]
    duration_mean_3 = loc_data['duration_mins'].tail(3).mean()
    duration_delta = 0
else:
    duration_lag1 = 240  # Default 4 hours
    duration_mean_3 = 300  # Default 5 hours
    duration_delta = 0

# Simple rule-based probability calculation
base_prob = 0.3  # Base 30% chance

# Weather impact
if weather_severity <= 2:  # Clear/sunny
    weather_impact = -0.2
elif weather_severity <= 5:  # Mild conditions
    weather_impact = 0.0
elif weather_severity <= 10:  # Severe weather
    weather_impact = 0.3
else:  # Extreme weather
    weather_impact = 0.5

# Time impact
if 6 <= hour <= 18 and not is_weekend:  # Business hours, weekday
    time_impact = -0.1
elif hour < 6 or hour > 20:  # Night
    time_impact = 0.1
else:
    time_impact = 0.0

# Calculate final probability
outage_prob = base_prob + weather_impact + time_impact
outage_prob = max(0.05, min(0.95, outage_prob))  # Clamp between 5-95%

print(f"Outage Probability: {outage_prob:.1%}")

if outage_prob < 0.4:
    print("Prediction: No outage expected")
else:
    # Build feature vector for duration prediction
    feature_vector = [
        loc_code, hour, dow, month, is_weekend,
        duration_lag1, duration_mean_3, duration_delta,
        w_rain, w_snow, w_windy, w_clear, w_storm,
        weather_severity, severe_night, severe_weekend
    ]
    
    feature_df = pd.DataFrame([feature_vector], columns=features)
    duration = duration_model.predict(feature_df)[0]
    print(f"Prediction: Outage expected, duration: {duration:.1f} minutes ({duration/60:.1f} hours)")