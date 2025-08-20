import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder

# Read and standardize columns
df = pd.read_csv('./data/raw.csv')
df.columns = df.columns.str.lower()
col_map = {col: ['timestamp', 'location', 'weather', 'duration_mins'][i] 
           for i, col in enumerate(df.columns[:4])}
df = df.rename(columns=col_map)
# Convert hours to minutes
df['duration_mins'] = df['duration_mins'] * 60

# Parse timestamp and sort
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values(['location', 'timestamp']).reset_index(drop=True)

# Time features
df['hour'] = df['timestamp'].dt.hour
df['dow'] = df['timestamp'].dt.dayofweek
df['month'] = df['timestamp'].dt.month
df['is_weekend'] = (df['dow'] >= 5).astype(int)

# Weather flags and severity scoring
weather_lower = df['weather'].str.lower()
df['w_rain'] = weather_lower.str.contains('rain', na=False).astype(int)
df['w_snow'] = weather_lower.str.contains('snow', na=False).astype(int)
df['w_windy'] = weather_lower.str.contains('wind', na=False).astype(int)
df['w_clear'] = weather_lower.str.contains('clear|sunny', na=False).astype(int)
df['w_storm'] = weather_lower.str.contains('storm|thunder', na=False).astype(int)

# Weather severity score (higher = more severe)
weather_severity = {
    'lightning': 8, 'hurricane': 12, 'thunderstorm': 10, 'equipment failure': 15,
    'tree fall': 9, 'monsoon': 11, 'snowstorm': 7, 'heavy rain': 6,
    'overload': 5, 'heatwave': 3, 'fog': 2, 'clear': 1, 'sunny': 1
}
df['weather_severity'] = weather_lower.apply(
    lambda x: max([score for keyword, score in weather_severity.items() if keyword in x] + [4])
)

# Weather-time interactions
df['severe_night'] = ((df['hour'] < 6) | (df['hour'] > 20)) * df['weather_severity']
df['severe_weekend'] = df['is_weekend'] * df['weather_severity']

# Location encoding
le = LabelEncoder()
df['loc_code'] = le.fit_transform(df['location'])
loc_map = {int(code): location for code, location in enumerate(le.classes_)}
with open('./artifacts/loc_map.json', 'w') as f:
    json.dump(loc_map, f)

# Lag/rolling features per location
df['duration_lag1'] = df.groupby('location')['duration_mins'].shift(1)
df['duration_mean_3'] = df.groupby('location')['duration_mins'].rolling(3, min_periods=1).mean().reset_index(0, drop=True)
df['duration_delta'] = df['duration_mins'] - df['duration_lag1']

# Select features
features = ['loc_code','hour','dow','month','is_weekend','duration_lag1','duration_mean_3','duration_delta','w_rain','w_snow','w_windy','w_clear','w_storm','weather_severity','severe_night','severe_weekend']
df_features = df[features + ['duration_mins']].dropna()

# Time-based split
split_idx = int(len(df_features) * 0.8)
train = df_features.iloc[:split_idx]
validation = df_features.iloc[split_idx:]

# Save datasets
train.to_csv('./data/train.csv', index=False)
validation.to_csv('./data/validation.csv', index=False)
df_features.head(100).to_parquet('./data/sample_features.parquet', index=False)

# Save feature config
with open('./artifacts/feature_config.json', 'w') as f:
    json.dump(features, f)

print(f"Train shape: {train.shape}")
print(f"Validation shape: {validation.shape}")