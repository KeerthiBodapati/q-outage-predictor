import pandas as pd
import numpy as np
import json
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, accuracy_score

# Load outage data
train = pd.read_csv('./data/train.csv')
validation = pd.read_csv('./data/validation.csv')

# Load features
with open('./artifacts/feature_config.json', 'r') as f:
    features = json.load(f)

# Create outage data with labels
train_outage = train.copy()
train_outage['outage_occurred'] = 1

val_outage = validation.copy()
val_outage['outage_occurred'] = 1

# Generate no-outage data directly as DataFrame - MUCH MORE to balance
no_outage_data = []
for loc_code in train['loc_code'].unique():
    loc_data = train[train['loc_code'] == loc_code]
    n_samples = int(len(loc_data) * 3)  # 3x more no-outage samples
    
    for _ in range(n_samples):
        # Ideal conditions for no outages
        hour = np.random.choice([9, 10, 11, 12, 13, 14, 15, 16])  # Business hours
        dow = np.random.choice([0, 1, 2, 3, 4])  # Weekdays only
        
        row = {
            'loc_code': loc_code,
            'hour': hour,
            'dow': dow,
            'month': np.random.randint(1, 13),
            'is_weekend': 0,  # Always weekday
            'duration_lag1': 0,  # No recent outages
            'duration_mean_3': 0,  # No historical pattern
            'duration_delta': 0,
            'w_rain': 0, 'w_snow': 0, 'w_windy': 0, 'w_clear': 1, 'w_storm': 0,
            'weather_severity': 1,  # Always clear weather
            'severe_night': 0, 'severe_weekend': 0,
            'duration_mins': 0,
            'outage_occurred': 0
        }
        no_outage_data.append(row)

train_no_outage = pd.DataFrame(no_outage_data)
val_no_outage = train_no_outage.sample(frac=0.3).reset_index(drop=True)
train_no_outage = train_no_outage.drop(val_no_outage.index).reset_index(drop=True)

# Combine outage and no-outage data
train_combined = pd.concat([train_outage, train_no_outage], ignore_index=True)
val_combined = pd.concat([val_outage, val_no_outage], ignore_index=True)

# Train probability classifier
X_train_prob = train_combined[features]
y_train_prob = train_combined['outage_occurred']
X_val_prob = val_combined[features]
y_val_prob = val_combined['outage_occurred']

prob_model = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42)
prob_model.fit(X_train_prob, y_train_prob)

# Evaluate probability model
y_pred_prob = prob_model.predict(X_val_prob)
prob_accuracy = accuracy_score(y_val_prob, y_pred_prob)

# Train duration model (only on outage data)
X_train_dur = train[features]
y_train_dur = train['duration_mins']
X_val_dur = validation[features]
y_val_dur = validation['duration_mins']

duration_model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42)
duration_model.fit(X_train_dur, y_train_dur)

# Save both models
joblib.dump(prob_model, './artifacts/probability_model.joblib')
joblib.dump(duration_model, './artifacts/duration_model.joblib')

# Save metrics
metrics = {
    'probability_accuracy': prob_accuracy,
    'outage_samples': len(train_outage) + len(val_outage),
    'no_outage_samples': len(train_no_outage) + len(val_no_outage)
}

with open('./artifacts/probability_metrics.json', 'w') as f:
    json.dump(metrics, f)

print(f"Probability Model Accuracy: {prob_accuracy:.3f}")
print(f"Outage samples: {metrics['outage_samples']}")
print(f"No-outage samples: {metrics['no_outage_samples']}")
print("Models saved successfully!")