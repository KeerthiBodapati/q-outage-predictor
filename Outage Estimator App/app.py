import streamlit as st
import pandas as pd
import json
import joblib
from datetime import datetime, time
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Outage Prediction System",
    page_icon="⚡",
    layout="wide"
)

@st.cache_data
def load_artifacts():
    """Load model artifacts and data"""
    try:
        duration_model = joblib.load('./artifacts/model.joblib')
        with open('./artifacts/feature_config.json', 'r') as f:
            features = json.load(f)
        with open('./artifacts/loc_map.json', 'r') as f:
            loc_map = json.load(f)
        
        train = pd.read_csv('./data/train.csv')
        validation = pd.read_csv('./data/validation.csv')
        all_data = pd.concat([train, validation]).reset_index(drop=True)
        
        return duration_model, features, loc_map, all_data
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        return None, None, None, None

def predict_outage(timestamp, location, weather, duration_model, features, loc_map, all_data):
    """Prediction logic from the script"""
    # Parse timestamp
    hour = timestamp.hour
    dow = timestamp.weekday()
    month = timestamp.month
    is_weekend = int(dow >= 5)
    
    # Get location code
    loc_code = 0
    for code, loc_name in loc_map.items():
        if loc_name.lower() == location.lower():
            loc_code = int(code)
            break
    
    # Weather features
    weather_lower = weather.lower()
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
    
    # Get historical data for this location
    loc_data = all_data[all_data['loc_code'] == loc_code].copy()
    if len(loc_data) > 0:
        duration_lag1 = loc_data['duration_mins'].iloc[-1]
        duration_mean_3 = loc_data['duration_mins'].tail(3).mean()
        duration_delta = 0
    else:
        duration_lag1 = 240
        duration_mean_3 = 300
        duration_delta = 0
    
    # Rule-based probability calculation
    base_prob = 0.3
    
    # Weather impact
    if weather_severity <= 2:
        weather_impact = -0.2
    elif weather_severity <= 5:
        weather_impact = 0.0
    elif weather_severity <= 10:
        weather_impact = 0.3
    else:
        weather_impact = 0.5
    
    # Time impact
    if 6 <= hour <= 18 and not is_weekend:
        time_impact = -0.1
    elif hour < 6 or hour > 20:
        time_impact = 0.1
    else:
        time_impact = 0.0
    
    # Calculate final probability
    outage_prob = base_prob + weather_impact + time_impact
    outage_prob = max(0.05, min(0.95, outage_prob))
    
    # Duration prediction if outage expected
    duration = None
    if outage_prob >= 0.4:
        feature_vector = [
            loc_code, hour, dow, month, is_weekend,
            duration_lag1, duration_mean_3, duration_delta,
            w_rain, w_snow, w_windy, w_clear, w_storm,
            weather_severity, severe_night, severe_weekend
        ]
        
        feature_df = pd.DataFrame([feature_vector], columns=features)
        duration = duration_model.predict(feature_df)[0]
    
    return outage_prob, duration

def create_probability_gauge(probability):
    """Create a gauge chart for probability"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Outage Probability (%)"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 40], 'color': "lightgreen"},
                {'range': [40, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

# Main app
def main():
    st.title("⚡ Power Outage Prediction System")
    st.markdown("Predict outage probability and duration based on weather conditions and location")
    
    # Load artifacts
    duration_model, features, loc_map, all_data = load_artifacts()
    
    if duration_model is None:
        st.error("Failed to load model artifacts. Please ensure all files are present.")
        return
    
    # Sidebar inputs
    st.sidebar.header("Input Parameters")
    
    # Date and time inputs
    date_input = st.sidebar.date_input("Date", datetime.now().date())
    time_input = st.sidebar.time_input("Time", time(12, 0))
    timestamp = datetime.combine(date_input, time_input)
    
    # Location dropdown
    locations = list(loc_map.values())
    location = st.sidebar.selectbox("Location", locations)
    
    # Weather condition
    weather_options = [
        "Clear", "Sunny", "Rain", "Heavy Rain", "Snow", "Snowstorm",
        "Thunderstorm", "Lightning", "Hurricane", "Fog", "Heatwave",
        "Equipment Failure", "Tree Fall", "Overload", "Monsoon"
    ]
    weather = st.sidebar.selectbox("Weather Condition", weather_options)
    
    # Custom weather input
    custom_weather = st.sidebar.text_input("Or enter custom weather condition:")
    if custom_weather:
        weather = custom_weather
    
    # Prediction button
    if st.sidebar.button("Predict Outage", type="primary"):
        with st.spinner("Analyzing conditions..."):
            outage_prob, duration = predict_outage(
                timestamp, location, weather, duration_model, features, loc_map, all_data
            )
        
        # Display results
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Probability Analysis")
            fig = create_probability_gauge(outage_prob)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Prediction Result")
            
            if outage_prob < 0.4:
                st.success("✅ No outage expected")
                st.metric("Outage Probability", f"{outage_prob:.1%}")
            else:
                st.error("⚠️ Outage expected")
                st.metric("Outage Probability", f"{outage_prob:.1%}")
                if duration:
                    st.metric("Expected Duration", f"{duration:.0f} minutes")
                    st.metric("Duration (Hours)", f"{duration/60:.1f} hours")
        
        # Additional info
        st.subheader("Analysis Details")
        col3, col4, col5 = st.columns(3)
        
        with col3:
            st.info(f"**Location:** {location}")
            st.info(f"**Date/Time:** {timestamp.strftime('%Y-%m-%d %H:%M')}")
        
        with col4:
            st.info(f"**Weather:** {weather}")
            day_type = "Weekend" if timestamp.weekday() >= 5 else "Weekday"
            st.info(f"**Day Type:** {day_type}")
        
        with col5:
            time_period = "Night" if timestamp.hour < 6 or timestamp.hour > 20 else "Day"
            st.info(f"**Time Period:** {time_period}")
            business_hours = "Yes" if 6 <= timestamp.hour <= 18 and timestamp.weekday() < 5 else "No"
            st.info(f"**Business Hours:** {business_hours}")

if __name__ == "__main__":
    main()