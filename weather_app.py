import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from meteostat import Point, Daily, Hourly
import warnings
import os
import pickle
warnings.filterwarnings('ignore')

# Configure the page
st.set_page_config(page_title="Weather Forecast App", layout="wide")
st.title("Weather Forecast Analysis and Prediction")

def get_weather_data(lat, lon, start_date, end_date, use_cache=True):
    """Fetch historical weather data for the given coordinates"""
    try:
        # Create cache directory if it doesn't exist
        cache_dir = "weather_cache"
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = f"{cache_dir}/weather_data_{lat}_{lon}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.pkl"
        
        # Try to load from cache if enabled
        if use_cache and os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
                st.info("Using cached weather data")
                return data
        
        # Create Point for the location
        location = Point(lat, lon)
        
        # Get hourly data
        data = Hourly(location, start_date, end_date)
        data = data.fetch()
        
        if data.empty:
            st.error("No data available for this location. Please try different coordinates.")
            return None
        
        # Save to cache if data was successfully retrieved
        if use_cache:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            st.success("Weather data cached for future use")
            
        return data
    except Exception as e:
        st.error(f"Error fetching weather data: {str(e)}")
        return None

def prepare_data(df):
    """Prepare the data for modeling"""
    if df is None or df.empty:
        return None
    
    try:
        # Reset index to make date a column
        df = df.reset_index()
        
        # Rename columns for consistency
        df = df.rename(columns={
            'time': 'datetime',
            'temp': 'temperature',
            'prcp': 'precipitation',
            'wspd': 'wind_speed'
        })
        
        # Fill missing values with forward fill then backward fill
        df['temperature'] = df['temperature'].fillna(method='ffill').fillna(method='bfill')
        df['precipitation'] = df['precipitation'].fillna(0)
        df['wind_speed'] = df['wind_speed'].fillna(method='ffill').fillna(method='bfill')
        
        return df
    except Exception as e:
        st.error(f"Error preparing data: {str(e)}")
        return None

def train_model(df):
    """Train a Random Forest model on the weather data"""
    try:
        # Create features from datetime
        df['hour'] = df['datetime'].dt.hour
        df['day'] = df['datetime'].dt.day
        df['month'] = df['datetime'].dt.month
        df['year'] = df['datetime'].dt.year
        df['day_of_year'] = df['datetime'].dt.dayofyear
        
        # Add cyclical time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year']/365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year']/365)
        
        # Prepare features and target
        features = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 
                   'precipitation', 'wind_speed']
        X = df[features].fillna(0)  # Ensure no NaN values
        y = df['temperature']
        
        # Train model with increased number of trees for better accuracy
        with st.spinner("Training model on historical data..."):
            model = RandomForestRegressor(
                n_estimators=200,  # Increased number of trees
                max_depth=15,      # Limit tree depth to prevent overfitting
                n_jobs=-1,         # Use all CPU cores
                random_state=42
            )
            model.fit(X, y)
            
            # Calculate and display feature importance
            feature_importance = pd.DataFrame({
                'feature': features,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            st.subheader("Feature Importance")
            st.dataframe(feature_importance)
        
        return model, features
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None, None

def predict_future(model, features, last_data, hours=240):  # 10 days * 24 hours
    """Generate predictions for future hours"""
    try:
        last_date = last_data['datetime'].iloc[-1]
        future_dates = pd.date_range(
            start=last_date + timedelta(hours=1),
            periods=hours,
            freq='H'
        )
        
        future_data = pd.DataFrame({'datetime': future_dates})
        
        # Create the same features as in training
        future_data['hour'] = future_data['datetime'].dt.hour
        future_data['day_of_year'] = future_data['datetime'].dt.dayofyear
        
        # Add cyclical time features
        future_data['hour_sin'] = np.sin(2 * np.pi * future_data['hour']/24)
        future_data['hour_cos'] = np.cos(2 * np.pi * future_data['hour']/24)
        future_data['day_sin'] = np.sin(2 * np.pi * future_data['day_of_year']/365)
        future_data['day_cos'] = np.cos(2 * np.pi * future_data['day_of_year']/365)
        
        # Use moving averages of last day for other features
        for feat in ['precipitation', 'wind_speed']:
            future_data[feat] = last_data[feat].tail(24).mean()
        
        predictions = model.predict(future_data[features])
        future_data['temperature'] = predictions
        
        return future_data
    except Exception as e:
        st.error(f"Error making predictions: {str(e)}")
        return None

def evaluate_model_performance(df, model, feature_cols, validation_days=7):
    """Evaluate model performance by comparing predictions with actual data"""
    # Split data into training and validation sets
    validation_split_date = df['datetime'].max() - timedelta(days=validation_days)
    train_df = df[df['datetime'] < validation_split_date].copy()
    val_df = df[df['datetime'] >= validation_split_date].copy()
    
    if len(train_df) < 100 or len(val_df) < 24:  # Minimum required data
        return None, None
    
    # Train model on earlier data
    model_val = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        n_jobs=-1,
        random_state=42
    )
    
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df['temperature']
    model_val.fit(X_train, y_train)
    
    # Make predictions on validation set
    X_val = val_df[feature_cols].fillna(0)
    val_df['predicted_temp'] = model_val.predict(X_val)
    
    return train_df, val_df

# Sidebar inputs
st.sidebar.header("Location Settings")
latitude = st.sidebar.number_input("Latitude", value=32.6689, format="%.4f", min_value=-90.0, max_value=90.0)
longitude = st.sidebar.number_input("Longitude", value=71.8107, format="%.4f", min_value=-180.0, max_value=180.0)

st.sidebar.header("Data Settings")
date_option = st.sidebar.selectbox("Select Date Range Type", ["Years Back", "Custom Period", "Days Back"])

end_date = datetime.now()

if date_option == "Years Back":
    years_back = st.sidebar.slider("Historical Years", min_value=1, max_value=20, value=5)
    start_date = end_date - timedelta(days=365*years_back)
elif date_option == "Custom Period":
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=end_date - timedelta(days=365))
    with col2:
        end_date = st.date_input("End Date", value=end_date)
else:  # Days Back
    historical_days = st.sidebar.slider("Historical Days", min_value=30, max_value=365, value=90)
    start_date = end_date - timedelta(days=historical_days)

future_days = st.sidebar.slider("Future Prediction Period (days)", min_value=1, max_value=30, value=10)
use_cached_data = st.sidebar.checkbox("Use Cached Data", value=True)

st.sidebar.info(f"Training on historical data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

if st.sidebar.button("Fetch and Analyze Data"):
    with st.spinner("Fetching weather data (this might take a while for 20 years of data)..."):
        # Get weather data
        weather_data = get_weather_data(latitude, longitude, start_date, end_date, use_cache=use_cached_data)
        
        if weather_data is not None:
            st.info(f"Successfully fetched {len(weather_data)} hourly records")
            
            # Prepare the data
            df = prepare_data(weather_data)
            
            if df is not None:
                # Display raw data
                st.subheader("Raw Weather Data Sample")
                st.dataframe(df.tail())
                
                # Train model and make predictions
                model, feature_cols = train_model(df)
                if model is not None and feature_cols is not None:
                    future_predictions = predict_future(model, feature_cols, df, hours=future_days*24)
                    
                    if future_predictions is not None:
                        # Plotting predictions
                        st.subheader("Temperature Forecast")
                        fig = px.line(future_predictions, x='datetime', y='temperature',
                                    title='Temperature Forecast',
                                    labels={'value': 'Temperature (°C)', 'datetime': 'Date/Time'})
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Model Validation Plot
                        st.subheader("Model Validation (Last 7 Days)")
                        train_df, val_df = evaluate_model_performance(df, model, feature_cols)
                        
                        if val_df is not None:
                            fig_val = px.line(val_df, x='datetime', y=['temperature', 'predicted_temp'],
                                            title='Model Validation - Actual vs Predicted (Last 7 Days)',
                                            labels={'value': 'Temperature (°C)', 'datetime': 'Date/Time',
                                                   'temperature': 'Actual Temperature',
                                                   'predicted_temp': 'Predicted Temperature'})
                            st.plotly_chart(fig_val, use_container_width=True)
                            
                            # Calculate and display error metrics
                            mae = np.mean(np.abs(val_df['temperature'] - val_df['predicted_temp']))
                            rmse = np.sqrt(np.mean((val_df['temperature'] - val_df['predicted_temp'])**2))
                            st.info(f"Validation Metrics - MAE: {mae:.2f}°C, RMSE: {rmse:.2f}°C")
