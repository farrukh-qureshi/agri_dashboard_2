import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from meteostat import Point, Daily
import warnings
warnings.filterwarnings('ignore')

def get_weather_data(lat, lon, start_date, end_date):
    """Fetch historical weather data for the given coordinates"""
    try:
        location = Point(lat, lon)
        data = Daily(location, start_date, end_date)
        data = data.fetch()
        
        if data.empty:
            print("No data available for this location.")
            return None
            
        return data
    except Exception as e:
        print(f"Error fetching weather data: {str(e)}")
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
            'tavg': 'temperature',
            'prcp': 'precipitation',
            'wspd': 'wind_speed'
        })
        
        # Fill missing values
        df['temperature'] = df['temperature'].fillna(method='ffill').fillna(method='bfill')
        df['precipitation'] = df['precipitation'].fillna(0)
        df['wind_speed'] = df['wind_speed'].fillna(method='ffill').fillna(method='bfill')
        
        return df
    except Exception as e:
        print(f"Error preparing data: {str(e)}")
        return None

def train_model(df):
    """Train a Random Forest model on the weather data"""
    try:
        # Create features from datetime
        df['day'] = df['datetime'].dt.day
        df['month'] = df['datetime'].dt.month
        df['year'] = df['datetime'].dt.year
        
        # Prepare features and target
        features = ['day', 'month', 'year', 'precipitation', 'wind_speed']
        X = df[features].fillna(0)
        y = df['temperature']
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        return model, features
    except Exception as e:
        print(f"Error training model: {str(e)}")
        return None, None

def predict_future(model, features, last_data, days=10):
    """Generate predictions for future days"""
    try:
        last_date = last_data['datetime'].iloc[-1]
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=days,
            freq='D'
        )
        
        future_data = pd.DataFrame({'datetime': future_dates})
        future_data['day'] = future_data['datetime'].dt.day
        future_data['month'] = future_data['datetime'].dt.month
        future_data['year'] = future_data['datetime'].dt.year
        
        # Use moving averages of last week for other features
        for feat in ['precipitation', 'wind_speed']:
            future_data[feat] = last_data[feat].tail(7).mean()
        
        predictions = model.predict(future_data[features])
        future_data['temperature'] = predictions
        
        return future_data
    except Exception as e:
        print(f"Error making predictions: {str(e)}")
        return None

def main():
    # Test coordinates (Namal)
    latitude = 32.6689
    longitude = 71.8107

    # Date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*20)
    
    print(f"Fetching weather data for coordinates: ({latitude}, {longitude})")
    weather_data = get_weather_data(latitude, longitude, start_date, end_date)
    
    if weather_data is not None:
        df = prepare_data(weather_data)
        
        if df is not None and not df.empty:
            print("\nCurrent Weather Conditions:")
            current = df.iloc[-1]
            print(f"Temperature: {current['temperature']:.1f}°C")
            print(f"Precipitation: {current['precipitation']:.1f} mm")
            print(f"Wind Speed: {current['wind_speed']:.1f} km/h")
            
            model, feature_cols = train_model(df)
            if model is not None and feature_cols is not None:
                future_predictions = predict_future(model, feature_cols, df)
                
                if future_predictions is not None:
                    print("\n10-Day Temperature Forecast:")
                    forecast_table = future_predictions[['datetime', 'temperature']].copy()
                    forecast_table['temperature'] = forecast_table['temperature'].round(1)
                    print(forecast_table.to_string(index=False))
                    
                    # Create and save plot
                    plot_data = pd.concat([
                        df[['datetime', 'temperature']].tail(30),
                        future_predictions[['datetime', 'temperature']]
                    ])
                    
                    fig = px.line(
                        plot_data,
                        x='datetime',
                        y='temperature',
                        title='Historical and Predicted Temperature'
                    )
                    
                    # Save plot to HTML file
                    fig.write_html("temperature_forecast.html")
                    print("\nPlot saved as 'temperature_forecast.html'")

if __name__ == "__main__":
    main()
