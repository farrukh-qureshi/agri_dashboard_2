# Weather Forecast Analysis and Prediction App

A Streamlit-based web application that provides weather forecasting, historical weather analysis, and temperature predictions using machine learning.

## Features

- **Historical Weather Data**: Fetch and display historical weather data for any location using latitude and longitude coordinates
- **Weather Forecasting**: Predict temperature for the next 10 days using machine learning (Random Forest)
- **Data Visualization**: Interactive plots and charts for weather trends
- **Feature Importance Analysis**: Understand which factors most influence temperature predictions
- **Model Performance Evaluation**: Validate prediction accuracy using historical data
- **Data Caching**: Efficient data management with local caching of weather data

## Requirements

- Python 3.7+
- streamlit
- pandas
- numpy
- plotly
- scikit-learn
- meteostat
- pickle

## Installation

1. Clone this repository
2. Install the required packages:
   ```bash
   pip install streamlit pandas numpy plotly scikit-learn meteostat
   ```

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run weather_app.py
   ```

2. In the sidebar:
   - Enter the latitude and longitude for your location of interest
   - Choose your preferred date range type:
     - Years Back
     - Custom Period
     - Days Back
   - Adjust other parameters as needed

3. The app will display:
   - Historical weather data visualization
   - Temperature predictions
   - Model performance metrics
   - Feature importance analysis

## Data Sources

The application uses the Meteostat Python library to fetch historical weather data, which includes:
- Temperature
- Precipitation
- Wind Speed

## Machine Learning Model

The app employs a Random Forest Regressor with the following features:
- Cyclical time encoding (hour, day of year)
- Precipitation
- Wind speed

The model is trained on historical data and can predict temperatures for the next 10 days (240 hours).

## Machine Learning Model Details

### Model Architecture
The application uses a Random Forest Regressor, which is particularly well-suited for this task because:
- It handles non-linear relationships in weather data effectively
- It's robust against outliers and noisy data
- It provides feature importance analysis
- It can process large datasets efficiently through parallel processing

### Data Processing and Feature Engineering
1. **Time-Based Features**:
   - Hours are converted to cyclical features using sine and cosine transformations
   - Days of the year are similarly encoded to capture seasonal patterns
   - This ensures the model understands the cyclical nature of weather patterns

2. **Weather Parameters**:
   - Temperature (target variable)
   - Precipitation (feature)
   - Wind Speed (feature)

### Handling Large Historical Data
1. **Efficient Data Loading**:
   - Data is fetched in chunks through the Meteostat API
   - Local caching system stores previously fetched data to minimize API calls
   - Cache is organized by location coordinates and date ranges

2. **Memory Management**:
   - The model uses streaming data processing where possible
   - Data is preprocessed in batches when handling large historical datasets
   - Missing values are handled through forward and backward filling to maintain data continuity

3. **Performance Optimization**:
   - Parallel processing using all available CPU cores (n_jobs=-1)
   - Optimized hyperparameters for large datasets:
     - 200 decision trees in the forest
     - Maximum depth of 15 to prevent overfitting
     - Balanced complexity and accuracy

### Prediction Process
1. **Training**:
   - Model learns patterns from historical data
   - Features are normalized and preprocessed
   - Cross-validation ensures robust performance

2. **Forecasting**:
   - Generates hourly predictions for the next 10 days
   - Uses rolling averages for precipitation and wind speed
   - Incorporates seasonal and daily patterns through cyclical features

3. **Validation**:
   - Model performance is evaluated on recent historical data
   - Predictions are compared with actual temperatures
   - Error metrics help assess forecast reliability

### Model Limitations and Considerations
- Accuracy depends on the quality and quantity of historical data
- Extreme weather events may be harder to predict
- Local microclimate effects might not be captured
- Predictions become less accurate as forecast horizon increases

## Caching

Weather data is cached locally to improve performance and reduce API calls. Cached data is stored in a `weather_cache` directory.

## Error Handling

The application includes comprehensive error handling for:
- Data fetching issues
- Invalid coordinates
- Missing data
- Model training errors

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

This project is open source and available under the MIT License.
