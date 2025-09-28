import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import logging

class WeatherPredictor:
    def __init__(self):
        """Initialize the WeatherPredictor by loading all required models"""
        self.logger = logging.getLogger(__name__)
        self.models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
        
        try:
            # Load rain prediction model and scaler
            rain_dir = os.path.join(self.models_dir, "rain_or_not")
            self.rain_model = joblib.load(os.path.join(rain_dir, "model.joblib"))
            self.rain_features = joblib.load(os.path.join(rain_dir, "feature_names.joblib"))
            self.rain_scaler = joblib.load(os.path.join(rain_dir, "scaler.joblib"))
            
            # Load precipitation model and scaler  
            precip_dir = os.path.join(self.models_dir, "precipitation_fall")
            self.precip_model = joblib.load(os.path.join(precip_dir, "xgb_precipitation_model.joblib"))
            
            # Load precipitation features from text file
            with open(os.path.join(precip_dir, "feature_columns.txt"), "r") as f:
                self.precip_features = [line.strip() for line in f.readlines()]
            
            self.precip_scaler = joblib.load(os.path.join(precip_dir, "precipitation_scaler.joblib"))
            
            self.logger.info("Models loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            raise

    def _generate_rain_features(self, date):
        """Generate features for rain prediction model"""
        # Base weather values for Sydney (using seasonal averages)
        month = date.month
        
        # Seasonal temperature adjustments
        if month in [12, 1, 2]:  # Summer
            temp_base = 26.0
        elif month in [6, 7, 8]:  # Winter
            temp_base = 16.0
        elif month in [3, 4, 5]:  # Autumn
            temp_base = 21.0
        else:  # Spring
            temp_base = 22.0
            
        base_values = {
            'temperature_2m': temp_base + np.random.normal(0, 2),
            'relative_humidity_2m': 65.0 + np.random.normal(0, 10),
            'dew_point_2m': temp_base - 5 + np.random.normal(0, 2),
            'pressure_msl': 1013.0 + np.random.normal(0, 5),
            'surface_pressure': 1013.0 + np.random.normal(0, 5),
            'wind_speed_10m': 8.0 + np.random.normal(0, 3),
            'wind_direction_10m': np.random.uniform(0, 360),
            'hour': date.hour,
            'month': month,
            'day_of_week': date.weekday(),
            'day_of_year': int(date.strftime('%j'))
        }
        
        features = base_values.copy()
        
        # Rolling features for different periods
        for period in [24, 72, 168]:
            features.update({
                f'temp_mean_{period}h': base_values['temperature_2m'] + np.random.normal(0, 1),
                f'humidity_mean_{period}h': base_values['relative_humidity_2m'] + np.random.normal(0, 5),
                f'pressure_mean_{period}h': base_values['pressure_msl'] + np.random.normal(0, 2),
                f'wind_speed_mean_{period}h': base_values['wind_speed_10m'] + np.random.normal(0, 1),
                f'temp_std_{period}h': abs(np.random.normal(2, 0.5)),
                f'humidity_std_{period}h': abs(np.random.normal(5, 1)),
                f'pressure_std_{period}h': abs(np.random.normal(3, 0.5)),
                f'wind_speed_std_{period}h': abs(np.random.normal(2, 0.5))
            })
        
        # Interaction features
        temp = base_values['temperature_2m']
        humidity = base_values['relative_humidity_2m']
        pressure = base_values['pressure_msl']
        wind = base_values['wind_speed_10m']
        
        features.update({
            'temp_humidity': temp * humidity,
            'wind_pressure': wind * pressure,
            'temp_squared': temp * temp,
            'humidity_squared': humidity * humidity,
            'pressure_squared': pressure * pressure,
            'wind_temp_humidity': wind * temp * humidity,
            'temp_humidity_pressure': temp * humidity * pressure
        })
        
        return features
    
    def _generate_precip_features(self, date):
        """Generate features for precipitation prediction model"""
        month = date.month
        
        # Seasonal adjustments
        if month in [12, 1, 2]:  # Summer - more rain
            precip_factor = 1.2
            temp_base = 26.0
        elif month in [6, 7, 8]:  # Winter - less rain
            precip_factor = 0.8
            temp_base = 16.0
        elif month in [3, 4, 5]:  # Autumn
            precip_factor = 1.0
            temp_base = 21.0
        else:  # Spring
            precip_factor = 1.1
            temp_base = 22.0
        
        # Daily weather features
        features = {
            'temperature_2m_max': temp_base + 3 + np.random.normal(0, 2),
            'temperature_2m_min': temp_base - 3 + np.random.normal(0, 2),
            'temperature_2m_mean': temp_base + np.random.normal(0, 2),
            'precipitation_sum': max(0, np.random.exponential(2) * precip_factor),
            'precipitation_hours': max(0, np.random.poisson(2)),
            'windspeed_10m_max': 15 + np.random.normal(0, 5),
            'windgusts_10m_max': 25 + np.random.normal(0, 8),
            'shortwave_radiation_sum': 25 + np.random.normal(0, 5),
            'relative_humidity_2m_mean': int(65 + np.random.normal(0, 15))
        }
        
        # 7-day rolling features
        features.update({
            'temp_mean_7d': features['temperature_2m_mean'] + np.random.normal(0, 1),
            'temp_std_7d': abs(np.random.normal(3, 1)),
            'precip_sum_7d': max(0, features['precipitation_sum'] * 7 + np.random.normal(0, 5)),
            'precip_max_7d': max(features['precipitation_sum'], np.random.exponential(5)),
            'rain_days_7d': max(0, int(np.random.poisson(2))),
            'wind_mean_7d': features['windspeed_10m_max'] + np.random.normal(0, 2),
            'wind_max_7d': features['windgusts_10m_max'] + np.random.normal(0, 3),
            'humidity_mean_7d': features['relative_humidity_2m_mean'] + np.random.normal(0, 5)
        })
        
        return features



    def predict_rain(self, date_str):
        """
        Predict if it will rain in exactly 7 days from the given date in Sydney.
        
        Args:
            date_str (str): Date in YYYY-MM-DD format
            
        Returns:
            dict: Prediction result with input date and prediction details
        """
        try:
            # Validate and parse date
            input_date = datetime.strptime(date_str, "%Y-%m-%d")
            prediction_date = input_date + timedelta(days=7)
            
            # Generate features for rain model
            features = self._generate_rain_features(input_date)
            
            # Create DataFrame with all features the model expects
            feature_df = pd.DataFrame([features])
            
            # Ensure we only use features the model was trained on
            feature_df = feature_df.reindex(columns=self.rain_features, fill_value=0)
            
            # Scale features
            features_scaled = self.rain_scaler.transform(feature_df)
            
            # Make prediction
            prediction_proba = self.rain_model.predict_proba(features_scaled)[0]
            will_rain = bool(self.rain_model.predict(features_scaled)[0])
            
            self.logger.info(f"Rain prediction for {prediction_date}: {will_rain} (probability: {prediction_proba[1]:.3f})")
            
            return {
                "input_date": date_str,
                "prediction": {
                    "date": prediction_date.strftime("%Y-%m-%d"),
                    "will_rain": will_rain
                }
            }
        except ValueError as e:
            self.logger.error(f"Date validation error: {e}")
            raise ValueError("Invalid date format. Use YYYY-MM-DD")
        except Exception as e:
            self.logger.error(f"Error predicting rain: {e}")
            raise

    def predict_precipitation(self, date_str):
        """
        Predict cumulative precipitation amount within the next 3 days from the given date in Sydney.
        
        Args:
            date_str (str): Date in YYYY-MM-DD format
            
        Returns:
            dict: Prediction result with input date and precipitation details
        """
        try:
            # Validate and parse date
            input_date = datetime.strptime(date_str, "%Y-%m-%d")
            start_date = input_date + timedelta(days=1)
            end_date = start_date + timedelta(days=2)
            
            # Generate features for precipitation model
            features = self._generate_precip_features(input_date)
            
            # Create DataFrame with features
            feature_df = pd.DataFrame([features])
            
            # Ensure we only use features the model was trained on
            feature_df = feature_df.reindex(columns=self.precip_features, fill_value=0)
            
            # Scale features
            features_scaled = self.precip_scaler.transform(feature_df)
            
            # Make prediction
            precipitation = float(self.precip_model.predict(features_scaled)[0])
            
            # Ensure non-negative precipitation
            precipitation = max(0, precipitation)
            
            self.logger.info(f"Precipitation prediction for {start_date} to {end_date}: {precipitation:.1f}mm")
            
            return {
                "input_date": date_str,
                "prediction": {
                    "start_date": start_date.strftime("%Y-%m-%d"),
                    "end_date": end_date.strftime("%Y-%m-%d"),
                    "precipitation_fall": f"{precipitation:.1f}"
                }
            }
        except ValueError as e:
            self.logger.error(f"Date validation error: {e}")
            raise ValueError("Invalid date format. Use YYYY-MM-DD")
        except Exception as e:
            self.logger.error(f"Error predicting precipitation: {e}")
            raise
