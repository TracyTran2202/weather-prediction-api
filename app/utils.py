"""
Utility functions using weather-prediction-utils-tracytran package
"""

# Example usage of custom package
# Uncomment after installing package:

"""
from weather_prediction_utils import WeatherValidator, WeatherMetrics

def validate_prediction_input(date: str) -> tuple[bool, list[str]]:
    '''Validate input using custom package'''
    validator = WeatherValidator()
    
    # Validate date format
    is_valid_date = validator.validate_date(date)
    
    errors = []
    if not is_valid_date:
        errors.append("Invalid date format. Use YYYY-MM-DD")
    
    return len(errors) == 0, errors

def calculate_model_performance(y_true, y_pred, model_type="rain"):
    '''Calculate performance metrics using custom package'''
    metrics_calc = WeatherMetrics()
    
    if model_type == "rain":
        return metrics_calc.calculate_classification_metrics(y_true, y_pred)
    else:
        return metrics_calc.calculate_regression_metrics(y_true, y_pred)
"""

# Placeholder functions for now
def validate_prediction_input(date: str) -> tuple[bool, list[str]]:
    """Validate input date format"""
    try:
        from datetime import datetime
        datetime.strptime(date, "%Y-%m-%d")
        return True, []
    except ValueError:
        return False, ["Invalid date format. Use YYYY-MM-DD"]

def log_prediction_metrics(model_name: str, prediction_result: dict):
    """Log prediction for monitoring"""
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"{model_name} prediction: {prediction_result}")