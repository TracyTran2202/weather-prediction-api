from fastapi import FastAPI, HTTPException
import logging
from .predictor import WeatherPredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Weather Prediction API",
    description="API for predicting rain and precipitation in Sydney",
    version="1.0.0"
)

# Initialize weather predictor
predictor = WeatherPredictor()

@app.get("/")
async def root():
    """Displaying project objectives, endpoints, input parameters and output format"""
    return {
        "name": "Weather Prediction API",
        "description": "API for predicting rain and precipitation in Sydney using machine learning models",
        "objectives": [
            "Predict if it will rain in exactly 7 days from a given date",
            "Predict cumulative precipitation within the next 3 days from a given date"
        ],
        "endpoints": {
            "/health/": "Health check endpoint",
            "/predict/rain/": "Predict rain in 7 days (GET, param: date)",
            "/predict/precipitation/fall/": "Predict precipitation amount in 3 days (GET, param: date)"
        },
        "input_format": {
            "date": "YYYY-MM-DD (e.g., 2025-09-26)"
        },
        "output_format": {
            "rain": {
                "input_date": "2025-09-26",
                "prediction": {
                    "date": "2025-10-03",
                    "will_rain": True
                }
            },
            "precipitation": {
                "input_date": "2025-09-26",
                "prediction": {
                    "start_date": "2025-09-27",
                    "end_date": "2025-09-29",
                    "precipitation_fall": "28.2"
                }
            }
        },
        "github_repo": "Replace with your actual GitHub repository URL for the API project"
    }

@app.get("/health/")
async def health():
    """Health check endpoint"""
    return "Welcome to the Weather Prediction API! The service is healthy and ready to serve predictions."

@app.get("/predict/rain/")
async def predict_rain(date: str):
    """
    Predict if it will rain in exactly 7 days from the given date in Sydney
    
    Parameters:
    - date: Date in YYYY-MM-DD format
    """
    try:
        result = predictor.predict_rain(date)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error predicting rain: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict/precipitation/fall/")
async def predict_precipitation(date: str):
    """
    Predict cumulative precipitation amount within the next 3 days from the given date in Sydney
    
    Parameters:
    - date: Date in YYYY-MM-DD format
    """
    try:
        result = predictor.predict_precipitation(date)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error predicting precipitation: {e}")
        raise HTTPException(status_code=500, detail=str(e))
