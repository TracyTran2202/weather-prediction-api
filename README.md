# Weather Prediction API

This repository contains a FastAPI application for weather prediction in Sydney, Australia. The API provides endpoints for predicting rain and precipitation.

## Features

- Predict whether it will rain in exactly 7 days
- Predict cumulative precipitation for the next 3 days
- RESTful API endpoints
- Docker containerization
- Deployment ready for Render

## API Endpoints

- `/`: Documentation and API information
- `/health/`: Health check endpoint
- `/predict/rain/`: Predict rain 7 days ahead
- `/predict/precipitation/fall/`: Predict 3-day precipitation

## Setup Instructions

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the API locally:
   ```bash
   uvicorn app.main:app --reload
   ```

4. Build and run with Docker:
   ```bash
   docker build -t weather-api .
   docker run -p 80:80 weather-api
   ```

## Deployment

The API is deployed on Render. Visit [API URL] to access the live version.

## Model Information

The API uses two machine learning models:
- Binary classification model for rain prediction
- Regression model for precipitation prediction

Models are trained on historical weather data from Sydney (2015-2024).

## Environment Variables

None required for basic operation.

## Contributing

This is a private repository. Please contact the repository owner for access.

## License

Private - All rights reserved
