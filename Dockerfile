FROM python:3.9-slim

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --extra-index-url https://test.pypi.org/simple/ weather-prediction-utils-tracytran>=0.1.0

COPY ./app /app/app
COPY ./models /app/models

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
