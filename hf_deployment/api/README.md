# GetAround Price Prediction API

## Model Information
- **Model Type:** Pipeline
- **Training Date:** 2025-06-18
- **Source:** MLflow (Run ID: 7da1f983c7c34ae1a3c4f1f82e15ee7e)

## Metrics
- **R2:** 0.7500
- **RMSE:** 16.2276
- **MAE:** 10.3358

## Usage
This Space provides a REST API for car rental price prediction.

### Endpoints
- `POST /predict` - Main prediction endpoint
- `GET /docs` - Interactive API documentation
- `GET /health` - API health check
- `GET /mlflow-stats` - Production statistics

### Example Request
```bash
curl -X POST "https://your-space.hf.space/predict" \
     -H "Content-Type: application/json" \
     -d '{"model_key": "Renault", "mileage": 50000, ...}'
```

## Development
Original MLflow experiment: `price_prediction_local`
MLflow Run ID: 7da1f983c7c34ae1a3c4f1f82e15ee7e

## Performance
The model achieved the following performance on test data:
- R-squared: 0.7500
- Root Mean Square Error: 16.23 EUR/day
- Mean Absolute Error: 10.34 EUR/day
