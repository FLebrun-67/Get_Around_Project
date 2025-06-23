# GetAround Dashboard

Interactive dashboard for car rental price prediction.

## Features
- Real-time price prediction
- Interactive parameter selection
- Connection to API backend
- Production statistics

## Connected API
This dashboard connects to the GetAround API Space for predictions.

**IMPORTANT: Update the API_URL in streamlit_app.py with your actual API Space URL.**

## Model Information
- **Source:** MLflow Run 7da1f983c7c34ae1a3c4f1f82e15ee7e
- **Training Date:** 2025-06-18
- **Performance:** R2 = 0.7500

## Setup
1. Create this Space with SDK=Streamlit
2. Upload streamlit_app.py (adapted from local app.py)
3. Update API_URL in streamlit_app.py to point to your API Space
4. Test the connection

Example API_URL:
```python
API_URL = "https://your-username-getaround-api.hf.space/predict"
```
