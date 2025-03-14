import pytest
from unittest.mock import patch
from src.pipeline.predict_pipeline import PredictPipeline
import pandas as pd

# Sample input data for testing the prediction
sample_input_data = {
    "Grid Position": 5,
    "Final Position": 3,
    "Fastest Lap Time": 75.432,
    "Points": 15,
    "Number of Laps": 55,
    "Temperature (°C)": 25.5,
    "Humidity (%)": 60.2,
    "Wind Speed (m/s)": 5.3,
    "Rain (mm)": 0.0,
    "Status": "Finished",
    "Weather": "No Rain",
    "Race Name": "Italian Grand Prix",
    "Driver ID": "Ve",  # Verstappen
    "Constructor Name": "RB"  # Red Bull
}

def test_predict_pipeline():
    #Preparing the mock prediction result
    mock_prediction = [2.0]

    #Mocking the 'predict' method of PredictPipeline
    with patch.object(PredictPipeline,'predict',return_value=mock_prediction):
        predict_pipeline = PredictPipeline()

    #Converting input data to DataFrame format as required by the pipeline
    pred_df = pd.DataFrame([sample_input_data])

    #Calling the predict method
    result = predict_pipeline.predict(pred_df)

    #Asserting the result as expected with the mocked prediction value
    assert result == mock_prediction
