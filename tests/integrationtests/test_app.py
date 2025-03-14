import pytest
from flask import Flask
from flask.testing import FlaskClient
from app import app
import sys

from src.logger import logging
from src.exception import CustomException

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

# Test for home page
def test_index(client: FlaskClient):
    logging.info("Testing the index route")
    response = client.get('/')
    assert response.status_code == 200

# Test for GET request on predict page
def test_predict_get(client: FlaskClient):
    logging.info("Testing the GET request")
    response = client.get('/predict')
    assert response.status_code == 200

# Test for POST request with missing data
def test_predict_post_missing_data(client: FlaskClient):
    logging.info("Testing the missing data POST route")
    response = client.post('/predict', data={})
    assert response.status_code == 200
    assert b"Missing value for" in response.data

# Test for POST request with invalid data type
def test_predict_post_invalid_data(client: FlaskClient):
    logging.info("Testing the invalid data POST route")
    try:
        response = client.post('/predict', data={
        'Grid Position': 'invalid',
        'Final Position': 'invalid',
        'Fastest Lap Time': 'invalid',
        'Points': 'invalid',
        'Number of Laps': 'invalid',
        'Temperature (°C)': 'invalid',
        'Humidity (%)': 'invalid',
        'Wind Speed (m/s)': 'invalid',
        'Rain (mm)': 'invalid',
        'Status': 'invalid',
        'Weather': 'invalid',
        'Race Name': 'invalid',
        'Driver ID': 'invalid',
        'Constructor Name': 'invalid'
    })
        assert response.status_code == 200
    except Exception as e:
        assert False, f"Unexpected CustomException: {str(e)}"
        raise CustomException(e,sys)

# Test for successful prediction
def test_predict_post_success(client: FlaskClient):
    logging.info("Testing the successful POST route")
    response = client.post('/predict', data={
        'Grid Position': '1',
        'Final Position': '1',
        'Fastest Lap Time': '90.5',
        'Points': '25',
        'Number of Laps': '58',
        'Temperature (°C)': '30',
        'Humidity (%)': '50',
        'Wind Speed (m/s)': '2',
        'Rain (mm)': '0',
        'Status': 'Finished',
        'Weather': 'Clear',
        'Race Name': 'Monaco GP',
        'Driver ID': '44',
        'Constructor Name': 'Mercedes'
    })
    assert response.status_code == 200

        
        