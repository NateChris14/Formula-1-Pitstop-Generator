import pytest
from src.pipeline.predict_pipeline import CustomData
import pandas as pd

def test_custom_data_transformation():
    '''This function tests the Custom Data class in the predict_pipeline'''
    #Sample Input Data
    input_data = {
    "Grid_Position": 5,
    "Final_Position": 3,
    "Fastest_Lap_Time": 75.432,
    "Points": 15,
    "Number_of_Laps": 55,
    "Temperature": 25.5,
    "Humidity": 60.2,
    "Wind_Speed": 5.3,
    "Rain": 0.0,
    "Status": "Finished",
    "Weather": "No Rain",
    "Race_Name": "Italian Grand Prix",
    "Driver_ID": "Ve",  # Verstappen
    "Constructor_Name": "RB"  # Red Bull
}

    #Create Custom Data object
    data = CustomData(**input_data)

    #Getting the data as a DataFrame
    df = data.get_data_as_frame()

    #Checking if the dataframe has the correct values
    assert isinstance(df,pd.DataFrame) #Ensuring it's a dataframe
    assert df["Grid Position"][0] == 5
    assert df["Points"][0] == 15
    assert df["Weather"][0] == 'No Rain'
    assert df["Race Name"][0] == 'Italian Grand Prix'