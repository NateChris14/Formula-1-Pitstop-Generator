import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

from src.components.model_trainer import ModelTrainer

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path = 'artifact/model.pkl'
            preprocessor_path = 'artifact/preprocessor.pkl'
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,
                Grid_Position,
                Final_Position,
                Fastest_Lap_Time,
                Points,
                Number_of_Laps,
                Temperature,
                Humidity,
                Wind_Speed,
                Rain,
                Status,
                Weather,
                Race_Name,
                Driver_ID,
                Constructor_Name):

            self.Grid_Position = Grid_Position;
            self.Final_Position = Final_Position;
            self.Fastest_Lap_Time = Fastest_Lap_Time;
            self.Points = Points;
            self.Number_of_Laps = Number_of_Laps;
            self.Temperature = Temperature;
            self.Humidity = Humidity;
            self.Wind_Speed = Wind_Speed;
            self.Rain = Rain;
            self.Status = Status;
            self.Weather = Weather;
            self.Race_Name = Race_Name;
            self.Driver_ID = Driver_ID;
            self.Constructor_Name = Constructor_Name

    def get_data_as_frame(self):
        try:
            custom_data_input_dict = {
                'Grid Position': [self.Grid_Position],
                'Final Position': [self.Final_Position],
                'Fastest Lap Time': [self.Fastest_Lap_Time],
                'Points': [self.Points],
                'Number of Laps': [self.Number_of_Laps],
                'Temperature (°C)':[self.Temperature],
                'Humidity (%)':[self.Humidity],
                'Wind Speed (m/s)':[self.Wind_Speed],
                'Rain (mm)':[self.Rain],
                'Status':[self.Status],
                'Weather' : [self.Weather],
                'Race Name':[self.Race_Name],
                'Driver ID':[self.Driver_ID],
                'Constructor Name':[self.Constructor_Name]

            }

            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e,sys)
            

    
    
              
                                             



