from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

app = Flask(__name__)

#Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            data = CustomData(
                Grid_Position=request.form.get('Grid Position'),
                Final_Position=request.form.get('Final Position'),
                Fastest_Lap_Time=request.form.get('Fastest Lap Time'),
                Points=request.form.get('Points'),
                Number_of_Laps=request.form.get('Number of Laps'),
                Temperature=request.form.get('Temperature (°C)'),
                Humidity=request.form.get('Humidity (%)'),
                Wind_Speed=request.form.get('Wind Speed (m/s)'),
                Rain=request.form.get('Rain (mm)'),
                Status=request.form.get('Status'),
                Weather=request.form.get('Weather'),
                Race_Name=request.form.get('Race Name'),
                Driver_ID=request.form.get('Driver ID'),
                Constructor_Name=request.form.get('Constructor Name')
            )

            pred_df = data.get_data_as_frame()
            print(pred_df)

            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)
            return render_template('home.html',results=results[0])
        
        except Exception as e:
            return render_template('home.html',error=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(host="0.0.0.0",debug=True)
