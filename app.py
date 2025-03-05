from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Enhanced predict route with server-side validation
@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            # Server-side validation
            required_fields = [
                'Grid Position', 'Final Position', 'Fastest Lap Time', 'Points', 'Number of Laps',
                'Temperature (°C)', 'Humidity (%)', 'Wind Speed (m/s)', 'Rain (mm)',
                'Status', 'Weather', 'Race Name', 'Driver ID', 'Constructor Name'
            ]

            # Check for missing fields
            for field in required_fields:
                if not request.form.get(field):
                    return render_template('home.html', error=f"Missing value for: {field}")

            # Creating data object with validated inputs
            data = CustomData(
                Grid_Position=int(request.form.get('Grid Position')),
                Final_Position=int(request.form.get('Final Position')),
                Fastest_Lap_Time=float(request.form.get('Fastest Lap Time')),
                Points=float(request.form.get('Points')),
                Number_of_Laps=int(request.form.get('Number of Laps')),
                Temperature=float(request.form.get('Temperature (°C)')),
                Humidity=float(request.form.get('Humidity (%)')),
                Wind_Speed=float(request.form.get('Wind Speed (m/s)')),
                Rain=float(request.form.get('Rain (mm)')),
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
            return render_template('home.html', results=results[0])

        except ValueError:
            return render_template('home.html', error="Invalid data type. Please check your inputs.")
        except Exception as e:
            return render_template('home.html', error=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
