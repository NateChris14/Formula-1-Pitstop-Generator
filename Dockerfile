#Python 3.12 as base image
FROM python:3.12-slim

#Setting the working directory
WORKDIR /app

#Copying the working directory
COPY . /app

#Upgrading pip and installing dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

#Setting the environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV FLASK_RUN_HOST=0.0.0.0
ENV PORT=5000

#Exposing the port
EXPOSE 5000

#Running the Flask App
CMD ["flask","run","--port=5000","--host=0.0.0.0"]
