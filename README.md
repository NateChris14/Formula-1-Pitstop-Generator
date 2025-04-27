
# 🏎️ Formula 1 Race Pitstop Prediction Project

## Project Overview

This project leverages Machine Learning to predict pitstop strategies for Formula 1 races, using historical data. The objective is to build a web application for this very purpose which can be freely used pre-race to determine the best possible race strategies.

## 🎬 Project Demo

Watch a quick demo of the F1 Pitstop Prediction app in action:

[![Project Demo](https://youtu.be/B7FerenWQZU)](https://youtu.be/B7FerenWQZU)


## Key Features

* 🏁 Pitstop Strategy Prediction: Build a predictive model to forecast the number of pit stops during a race based on various factors like weather, laps, and race conditions.

* 🔄 Data Pipeline: Implemented an efficient ML pipeline to automate data ingestion, data transformation, feature engineering, and model evaluation.

* ⚙️ Modeling: Used powerful Machine Learning algorithms such as XGBoost and SVM to predict pitstop strategies.

* 🐳 Deployment: Dockerized the Flask application and deployed it on AWS ECS Fargate using a fully automated CI/CD pipeline with GitHub Actions.

## Tech Stack

* Languages: Python

* Libraries: Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn

* APIs: Ergast API (Formula 1 Data)

* MLOps: Docker, GitHub Actions, AWS ECR, AWS ECS (Fargate)

* Web Framework: Flask

## Business Impact

* 📈 The predictive model helps optimize pitstop strategies, providing valuable insights for team decision-making.

* ⚡ Streamlined data processing and model evaluation workflows, improving development efficiency and reducing manual errors.

* 🌐 Deployed the solution with Docker and AWS ECS Fargate, ensuring scalability, performance, and automated updates through CI/CD pipelines.

## Getting Started

Follow these steps to set the project up locally:

### Prerequisites:

* Python 3.12

* Docker (for containerization)

* GitHub account (for CI/CD pipeline)

* AWS Account (for deployment via ECS)

### Steps to Set Up:

1. Clone the repository:

```bash
git clone https://github.com/NateChris14/Formula-1-Pitstop-Generator
cd Formula-1-Pitstop-Generator
```

2. Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install the project and its dependencies:

```bash
pip install -e .
```

This will install the project in "editable" mode, allowing you to make changes to the source code without reinstalling the package. It also installs any dependencies specified in the install_requires section of the setup.py file.

4. Running the Flask App:

```bash
python app.py
```

5. Deploying the app using Docker and AWS ECS: Follow the instructions below for Dockerizing the app and deploying it to AWS ECS.

## Deployment 

This project has been containerized using Docker and deployed to AWS ECS (Fargate). Below are the steps for deploying the application:

## 📊 Deployment Architecture

Below is the architecture used to deploy the F1 Pitstop Prediction app:

![Deployment Architecture](https://github.com/NateChris14/Formula-1-Pitstop-Generator/blob/main/F1%20deploy%20architecture.png)

**Key components:**
- **AWS ECS (Fargate)**: For running containerized Flask app.
- **AWS ECR**: For storing the Docker image.
- **GitHub Actions**: For CI/CD pipeline to automatically deploy the app.
- **Flask App**: Exposes the prediction model to users via a web interface.
- **Docker**: Containerizes the application for deployment consistency across environments.


1. Dockerize the app: Ensure the Dockerfile is correctly set up in your project folder. Then build the Docker image:

```bash
docker build -t f1-pitstop-prediction .
```

2. Push Docker image to AWS ECR: Follow the steps to push your Docker image to AWS ECR:

* Create a repository on AWS ECR.

* Authenticate Docker to AWS using the AWS CLI.

* Push the image to ECR.

3. Set up AWS ECS:

* Create an ECS Cluster and Task Definition.

* Deploy your app using ECS Fargate.

4. Set up CI/CD with GitHub Actions: GitHub Actions will automatically deploy updates to AWS ECS whenever changes are pushed to the main branch.