# 🏡 Housing Price Prediction – End-to-End MLOps Project

An end-to-end **MLOps project** that automates housing price prediction using Machine Learning, experiment tracking, model monitoring, containerization, and CI/CD automation.

The project demonstrates how to build, test, package, monitor, and continuously integrate a machine learning application using modern DevOps and MLOps tools.

## 🚀 Project Overview

This project uses the California Housing dataset to train an XGBoost regression model that predicts median house values.

It implements an end-to-end workflow covering:

- Data loading and preprocessing
- Machine learning model training
- Experiment tracking using MLflow
- Data drift detection
- Conditional model retraining
- REST API development using FastAPI
- Automated testing using Pytest
- Docker containerization
- Monitoring using Prometheus and Grafana
- CI/CD automation using GitHub Actions
- Docker image publishing to Docker Hub

## 🏗️ Architecture

```text
                 California Housing Dataset
                            |
                            v
                   Data Loading & Validation
                            |
                            v
                    XGBoost Model Training
                            |
                            v
                    MLflow Experiment Tracking
                            |
                            v
                    Trained Model (.pkl)
                            |
                            v
                    FastAPI Prediction API
                            |
             +--------------+--------------+
             |              |              |
             v              v              v
          Pytest        Prometheus      API Metrics
                            |
                            v
                         Grafana

             Data Drift Monitoring
                      |
                      v
              Conditional Retraining
                      |
                      v
              Model Evaluation

             GitHub Repository
                      |
                      v
                GitHub Actions
                      |
              +-------+-------+
              |               |
              v               v
          Run Tests       Build Docker Image
                              |
                              v
                         Docker Hub
```

## 🛠️ Technologies Used

| Category | Tools |
|---|---|
| Programming Language | Python 3.12 |
| Machine Learning | XGBoost, Scikit-learn |
| Data Processing | Pandas, NumPy, SciPy |
| Experiment Tracking | MLflow |
| Drift Monitoring | Evidently |
| API Development | FastAPI, Uvicorn |
| Testing | Pytest |
| Containerization | Docker |
| Container Orchestration | Docker Compose |
| Metrics Monitoring | Prometheus |
| Visualization | Grafana |
| CI/CD | GitHub Actions |
| Image Registry | Docker Hub |
| Version Control | Git, GitHub |

## 📂 Project Structure

```text
housing-mlops/
│
├── .github/
│   └── workflows/
│       └── ci-cd.yml
│
├── data/
│   └── housing.csv
│
├── models/
│   └── xgboost_housing_model.pkl
│
├── monitoring/
│   └── prometheus.yml
│
├── reports/
│   └── data_drift_report.html
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── train.py
│   ├── app.py
│   ├── monitor_drift.py
│   └── retrain.py
│
├── tests/
│   ├── conftest.py
│   ├── test_app.py
│   └── test_retrain.py
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-api.txt
├── .dockerignore
├── .gitignore
└── README.md
```

## ⚙️ Project Implementation

### 1. Data Loading and Validation

- Downloaded the California Housing dataset using Scikit-learn.
- Validated dataset structure and missing values.
- Saved the dataset as a CSV file for model training.

### 2. Model Training

- Developed a regression model using XGBoost.
- Split the dataset into training and testing sets.
- Evaluated model performance using:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - R² Score

### 3. MLflow Experiment Tracking

- Integrated MLflow for experiment tracking.
- Logged model parameters and evaluation metrics.
- Created separate experiments for initial training and conditional retraining.
- Tracked model performance across experiments.

### 4. Data Drift Monitoring

- Integrated Evidently for data drift analysis.
- Used Statistical Kolmogorov-Smirnov tests for feature-level drift detection.
- Generated HTML drift reports.
- Configured a drift threshold to recommend retraining when drift exceeds the defined limit.

**Note:** The current drift demonstration uses simulated current data to demonstrate the monitoring workflow.

### 5. Conditional Model Retraining

- Implemented a conditional retraining pipeline.
- Triggered retraining recommendations based on the drift threshold.
- Compared candidate model performance against the baseline.
- Promoted the candidate model only when it met the defined performance criteria.

### 6. FastAPI Model Serving

Developed a REST API to serve predictions.

Available endpoints:

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | API root |
| `/health` | GET | Health check |
| `/predict` | POST | Housing price prediction |
| `/metrics` | GET | Prometheus metrics |

### 7. Automated Testing

Implemented automated tests using Pytest.

Test coverage includes:

- API health check
- Valid prediction requests
- Invalid input validation
- Prometheus metrics endpoint
- Drift detection logic
- Conditional retraining logic

**Test result:** 7 tests passed.

### 8. Docker Containerization

- Created a Dockerfile for the FastAPI application.
- Built a custom Docker image.
- Containerized the prediction service.
- Published the image to Docker Hub.

**Docker Hub Repository:**

[https://hub.docker.com/r/dockernavaneeth/housing-mlops](https://hub.docker.com/r/dockernavaneeth/housing-mlops)

Pull the image:

```bash
docker pull dockernavaneeth/housing-mlops:v1
```

Run the container:

```bash
docker run -d \
  -p 8000:8000 \
  --name housing-api \
  dockernavaneeth/housing-mlops:v1
```

### 9. Prometheus and Grafana Monitoring

Configured Prometheus and Grafana using Docker Compose.

Prometheus collects application metrics from the FastAPI `/metrics` endpoint.

Grafana dashboards visualize:

- Total prediction requests
- Prediction errors
- Prediction request rate
- Average prediction latency
- API availability

**Access the services locally:**

| Service | URL |
|---|---|
| FastAPI | http://localhost:8000 |
| Swagger UI | http://localhost:8000/docs |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

### 10. CI/CD Automation with GitHub Actions

Implemented a GitHub Actions workflow to automate the CI/CD process.

**Pipeline workflow:**

1. Trigger the pipeline when code is pushed to the `main` branch.
2. Check out the source code.
3. Set up the Python environment.
4. Install project dependencies.
5. Run automated Pytest tests.
6. Build the Docker image.
7. Authenticate with Docker Hub using GitHub Secrets.
8. Push the Docker image to Docker Hub.

Docker images are published with:

- `latest` tag
- Git commit SHA tag

Docker publishing runs only after the test job succeeds and only pushes images for `push` events, not pull requests.

## ▶️ Run the Project Locally

### Prerequisites

- Python 3.12
- Git
- Docker Desktop
- Docker Compose

### Step 1: Clone the Repository

```bash
git clone https://github.com/Navaneethkrishna-coder/housing-mlops.git

cd housing-mlops
```

### Step 2: Create Virtual Environment

```bash
python3 -m venv .venv

source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Train the Model

```bash
python src/data_loader.py
python src/train.py
```

### Step 5: Start the FastAPI Application

```bash
uvicorn src.app:app --reload
```

Open Swagger UI:

http://localhost:8000/docs

### Step 6: Run Automated Tests

```bash
pytest -v
```

### Step 7: Start Monitoring Services

```bash
docker compose up -d
```

Check running containers:

```bash
docker compose ps
```

Stop the services:

```bash
docker compose down
```

## 📊 Key MLOps and DevOps Highlights

- End-to-end machine learning lifecycle implementation.
- MLflow-based experiment tracking.
- Automated data drift detection.
- Conditional model retraining and model evaluation.
- REST API deployment using FastAPI.
- Automated testing using Pytest.
- Docker-based application packaging.
- Prometheus and Grafana observability.
- GitHub Actions CI/CD automation.
- Automated Docker image publishing to Docker Hub.

## 🔮 Future Enhancements

- Deploy the application on AWS EC2 using Terraform.
- Implement automated continuous deployment after Docker image publishing.
- Integrate a cloud-based model registry.
- Add authentication and API security.
- Implement advanced monitoring and alerting.
- Add automated model versioning and rollback.
- Integrate a production data ingestion pipeline.

## 👨‍💻 Author

**Navaneeth Krishna**

B.Tech Computer Science Graduate | DevOps & MLOps Enthusiast

- GitHub: [Navaneethkrishna-coder](https://github.com/Navaneethkrishna-coder)
- LinkedIn: [Connect with me](https://www.linkedin.com/in/navaneeth-krishna-22b482278/)

---

⭐ If you find this project useful, consider giving the repository a star!

**Built to explore the integration of Machine Learning, MLOps, and DevOps practices in an end-to-end project.**
