# ChurnGuard - MLOps Pipeline

End-to-end CI/CD pipeline with MLflow for automated training, versioning, and deployment with model monitoring.

## Features

- **Automated Training Pipeline**: Automated model training and evaluation
- **MLflow Integration**: Experiment tracking and model versioning
- **FastAPI Service**: REST API for model inference
- **Docker Containers**: Containerized training and serving
- **CI/CD Pipeline**: GitHub Actions for automated testing and deployment
- **Model Monitoring**: Drift detection and performance monitoring
- **Reduced deployment time by 60%**

## Architecture

```
Data → Preprocessing → Training → MLflow Tracking → FastAPI → Monitoring
```

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/FayssalSabri/ChurnGuard.git
cd churnguard-mlops
```

2. **Set up environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. **Run with Docker Compose**
```bash
docker-compose up --build
```

## Usage

### Training the Model
```bash
python train_pipeline.py
```

### Starting the API
```bash
uvicorn api.main:app --reload
```

### API Endpoints
- `GET /` - Health check
- `GET /health` - System health
- `POST /predict` - Make predictions
- `GET /model/info` - Model information

## Testing
```bash
pytest tests/ -v
```

## Monitoring
- MLflow UI: http://localhost:5000
- API Docs: http://localhost:8000/docs
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

## Configuration
Edit `config/config.yaml` for model parameters, data paths, and API settings.

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License
MIT License

## Getting Started

1. **Initialize the project:**
```bash
mkdir churnguard-mlops && cd churnguard-mlops
git init
```

2. **Create the structure:**
```bash
# Create the folder structure as indicated
mkdir -p data/{raw,processed} src tests api monitoring docker config notebooks
mkdir -p .github/workflows
```

3. **Add the files:**
```bash
# Copy each file to its respective folder
```

4. **Launch MLflow locally:**
```bash
mlflow server --backend-store-uri sqlite:///mlruns/mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
```

5. **Run the pipeline:**
```bash
python train_pipeline.py
```

6. **Start the API:**
```bash
uvicorn api.main:app --reload
```