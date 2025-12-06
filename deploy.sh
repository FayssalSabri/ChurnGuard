#!/bin/bash

echo "🚀 Deploying ChurnGuard MLOps Pipeline..."
echo "========================================="

# 1. Vérifier les dépendances
echo "📦 Checking dependencies..."
python -c "import mlflow, sklearn, pandas, numpy, fastapi" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️ Some dependencies missing. Installing..."
    pip install -r requirements.txt
fi

# 2. Démarrer MLflow (en arrière-plan)
echo "🔧 Starting MLflow server..."
mlflow server --backend-store-uri sqlite:///mlruns/mlflow.db \
              --default-artifact-root ./mlruns \
              --host 0.0.0.0 \
              --port 5000 &
MLFLOW_PID=$!
echo "✅ MLflow started with PID: $MLFLOW_PID"

# 3. Attendre que MLflow démarre
echo "⏳ Waiting for MLflow to start..."
sleep 5

# 4. Vérifier l'état des expériences
echo "📊 Setting up MLflow experiments..."
python setup_mlflow.py

# 5. Entraîner le modèle (si non déjà fait)
if [ ! -f "models/churn_model.pkl" ]; then
    echo "🤖 Training model..."
    python train_pipeline.py
else
    echo "✅ Model already trained: models/churn_model.pkl"
fi

# 6. Démarrer l'API
echo "🌐 Starting FastAPI..."
python api/main.py &
API_PID=$!
echo "✅ API started with PID: $API_PID"

# 7. Attendre que l'API démarre
echo "⏳ Waiting for API to start..."
sleep 5

# 8. Tester l'API
echo "🧪 Testing API..."
python test_api.py

# 9. Afficher les URLs
echo ""
echo "========================================="
echo "🎉 Deployment Complete!"
echo ""
echo "🌐 API Documentation: http://localhost:8000/docs"
echo "📊 MLflow UI: http://localhost:5000"
echo ""
echo "📋 Endpoints disponibles:"
echo "   - GET /                 - Service info"
echo "   - GET /health           - Health check"
echo "   - GET /model/info       - Model information"
echo "   - POST /predict         - Single prediction"
echo "   - POST /predict/batch   - Batch prediction"
echo ""
echo "🛑 Pour arrêter les services:"
echo "   kill $API_PID $MLFLOW_PID"
echo "========================================="

# Garder le script actif
wait