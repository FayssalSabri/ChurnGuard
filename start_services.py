#!/usr/bin/env python3
"""
Script pour démarrer tous les services ChurnGuard
"""

import subprocess
import time
import sys
import os
from loguru import logger

def start_mlflow():
    """Démarre le serveur MLflow"""
    logger.info("🚀 Starting MLflow server...")
    
    # Créer le dossier mlruns s'il n'existe pas
    os.makedirs("mlruns", exist_ok=True)
    
    # Commande pour démarrer MLflow
    cmd = [
        sys.executable, "-m", "mlflow", "server",
        "--backend-store-uri", "sqlite:///mlruns/mlflow.db",
        "--default-artifact-root", "./mlruns",
        "--host", "0.0.0.0",
        "--port", "5000"
    ]
    
    try:
        # Démarrer MLflow en arrière-plan
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Attendre un peu
        time.sleep(3)
        
        # Vérifier si MLflow est en cours d'exécution
        if process.poll() is None:
            logger.info(f"✅ MLflow started with PID: {process.pid}")
            return process
        else:
            stderr = process.stderr.read()
            logger.error(f"❌ MLflow failed to start: {stderr}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Error starting MLflow: {e}")
        return None

def start_api():
    """Démarre l'API FastAPI"""
    logger.info("🌐 Starting FastAPI...")
    
    cmd = [
        sys.executable, "-m", "uvicorn",
        "api.main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload"
    ]
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        time.sleep(5)
        
        if process.poll() is None:
            logger.info(f"✅ FastAPI started with PID: {process.pid}")
            return process
        else:
            stderr = process.stderr.read()
            logger.error(f"❌ FastAPI failed to start: {stderr}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Error starting FastAPI: {e}")
        return None

def check_services():
    """Vérifie que les services sont en cours d'exécution"""
    logger.info("🔍 Checking services...")
    
    import requests
    
    services = [
        ("MLflow", "http://localhost:5000", 10),
        ("FastAPI", "http://localhost:8000/health", 30)
    ]
    
    all_ready = True
    
    for service_name, url, timeout in services:
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code == 200:
                logger.info(f"✅ {service_name} is ready at {url}")
            else:
                logger.warning(f"⚠️ {service_name} returned status {response.status_code}")
                all_ready = False
        except Exception as e:
            logger.error(f"❌ {service_name} not ready: {e}")
            all_ready = False
    
    return all_ready

def setup_logging():
    """Configure le logging"""
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    logger.add(
        "logs/churnguard.log",
        rotation="500 MB",
        retention="10 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="INFO"
    )

def main():
    """Fonction principale"""
    
    # Configuration
    setup_logging()
    
    # Créer les dossiers nécessaires
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    print("\n" + "="*60)
    print("🚀 Starting ChurnGuard MLOps Services")
    print("="*60 + "\n")
    
    # Vérifier si le modèle existe
    if not os.path.exists("models/churn_model.pkl"):
        logger.warning("⚠️ Model not found. Running training pipeline...")
        try:
            subprocess.run([sys.executable, "train_pipeline.py"], check=True)
            logger.info("✅ Model trained successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Training failed: {e}")
            return
    
    # Démarrer MLflow
    mlflow_process = start_mlflow()
    if not mlflow_process:
        logger.error("❌ Cannot continue without MLflow")
        return
    
    # Démarrer l'API
    api_process = start_api()
    if not api_process:
        logger.error("❌ Cannot continue without API")
        mlflow_process.terminate()
        return
    
    # Vérifier les services
    if check_services():
        print("\n" + "="*60)
        print("🎉 All services started successfully!")
        print("="*60)
        print("\n🔗 Service URLs:")
        print("  📊 MLflow UI: http://localhost:5000")
        print("  🌐 API Docs: http://localhost:8000/docs")
        print("  🔧 API Health: http://localhost:8000/health")
        print("\n📋 Available Endpoints:")
        print("  GET  /                 - Service info")
        print("  GET  /health           - Health check")
        print("  GET  /model/info       - Model information")
        print("  POST /predict          - Single prediction")
        print("  POST /predict/batch    - Batch prediction")
        print("\n🛑 To stop services, press Ctrl+C")
        print("="*60 + "\n")
    else:
        logger.error("❌ Some services failed to start")
    
    try:
        # Garder le script en cours d'exécution
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping services...")
        
        if api_process:
            api_process.terminate()
            logger.info("✅ FastAPI stopped")
        
        if mlflow_process:
            mlflow_process.terminate()
            logger.info("✅ MLflow stopped")
        
        print("👋 Services stopped. Goodbye!")

if __name__ == "__main__":
    main()