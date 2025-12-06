import mlflow
import os
from loguru import logger

def setup_mlflow_experiments():
    """Crée les expériences MLflow nécessaires"""
    
    # URI de tracking
    tracking_uri = "http://localhost:5000"
    mlflow.set_tracking_uri(tracking_uri)
    
    # Liste des expériences à créer
    experiments = [
        "churn-prediction",
        "churn-prediction-pro",
        "churn-prediction-experiments",
        "churn-model-comparison"
    ]
    
    for exp_name in experiments:
        try:
            # Vérifier si l'expérience existe déjà
            exp = mlflow.get_experiment_by_name(exp_name)
            if exp is None:
                # Créer l'expérience
                mlflow.create_experiment(exp_name)
                logger.info(f"✅ Created experiment: {exp_name}")
            else:
                logger.info(f"ℹ️  Experiment already exists: {exp_name}")
        except Exception as e:
            logger.error(f"❌ Error creating experiment {exp_name}: {e}")
    
    # Vérifier la connexion
    try:
        experiments = mlflow.search_experiments()
        logger.info(f"✅ MLflow setup complete. Found {len(experiments)} experiments")
        for exp in experiments:
            logger.info(f"  - {exp.name} (ID: {exp.experiment_id})")
    except Exception as e:
        logger.error(f"❌ Cannot connect to MLflow at {tracking_uri}: {e}")

if __name__ == "__main__":
    setup_mlflow_experiments()