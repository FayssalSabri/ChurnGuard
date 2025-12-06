import joblib
import pandas as pd
import numpy as np
from loguru import logger
import yaml
import os
import json

def save_artifacts(artifacts, config):
    """Sauvegarde les artefacts du modèle"""
    os.makedirs('models', exist_ok=True)
    
    # Sauvegarder le modèle
    if 'model' in artifacts:
        joblib.dump(artifacts['model'], 'models/churn_model.pkl')
        logger.info("Model saved to models/churn_model.pkl")
    
    # Sauvegarder les encodeurs
    if 'label_encoders' in artifacts:
        joblib.dump(artifacts['label_encoders'], 'models/label_encoders.pkl')
        logger.info("Label encoders saved")
    
    # Sauvegarder les noms de features
    if 'feature_names' in artifacts:
        joblib.dump(artifacts['feature_names'], 'models/feature_names.pkl')
        logger.info("Feature names saved")
    
    # Sauvegarder un échantillon des données
    if 'X_train_sample' in artifacts and 'y_train_sample' in artifacts:
        artifacts['X_train_sample'].to_csv('models/train_sample.csv', index=False)
        artifacts['y_train_sample'].to_csv('models/target_sample.csv', index=False)
    
    # Sauvegarder la configuration
    with open('models/config.json', 'w') as f:
        json.dump(config, f, indent=2)

def load_artifacts():
    """Charge les artefacts du modèle"""
    artifacts = {}
    
    try:
        if os.path.exists('models/churn_model.pkl'):
            artifacts['model'] = joblib.load('models/churn_model.pkl')
        
        if os.path.exists('models/label_encoders.pkl'):
            artifacts['label_encoders'] = joblib.load('models/label_encoders.pkl')
        
        if os.path.exists('models/feature_names.pkl'):
            artifacts['feature_names'] = joblib.load('models/feature_names.pkl')
        
        return artifacts
    except Exception as e:
        logger.error(f"Error loading artifacts: {e}")
        return {}

def preprocess_input(input_df, label_encoders=None, expected_features=None):
    """Prétraite l'input pour la prédiction"""
    df = input_df.copy()
    
    # Appliquer les encodeurs si disponibles
    if label_encoders:
        for col, encoder in label_encoders.items():
            if col in df.columns:
                df[col] = encoder.transform(df[col].astype(str))
    
    # S'assurer que toutes les features attendues sont présentes
    if expected_features:
        for feature in expected_features:
            if feature not in df.columns:
                df[feature] = 0  # Valeur par défaut pour les features manquantes
    
    # Garder seulement les features attendues
    if expected_features:
        df = df[expected_features]
    
    return df

def log_model_info(model, X_train, y_train):
    """Log les informations du modèle"""
    info = {
        'model_type': str(type(model)),
        'train_samples': len(X_train),
        'features_count': X_train.shape[1],
        'target_distribution': y_train.value_counts().to_dict(),
        'feature_names': X_train.columns.tolist()
    }
    
    logger.info(f"Model info: {info}")
    return info