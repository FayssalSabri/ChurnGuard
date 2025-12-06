import mlflow
import mlflow.sklearn
import mlflow.pyfunc
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
import joblib
from loguru import logger
import json
import os

def train_multiple_models(X_train, y_train, config):
    """Entraîne plusieurs modèles et sélectionne le meilleur"""
    
    # Identifier les types de colonnes
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Prétraitement
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    # Définir les modèles
    models = {
        'random_forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        ),
        'gradient_boosting': GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        ),
        'logistic_regression': LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
    }
    
    best_model = None
    best_score = 0
    best_model_name = ""
    
    for model_name, model in models.items():
        try:
            logger.info(f"Training {model_name}...")
            
            # Créer le pipeline
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])
            
            # Entraîner le modèle
            pipeline.fit(X_train, y_train)
            
            # Validation croisée
            cv_scores = cross_val_score(pipeline, X_train, y_train, 
                                      cv=5, scoring='roc_auc', n_jobs=-1)
            mean_cv_score = cv_scores.mean()
            
            # Log dans MLflow
            mlflow.log_param(f"{model_name}_type", model_name)
            mlflow.log_metric(f"{model_name}_cv_roc_auc_mean", mean_cv_score)
            mlflow.log_metric(f"{model_name}_cv_roc_auc_std", cv_scores.std())
            
            # Sauvegarder le modèle si c'est le meilleur
            if mean_cv_score > best_score:
                best_score = mean_cv_score
                best_model = pipeline
                best_model_name = model_name
                
            logger.info(f"{model_name} - CV ROC AUC: {mean_cv_score:.4f}")
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")
            continue
    
    logger.info(f"Best model: {best_model_name} with score: {best_score:.4f}")
    return best_model, best_model_name, best_score