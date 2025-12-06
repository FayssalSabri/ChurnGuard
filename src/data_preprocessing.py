import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import logging
from loguru import logger

def load_data(train_path='data/customer_churn_dataset-training-master.csv', 
              test_path='data/customer_churn_dataset-testing-master.csv'):
    """Charge les données d'entraînement et de test"""
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        logger.info(f"Train data shape: {train_df.shape}")
        logger.info(f"Test data shape: {test_df.shape}")
        logger.info(f"Train columns: {train_df.columns.tolist()}")
        
        return train_df, test_df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def clean_data(df):
    """Nettoie les données"""
    # Faire une copie
    df_clean = df.copy()
    
    # Gérer les valeurs manquantes
    if df_clean.isnull().sum().sum() > 0:
        logger.warning(f"Missing values detected: {df_clean.isnull().sum().sum()}")
        
        # Créer un dictionnaire pour le remplissage
        fill_values = {}
        
        for col in df_clean.columns:
            if df_clean[col].dtype in ['int64', 'float64']:
                fill_values[col] = df_clean[col].median()
            else:
                fill_values[col] = df_clean[col].mode()[0] if not df_clean[col].mode().empty else ''
        
        # Remplir les valeurs manquantes
        df_clean = df_clean.fillna(fill_values)
    
    # Supprimer les doublons
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    duplicates_removed = initial_rows - len(df_clean)
    if duplicates_removed > 0:
        logger.info(f"Removed {duplicates_removed} duplicates")
    
    # Traiter les outliers pour les colonnes numériques
    numerical_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    if 'CustomerID' in numerical_cols:
        numerical_cols.remove('CustomerID')
    
    for col in numerical_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Limiter les outliers sans les supprimer
        df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
    
    return df_clean

def encode_categorical_features(df):
    """Encode les features catégorielles"""
    df_encoded = df.copy()
    
    categorical_cols = df_encoded.select_dtypes(include=['object']).columns.tolist()
    label_encoders = {}
    
    for col in categorical_cols:
        # Vérifier si la colonne existe et n'est pas vide
        if col in df_encoded.columns and not df_encoded[col].empty:
            try:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                label_encoders[col] = le
            except Exception as e:
                logger.warning(f"Could not encode column {col}: {e}")
                # Si l'encodage échoue, supprimer la colonne
                df_encoded = df_encoded.drop(col, axis=1)
    
    return df_encoded, label_encoders

def prepare_features(df):
    """Prépare les features pour l'entraînement"""
    # Faire une copie
    df_features = df.copy()
    
    # Supprimer CustomerID s'il existe (pas utile pour la prédiction)
    if 'CustomerID' in df_features.columns:
        df_features = df_features.drop('CustomerID', axis=1)
    
    # Vérifier si 'Churn' existe
    if 'Churn' not in df_features.columns:
        raise ValueError("Target column 'Churn' not found in dataset")
    
    # Convertir Churn en int si ce n'est pas déjà le cas
    if df_features['Churn'].dtype != 'int64':
        df_features['Churn'] = df_features['Churn'].astype(int)
    
    # Séparer features et target
    X = df_features.drop('Churn', axis=1)
    y = df_features['Churn']
    
    # Convertir les colonnes catégorielles en string pour l'encodage OneHot
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        X[col] = X[col].astype(str)
    
    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Target distribution:\n{y.value_counts(normalize=True)}")
    logger.info(f"Categorical columns: {categorical_cols.tolist()}")
    
    return X, y