import pandas as pd
import numpy as np
from loguru import logger

def create_features(df):
    """Crée de nouvelles features"""
    df_features = df.copy()
    
    # Features basées sur l'interaction
    if 'Last Interaction' in df_features.columns and 'Tenure' in df_features.columns:
        df_features['Interaction_Rate'] = df_features['Last Interaction'] / df_features['Tenure'].replace(0, 1)
        df_features['Activity_Level'] = df_features['Usage Frequency'] * df_features['Tenure']
    
    # Features basées sur les paiements
    if 'Payment Delay' in df_features.columns:
        df_features['Payment_Issue'] = (df_features['Payment Delay'] > 14).astype(int)
        df_features['High_Delay'] = (df_features['Payment Delay'] > 30).astype(int)
    
    # Features de valeur client
    if 'Total Spend' in df_features.columns and 'Tenure' in df_features.columns:
        df_features['Avg_Monthly_Spend'] = df_features['Total Spend'] / df_features['Tenure'].replace(0, 1)
        df_features['Customer_Value'] = df_features['Total Spend'] * df_features['Tenure']
    
    # Features d'engagement
    if 'Support Calls' in df_features.columns and 'Tenure' in df_features.columns:
        df_features['Avg_Support_Calls'] = df_features['Support Calls'] / df_features['Tenure'].replace(0, 1)
        df_features['Support_Intensity'] = df_features['Support Calls'] * df_features['Usage Frequency']
    
    # Features d'âge segmenté
    if 'Age' in df_features.columns:
        bins = [0, 30, 50, 65, 100]
        labels = ['Young', 'Adult', 'Senior', 'Elderly']
        df_features['Age_Group'] = pd.cut(df_features['Age'], bins=bins, labels=labels)
    
    # Feature de saisonnalité (si date disponible)
    if 'Last Interaction' in df_features.columns:
        df_features['Recent_Activity'] = (df_features['Last Interaction'] < 7).astype(int)
    
    logger.info(f"Added {len(df_features.columns) - len(df.columns)} new features")
    logger.info(f"New columns: {[col for col in df_features.columns if col not in df.columns]}")
    
    return df_features