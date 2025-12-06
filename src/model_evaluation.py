import mlflow
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, 
                           recall_score, f1_score, confusion_matrix, 
                           classification_report, roc_curve, auc,
                           average_precision_score)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
import json
import os

def evaluate_model(y_true, y_pred, y_pred_proba, model_name=""):
    """Évalue le modèle et log les métriques"""
    
    try:
        metrics = {
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'average_precision': average_precision_score(y_true, y_pred_proba)
        }
        
        # Log dans MLflow
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
            logger.info(f"{metric_name}: {metric_value:.4f}")
        
        # Matrice de confusion
        cm = confusion_matrix(y_true, y_pred)
        cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Log la matrice de confusion
        mlflow.log_dict({
            'confusion_matrix': cm.tolist(),
            'confusion_matrix_percentage': cm_percentage.tolist()
        }, f"confusion_matrix_{model_name}.json")
        
        # Classification report
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        mlflow.log_dict(report, f"classification_report_{model_name}.json")
        
        # Créer des visualisations
        create_evaluation_plots(y_true, y_pred, y_pred_proba, model_name)
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error in model evaluation: {e}")
        return {}

def create_evaluation_plots(y_true, y_pred, y_pred_proba, model_name=""):
    """Crée des visualisations d'évaluation"""
    
    try:
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(12, 10))
        
        # Plot ROC
        plt.subplot(2, 2, 1)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        # Confusion Matrix Heatmap
        plt.subplot(2, 2, 2)
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Churn', 'Churn'],
                   yticklabels=['No Churn', 'Churn'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        # Distribution des probabilités
        plt.subplot(2, 2, 3)
        plt.hist(y_pred_proba[y_true == 0], bins=30, alpha=0.5, label='Non-Churn', color='blue', density=True)
        plt.hist(y_pred_proba[y_true == 1], bins=30, alpha=0.5, label='Churn', color='red', density=True)
        plt.xlabel('Predicted Probability')
        plt.ylabel('Density')
        plt.title('Probability Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Precision-Recall curve
        plt.subplot(2, 2, 4)
        from sklearn.metrics import precision_recall_curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        plt.plot(recall, precision, color='green', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Sauvegarder et log
        plot_path = f"evaluation_plots_{model_name}.png"
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        mlflow.log_artifact(plot_path)
        plt.close()
        
        # Nettoyer le fichier temporaire
        if os.path.exists(plot_path):
            os.remove(plot_path)
            
    except Exception as e:
        logger.error(f"Error creating evaluation plots: {e}")