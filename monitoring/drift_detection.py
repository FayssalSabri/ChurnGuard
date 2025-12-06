import pandas as pd
import numpy as np
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import DataDriftTable, DatasetDriftMetric
from loguru import logger
import mlflow
import joblib
from datetime import datetime

class DriftDetector:
    def __init__(self, reference_data_path, model_path='models/churn_model.pkl'):
        self.reference_data = pd.read_csv(reference_data_path)
        self.model = joblib.load(model_path)
        self.column_mapping = ColumnMapping(
            target='churn',
            numerical_features=['tenure', 'monthly_charges', 'total_charges'],
            categorical_features=['contract', 'payment_method', 'internet_service']
        )
    
    def detect_drift(self, current_data):
        """Detect data drift between reference and current data"""
        try:
            report = Report(metrics=[
                DataDriftTable(),
                DatasetDriftMetric()
            ])
            
            report.run(
                reference_data=self.reference_data,
                current_data=current_data,
                column_mapping=self.column_mapping
            )
            
            result = report.as_dict()
            drift_detected = result['metrics'][1]['result']['dataset_drift']
            drift_score = result['metrics'][1]['result']['drift_score']
            
            # Log to MLflow
            mlflow.log_metric("drift_score", drift_score)
            mlflow.log_metric("drift_detected", int(drift_detected))
            
            return {
                'drift_detected': drift_detected,
                'drift_score': drift_score,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Drift detection failed: {e}")
            return None