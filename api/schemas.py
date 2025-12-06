from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd

class PredictionRequest(BaseModel):
    """Schéma pour la requête de prédiction"""
    CustomerID: Optional[int] = None
    Age: int = Field(..., ge=18, le=100, description="Âge du client")
    Gender: str = Field(..., description="Genre du client")
    Tenure: int = Field(..., ge=0, le=100, description="Ancienneté en mois")
    Usage_Frequency: int = Field(..., ge=0, le=100, description="Fréquence d'utilisation")
    Support_Calls: int = Field(..., ge=0, le=50, description="Nombre d'appels support")
    Payment_Delay: int = Field(..., ge=0, le=90, description="Retard de paiement en jours")
    Subscription_Type: str = Field(..., description="Type d'abonnement")
    Contract_Length: str = Field(..., description="Durée du contrat")
    Total_Spend: float = Field(..., ge=0, description="Dépenses totales")
    Last_Interaction: int = Field(..., ge=0, le=365, description="Jours depuis dernière interaction")
    
    @validator('Gender')
    def validate_gender(cls, v):
        valid_genders = ['Male', 'Female', 'Other']
        if v not in valid_genders:
            raise ValueError(f'Gender must be one of {valid_genders}')
        return v
    
    @validator('Subscription_Type')
    def validate_subscription(cls, v):
        valid_types = ['Basic', 'Standard', 'Premium']
        if v not in valid_types:
            raise ValueError(f'Subscription type must be one of {valid_types}')
        return v
    
    @validator('Contract_Length')
    def validate_contract(cls, v):
        valid_lengths = ['Monthly', 'Quarterly', 'Annual']
        if v not in valid_lengths:
            raise ValueError(f'Contract length must be one of {valid_lengths}')
        return v

class PredictionResponse(BaseModel):
    """Schéma pour la réponse de prédiction"""
    customer_id: Optional[int] = None
    churn_prediction: bool = Field(..., description="Prédiction de churn (True = churn)")
    churn_probability: float = Field(..., ge=0, le=1, description="Probabilité de churn")
    interpretation: str = Field(..., description="Interprétation du risque")
    recommendations: List[str] = Field(..., description="Recommandations d'action")
    confidence_level: Optional[str] = Field(None, description="Niveau de confiance")
    
    @validator('confidence_level', always=True)
    def set_confidence_level(cls, v, values):
        probability = values.get('churn_probability', 0.5)
        if probability > 0.8 or probability < 0.2:
            return "High"
        elif probability > 0.6 or probability < 0.4:
            return "Medium"
        else:
            return "Low"