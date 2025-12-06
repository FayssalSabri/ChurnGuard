from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from contextlib import asynccontextmanager
import pandas as pd
import numpy as np
import joblib
from loguru import logger
import os
import sys
from typing import List, Optional, Dict, Any

# Obtenir le chemin racine du projet
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Variables globales
model = None
label_encoders = None
feature_names = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gère le cycle de vie de l'application"""
    # Startup
    await load_model()
    logger.info(" API started successfully!")
    yield
    # Shutdown
    logger.info(" API shutting down...")

app = FastAPI(
    title="ChurnGuard API",
    description="Customer Churn Prediction API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modèles Pydantic V2
class CustomerData(BaseModel):
    Age: int = Field(..., ge=18, le=100, description="Customer age")
    Gender: str = Field(..., description="Gender (Male/Female/Other)")
    Tenure: int = Field(..., ge=0, le=100, description="Tenure in months")
    Usage_Frequency: int = Field(..., ge=0, le=100, description="Usage frequency")
    Support_Calls: int = Field(..., ge=0, le=50, description="Number of support calls")
    Payment_Delay: int = Field(..., ge=0, le=100, description="Payment delay in days")
    Subscription_Type: str = Field(..., description="Subscription type")
    Contract_Length: str = Field(..., description="Contract length")
    Total_Spend: float = Field(..., ge=0, description="Total spend")
    Last_Interaction: int = Field(..., ge=0, le=365, description="Days since last interaction")
    
    @field_validator('Gender')
    @classmethod
    def validate_gender(cls, v: str) -> str:
        valid_genders = ['Male', 'Female', 'Other']
        if v not in valid_genders:
            raise ValueError(f'Gender must be one of {valid_genders}')
        return v
    
    @field_validator('Subscription_Type')
    @classmethod
    def validate_subscription(cls, v: str) -> str:
        valid_types = ['Basic', 'Standard', 'Premium']
        if v not in valid_types:
            raise ValueError(f'Subscription type must be one of {valid_types}')
        return v
    
    @field_validator('Contract_Length')
    @classmethod
    def validate_contract(cls, v: str) -> str:
        valid_lengths = ['Monthly', 'Quarterly', 'Annual']
        if v not in valid_lengths:
            raise ValueError(f'Contract length must be one of {valid_lengths}')
        return v

class PredictionResponse(BaseModel):
    churn_prediction: bool = Field(..., description="True if customer is predicted to churn")
    churn_probability: float = Field(..., ge=0, le=1, description="Probability of churn")
    risk_level: str = Field(..., description="Risk level (HIGH/MEDIUM/LOW)")
    confidence: str = Field(..., description="Confidence level")
    recommendations: List[str] = Field(..., description="Recommended actions")

async def load_model():
    """Charge le modèle et les artefacts"""
    global model, label_encoders, feature_names
    
    try:
        logger.info(f" Looking for models in: {MODELS_DIR}")
        
        # Charger le modèle
        model_path = os.path.join(MODELS_DIR, "churn_model.pkl")
        logger.info(f" Model path: {model_path}")
        
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            logger.info(f" Model loaded from {model_path}")
            
            # Vérifier le type de modèle
            if hasattr(model, 'named_steps'):
                classifier = model.named_steps.get('classifier')
                if classifier:
                    logger.info(f" Model type: {type(classifier).__name__}")
        else:
            # Lister les fichiers disponibles
            available_files = os.listdir(MODELS_DIR) if os.path.exists(MODELS_DIR) else []
            logger.error(f" Model file not found: {model_path}")
            logger.error(f" Available files in {MODELS_DIR}:")
            for file in available_files:
                logger.error(f"   - {file}")
            
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Charger les encodeurs
        encoders_path = os.path.join(MODELS_DIR, "label_encoders.pkl")
        if os.path.exists(encoders_path):
            label_encoders = joblib.load(encoders_path)
            logger.info(f" Label encoders loaded ({len(label_encoders)} encoders)")
        else:
            logger.warning(f" Label encoders not found: {encoders_path}")
        
        # Charger les noms de features
        features_path = os.path.join(MODELS_DIR, "feature_names.pkl")
        if os.path.exists(features_path):
            feature_names = joblib.load(features_path)
            logger.info(f" Feature names loaded ({len(feature_names)} features)")
        else:
            logger.warning(f" Feature names not found: {features_path}")
            # Essayer d'extraire les features du modèle
            if model is not None and hasattr(model, 'named_steps'):
                try:
                    preprocessor = model.named_steps.get('preprocessor')
                    if hasattr(preprocessor, 'get_feature_names_out'):
                        feature_names = preprocessor.get_feature_names_out().tolist()
                        logger.info(f" Extracted {len(feature_names)} features from model")
                except:
                    pass
        
        # Charger la comparaison des modèles
        comparison_path = os.path.join(MODELS_DIR, "model_comparison.csv")
        if os.path.exists(comparison_path):
            comparison = pd.read_csv(comparison_path)
            logger.info("📈 Model comparison loaded:")
            for _, row in comparison.iterrows():
                logger.info(f"  {row['model']}: AUC={row['test_roc_auc']:.4f}, F1={row['test_f1']:.4f}")
        
    except Exception as e:
        logger.error(f" Failed to load model: {e}")
        logger.error("💡 Solution: Run 'python train_pipeline.py' from project root first")
        raise

@app.get("/")
async def root():
    """Endpoint racine avec informations du service"""
    return {
        "service": "ChurnGuard API",
        "version": "1.0.0",
        "status": "operational" if model is not None else "unhealthy",
        "model": "Logistic Regression (v2)" if model is not None else "No model loaded",
        "project_root": PROJECT_ROOT,
        "models_dir": MODELS_DIR,
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "model_info": "/model/info",
            "predict": "/predict",
            "batch_predict": "/predict/batch"
        }
    }

@app.get("/health")
async def health_check():
    """Vérifie l'état de santé de l'API"""
    status = "healthy" if model is not None else "unhealthy"
    return {
        "status": status,
        "model_loaded": model is not None,
        "features_loaded": feature_names is not None,
        "label_encoders_loaded": label_encoders is not None,
        "project_root": PROJECT_ROOT,
        "models_dir": MODELS_DIR,
        "timestamp": pd.Timestamp.now().isoformat()
    }

@app.get("/model/info")
async def model_info():
    """Informations détaillées sur le modèle"""
    if model is not None:
        info = {
            "model_type": type(model).__name__,
            "version": "2.0",
            "features_count": len(feature_names) if feature_names else "unknown",
            "project_root": PROJECT_ROOT,
            "models_dir": MODELS_DIR
        }
        
        # Ajouter des informations spécifiques au modèle
        if hasattr(model, 'named_steps'):
            info['pipeline_steps'] = list(model.named_steps.keys())
            
            classifier = model.named_steps.get('classifier')
            if classifier is not None:
                info['classifier_type'] = type(classifier).__name__
                
                # Ajouter quelques paramètres importants
                if hasattr(classifier, 'get_params'):
                    params = classifier.get_params()
                    important_params = {
                        k: v for k, v in params.items() 
                        if k in ['C', 'penalty', 'solver', 'max_iter', 'class_weight']
                    }
                    info['classifier_params'] = important_params
        
        return info
    
    return {
        "error": "Model not loaded",
        "solution": "Run 'python train_pipeline.py' from project root first",
        "project_root": PROJECT_ROOT,
        "models_dir": MODELS_DIR
    }

def create_features_for_prediction(input_dict: Dict[str, Any]) -> pd.DataFrame:
    """Crée les features pour la prédiction"""
    df = pd.DataFrame([input_dict])
    
    # Feature engineering (identique à l'entraînement)
    # 1. Features financières
    if 'Total Spend' in df.columns and 'Tenure' in df.columns:
        df['Monthly_Spend'] = df['Total Spend'] / df['Tenure'].replace(0, 1)
    
    if 'Total Spend' in df.columns and 'Usage Frequency' in df.columns:
        df['Spend_per_Interaction'] = df['Total Spend'] / df['Usage Frequency'].replace(0, 1)
    
    # 2. Features de support
    if 'Support Calls' in df.columns and 'Tenure' in df.columns:
        df['Support_Calls_per_Month'] = df['Support Calls'] / df['Tenure'].replace(0, 1)
    
    # 3. Features de paiement
    if 'Payment Delay' in df.columns:
        df['Has_Payment_Delay'] = (df['Payment Delay'] > 0).astype(int)
        df['Severe_Delay'] = (df['Payment Delay'] > 30).astype(int)
    
    # 4. Features d'engagement
    if 'Last Interaction' in df.columns:
        df['Is_Active'] = (df['Last Interaction'] <= 7).astype(int)
    
    # 5. Features démographiques
    if 'Age' in df.columns:
        # Catégorisation de l'âge
        bins = [0, 25, 35, 50, 65, 100]
        labels = ['GenZ', 'Millennials', 'GenX', 'Boomers', 'Seniors']
        df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels)
    
    # Encoder les colonnes catégorielles si des encodeurs sont disponibles
    if label_encoders:
        for col, encoder in label_encoders.items():
            if col in df.columns:
                try:
                    df[col] = encoder.transform(df[col].astype(str))
                except ValueError:
                    # Si une nouvelle valeur est rencontrée, utiliser une valeur par défaut
                    df[col] = 0
                    logger.warning(f" Unknown value in column {col}, using default encoding")
    
    # S'assurer que toutes les features attendues sont présentes
    if feature_names is not None:
        missing_features = []
        for feature in feature_names:
            if feature not in df.columns:
                df[feature] = 0  # Valeur par défaut pour les features manquantes
                missing_features.append(feature)
        
        if missing_features:
            logger.warning(f" {len(missing_features)} features missing, set to 0")
        
        # Sélectionner seulement les features attendues (dans le bon ordre)
        df = df[feature_names]
    
    return df

def generate_recommendations(customer_data: Dict[str, Any], churn_probability: float) -> List[str]:
    """Génère des recommandations personnalisées basées sur les données client"""
    recommendations = []
    
    # Recommandations basées sur le niveau de risque
    if churn_probability >= 0.7:
        recommendations.append("🚨 HIGH RISK: Immediate action required")
        recommendations.append("📞 Contact customer within 24 hours")
        recommendations.append("💰 Offer retention discount or special offer")
    elif churn_probability >= 0.4:
        recommendations.append(" MEDIUM RISK: Proactive engagement needed")
        recommendations.append("📧 Send personalized engagement email")
        recommendations.append("🎁 Offer loyalty rewards or upgrade incentives")
    else:
        recommendations.append(" LOW RISK: Maintain current relationship")
        recommendations.append("👍 Continue regular communication")
        recommendations.append("💡 Suggest premium features or add-ons")
    
    # Recommandations spécifiques basées sur les données client
    payment_delay = customer_data.get('Payment_Delay', 0)
    if payment_delay > 30:
        recommendations.append("💳 CRITICAL: Payment plan restructuring needed")
    elif payment_delay > 14:
        recommendations.append("💳 Offer flexible payment options")
    
    support_calls = customer_data.get('Support_Calls', 0)
    if support_calls > 8:
        recommendations.append("🛠️ URGENT: Technical issue resolution priority")
    elif support_calls > 4:
        recommendations.append("🛠️ Provide proactive technical support")
    
    last_interaction = customer_data.get('Last_Interaction', 365)
    if last_interaction > 60:
        recommendations.append("📞 CRITICAL: Re-engagement campaign required")
    elif last_interaction > 30:
        recommendations.append("📞 Initiate re-engagement campaign")
    
    usage_frequency = customer_data.get('Usage_Frequency', 0)
    if usage_frequency < 5:
        recommendations.append("🎯 HIGH PRIORITY: Onboarding assistance needed")
    elif usage_frequency < 10:
        recommendations.append("🎯 Provide usage tutorials and tips")
    
    # Recommandation finale
    recommendations.append(f"📈 Predicted churn probability: {churn_probability:.1%}")
    
    return recommendations

@app.post("/predict", response_model=PredictionResponse)
async def predict(customer_data: CustomerData):
    """
    Endpoint de prédiction pour un client unique
    """
    
    if model is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Model not loaded",
                "solution": "Run 'python train_pipeline.py' from project root first",
                "project_root": PROJECT_ROOT,
                "models_dir": MODELS_DIR
            }
        )
    
    try:
        # Convertir les données en dictionnaire
        input_dict = customer_data.model_dump()
        
        # Journaliser la requête
        logger.info(f" Prediction request received")
        
        # Créer les features pour la prédiction
        input_df = create_features_for_prediction(input_dict)
        
        # Vérifier les features
        logger.info(f" Input features shape: {input_df.shape}")
        
        # Faire la prédiction
        try:
            churn_probability = float(model.predict_proba(input_df)[0, 1])
            churn_prediction = churn_probability >= 0.5
        except Exception as e:
            logger.error(f" Prediction error: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
        
        # Déterminer le niveau de risque
        if churn_probability >= 0.7:
            risk_level = "HIGH"
            confidence = "High"
        elif churn_probability >= 0.4:
            risk_level = "MEDIUM"
            confidence = "Medium"
        else:
            risk_level = "LOW"
            confidence = "High"
        
        # Générer les recommandations
        recommendations = generate_recommendations(input_dict, churn_probability)
        
        # Journaliser la réponse
        logger.info(f" Prediction response: {churn_probability:.2%} ({risk_level} risk)")
        
        return PredictionResponse(
            churn_prediction=churn_prediction,
            churn_probability=churn_probability,
            risk_level=risk_level,
            confidence=confidence,
            recommendations=recommendations
        )
        
    except ValueError as e:
        logger.error(f" Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f" Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(customers: List[CustomerData]):
    """
    Endpoint de prédiction par lot pour plusieurs clients
    """
    
    if model is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Model not loaded",
                "solution": "Run 'python train_pipeline.py' from project root first"
            }
        )
    
    try:
        results = []
        predictions_data = []
        
        for idx, customer_data in enumerate(customers):
            try:
                # Convertir en dictionnaire
                input_dict = customer_data.model_dump()
                
                # Créer les features
                input_df = create_features_for_prediction(input_dict)
                
                # Faire la prédiction
                churn_probability = float(model.predict_proba(input_df)[0, 1])
                churn_prediction = churn_probability >= 0.5
                
                # Déterminer le niveau de risque
                risk_level = "HIGH" if churn_probability >= 0.7 else "MEDIUM" if churn_probability >= 0.4 else "LOW"
                
                results.append({
                    "customer_index": idx,
                    "churn_prediction": churn_prediction,
                    "churn_probability": churn_probability,
                    "risk_level": risk_level,
                    "success": True
                })
                
                predictions_data.append(churn_probability)
                
            except Exception as e:
                logger.error(f" Error processing customer {idx}: {e}")
                results.append({
                    "customer_index": idx,
                    "error": str(e),
                    "success": False
                })
        
        # Calculer les statistiques
        if predictions_data:
            successful = [r for r in results if r.get('success', False)]
            if successful:
                probabilities = [r['churn_probability'] for r in successful]
                
                summary = {
                    "total_customers": len(customers),
                    "successful_predictions": len(successful),
                    "failed_predictions": len(results) - len(successful),
                    "average_churn_probability": float(np.mean(probabilities)),
                    "median_churn_probability": float(np.median(probabilities)),
                    "std_churn_probability": float(np.std(probabilities)),
                    "high_risk_customers": len([p for p in probabilities if p >= 0.7]),
                    "medium_risk_customers": len([p for p in probabilities if p >= 0.4]),
                    "low_risk_customers": len([p for p in probabilities if p < 0.4]),
                    "success_rate": len(successful) / len(customers) if customers else 0
                }
            else:
                summary = {
                    "error": "No successful predictions",
                    "success_rate": 0
                }
        else:
            summary = {
                "error": "No predictions generated",
                "success_rate": 0
            }
        
        logger.info(f" Batch prediction summary: {summary['successful_predictions']}/{summary['total_customers']} successful")
        
        return {
            "results": results,
            "summary": summary,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f" Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)