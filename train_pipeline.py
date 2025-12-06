import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
import joblib
import yaml
from loguru import logger
from datetime import datetime
import os
import sys
import warnings
warnings.filterwarnings('ignore')

def load_config():
    """Charge la configuration"""
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def load_and_preprocess_data():
    """Charge et prétraite les données"""
    
    # Charger les données
    train_df = pd.read_csv('data/customer_churn_dataset-training-master.csv')
    test_df = pd.read_csv('data/customer_churn_dataset-testing-master.csv')
    
    logger.info(f"📁 Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    
    # Supprimer CustomerID (data leakage potentiel)
    train_df = train_df.drop('CustomerID', axis=1, errors='ignore')
    test_df = test_df.drop('CustomerID', axis=1, errors='ignore')
    
    # Gérer les valeurs manquantes
    logger.info("🧹 Handling missing values...")
    
    # Remplir les valeurs numériques avec la médiane
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns
    train_df[numeric_cols] = train_df[numeric_cols].fillna(train_df[numeric_cols].median())
    test_df[numeric_cols] = test_df[numeric_cols].fillna(test_df[numeric_cols].median())
    
    # Remplir les valeurs catégorielles avec le mode
    categorical_cols = train_df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col in train_df.columns:
            train_df[col] = train_df[col].fillna(train_df[col].mode()[0] if not train_df[col].mode().empty else 'Unknown')
        if col in test_df.columns:
            test_df[col] = test_df[col].fillna(test_df[col].mode()[0] if not test_df[col].mode().empty else 'Unknown')
    
    return train_df, test_df

def create_smart_features(df):
    """Crée des features intelligentes"""
    df_features = df.copy()
    
    # 1. Features financières
    if 'Total Spend' in df_features.columns and 'Tenure' in df_features.columns:
        df_features['Monthly_Spend'] = df_features['Total Spend'] / df_features['Tenure'].replace(0, 1)
        df_features['Spend_per_Interaction'] = df_features['Total Spend'] / df_features['Usage Frequency'].replace(0, 1)
    
    # 2. Features d'engagement
    if 'Support Calls' in df_features.columns and 'Tenure' in df_features.columns:
        df_features['Support_Calls_per_Month'] = df_features['Support Calls'] / df_features['Tenure'].replace(0, 1)
    
    # 3. Features de paiement
    if 'Payment Delay' in df_features.columns:
        df_features['Has_Payment_Delay'] = (df_features['Payment Delay'] > 0).astype(int)
        df_features['Severe_Delay'] = (df_features['Payment Delay'] > 30).astype(int)
    
    # 4. Features temporelles
    if 'Last Interaction' in df_features.columns:
        df_features['Is_Active'] = (df_features['Last Interaction'] <= 7).astype(int)
        df_features['Interaction_Category'] = pd.cut(
            df_features['Last Interaction'],
            bins=[-1, 3, 7, 14, 30, 365],
            labels=['Very Active', 'Active', 'Moderate', 'Inactive', 'Very Inactive']
        )
    
    # 5. Features démographiques
    if 'Age' in df_features.columns:
        df_features['Age_Group'] = pd.cut(
            df_features['Age'],
            bins=[0, 25, 35, 50, 65, 100],
            labels=['GenZ', 'Millennials', 'GenX', 'Boomers', 'Seniors']
        )
    
    logger.info(f"🔧 Created {len(df_features.columns) - len(df.columns)} new features")
    return df_features

def prepare_features_for_training(df):
    """Prépare les features pour l'entraînement"""
    
    # Séparer features et target
    if 'Churn' not in df.columns:
        raise ValueError("Target column 'Churn' not found")
    
    X = df.drop('Churn', axis=1)
    y = df['Churn'].astype(int)
    
    # Encoder les colonnes catégorielles
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    logger.info(f"📊 Features shape: {X.shape}")
    logger.info(f"🎯 Target distribution: 0={sum(y==0)/len(y):.2%}, 1={sum(y==1)/len(y):.2%}")
    
    return X, y, label_encoders

def create_model_pipeline(model_type='random_forest', categorical_cols=None, numerical_cols=None):
    """Crée un pipeline de modèle"""
    
    if categorical_cols is None:
        categorical_cols = []
    if numerical_cols is None:
        numerical_cols = []
    
    # Transformer numérique
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Transformer catégoriel
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Préprocesseur
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    # Sélection du modèle
    if model_type == 'random_forest':
        classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
    elif model_type == 'gradient_boosting':
        classifier = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        )
    elif model_type == 'logistic_regression':
        classifier = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced',
            C=0.1,
            solver='liblinear'
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Pipeline complet
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    
    return pipeline

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Évalue un modèle et retourne les métriques"""
    
    # Prédictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calcul des métriques
    metrics = {
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'average_precision': average_precision_score(y_test, y_pred_proba)
    }
    
    # Rapport de classification
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Courbe ROC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_data = {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    
    logger.info(f"📈 {model_name} Evaluation:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    return {
        'metrics': metrics,
        'report': report,
        'roc_curve': roc_data,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

def main():
    """Pipeline d'entraînement principal"""
    
    # Charger la configuration
    config = load_config()
    
    # Configurer MLflow
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    
    # Créer ou récupérer l'expérience
    experiment_name = config['mlflow']['experiment_name']
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            logger.info(f"✅ Created new experiment: {experiment_name}")
        else:
            experiment_id = experiment.experiment_id
            logger.info(f"ℹ️  Using existing experiment: {experiment_name}")
    except Exception as e:
        logger.warning(f"Could not get/create experiment: {e}")
        experiment_id = 0
    
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        try:
            # 1. Charger les données
            logger.info("🚀 Step 1: Loading data...")
            train_df, test_df = load_and_preprocess_data()
            
            # 2. Feature engineering
            logger.info("🔧 Step 2: Creating features...")
            train_df = create_smart_features(train_df)
            test_df = create_smart_features(test_df)
            
            # 3. Préparer les données
            logger.info("📊 Step 3: Preparing features for training...")
            X_train, y_train, label_encoders = prepare_features_for_training(train_df)
            X_test, y_test, _ = prepare_features_for_training(test_df)
            
            # Identifier les types de colonnes
            categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
            numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            logger.info(f"🔢 Numerical columns: {len(numerical_cols)}")
            logger.info(f"🔤 Categorical columns: {len(categorical_cols)}")
            
            # Log des informations des données
            mlflow.log_params({
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'features': X_train.shape[1],
                'categorical_features': len(categorical_cols),
                'numerical_features': len(numerical_cols),
                'churn_rate_train': y_train.mean(),
                'churn_rate_test': y_test.mean()
            })
            
            # 4. Entraîner plusieurs modèles
            logger.info("🤖 Step 4: Training models...")
            
            models_to_train = ['random_forest', 'gradient_boosting', 'logistic_regression']
            results = {}
            best_score = 0
            best_model = None
            best_model_name = ""
            
            for model_name in models_to_train:
                try:
                    logger.info(f"  Training {model_name}...")
                    
                    # Créer le modèle
                    model = create_model_pipeline(model_name, categorical_cols, numerical_cols)
                    
                    # Validation croisée
                    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                    cv_scores = cross_val_score(
                        model, X_train, y_train,
                        cv=cv,
                        scoring='roc_auc',
                        n_jobs=-1
                    )
                    
                    # Entraîner sur toutes les données
                    model.fit(X_train, y_train)
                    
                    # Évaluer sur le test set
                    eval_results = evaluate_model(model, X_test, y_test, model_name)
                    
                    # Stocker les résultats
                    results[model_name] = {
                        'model': model,
                        'cv_scores': cv_scores,
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std(),
                        'test_metrics': eval_results['metrics']
                    }
                    
                    # Log dans MLflow
                    with mlflow.start_run(nested=True, run_name=model_name):
                        mlflow.log_params({
                            'model_type': model_name,
                            'cv_folds': 5
                        })
                        
                        mlflow.log_metrics({
                            'cv_roc_auc_mean': cv_scores.mean(),
                            'cv_roc_auc_std': cv_scores.std(),
                            'test_roc_auc': eval_results['metrics']['roc_auc'],
                            'test_f1_score': eval_results['metrics']['f1_score']
                        })
                        
                        # Log le modèle
                        mlflow.sklearn.log_model(model, "model")
                    
                    # Mettre à jour le meilleur modèle
                    if eval_results['metrics']['roc_auc'] > best_score:
                        best_score = eval_results['metrics']['roc_auc']
                        best_model = model
                        best_model_name = model_name
                    
                    logger.info(f"    CV AUC: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
                    logger.info(f"    Test AUC: {eval_results['metrics']['roc_auc']:.4f}")
                    
                except Exception as e:
                    logger.error(f"  Error training {model_name}: {e}")
                    continue
            
            if best_model is None:
                raise ValueError("❌ No model was successfully trained")
            
            # 5. Évaluation finale du meilleur modèle
            logger.info(f"\n🏆 Best Model: {best_model_name} (AUC: {best_score:.4f})")
            
            final_eval = evaluate_model(best_model, X_test, y_test, "Best Model")
            
            # Log des métriques finales
            mlflow.log_params({
                'best_model': best_model_name,
                'final_features': X_train.shape[1]
            })
            
            mlflow.log_metrics({
                'final_roc_auc': final_eval['metrics']['roc_auc'],
                'final_f1_score': final_eval['metrics']['f1_score'],
                'final_accuracy': final_eval['metrics']['accuracy']
            })
            
            # 6. Sauvegarder les artefacts
            logger.info("💾 Step 5: Saving artifacts...")
            os.makedirs('models', exist_ok=True)
            
            # Sauvegarder le meilleur modèle
            joblib.dump(best_model, 'models/churn_model.pkl')
            
            # Sauvegarder les encodeurs
            joblib.dump(label_encoders, 'models/label_encoders.pkl')
            
            # Sauvegarder les noms de features
            joblib.dump(X_train.columns.tolist(), 'models/feature_names.pkl')
            
            # Sauvegarder les résultats
            results_df = pd.DataFrame([
                {
                    'model': name,
                    'cv_mean': data['cv_mean'],
                    'cv_std': data['cv_std'],
                    'test_roc_auc': data['test_metrics']['roc_auc'],
                    'test_f1': data['test_metrics']['f1_score']
                }
                for name, data in results.items()
            ])
            results_df.to_csv('models/model_comparison.csv', index=False)
            
            # 7. Enregistrer le modèle dans MLflow
            try:
                mlflow.sklearn.log_model(best_model, "best_model")
                
                mlflow.register_model(
                    f"runs:/{mlflow.active_run().info.run_id}/best_model",
                    "churnguard-model"
                )
                logger.info("✅ Model registered in MLflow")
            except Exception as e:
                logger.warning(f"Could not register model: {e}")
            
            # 8. Rapport final
            logger.info("\n" + "="*60)
            logger.info("🎯 TRAINING COMPLETED SUCCESSFULLY!")
            logger.info("="*60)
            
            logger.info("\n📊 Model Performance Summary:")
            for model_name, data in results.items():
                logger.info(f"  {model_name:20} Test AUC: {data['test_metrics']['roc_auc']:.4f} | F1: {data['test_metrics']['f1_score']:.4f}")
            
            logger.info(f"\n📁 Artifacts saved in 'models/' folder:")
            logger.info(f"  - churn_model.pkl (best model)")
            logger.info(f"  - label_encoders.pkl")
            logger.info(f"  - feature_names.pkl")
            logger.info(f"  - model_comparison.csv")
            
            # Feature importance
            if hasattr(best_model.named_steps['classifier'], 'feature_importances_'):
                try:
                    importances = best_model.named_steps['classifier'].feature_importances_
                    
                    # Récupérer les noms de features
                    if hasattr(best_model.named_steps['preprocessor'], 'get_feature_names_out'):
                        feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()
                    else:
                        feature_names = [f"feature_{i}" for i in range(len(importances))]
                    
                    # Créer un DataFrame
                    importance_df = pd.DataFrame({
                        'feature': feature_names,
                        'importance': importances
                    }).sort_values('importance', ascending=False)
                    
                    logger.info("\n🎯 Top 5 Most Important Features:")
                    for idx, row in importance_df.head(5).iterrows():
                        logger.info(f"  {row['feature']}: {row['importance']:.4f}")
                    
                    importance_df.to_csv('models/feature_importance.csv', index=False)
                    
                except Exception as e:
                    logger.warning(f"Could not extract feature importance: {e}")
            
        except Exception as e:
            logger.error(f"❌ Training pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            raise

if __name__ == "__main__":
    main()