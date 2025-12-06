"""
Dashboard de visualisation ChurnGuard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import requests
import json
from datetime import datetime
import os
import sys

# Configuration de la page
st.set_page_config(
    page_title="ChurnGuard Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
        font-weight: bold;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .risk-high {
        color: #DC2626;
        font-weight: bold;
    }
    .risk-medium {
        color: #F59E0B;
        font-weight: bold;
    }
    .risk-low {
        color: #10B981;
        font-weight: bold;
    }
    .info-box {
        background-color: #EFF6FF;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #BFDBFE;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

class ChurnGuardDashboard:
    def __init__(self):
        self.api_url = "http://localhost:8000"
        self.models_dir = "models"
        self.data_dir = "data"
        
    def load_data(self):
        """Charge les données et modèles"""
        try:
            # Charger les données d'entraînement
            train_path = os.path.join(self.data_dir, "customer_churn_dataset-training-master.csv")
            test_path = os.path.join(self.data_dir, "customer_churn_dataset-testing-master.csv")
            
            if os.path.exists(train_path):
                self.train_df = pd.read_csv(train_path)
            else:
                self.train_df = None
                
            if os.path.exists(test_path):
                self.test_df = pd.read_csv(test_path)
            else:
                self.test_df = None
            
            # Charger le modèle
            model_path = os.path.join(self.models_dir, "churn_model.pkl")
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
            else:
                self.model = None
            
            # Charger les résultats du modèle
            results_path = os.path.join(self.models_dir, "model_comparison.csv")
            if os.path.exists(results_path):
                self.model_results = pd.read_csv(results_path)
            else:
                self.model_results = None
            
            # Charger les features
            features_path = os.path.join(self.models_dir, "feature_names.pkl")
            if os.path.exists(features_path):
                self.feature_names = joblib.load(features_path)
            else:
                self.feature_names = None
            
            return True
            
        except Exception as e:
            st.error(f"Erreur lors du chargement des données: {e}")
            return False
    
    def check_api_status(self):
        """Vérifie le statut de l'API"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return data.get('status') == 'healthy'
            return False
        except:
            return False
    
    def render_header(self):
        """Afficher l'en-tête du dashboard"""
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown('<div class="main-header">ChurnGuard - Dashboard de Visualisation</div>', 
                       unsafe_allow_html=True)
            st.markdown("Système de prédiction et d'analyse du churn client")
        
        with col2:
            api_status = self.check_api_status()
            status_color = "green" if api_status else "red"
            status_text = "Connecté" if api_status else "Non connecté"
            
            st.markdown(f"""
            <div class="info-box">
                <h4>Statut du Système</h4>
                <p><strong>API:</strong> <span style="color:{status_color}">{status_text}</span></p>
                <p><strong>Modèle:</strong> {'Chargé' if self.model is not None else 'Non chargé'}</p>
                <p><strong>Données:</strong> {'Disponibles' if self.train_df is not None else 'Non disponibles'}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
    
    def render_overview_metrics(self):
        """Affiche les métriques d'aperçu"""
        st.markdown('<div class="sub-header">Aperçu Global</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if self.train_df is not None and 'Churn' in self.train_df.columns:
                churn_rate = self.train_df['Churn'].mean() * 100
                st.metric("Taux de Churn (Train)", f"{churn_rate:.1f}%")
            else:
                st.metric("Taux de Churn (Train)", "N/A")
        
        with col2:
            if self.test_df is not None and 'Churn' in self.test_df.columns:
                churn_rate = self.test_df['Churn'].mean() * 100
                st.metric("Taux de Churn (Test)", f"{churn_rate:.1f}%")
            else:
                st.metric("Taux de Churn (Test)", "N/A")
        
        with col3:
            if self.model_results is not None:
                best_model = self.model_results.loc[self.model_results['test_roc_auc'].idxmax()]
                st.metric("Meilleur Modèle", best_model['model'])
            else:
                st.metric("Meilleur Modèle", "N/A")
        
        with col4:
            if self.model_results is not None:
                best_auc = self.model_results['test_roc_auc'].max()
                st.metric("ROC AUC (Test)", f"{best_auc:.3f}")
            else:
                st.metric("ROC AUC (Test)", "N/A")
    
    def render_data_distribution(self):
        """Visualise la distribution des données"""
        st.markdown('<div class="sub-header">Distribution des Données</div>', unsafe_allow_html=True)
        
        if self.train_df is None:
            st.warning("Données d'entraînement non disponibles")
            return
        
        tab1, tab2, tab3 = st.tabs(["Distribution du Churn", "Caractéristiques Numériques", "Caractéristiques Catégorielles"])
        
        with tab1:
            if 'Churn' in self.train_df.columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Distribution du churn
                    churn_counts = self.train_df['Churn'].value_counts()
                    fig = go.Figure(data=[
                        go.Pie(
                            labels=['Non-Churn', 'Churn'],
                            values=churn_counts.values,
                            hole=0.3,
                            marker_colors=['#10B981', '#EF4444']
                        )
                    ])
                    fig.update_layout(
                        title="Distribution du Churn",
                        height=400
                    )
                    st.plotly_chart(fig, width='stretch')
                
                with col2:
                    # Distribution par segment
                    if 'Age' in self.train_df.columns:
                        # Créer des groupes d'âge
                        self.train_df['Age_Group'] = pd.cut(
                            self.train_df['Age'],
                            bins=[0, 25, 35, 50, 65, 100],
                            labels=['18-25', '26-35', '36-50', '51-65', '65+']
                        )
                        
                        age_churn = self.train_df.groupby('Age_Group')['Churn'].mean().reset_index()
                        
                        fig = px.bar(
                            age_churn,
                            x='Age_Group',
                            y='Churn',
                            title="Taux de Churn par Groupe d'Âge",
                            color='Churn',
                            color_continuous_scale='Reds'
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, width='stretch')
        
        with tab2:
            # Distribution des caractéristiques numériques
            numeric_cols = self.train_df.select_dtypes(include=[np.number]).columns.tolist()
            if 'CustomerID' in numeric_cols:
                numeric_cols.remove('CustomerID')
            if 'Churn' in numeric_cols:
                numeric_cols.remove('Churn')
            
            selected_feature = st.selectbox(
                "Sélectionner une caractéristique numérique",
                numeric_cols[:10] if len(numeric_cols) > 10 else numeric_cols
            )
            
            if selected_feature:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Histogramme
                    fig = px.histogram(
                        self.train_df,
                        x=selected_feature,
                        color='Churn' if 'Churn' in self.train_df.columns else None,
                        nbins=50,
                        title=f"Distribution de {selected_feature}",
                        barmode='overlay',
                        opacity=0.7
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, width='stretch')
                
                with col2:
                    # Box plot
                    if 'Churn' in self.train_df.columns:
                        fig = px.box(
                            self.train_df,
                            x='Churn',
                            y=selected_feature,
                            color='Churn',
                            title=f"{selected_feature} par Statut de Churn",
                            points="outliers"
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, width='stretch')
        
        with tab3:
            # Distribution des caractéristiques catégorielles
            categorical_cols = self.train_df.select_dtypes(include=['object']).columns.tolist()
            
            if categorical_cols:
                selected_cat = st.selectbox(
                    "Sélectionner une caractéristique catégorielle",
                    categorical_cols
                )
                
                if selected_cat and 'Churn' in self.train_df.columns:
                    # Taux de churn par catégorie
                    cat_churn = self.train_df.groupby(selected_cat)['Churn'].agg(['mean', 'count']).reset_index()
                    cat_churn.columns = [selected_cat, 'Churn_Rate', 'Count']
                    cat_churn['Churn_Rate'] = cat_churn['Churn_Rate'] * 100
                    
                    fig = make_subplots(
                        rows=2, cols=1,
                        subplot_titles=(
                            f"Taux de Churn par {selected_cat}",
                            f"Nombre de Clients par {selected_cat}"
                        ),
                        vertical_spacing=0.15
                    )
                    
                    # Bar chart pour le taux de churn
                    fig.add_trace(
                        go.Bar(
                            x=cat_churn[selected_cat],
                            y=cat_churn['Churn_Rate'],
                            name="Taux de Churn (%)",
                            marker_color='indianred'
                        ),
                        row=1, col=1
                    )
                    
                    # Bar chart pour le nombre de clients
                    fig.add_trace(
                        go.Bar(
                            x=cat_churn[selected_cat],
                            y=cat_churn['Count'],
                            name="Nombre de Clients",
                            marker_color='lightblue'
                        ),
                        row=2, col=1
                    )
                    
                    fig.update_layout(height=600, showlegend=False)
                    st.plotly_chart(fig, width='stretch')
    
    def render_model_performance(self):
        """Visualise la performance du modèle"""
        st.markdown('<div class="sub-header">Performance du Modèle</div>', unsafe_allow_html=True)
        
        if self.model_results is None:
            st.warning("Résultats du modèle non disponibles")
            return
        
        tab1, tab2, tab3 = st.tabs(["Comparaison des Modèles", "Métriques Détaillées", "Importance des Features"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                # Comparaison ROC AUC
                fig = px.bar(
                    self.model_results,
                    x='model',
                    y='test_roc_auc',
                    title="ROC AUC par Modèle (Test Set)",
                    color='test_roc_auc',
                    color_continuous_scale='Viridis',
                    text='test_roc_auc'
                )
                fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                fig.update_layout(height=400)
                st.plotly_chart(fig, width='stretch')
            
            with col2:
                # Comparaison F1 Score
                fig = px.bar(
                    self.model_results,
                    x='model',
                    y='test_f1',
                    title="F1 Score par Modèle (Test Set)",
                    color='test_f1',
                    color_continuous_scale='Plasma',
                    text='test_f1'
                )
                fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                fig.update_layout(height=400)
                st.plotly_chart(fig, width='stretch')
        
        with tab2:
            # Matrice de corrélation des métriques
            metrics_df = self.model_results[['cv_mean', 'cv_std', 'test_roc_auc', 'test_f1']]
            corr_matrix = metrics_df.corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=corr_matrix.round(2).values,
                texttemplate='%{text}',
                textfont={"size": 10}
            ))
            
            fig.update_layout(
                title="Corrélation entre les Métriques",
                height=500
            )
            st.plotly_chart(fig, width='stretch')
        
        with tab3:
            if self.model is not None and hasattr(self.model.named_steps['classifier'], 'feature_importances_'):
                try:
                    # Extraire l'importance des features
                    importances = self.model.named_steps['classifier'].feature_importances_
                    
                    if self.feature_names is not None and len(self.feature_names) == len(importances):
                        importance_df = pd.DataFrame({
                            'feature': self.feature_names,
                            'importance': importances
                        }).sort_values('importance', ascending=False).head(15)
                        
                        fig = px.bar(
                            importance_df,
                            x='importance',
                            y='feature',
                            orientation='h',
                            title="Top 15 Features par Importance",
                            color='importance',
                            color_continuous_scale='Blues'
                        )
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, width='stretch')
                except:
                    st.info("L'importance des features n'est pas disponible pour ce modèle")
    
    def render_prediction_interface(self):
        """Interface de prédiction interactive"""
        st.markdown('<div class="sub-header">Prédiction Interactive</div>', unsafe_allow_html=True)
        
        if not self.check_api_status():
            st.error("L'API n'est pas disponible. Assurez-vous que le serveur API est en cours d'exécution.")
            return
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            with st.form("prediction_form"):
                st.markdown("#### Informations Client")
                
                # Formulaire de saisie
                age = st.slider("Âge", 18, 100, 35)
                gender = st.selectbox("Genre", ["Male", "Female", "Other"])
                tenure = st.slider("Ancienneté (mois)", 0, 100, 24)
                usage_frequency = st.slider("Fréquence d'Utilisation", 0, 100, 20)
                support_calls = st.slider("Appels Support", 0, 50, 3)
                payment_delay = st.slider("Retard Paiement (jours)", 0, 100, 5)
                subscription_type = st.selectbox("Type d'Abonnement", ["Basic", "Standard", "Premium"])
                contract_length = st.selectbox("Durée Contrat", ["Monthly", "Quarterly", "Annual"])
                total_spend = st.number_input("Dépenses Totales", min_value=0.0, value=1200.0, step=100.0)
                last_interaction = st.slider("Jours depuis dernière interaction", 0, 365, 7)
                
                submitted = st.form_submit_button("Prédire le Risque de Churn")
        
        with col2:
            st.markdown("#### Configuration")
            
            prediction_threshold = st.slider(
                "Seuil de Prédiction",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="Seuil de probabilité pour classer comme churn"
            )
            
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown("**Légende des Risques:**")
            st.markdown('- <span class="risk-low">Faible Risque</span>: < 40%', unsafe_allow_html=True)
            st.markdown('- <span class="risk-medium">Risque Modéré</span>: 40-70%', unsafe_allow_html=True)
            st.markdown('- <span class="risk-high">Haut Risque</span>: > 70%', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        if submitted:
            with st.spinner("Calcul de la prédiction..."):
                try:
                    # Préparer les données pour la requête
                    customer_data = {
                        "Age": age,
                        "Gender": gender,
                        "Tenure": tenure,
                        "Usage_Frequency": usage_frequency,
                        "Support_Calls": support_calls,
                        "Payment_Delay": payment_delay,
                        "Subscription_Type": subscription_type,
                        "Contract_Length": contract_length,
                        "Total_Spend": total_spend,
                        "Last_Interaction": last_interaction
                    }
                    
                    # Envoyer la requête à l'API
                    response = requests.post(
                        f"{self.api_url}/predict",
                        json=customer_data,
                        headers={"Content-Type": "application/json"},
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Afficher les résultats
                        st.markdown("---")
                        st.markdown("#### Résultats de la Prédiction")
                        
                        # Métriques
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            prob = result['churn_probability']
                            if prob >= 0.7:
                                risk_class = "risk-high"
                            elif prob >= 0.4:
                                risk_class = "risk-medium"
                            else:
                                risk_class = "risk-low"
                            
                            st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
                            st.markdown(f'<h3 style="margin-bottom: 0;">Probabilité de Churn</h3>', unsafe_allow_html=True)
                            st.markdown(f'<h1 class="{risk_class}">{prob:.1%}</h1>', unsafe_allow_html=True)
                            st.markdown(f'</div>', unsafe_allow_html=True)
                        
                        with col2:
                            prediction = "CHURN" if result['churn_prediction'] else "NO CHURN"
                            pred_color = "#EF4444" if result['churn_prediction'] else "#10B981"
                            
                            st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
                            st.markdown(f'<h3 style="margin-bottom: 0;">Prédiction</h3>', unsafe_allow_html=True)
                            st.markdown(f'<h1 style="color: {pred_color};">{prediction}</h1>', unsafe_allow_html=True)
                            st.markdown(f'</div>', unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
                            st.markdown(f'<h3 style="margin-bottom: 0;">Niveau de Risque</h3>', unsafe_allow_html=True)
                            st.markdown(f'<h1 class="{risk_class}">{result["risk_level"]}</h1>', unsafe_allow_html=True)
                            st.markdown(f'</div>', unsafe_allow_html=True)
                        
                        # Visualisation de la probabilité
                        fig = go.Figure()
                        
                        fig.add_trace(go.Indicator(
                            mode="gauge+number",
                            value=prob * 100,
                            title={'text': "Score de Risque"},
                            domain={'x': [0, 1], 'y': [0, 1]},
                            gauge={
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 40], 'color': "lightgreen"},
                                    {'range': [40, 70], 'color': "yellow"},
                                    {'range': [70, 100], 'color': "red"}
                                ],
                                'threshold': {
                                    'line': {'color': "black", 'width': 4},
                                    'thickness': 0.75,
                                    'value': prob * 100
                                }
                            }
                        ))
                        
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, width='stretch')
                        
                        # Recommandations
                        st.markdown("#### Recommandations")
                        for i, rec in enumerate(result['recommendations'][:5], 1):
                            st.markdown(f"{i}. {rec}")
                    
                    else:
                        st.error(f"Erreur de l'API: {response.status_code}")
                        st.text(response.text)
                        
                except Exception as e:
                    st.error(f"Erreur lors de la prédiction: {e}")
    
    def render_batch_analysis(self):
        """Analyse par lot"""
        st.markdown('<div class="sub-header">Analyse par Lot</div>', unsafe_allow_html=True)
        
        if not self.check_api_status():
            st.warning("L'API n'est pas disponible pour l'analyse par lot")
            return
        
        tab1, tab2 = st.tabs(["Téléchargement de Fichier", "Génération Automatique"])
        
        with tab1:
            st.markdown("Téléchargez un fichier CSV avec les données des clients")
            
            uploaded_file = st.file_uploader("Choisir un fichier CSV", type="csv")
            
            if uploaded_file is not None:
                try:
                    # Lire le fichier CSV
                    batch_df = pd.read_csv(uploaded_file)
                    
                    # Afficher un aperçu
                    st.markdown("**Aperçu des données:**")
                    st.dataframe(batch_df.head())
                    
                    # Vérifier les colonnes requises
                    required_columns = [
                        'Age', 'Gender', 'Tenure', 'Usage_Frequency', 
                        'Support_Calls', 'Payment_Delay', 'Subscription_Type',
                        'Contract_Length', 'Total_Spend', 'Last_Interaction'
                    ]
                    
                    missing_cols = [col for col in required_columns if col not in batch_df.columns]
                    
                    if missing_cols:
                        st.error(f"Colonnes manquantes: {missing_cols}")
                    else:
                        if st.button("Analyser le Lot", type="primary"):
                            with st.spinner("Analyse en cours..."):
                                # Convertir en format API
                                customers = batch_df[required_columns].to_dict('records')
                                
                                # Limiter à 100 clients pour les performances
                                if len(customers) > 100:
                                    st.warning(f"Le fichier contient {len(customers)} clients. Seuls les 100 premiers seront analysés.")
                                    customers = customers[:100]
                                
                                # Envoyer la requête batch
                                response = requests.post(
                                    f"{self.api_url}/predict/batch",
                                    json=customers,
                                    headers={"Content-Type": "application/json"},
                                    timeout=60
                                )
                                
                                if response.status_code == 200:
                                    results = response.json()
                                    
                                    # Afficher les résultats
                                    st.success(f"Analyse terminée: {results['summary']['successful_predictions']}/{results['summary']['total_customers']} prédictions réussies")
                                    
                                    # Visualiser les résultats
                                    self.visualize_batch_results(results)
                                
                                else:
                                    st.error(f"Erreur lors de l'analyse batch: {response.status_code}")
                
                except Exception as e:
                    st.error(f"Erreur lors de la lecture du fichier: {e}")
        
        with tab2:
            st.markdown("Générer automatiquement un lot de clients pour l'analyse")
            
            num_customers = st.slider("Nombre de clients", 5, 50, 20)
            
            if st.button("Générer et Analyser", type="primary"):
                with st.spinner("Génération et analyse en cours..."):
                    try:
                        # Générer des clients aléatoires
                        customers = []
                        for _ in range(num_customers):
                            customer = {
                                "Age": np.random.randint(20, 70),
                                "Gender": np.random.choice(["Male", "Female"]),
                                "Tenure": np.random.randint(0, 100),
                                "Usage_Frequency": np.random.randint(0, 100),
                                "Support_Calls": np.random.randint(0, 20),
                                "Payment_Delay": np.random.randint(0, 60),
                                "Subscription_Type": np.random.choice(["Basic", "Standard", "Premium"]),
                                "Contract_Length": np.random.choice(["Monthly", "Quarterly", "Annual"]),
                                "Total_Spend": np.random.uniform(100, 5000),
                                "Last_Interaction": np.random.randint(0, 90)
                            }
                            customers.append(customer)
                        
                        # Envoyer la requête batch
                        response = requests.post(
                            f"{self.api_url}/predict/batch",
                            json=customers,
                            headers={"Content-Type": "application/json"},
                            timeout=30
                        )
                        
                        if response.status_code == 200:
                            results = response.json()
                            
                            # Afficher les résultats
                            st.success(f"Analyse terminée: {results['summary']['successful_predictions']}/{results['summary']['total_customers']} prédictions réussies")
                            
                            # Visualiser les résultats
                            self.visualize_batch_results(results)
                        
                        else:
                            st.error(f"Erreur lors de l'analyse batch: {response.status_code}")
                    
                    except Exception as e:
                        st.error(f"Erreur lors de la génération: {e}")
    
    def visualize_batch_results(self, results):
        """Visualise les résultats de l'analyse batch"""
        summary = results['summary']
        
        # Métriques
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Taux de Churn Moyen", f"{summary.get('average_churn_probability', 0):.1%}")
        
        with col2:
            st.metric("Clients à Haut Risque", summary.get('high_risk_customers', 0))
        
        with col3:
            st.metric("Clients à Risque Modéré", summary.get('medium_risk_customers', 0))
        
        with col4:
            st.metric("Clients à Faible Risque", summary.get('low_risk_customers', 0))
        
        # Distribution des risques
        fig = px.pie(
            values=[
                summary.get('high_risk_customers', 0),
                summary.get('medium_risk_customers', 0),
                summary.get('low_risk_customers', 0)
            ],
            names=['Haut Risque', 'Risque Modéré', 'Faible Risque'],
            title="Distribution des Niveaux de Risque",
            color_discrete_sequence=['#EF4444', '#F59E0B', '#10B981']
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400)
        st.plotly_chart(fig, width='stretch')
        
        # Histogramme des probabilités
        probabilities = [r['churn_probability'] for r in results['results'] if r.get('success', False)]
        
        if probabilities:
            fig = px.histogram(
                x=probabilities,
                nbins=20,
                title="Distribution des Probabilités de Churn",
                labels={'x': 'Probabilité de Churn'},
                color_discrete_sequence=['#3B82F6']
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, width='stretch')
    
    def render_trend_analysis(self):
        """Analyse des tendances"""
        st.markdown('<div class="sub-header">Analyse des Tendances</div>', unsafe_allow_html=True)
        
        if self.train_df is None:
            st.warning("Données non disponibles pour l'analyse des tendances")
            return
        
        # Ajouter une colonne de date simulée pour l'analyse temporelle
        if 'Last_Interaction' in self.train_df.columns:
            # Utiliser Last_Interaction comme proxy temporel
            self.train_df['Interaction_Date'] = pd.to_datetime('2024-01-01') - pd.to_timedelta(self.train_df['Last_Interaction'], unit='D')
            
            # Agréger par mois
            self.train_df['Month'] = self.train_df['Interaction_Date'].dt.to_period('M').astype(str)
            
            monthly_churn = self.train_df.groupby('Month')['Churn'].agg(['mean', 'count']).reset_index()
            monthly_churn.columns = ['Month', 'Churn_Rate', 'Customer_Count']
            monthly_churn['Churn_Rate'] = monthly_churn['Churn_Rate'] * 100
            
            # Trier par mois
            monthly_churn = monthly_churn.sort_values('Month')
            
            # Graphique à double axe
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Taux de churn (axe gauche)
            fig.add_trace(
                go.Scatter(
                    x=monthly_churn['Month'],
                    y=monthly_churn['Churn_Rate'],
                    name="Taux de Churn (%)",
                    line=dict(color='red', width=2)
                ),
                secondary_y=False
            )
            
            # Nombre de clients (axe droit)
            fig.add_trace(
                go.Bar(
                    x=monthly_churn['Month'],
                    y=monthly_churn['Customer_Count'],
                    name="Nombre de Clients",
                    opacity=0.3
                ),
                secondary_y=True
            )
            
            fig.update_layout(
                title="Évolution du Taux de Churn",
                xaxis_title="Mois",
                height=400
            )
            
            fig.update_yaxes(title_text="Taux de Churn (%)", secondary_y=False)
            fig.update_yaxes(title_text="Nombre de Clients", secondary_y=True)
            
            st.plotly_chart(fig, width='stretch')
    
    def render_export_options(self):
        """Options d'exportation"""
        st.markdown('<div class="sub-header">Exportation des Données</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Exporter les Résultats du Modèle", type="secondary"):
                if self.model_results is not None:
                    csv = self.model_results.to_csv(index=False)
                    st.download_button(
                        label="Télécharger CSV",
                        data=csv,
                        file_name="model_results.csv",
                        mime="text/csv"
                    )
        
        with col2:
            if st.button("Exporter la Distribution des Données", type="secondary"):
                if self.train_df is not None:
                    # Créer un résumé des données
                    summary_stats = self.train_df.describe().transpose()
                    csv = summary_stats.to_csv()
                    st.download_button(
                        label="Télécharger CSV",
                        data=csv,
                        file_name="data_summary.csv",
                        mime="text/csv"
                    )
        
        with col3:
            if st.button("Exporter les Métriques", type="secondary"):
                # Créer un rapport de métriques
                metrics_report = {
                    "timestamp": datetime.now().isoformat(),
                    "best_model": self.model_results.loc[self.model_results['test_roc_auc'].idxmax(), 'model'] if self.model_results is not None else "N/A",
                    "best_auc": self.model_results['test_roc_auc'].max() if self.model_results is not None else "N/A",
                    "best_f1": self.model_results['test_f1'].max() if self.model_results is not None else "N/A",
                    "total_customers": len(self.train_df) if self.train_df is not None else "N/A",
                    "churn_rate": self.train_df['Churn'].mean() if self.train_df is not None and 'Churn' in self.train_df.columns else "N/A"
                }
                
                json_report = json.dumps(metrics_report, indent=2)
                st.download_button(
                    label="Télécharger JSON",
                    data=json_report,
                    file_name="metrics_report.json",
                    mime="application/json"
                )

def main():
    """Fonction principale du dashboard"""
    dashboard = ChurnGuardDashboard()
    
    # Charger les données
    if not dashboard.load_data():
        st.error("Impossible de charger les données. Vérifiez que les fichiers de données et de modèle existent.")
        return
    
    # Afficher l'en-tête
    dashboard.render_header()
    
    # Créer les onglets
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Aperçu",
        "Distribution des Données",
        "Performance du Modèle",
        "Prédiction",
        "Analyse Avancée"
    ])
    
    with tab1:
        dashboard.render_overview_metrics()
        dashboard.render_trend_analysis()
    
    with tab2:
        dashboard.render_data_distribution()
    
    with tab3:
        dashboard.render_model_performance()
    
    with tab4:
        dashboard.render_prediction_interface()
        dashboard.render_batch_analysis()
    
    with tab5:
        st.markdown('<div class="sub-header">Outils Avancés</div>', unsafe_allow_html=True)
        dashboard.render_export_options()
        
        # Section de débogage
        with st.expander("Informations de Débogage"):
            st.json({
                "api_status": dashboard.check_api_status(),
                "model_loaded": dashboard.model is not None,
                "train_data_shape": dashboard.train_df.shape if dashboard.train_df is not None else "N/A",
                "test_data_shape": dashboard.test_df.shape if dashboard.test_df is not None else "N/A",
                "model_results_available": dashboard.model_results is not None,
                "feature_names_count": len(dashboard.feature_names) if dashboard.feature_names is not None else 0
            })

if __name__ == "__main__":
    main()