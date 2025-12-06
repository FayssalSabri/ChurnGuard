"""
Utilitaires pour le dashboard ChurnGuard
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class DataProcessor:
    """Processeur de données pour le dashboard"""
    
    @staticmethod
    def calculate_risk_segments(data, risk_scores):
        """Calcule les segments de risque"""
        segments = []
        for score in risk_scores:
            if score < 0.3:
                segments.append('Faible')
            elif score < 0.7:
                segments.append('Moyen')
            else:
                segments.append('Élevé')
        return segments
    
    @staticmethod
    def generate_sample_predictions(n_samples=1000):
        """Génère des prédictions d'échantillon pour la visualisation"""
        np.random.seed(42)
        
        # Générer des données d'échantillon
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        
        # Tendance simulée
        base_performance = 0.75
        daily_variation = np.random.randn(30) * 0.02
        performance = base_performance + daily_variation.cumsum()
        performance = np.clip(performance, 0.7, 0.85)
        
        # Créer le DataFrame
        df = pd.DataFrame({
            'date': dates,
            'performance': performance,
            'moving_avg_7': performance.rolling(7).mean(),
            'predictions': np.random.randint(50, 200, 30),
            'actual_churn': np.random.randint(40, 180, 30)
        })
        
        return df
    
    @staticmethod
    def calculate_metrics(data, churn_col='Churn'):
        """Calcule les métriques de performance"""
        if data is None or churn_col not in data.columns:
            return None
        
        metrics = {
            'total_customers': len(data),
            'churn_rate': data[churn_col].mean() * 100,
            'churn_count': data[churn_col].sum(),
            'retention_rate': (1 - data[churn_col].mean()) * 100
        }
        
        # Calculer les statistiques par segment (si disponibles)
        if 'Age' in data.columns:
            # Segmentation par âge
            age_bins = [0, 25, 35, 50, 65, 100]
            age_labels = ['18-25', '26-35', '36-50', '51-65', '65+']
            data['age_group'] = pd.cut(data['Age'], bins=age_bins, labels=age_labels)
            
            age_churn = data.groupby('age_group')[churn_col].mean() * 100
            metrics['age_churn_rates'] = age_churn.to_dict()
        
        return metrics

class VisualizationTools:
    """Outils de visualisation"""
    
    def __init__(self, config):
        self.config = config
    
    def create_metric_card(self, title, value, change=None, change_label=None):
        """Crée une carte de métrique"""
        if change is not None:
            change_class = "change-positive" if change > 0 else "change-negative"
            change_icon = "↗" if change > 0 else "↘"
            change_html = f"""
            <div class="metric-change {change_class}">
                {change_icon} {abs(change):.1f}% {change_label if change_label else ''}
            </div>
            """
        else:
            change_html = ""
        
        return f"""
        <div class="metric-card">
            <div class="metric-label">{title}</div>
            <div class="metric-value">{value}</div>
            {change_html}
        </div>
        """
    
    def create_risk_gauge(self, risk_score, title="Score de risque"):
        """Crée un indicateur de risque"""
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score * 100,
            title={'text': title, 'font': {'size': 16}},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': self.config.COLORS['primary']['blue']},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 30], 'color': self.config.COLORS['status']['success']},
                    {'range': [30, 70], 'color': self.config.COLORS['status']['warning']},
                    {'range': [70, 100], 'color': self.config.COLORS['status']['danger']}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': risk_score * 100
                }
            }
        ))
        
        fig.update_layout(
            height=250,
            margin=dict(t=50, b=10, l=10, r=10),
            font={'family': self.config.TYPOGRAPHY['font_family']}
        )
        
        return fig
    
    def create_performance_trend(self, data, date_col='date', value_col='performance'):
        """Crée un graphique de tendance de performance"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=data[date_col],
            y=data[value_col],
            name='Performance quotidienne',
            mode='lines',
            line=dict(
                color=self.config.COLORS['primary']['blue'],
                width=2
            ),
            opacity=0.6
        ))
        
        if 'moving_avg_7' in data.columns:
            fig.add_trace(go.Scatter(
                x=data[date_col],
                y=data['moving_avg_7'],
                name='Moyenne mobile (7 jours)',
                mode='lines',
                line=dict(
                    color=self.config.COLORS['primary']['blue'],
                    width=3
                )
            ))
        
        fig.update_layout(
            title="Tendance de la performance",
            yaxis_title="Score",
            xaxis_title="Date",
            height=400,
            template="plotly_white",
            hovermode="x unified",
            font={'family': self.config.TYPOGRAPHY['font_family']}
        )
        
        return fig
    
    def create_risk_distribution(self, risk_scores, risk_categories):
        """Crée une visualisation de distribution des risques"""
        fig = go.Figure()
        
        colors = {
            'Faible': self.config.COLORS['status']['success'],
            'Moyen': self.config.COLORS['status']['warning'],
            'Élevé': self.config.COLORS['status']['danger']
        }
        
        for category in ['Faible', 'Moyen', 'Élevé']:
            category_data = [score for score, cat in zip(risk_scores, risk_categories) if cat == category]
            if category_data:
                fig.add_trace(go.Violin(
                    y=category_data,
                    name=category,
                    box_visible=True,
                    meanline_visible=True,
                    fillcolor=colors[category],
                    line_color=colors[category],
                    opacity=0.6
                ))
        
        fig.update_layout(
            title="Distribution des scores de risque",
            yaxis_title="Score de risque",
            xaxis_title="Catégorie de risque",
            height=400,
            showlegend=True,
            template="plotly_white",
            font={'family': self.config.TYPOGRAPHY['font_family']}
        )
        
        return fig

class RiskAnalyzer:
    """Analyseur de risque"""
    
    @staticmethod
    def analyze_customer_profile(customer_data):
        """Analyse un profil client pour identifier les facteurs de risque"""
        risk_factors = []
        
        # Vérifier les retards de paiement
        if customer_data.get('Payment_Delay', 0) > 30:
            risk_factors.append({
                'factor': 'Retard de paiement élevé',
                'severity': 'high',
                'message': f'Retard de {customer_data["Payment_Delay"]} jours'
            })
        elif customer_data.get('Payment_Delay', 0) > 14:
            risk_factors.append({
                'factor': 'Retard de paiement modéré',
                'severity': 'medium',
                'message': f'Retard de {customer_data["Payment_Delay"]} jours'
            })
        
        # Vérifier les appels support
        if customer_data.get('Support_Calls', 0) > 10:
            risk_factors.append({
                'factor': 'Nombre élevé d\'appels support',
                'severity': 'high',
                'message': f'{customer_data["Support_Calls"]} appels'
            })
        elif customer_data.get('Support_Calls', 0) > 5:
            risk_factors.append({
                'factor': 'Appels support fréquents',
                'severity': 'medium',
                'message': f'{customer_data["Support_Calls"]} appels'
            })
        
        # Vérifier l'inactivité
        if customer_data.get('Last_Interaction', 0) > 60:
            risk_factors.append({
                'factor': 'Inactivité prolongée',
                'severity': 'high',
                'message': f'{customer_data["Last_Interaction"]} jours depuis dernière interaction'
            })
        elif customer_data.get('Last_Interaction', 0) > 30:
            risk_factors.append({
                'factor': 'Faible activité récente',
                'severity': 'medium',
                'message': f'{customer_data["Last_Interaction"]} jours depuis dernière interaction'
            })
        
        # Vérifier le type d'abonnement
        if customer_data.get('Subscription_Type') == 'Basic':
            risk_factors.append({
                'factor': 'Abonnement basique',
                'severity': 'low',
                'message': 'Taux de churn plus élevé pour les abonnements basiques'
            })
        
        # Vérifier la durée du contrat
        if customer_data.get('Contract_Length') == 'Monthly':
            risk_factors.append({
                'factor': 'Contrat mensuel',
                'severity': 'medium',
                'message': 'Engagement à court terme'
            })
        
        return risk_factors
    
    @staticmethod
    def generate_recommendations(risk_factors, churn_probability):
        """Génère des recommandations basées sur les facteurs de risque"""
        recommendations = []
        
        # Recommandations basées sur la probabilité
        if churn_probability >= 0.7:
            recommendations.append("🚨 Contact immédiat requis - Client à très haut risque")
            recommendations.append("💰 Offrir une incitation à la rétention personnalisée")
            recommendations.append("👑 Assigner un gestionnaire de compte dédié")
        elif churn_probability >= 0.4:
            recommendations.append("⚠️ Engagement proactif nécessaire")
            recommendations.append("📧 Envoyer un email d'engagement personnalisé")
            recommendations.append("🎁 Proposer des avantages de fidélité")
        else:
            recommendations.append("✅ Maintenir la communication régulière")
            recommendations.append("💡 Suggérer des fonctionnalités premium")
        
        # Recommandations spécifiques basées sur les facteurs de risque
        for factor in risk_factors:
            if factor['severity'] == 'high':
                if 'paiement' in factor['factor'].lower():
                    recommendations.append("💳 Proposer un plan de paiement flexible")
                elif 'support' in factor['factor'].lower():
                    recommendations.append("🛠️ Fournir un support technique proactif")
                elif 'inactivité' in factor['factor'].lower():
                    recommendations.append("📞 Campagne de réengagement immédiate")
        
        return recommendations