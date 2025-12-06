"""
Configuration du dashboard ChurnGuard
"""

class DashboardConfig:
    # Couleurs inspirées de churnguard.sh
    COLORS = {
        'primary': {
            'blue': '#0066FF',
            'dark': '#0A2540',
            'light': '#E6F0FF'
        },
        'status': {
            'success': '#00D4AA',
            'warning': '#FF9F00',
            'danger': '#FF4757',
            'neutral': '#8C8C9B'
        },
        'background': {
            'light': '#F8FAFC',
            'card': '#FFFFFF'
        }
    }
    
    # Typographie
    TYPOGRAPHY = {
        'font_family': "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
        'headers': {
            'h1': {'size': '2.8rem', 'weight': 700},
            'h2': {'size': '2rem', 'weight': 600},
            'h3': {'size': '1.5rem', 'weight': 600},
            'h4': {'size': '1.25rem', 'weight': 600}
        },
        'body': {
            'large': {'size': '1.125rem', 'weight': 400},
            'regular': {'size': '1rem', 'weight': 400},
            'small': {'size': '0.875rem', 'weight': 400},
            'caption': {'size': '0.75rem', 'weight': 400}
        }
    }
    
    # Espacements
    SPACING = {
        'xs': '0.25rem',
        'sm': '0.5rem',
        'md': '1rem',
        'lg': '1.5rem',
        'xl': '2rem',
        'xxl': '3rem'
    }
    
    # Bordures et ombres
    BORDERS = {
        'radius': {
            'sm': '4px',
            'md': '8px',
            'lg': '12px',
            'xl': '16px'
        },
        'shadow': {
            'sm': '0px 1px 3px rgba(0, 0, 0, 0.08)',
            'md': '0px 2px 8px rgba(0, 0, 0, 0.08)',
            'lg': '0px 4px 16px rgba(0, 0, 0, 0.08)'
        }
    }
    
    # Configuration API
    API = {
        'url': 'http://localhost:8000',
        'timeout': 30,
        'endpoints': {
            'health': '/health',
            'predict': '/predict',
            'batch': '/predict/batch',
            'info': '/model/info'
        }
    }
    
    # Configuration des données
    DATA = {
        'paths': {
            'train': 'data/customer_churn_dataset-training-master.csv',
            'test': 'data/customer_churn_dataset-testing-master.csv',
            'models': 'models/'
        },
        'required_columns': [
            'Age', 'Gender', 'Tenure', 'Usage_Frequency', 
            'Support_Calls', 'Payment_Delay', 'Subscription_Type',
            'Contract_Length', 'Total_Spend', 'Last_Interaction'
        ]
    }
    
    # Configuration des visualisations
    VISUALIZATIONS = {
        'charts': {
            'height': 400,
            'template': 'plotly_white',
            'color_scale': 'Viridis'
        },
        'metrics': {
            'precision': 2,
            'percentage_precision': 1
        }
    }

config = DashboardConfig()