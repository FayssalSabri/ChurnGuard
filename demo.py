"""
Démonstration complète du système ChurnGuard
"""

import requests
import json
import pandas as pd
import numpy as np
import time
import sys
from colorama import init, Fore, Style

# Initialiser colorama pour Windows
init(autoreset=True)

API_URL = "http://localhost:8000"

def print_header(text):
    """Affiche un en-tête coloré"""
    print(f"\n{Fore.CYAN}{'='*70}")
    print(f" {text}")
    print(f"{'='*70}{Style.RESET_ALL}")

def print_success(text):
    """Affiche un message de succès"""
    print(f"{Fore.GREEN} {text}{Style.RESET_ALL}")

def print_warning(text):
    """Affiche un message d'avertissement"""
    print(f"{Fore.YELLOW}  {text}{Style.RESET_ALL}")

def print_error(text):
    """Affiche un message d'erreur"""
    print(f"{Fore.RED} {text}{Style.RESET_ALL}")

def print_info(text):
    """Affiche un message d'information"""
    print(f"{Fore.BLUE} {text}{Style.RESET_ALL}")

def wait_for_api():
    """Attend que l'API soit disponible"""
    print_header("Waiting for API")
    
    max_attempts = 30
    for attempt in range(max_attempts):
        try:
            response = requests.get(f"{API_URL}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'healthy':
                    print_success(f"API is ready! (Attempt {attempt + 1}/{max_attempts})")
                    print_info(f"Model loaded: {data.get('model_loaded')}")
                    print_info(f"Features loaded: {data.get('features_loaded')}")
                    return True
        except:
            pass
        
        if attempt < max_attempts - 1:
            print(f" Waiting for API... ({attempt + 1}/{max_attempts})")
            time.sleep(2)
    
    print_error("API not ready after waiting")
    return False

def demo_health_check():
    """Démonstration du health check"""
    print_header("Health Check Demo")
    
    try:
        response = requests.get(f"{API_URL}/health", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            # Afficher les informations de santé
            status = data.get('status', 'unknown')
            status_color = Fore.GREEN if status == 'healthy' else Fore.RED
            
            print(f"{status_color}Status: {status}{Style.RESET_ALL}")
            print(f"Model loaded: {data.get('model_loaded')}")
            print(f"Features loaded: {data.get('features_loaded')}")
            print(f"Timestamp: {data.get('timestamp')}")
            print(f"Project root: {data.get('project_root')}")
            
            return True
        else:
            print_error(f"Health check failed: {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Health check error: {e}")
        return False

def demo_model_info():
    """Démonstration des informations du modèle"""
    print_header("Model Information Demo")
    
    try:
        response = requests.get(f"{API_URL}/model/info", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            print_success("Model information retrieved successfully")
            print(f"\n Model Details:")
            print(f"  Type: {data.get('model_type')}")
            print(f"  Version: {data.get('version')}")
            print(f"  Features count: {data.get('features_count')}")
            
            if 'pipeline_steps' in data:
                print(f"\n Pipeline Steps:")
                for step in data.get('pipeline_steps', []):
                    print(f"  - {step}")
            
            if 'classifier_type' in data:
                print(f"\n Classifier: {data.get('classifier_type')}")
                if 'classifier_params' in data:
                    print(f"  Parameters:")
                    for param, value in data.get('classifier_params', {}).items():
                        print(f"    {param}: {value}")
            
            return True
        else:
            print_error(f"Model info failed: {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Model info error: {e}")
        return False

def demo_single_predictions():
    """Démonstration des prédictions individuelles"""
    print_header("Single Predictions Demo")
    
    # Cas de test variés
    test_cases = [
        {
            "name": " High Risk - New customer with issues",
            "data": {
                "Age": 25,
                "Gender": "Male",
                "Tenure": 3,
                "Usage_Frequency": 2,
                "Support_Calls": 15,
                "Payment_Delay": 45,
                "Subscription_Type": "Basic",
                "Contract_Length": "Monthly",
                "Total_Spend": 150.0,
                "Last_Interaction": 45
            }
        },
        {
            "name": " Medium Risk - Regular customer",
            "data": {
                "Age": 35,
                "Gender": "Female",
                "Tenure": 24,
                "Usage_Frequency": 20,
                "Support_Calls": 3,
                "Payment_Delay": 10,
                "Subscription_Type": "Standard",
                "Contract_Length": "Quarterly",
                "Total_Spend": 1200.0,
                "Last_Interaction": 14
            }
        },
        {
            "name": " Low Risk - Loyal customer",
            "data": {
                "Age": 50,
                "Gender": "Male",
                "Tenure": 60,
                "Usage_Frequency": 40,
                "Support_Calls": 1,
                "Payment_Delay": 0,
                "Subscription_Type": "Premium",
                "Contract_Length": "Annual",
                "Total_Spend": 3000.0,
                "Last_Interaction": 3
            }
        }
    ]
    
    all_success = True
    
    for test_case in test_cases:
        print(f"\n {test_case['name']}")
        print(f"  Data: {json.dumps(test_case['data'], indent=2)}")
        
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json=test_case['data'],
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Déterminer la couleur en fonction du risque
                if result['risk_level'] == "HIGH":
                    risk_color = Fore.RED
                elif result['risk_level'] == "MEDIUM":
                    risk_color = Fore.YELLOW
                else:
                    risk_color = Fore.GREEN
                
                print(f"  {risk_color} Churn probability: {result['churn_probability']:.2%}{Style.RESET_ALL}")
                print(f"  {risk_color} Prediction: {'CHURN' if result['churn_prediction'] else 'NO CHURN'}{Style.RESET_ALL}")
                print(f"  {risk_color}  Risk level: {result['risk_level']}{Style.RESET_ALL}")
                print(f"   Confidence: {result['confidence']}")
                
                print(f"   Top 3 recommendations:")
                for i, rec in enumerate(result['recommendations'][:3], 1):
                    print(f"    {i}. {rec}")
                
                print_success("  Prediction successful!")
                
            else:
                print_error(f"  Prediction failed: {response.status_code}")
                print(f"  Error: {response.text[:100]}...")
                all_success = False
                
        except Exception as e:
            print_error(f"  Prediction error: {e}")
            all_success = False
    
    return all_success

def demo_batch_prediction():
    """Démonstration de la prédiction par lot"""
    print_header("Batch Prediction Demo")
    
    # Générer un lot de clients variés
    batch_size = 10
    batch_customers = []
    
    print_info(f"Generating batch of {batch_size} customers...")
    
    # Créer un mélange de clients
    for i in range(batch_size):
        if i < 3:  # 3 clients à haut risque
            customer = {
                "Age": np.random.randint(20, 40),
                "Gender": np.random.choice(["Male", "Female"]),
                "Tenure": np.random.randint(1, 6),
                "Usage_Frequency": np.random.randint(1, 10),
                "Support_Calls": np.random.randint(8, 20),
                "Payment_Delay": np.random.randint(30, 60),
                "Subscription_Type": "Basic",
                "Contract_Length": "Monthly",
                "Total_Spend": np.random.uniform(100, 500),
                "Last_Interaction": np.random.randint(30, 90)
            }
        elif i < 7:  # 4 clients à risque moyen
            customer = {
                "Age": np.random.randint(30, 50),
                "Gender": np.random.choice(["Male", "Female"]),
                "Tenure": np.random.randint(12, 36),
                "Usage_Frequency": np.random.randint(15, 30),
                "Support_Calls": np.random.randint(2, 6),
                "Payment_Delay": np.random.randint(5, 15),
                "Subscription_Type": np.random.choice(["Standard", "Premium"]),
                "Contract_Length": np.random.choice(["Quarterly", "Annual"]),
                "Total_Spend": np.random.uniform(800, 2000),
                "Last_Interaction": np.random.randint(7, 30)
            }
        else:  # 3 clients à faible risque
            customer = {
                "Age": np.random.randint(40, 65),
                "Gender": np.random.choice(["Male", "Female"]),
                "Tenure": np.random.randint(36, 84),
                "Usage_Frequency": np.random.randint(30, 60),
                "Support_Calls": np.random.randint(0, 3),
                "Payment_Delay": np.random.randint(0, 5),
                "Subscription_Type": "Premium",
                "Contract_Length": "Annual",
                "Total_Spend": np.random.uniform(2000, 5000),
                "Last_Interaction": np.random.randint(1, 7)
            }
        
        batch_customers.append(customer)
    
    try:
        print_info("Sending batch prediction request...")
        
        response = requests.post(
            f"{API_URL}/predict/batch",
            json=batch_customers,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            summary = result.get('summary', {})
            
            print_success("Batch prediction successful!")
            
            print(f"\n Batch Summary:")
            print(f"  Total customers: {summary.get('total_customers', 0)}")
            print(f"  Successful predictions: {summary.get('successful_predictions', 0)}")
            print(f"  Failed predictions: {summary.get('failed_predictions', 0)}")
            print(f"  Success rate: {summary.get('success_rate', 0):.1%}")
            print(f"  Average churn probability: {summary.get('average_churn_probability', 0):.2%}")
            
            print(f"\n Risk Distribution:")
            print(f"  {Fore.RED} High risk: {summary.get('high_risk_customers', 0)} customers{Style.RESET_ALL}")
            print(f"  {Fore.YELLOW}  Medium risk: {summary.get('medium_risk_customers', 0)} customers{Style.RESET_ALL}")
            print(f"  {Fore.GREEN} Low risk: {summary.get('low_risk_customers', 0)} customers{Style.RESET_ALL}")
            
            # Créer un DataFrame pour l'analyse
            successful_results = [r for r in result.get('results', []) if r.get('success', False)]
            
            if successful_results:
                df = pd.DataFrame(successful_results)
                
                print(f"\n Statistical Analysis:")
                print(f"  Min probability: {df['churn_probability'].min():.2%}")
                print(f"  Max probability: {df['churn_probability'].max():.2%}")
                print(f"  Median probability: {df['churn_probability'].median():.2%}")
                print(f"  Standard deviation: {df['churn_probability'].std():.2%}")
                
                # Analyse par genre
                if batch_customers:
                    gender_analysis = pd.DataFrame(batch_customers).merge(
                        df[['customer_index', 'churn_probability']], 
                        left_index=True, 
                        right_on='customer_index'
                    )
                    
                    if 'Gender' in gender_analysis.columns:
                        print(f"\n👥 Analysis by Gender:")
                        gender_stats = gender_analysis.groupby('Gender')['churn_probability'].mean()
                        for gender, prob in gender_stats.items():
                            print(f"  {gender}: {prob:.2%} average churn risk")
            
            print_info("Batch analysis complete!")
            return True
            
        else:
            print_error(f"Batch prediction failed: {response.status_code}")
            print(f"Error: {response.text[:200]}...")
            return False
            
    except Exception as e:
        print_error(f"Batch prediction error: {e}")
        return False

def demo_error_handling():
    """Démonstration de la gestion des erreurs"""
    print_header("Error Handling Demo")
    
    error_cases = [
        {
            "name": "Invalid age (too young)",
            "data": {
                "Age": 15,  # Trop jeune
                "Gender": "Male",
                "Tenure": 12,
                "Usage_Frequency": 20,
                "Support_Calls": 2,
                "Payment_Delay": 0,
                "Subscription_Type": "Standard",
                "Contract_Length": "Monthly",
                "Total_Spend": 500.0,
                "Last_Interaction": 7
            }
        },
        {
            "name": "Invalid subscription type",
            "data": {
                "Age": 30,
                "Gender": "Female",
                "Tenure": 24,
                "Usage_Frequency": 15,
                "Support_Calls": 3,
                "Payment_Delay": 5,
                "Subscription_Type": "InvalidType",  # Type invalide
                "Contract_Length": "Annual",
                "Total_Spend": 1000.0,
                "Last_Interaction": 14
            }
        },
        {
            "name": "Missing required field",
            "data": {
                "Age": 40,
                "Gender": "Male",
                # Missing Tenure field
                "Usage_Frequency": 25,
                "Support_Calls": 1,
                "Payment_Delay": 0,
                "Subscription_Type": "Premium",
                "Contract_Length": "Annual",
                "Total_Spend": 2000.0,
                "Last_Interaction": 3
            }
        }
    ]
    
    all_correct = True
    
    for error_case in error_cases:
        print(f"\n Testing: {error_case['name']}")
        
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json=error_case['data'],
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            # Nous nous attendons à une erreur 400 ou 422
            if response.status_code in [400, 422]:
                print_success(" Correctly rejected invalid data")
                print(f"  Status code: {response.status_code}")
                
                # Afficher le message d'erreur
                try:
                    error_detail = response.json()
                    if 'detail' in error_detail:
                        print(f"  Error: {error_detail['detail']}")
                except:
                    print(f"  Error: {response.text[:100]}...")
            else:
                print_error(f" Expected error, got {response.status_code}")
                all_correct = False
                
        except Exception as e:
            print_error(f" Request error: {e}")
            all_correct = False
    
    return all_correct

def demo_performance_test():
    """Test de performance de l'API"""
    print_header("Performance Test Demo")
    
    # Créer un client de test
    test_customer = {
        "Age": 35,
        "Gender": "Female",
        "Tenure": 24,
        "Usage_Frequency": 20,
        "Support_Calls": 3,
        "Payment_Delay": 5,
        "Subscription_Type": "Standard",
        "Contract_Length": "Annual",
        "Total_Spend": 1200.0,
        "Last_Interaction": 7
    }
    
    print_info("Testing API response time...")
    
    n_requests = 10
    response_times = []
    
    for i in range(n_requests):
        try:
            start_time = time.time()
            
            response = requests.post(
                f"{API_URL}/predict",
                json=test_customer,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            end_time = time.time()
            response_time = (end_time - start_time) * 1000  # en ms
            
            response_times.append(response_time)
            
            if response.status_code == 200:
                print(f"  Request {i+1}/{n_requests}: {response_time:.1f} ms")
            else:
                print_error(f"  Request {i+1}/{n_requests} failed: {response.status_code}")
                
        except Exception as e:
            print_error(f"  Request {i+1}/{n_requests} error: {e}")
    
    if response_times:
        print(f"\n Performance Summary:")
        print(f"  Total requests: {n_requests}")
        print(f"  Successful: {len(response_times)}")
        print(f"  Average response time: {np.mean(response_times):.1f} ms")
        print(f"  Min response time: {np.min(response_times):.1f} ms")
        print(f"  Max response time: {np.max(response_times):.1f} ms")
        print(f"  Standard deviation: {np.std(response_times):.1f} ms")
        
        if np.mean(response_times) < 100:
            print_success(" Performance: Excellent (< 100 ms)")
        elif np.mean(response_times) < 500:
            print_warning("  Performance: Good (< 500 ms)")
        else:
            print_warning("  Performance: Could be improved (> 500 ms)")
        
        return True
    else:
        print_error(" No successful requests")
        return False

def main():
    """Fonction principale"""
    
    print_header("ChurnGuard MLOps - Complete Demo")
    print("🚀 End-to-end demonstration of the customer churn prediction system")
    
    # Attendre que l'API soit prête
    if not wait_for_api():
        return
    
    # Exécuter les démonstrations
    demos = [
        ("Health Check", demo_health_check),
        ("Model Information", demo_model_info),
        ("Single Predictions", demo_single_predictions),
        ("Batch Prediction", demo_batch_prediction),
        ("Error Handling", demo_error_handling),
        ("Performance Test", demo_performance_test)
    ]
    
    results = []
    
    for demo_name, demo_func in demos:
        try:
            success = demo_func()
            results.append((demo_name, success))
            time.sleep(1)  # Petite pause entre les démos
        except Exception as e:
            print_error(f"Demo '{demo_name}' failed: {e}")
            results.append((demo_name, False))
    
    # Résumé final
    print_header("Demo Results Summary")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"\n Results: {passed}/{total} demos successful ({passed/total*100:.1f}%)")
    print("\n" + "="*70)
    
    for demo_name, success in results:
        status = f"{Fore.GREEN} PASS" if success else f"{Fore.RED} FAIL"
        print(f"{status}{Style.RESET_ALL} {demo_name}")
    
    print("\n" + "="*70)
    
    if passed == total:
        print_success(" All demonstrations completed successfully!")
        print("\n Next steps:")
        print("  1. Visit API documentation: http://localhost:8000/docs")
        print("  2. Check MLflow experiments: http://localhost:5000")
        print("  3. Integrate with your application using the API endpoints")
        print("  4. Monitor model performance and retrain as needed")
    else:
        print_warning(f" {total - passed} demo(s) failed")
        print("\n Troubleshooting tips:")
        print("  - Check that MLflow is running: http://localhost:5000")
        print("  - Verify the model is loaded: http://localhost:8000/health")
        print("  - Check API logs for errors")
        print("  - Ensure all dependencies are installed")
    
    print("\n" + "="*70)
    print("👋 Demo completed. Thank you for using ChurnGuard!")
    print("="*70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n Demo interrupted by user")
        sys.exit(0)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        sys.exit(1)