import requests
import json
import time

API_URL = "http://localhost:8000"

def print_section(title):
    """Affiche une section avec titre"""
    print(f"\n{'='*60}")
    print(f"🧪 {title}")
    print('='*60)

def test_health():
    """Test du endpoint health"""
    print_section("Testing Health Endpoint")
    try:
        response = requests.get(f"{API_URL}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Status: {data.get('status')}")
            print(f"✅ Model loaded: {data.get('model_loaded')}")
            print(f"✅ Features loaded: {data.get('features_loaded')}")
            print(f"✅ Timestamp: {data.get('timestamp')}")
            return True
        else:
            print(f"❌ Failed with status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_root():
    """Test du endpoint racine"""
    print_section("Testing Root Endpoint")
    try:
        response = requests.get(f"{API_URL}", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Service: {data.get('service')}")
            print(f"✅ Version: {data.get('version')}")
            print(f"✅ Status: {data.get('status')}")
            print(f"✅ Model: {data.get('model')}")
            print(f"\n📊 Performance:")
            perf = data.get('performance', {})
            for key, value in perf.items():
                print(f"  {key}: {value}")
            print(f"\n🔗 Endpoints:")
            for endpoint, path in data.get('endpoints', {}).items():
                print(f"  {endpoint}: {path}")
            return True
        else:
            print(f"❌ Failed with status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_model_info():
    """Test du endpoint model info"""
    print_section("Testing Model Info")
    try:
        response = requests.get(f"{API_URL}/model/info", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Model type: {data.get('model_type')}")
            print(f"✅ Version: {data.get('version')}")
            print(f"✅ Features count: {data.get('features_count')}")
            print(f"\n📋 Pipeline steps: {data.get('pipeline_steps', [])}")
            print(f"\n📈 Performance:")
            perf = data.get('performance', {})
            for key, value in perf.items():
                print(f"  {key}: {value}")
            return True
        else:
            print(f"❌ Failed with status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_single_prediction():
    """Test de prédiction unique"""
    print_section("Testing Single Prediction")
    
    # Cas 1: Client à haut risque
    high_risk_customer = {
        "Age": 42,
        "Gender": "Female",
        "Tenure": 3,
        "Usage_Frequency": 5,
        "Support_Calls": 12,
        "Payment_Delay": 60,
        "Subscription_Type": "Basic",
        "Contract_Length": "Monthly",
        "Total_Spend": 299.99,
        "Last_Interaction": 90
    }
    
    # Cas 2: Client à bas risque
    low_risk_customer = {
        "Age": 55,
        "Gender": "Male",
        "Tenure": 60,
        "Usage_Frequency": 40,
        "Support_Calls": 1,
        "Payment_Delay": 0,
        "Subscription_Type": "Premium",
        "Contract_Length": "Annual",
        "Total_Spend": 3000.00,
        "Last_Interaction": 7
    }
    
    test_cases = [
        ("High Risk Customer", high_risk_customer),
        ("Low Risk Customer", low_risk_customer)
    ]
    
    all_success = True
    
    for case_name, customer_data in test_cases:
        print(f"\n📋 Testing: {case_name}")
        print(f"  Data: {json.dumps(customer_data, indent=2)}")
        
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json=customer_data,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"  ✅ Prediction successful!")
                print(f"  📊 Churn probability: {result['churn_probability']:.2%}")
                print(f"  🎯 Prediction: {'CHURN' if result['churn_prediction'] else 'NO CHURN'}")
                print(f"  ⚠️  Risk level: {result['risk_level']}")
                print(f"  💡 Confidence: {result['confidence']}")
                print(f"  📋 Recommendations:")
                for i, rec in enumerate(result['recommendations'][:3], 1):
                    print(f"    {i}. {rec}")
                if len(result['recommendations']) > 3:
                    print(f"    ... and {len(result['recommendations']) - 3} more")
            else:
                print(f"  ❌ Failed with status: {response.status_code}")
                print(f"  Error: {response.text[:200]}...")
                all_success = False
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
            all_success = False
    
    return all_success

def test_batch_prediction():
    """Test de prédiction par lot"""
    print_section("Testing Batch Prediction")
    
    batch_customers = [
        {
            "Age": 28,
            "Gender": "Male",
            "Tenure": 6,  # Corrigé: était 6 (OK)
            "Usage_Frequency": 5,
            "Support_Calls": 8,
            "Payment_Delay": 60,
            "Subscription_Type": "Basic",
            "Contract_Length": "Monthly",
            "Total_Spend": 299.99,
            "Last_Interaction": 90
        },
        {
            "Age": 55,
            "Gender": "Female",
            "Tenure": 48,  # Corrigé: était 48 (OK)
            "Usage_Frequency": 30,
            "Support_Calls": 1,
            "Payment_Delay": 0,
            "Subscription_Type": "Premium",
            "Contract_Length": "Annual",
            "Total_Spend": 2500.00,
            "Last_Interaction": 7
        },
        {
            "Age": 38,
            "Gender": "Male",
            "Tenure": 18,  # Corrigé: était 18 (OK)
            "Usage_Frequency": 20,
            "Support_Calls": 4,
            "Payment_Delay": 15,
            "Subscription_Type": "Standard",
            "Contract_Length": "Quarterly",
            "Total_Spend": 800.00,
            "Last_Interaction": 21
        },
        {
            "Age": 65,
            "Gender": "Female",
            "Tenure": 84,  # Corrigé: était 120 (trop grand) -> 84
            "Usage_Frequency": 50,
            "Support_Calls": 0,
            "Payment_Delay": 0,
            "Subscription_Type": "Premium",
            "Contract_Length": "Annual",
            "Total_Spend": 5000.00,
            "Last_Interaction": 3
        }
    ]
    
    try:
        response = requests.post(
            f"{API_URL}/predict/batch",
            json=batch_customers,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            summary = result.get('summary', {})
            
            print(f"✅ Batch prediction successful!")
            print(f"\n📊 Summary Statistics:")
            print(f"  Total customers: {summary.get('total_customers', 0)}")
            print(f"  Successful predictions: {summary.get('successful_predictions', 0)}")
            print(f"  Failed predictions: {summary.get('failed_predictions', 0)}")
            print(f"  Success rate: {summary.get('success_rate', 0):.1%}")
            print(f"  Average churn probability: {summary.get('average_churn_probability', 0):.2%}")
            print(f"  High risk customers: {summary.get('high_risk_customers', 0)}")
            print(f"  Medium risk customers: {summary.get('medium_risk_calls', 0)}")
            print(f"  Low risk customers: {summary.get('low_risk_customers', 0)}")
            
            print(f"\n📋 Individual Results:")
            for res in result.get('results', []):
                if res.get('success'):
                    status = "✅" if res['churn_probability'] < 0.4 else "⚠️" if res['churn_probability'] < 0.7 else "🚨"
                    print(f"  {status} Customer {res['customer_index']}: {res['churn_probability']:.2%} ({res['risk_level']})")
                else:
                    print(f"  ❌ Customer {res['customer_index']}: Error: {res.get('error', 'Unknown')}")
            
            return True
        else:
            print(f"❌ Failed with status: {response.status_code}")
            print(f"Error: {response.text[:200]}...")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_error_cases():
    """Test des cas d'erreur"""
    print_section("Testing Error Cases")
    
    # Test 1: Données invalides
    print("\n1. Testing invalid data...")
    invalid_customer = {
        "Age": 150,  # Trop âgé
        "Gender": "Invalid",
        "Tenure": -10,
        "Usage_Frequency": 200,
        "Support_Calls": 100,
        "Payment_Delay": -5,
        "Subscription_Type": "Invalid",
        "Contract_Length": "Invalid",
        "Total_Spend": -100,
        "Last_Interaction": 400
    }
    
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=invalid_customer,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 422:  # Validation error
            print("✅ Correctly rejected invalid data")
        else:
            print(f"❌ Expected 422, got {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 2: Requête vide
    print("\n2. Testing empty request...")
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={},
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 422:
            print("✅ Correctly rejected empty request")
        else:
            print(f"❌ Expected 422, got {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    return True

def run_all_tests():
    """Exécute tous les tests"""
    print("🚀 Starting ChurnGuard API Tests...")
    print("="*60)
    
    test_results = []
    
    # Exécuter les tests
    test_results.append(("Health Check", test_health()))
    time.sleep(1)
    
    test_results.append(("Root Endpoint", test_root()))
    time.sleep(1)
    
    test_results.append(("Model Info", test_model_info()))
    time.sleep(1)
    
    test_results.append(("Single Prediction", test_single_prediction()))
    time.sleep(1)
    
    test_results.append(("Batch Prediction", test_batch_prediction()))
    time.sleep(1)
    
    test_results.append(("Error Cases", test_error_cases()))
    
    # Résumé
    print_section("Test Results Summary")
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    print(f"\n📊 Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print("\n" + "="*60)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print("\n" + "="*60)
    
    if passed == total:
        print("🎉 All tests passed successfully!")
        print("\n🔗 API is ready at: http://localhost:8000")
        print("📚 Documentation: http://localhost:8000/docs")
        print("📊 MLflow UI: http://localhost:5000")
    else:
        print(f"⚠️ {total - passed} test(s) failed")
    
    return passed == total

if __name__ == "__main__":
    # Attendre que l'API démarre
    print("⏳ Waiting for API to start...")
    time.sleep(3)
    
    # Exécuter les tests
    success = run_all_tests()
    
    # Code de sortie pour les pipelines CI/CD
    exit(0 if success else 1)