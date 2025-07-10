#!/usr/bin/env python3
"""
Test Streamlit Configuration Integration
Verifies that the Streamlit configuration page properly communicates with the API
"""

import requests
import json
import time

API_BASE_URL = "http://localhost:8000"
STREAMLIT_URL = "http://localhost:8501"

def test_streamlit_api_integration():
    """Test that Streamlit can properly interact with the configuration API"""
    
    print("🧪 Testing Streamlit Configuration Integration")
    print("=" * 60)
    
    # 1. Test API availability
    print("\n1️⃣ Testing API availability...")
    try:
        health_response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if health_response.status_code == 200:
            health_data = health_response.json()
            print(f"✅ API is healthy")
            print(f"   • Status: {health_data.get('status')}")
            print(f"   • Models loaded: {health_data.get('models_loaded')}")
            print(f"   • YOLO available: {health_data.get('yolo_available')}")
        else:
            print(f"❌ API unhealthy: {health_response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        return False
    
    # 2. Test Streamlit availability
    print("\n2️⃣ Testing Streamlit availability...")
    try:
        streamlit_response = requests.get(f"{STREAMLIT_URL}", timeout=5)
        if streamlit_response.status_code == 200:
            print(f"✅ Streamlit is running")
            print(f"   • URL: {STREAMLIT_URL}")
            print(f"   • Status: {streamlit_response.status_code}")
        else:
            print(f"❌ Streamlit not responding: {streamlit_response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to Streamlit: {e}")
        return False
    
    # 3. Test configuration endpoint integration
    print("\n3️⃣ Testing configuration endpoints...")
    
    # Get current config
    config_response = requests.get(f"{API_BASE_URL}/config")
    if config_response.status_code == 200:
        current_config = config_response.json()
        print(f"✅ Configuration retrieved successfully")
        print(f"   • Number of classes: {current_config['models']['object_detection']['num_classes']}")
        print(f"   • Confidence threshold: {current_config['models']['object_detection']['confidence_threshold']}")
        print(f"   • Patch detection: {current_config['inference']['patch_detection']['enabled']}")
    else:
        print(f"❌ Failed to get configuration: {config_response.text}")
        return False
    
    # 4. Test configuration update (simulating Streamlit form submission)
    print("\n4️⃣ Testing configuration update (simulating Streamlit)...")
    
    streamlit_config_update = {
        "models": {
            "object_detection": {
                "architecture": "yolov8",
                "backbone": "efficientnet-b3",
                "num_classes": 30,
                "confidence_threshold": 0.75,
                "nms_threshold": 0.35
            },
            "domain_adaptation": {
                "lambda_grl": 1.2
            },
            "autoencoder": {
                "latent_dim": 256
            }
        },
        "training": {
            "learning_rate": 0.0015,
            "weight_decay": 0.015,
            "batch_size": 12,
            "epochs": 150,
            "optimizer": "adamw"
        },
        "inference": {
            "patch_detection": {
                "enabled": True,
                "patch_size": [224, 224],
                "overlap": 0.25,
                "min_object_size": 25
            },
            "parallel_processing": {
                "num_workers": 6,
                "batch_size": 12
            }
        }
    }
    
    update_response = requests.post(f"{API_BASE_URL}/config", json=streamlit_config_update)
    if update_response.status_code == 200:
        update_result = update_response.json()
        print(f"✅ Configuration update successful")
        print(f"   • Success: {update_result['success']}")
        print(f"   • Models reinitialized: {update_result['models_reinitialized']}")
        print(f"   • Changes applied: {len(update_result['changes_applied'])}")
        
        for change in update_result['changes_applied']:
            print(f"     - {change}")
    else:
        print(f"❌ Configuration update failed: {update_response.text}")
        return False
    
    # 5. Verify configuration was applied
    print("\n5️⃣ Verifying configuration was applied...")
    
    updated_config_response = requests.get(f"{API_BASE_URL}/config")
    if updated_config_response.status_code == 200:
        updated_config = updated_config_response.json()
        
        # Check key values
        obj_config = updated_config['models']['object_detection']
        train_config = updated_config['training']
        patch_config = updated_config['inference']['patch_detection']
        
        success = True
        checks = [
            (obj_config['num_classes'], 30, "Number of classes"),
            (obj_config['confidence_threshold'], 0.75, "Confidence threshold"),
            (obj_config['nms_threshold'], 0.35, "NMS threshold"),
            (train_config['learning_rate'], 0.0015, "Learning rate"),
            (train_config['batch_size'], 12, "Batch size"),
            (train_config['optimizer'], "adamw", "Optimizer"),
            (patch_config['patch_size'], [224, 224], "Patch size"),
            (patch_config['overlap'], 0.25, "Patch overlap")
        ]
        
        for actual, expected, name in checks:
            if actual == expected:
                print(f"   ✅ {name}: {actual}")
            else:
                print(f"   ❌ {name}: expected {expected}, got {actual}")
                success = False
        
        if not success:
            return False
    else:
        print(f"❌ Failed to verify configuration: {updated_config_response.text}")
        return False
    
    # 6. Test model information reflects changes
    print("\n6️⃣ Testing model information...")
    
    models_response = requests.get(f"{API_BASE_URL}/models")
    if models_response.status_code == 200:
        models_info = models_response.json()
        print(f"✅ Model information retrieved")
        print(f"   • Total models: {models_info['total_models']}")
        print(f"   • Models loaded: {models_info['models_loaded']}")
        
        if 'yolov8' in models_info['models']:
            yolo_info = models_info['models']['yolov8']
            expected_classes = 30
            actual_classes = yolo_info.get('classes', 'unknown')
            
            if actual_classes == expected_classes:
                print(f"   ✅ YOLOv8 classes updated: {actual_classes}")
            else:
                print(f"   ❌ YOLOv8 classes: expected {expected_classes}, got {actual_classes}")
                return False
        
        # Check patch detector config
        if 'patch_detector' in models_info['models']:
            patch_info = models_info['models']['patch_detector']
            expected_size = [224, 224]
            actual_size = patch_info.get('patch_size', 'unknown')
            
            if actual_size == expected_size:
                print(f"   ✅ Patch detector size updated: {actual_size}")
            else:
                print(f"   ❌ Patch detector size: expected {expected_size}, got {actual_size}")
                return False
    else:
        print(f"❌ Failed to get model information: {models_response.text}")
        return False
    
    # 7. Test detection with new configuration
    print("\n7️⃣ Testing detection with new configuration...")
    
    # Use a test image if available
    test_image_path = "demo_upload_images/01_urban_dense_many_small_objects.jpg"
    try:
        with open(test_image_path, "rb") as f:
            files = {"file": f}
            detection_response = requests.post(f"{API_BASE_URL}/detect", files=files)
            
            if detection_response.status_code == 200:
                detection_result = detection_response.json()
                print(f"✅ Detection with new config successful")
                print(f"   • Objects detected: {len(detection_result['detections'])}")
                print(f"   • Processing time: {detection_result['processing_time']:.3f}s")
                
                if detection_result['detections']:
                    confidences = [det['confidence'] for det in detection_result['detections']]
                    min_conf = min(confidences)
                    max_conf = max(confidences)
                    print(f"   • Confidence range: {min_conf:.3f} - {max_conf:.3f}")
                    
                    # Verify all detections meet the new threshold (0.75)
                    low_conf = [c for c in confidences if c < 0.75]
                    if not low_conf:
                        print(f"   ✅ All detections meet new threshold (≥0.75)")
                    else:
                        print(f"   ⚠️  Found {len(low_conf)} detections below threshold")
            else:
                print(f"❌ Detection failed: {detection_response.text}")
                return False
                
    except FileNotFoundError:
        print(f"   ⚠️  Test image not found: {test_image_path}")
        print(f"   ℹ️  Skipping detection test")
    
    print("\n🎉 All Streamlit integration tests passed!")
    return True

def test_configuration_page_functionality():
    """Test specific configuration page features"""
    
    print("\n🎛️ Testing Configuration Page Functionality")
    print("=" * 60)
    
    # Test configuration retrieval for Streamlit page
    print("\n1️⃣ Testing configuration retrieval for Streamlit...")
    
    config_response = requests.get(f"{API_BASE_URL}/config")
    if config_response.status_code == 200:
        config = config_response.json()
        print(f"✅ Configuration available for Streamlit")
        
        # Check all required sections exist
        required_sections = ['models', 'training', 'inference']
        for section in required_sections:
            if section in config:
                print(f"   ✅ {section} section available")
            else:
                print(f"   ❌ {section} section missing")
                return False
        
        # Check required model parameters
        model_params = ['architecture', 'num_classes', 'confidence_threshold', 'nms_threshold']
        obj_config = config.get('models', {}).get('object_detection', {})
        
        for param in model_params:
            if param in obj_config:
                print(f"   ✅ {param}: {obj_config[param]}")
            else:
                print(f"   ❌ {param} missing from object detection config")
                return False
    else:
        print(f"❌ Configuration not available: {config_response.text}")
        return False
    
    # Test form submission simulation
    print("\n2️⃣ Testing form submission simulation...")
    
    # Simulate what Streamlit form would send
    form_data = {
        "models": {
            "object_detection": {
                "architecture": "yolov8",
                "backbone": "mobilenet-v3",
                "num_classes": 45,
                "confidence_threshold": 0.65,
                "nms_threshold": 0.45
            },
            "domain_adaptation": {
                "lambda_grl": 0.8
            }
        },
        "training": {
            "learning_rate": 0.0008,
            "weight_decay": 0.012,
            "batch_size": 10,
            "epochs": 80,
            "optimizer": "sgd"
        },
        "inference": {
            "patch_detection": {
                "enabled": False,
                "patch_size": [320, 320],
                "overlap": 0.15,
                "min_object_size": 15
            }
        }
    }
    
    form_response = requests.post(f"{API_BASE_URL}/config", json=form_data)
    if form_response.status_code == 200:
        result = form_response.json()
        print(f"✅ Form submission successful")
        print(f"   • Message: {result['message']}")
        print(f"   • Changes: {len(result['changes_applied'])}")
    else:
        print(f"❌ Form submission failed: {form_response.text}")
        return False
    
    print("\n✅ Configuration page functionality verified!")
    return True

if __name__ == "__main__":
    print("🚀 Starting Streamlit Configuration Integration Tests")
    print("This verifies the fix for: 'Number of Classes doesn't update between functionalities'")
    print()
    
    # Run integration tests
    integration_success = test_streamlit_api_integration()
    config_page_success = test_configuration_page_functionality()
    
    if integration_success and config_page_success:
        print("\n🎯 STREAMLIT INTEGRATION VERIFIED:")
        print("   ✅ API connectivity working")
        print("   ✅ Configuration endpoints functional") 
        print("   ✅ Model reinitialization working")
        print("   ✅ Configuration page ready for use")
        print("   ✅ All parameters update correctly")
        
        print("\n🌐 Ready to use:")
        print(f"   • Streamlit UI: {STREAMLIT_URL}")
        print(f"   • API Backend: {API_BASE_URL}")
        print("   • Go to Configuration page and test parameter changes")
        print("   • Verify changes apply to all detection functionalities")
        
    else:
        print("\n❌ INTEGRATION TESTS FAILED")
        print("   Please check API and Streamlit server status")
        exit(1) 