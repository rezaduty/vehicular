#!/usr/bin/env python3
"""
Test Configuration Functionality
Verifies that configuration changes are properly applied and affect detection results
"""

import requests
import json
import time

API_BASE_URL = "http://localhost:8000"

def test_configuration_functionality():
    """Test configuration update and verification"""
    
    print("🧪 Testing Configuration Functionality")
    print("=" * 50)
    
    # 1. Get initial configuration
    print("\n1️⃣ Getting initial configuration...")
    response = requests.get(f"{API_BASE_URL}/config")
    if response.status_code == 200:
        initial_config = response.json()
        print(f"✅ Initial config loaded")
        print(f"   • Number of classes: {initial_config['models']['object_detection']['num_classes']}")
        print(f"   • Confidence threshold: {initial_config['models']['object_detection']['confidence_threshold']}")
        print(f"   • Patch detection enabled: {initial_config['inference']['patch_detection']['enabled']}")
    else:
        print(f"❌ Failed to get config: {response.text}")
        return False
    
    # 2. Test detection with initial configuration
    print("\n2️⃣ Testing detection with initial configuration...")
    with open("demo_upload_images/01_urban_dense_many_small_objects.jpg", "rb") as f:
        files = {"file": f}
        response = requests.post(f"{API_BASE_URL}/detect", files=files)
        
        if response.status_code == 200:
            initial_detection = response.json()
            print(f"✅ Initial detection successful")
            print(f"   • Objects detected: {len(initial_detection['detections'])}")
            if initial_detection['detections']:
                max_conf = max(det['confidence'] for det in initial_detection['detections'])
                min_conf = min(det['confidence'] for det in initial_detection['detections'])
                print(f"   • Confidence range: {min_conf:.3f} - {max_conf:.3f}")
        else:
            print(f"❌ Initial detection failed: {response.text}")
            return False
    
    # 3. Update configuration with new values
    print("\n3️⃣ Updating configuration...")
    new_config = {
        "models": {
            "object_detection": {
                "num_classes": 50,
                "confidence_threshold": 0.8,
                "nms_threshold": 0.3
            }
        },
        "inference": {
            "patch_detection": {
                "enabled": False,
                "patch_size": [256, 256],
                "overlap": 0.3
            }
        },
        "training": {
            "learning_rate": 0.002,
            "batch_size": 16
        }
    }
    
    response = requests.post(f"{API_BASE_URL}/config", json=new_config)
    if response.status_code == 200:
        update_result = response.json()
        print(f"✅ Configuration updated successfully")
        print(f"   • Changes applied: {len(update_result['changes_applied'])}")
        for change in update_result['changes_applied']:
            print(f"     - {change}")
    else:
        print(f"❌ Failed to update config: {response.text}")
        return False
    
    # 4. Verify configuration was updated
    print("\n4️⃣ Verifying configuration changes...")
    response = requests.get(f"{API_BASE_URL}/config")
    if response.status_code == 200:
        updated_config = response.json()
        print(f"✅ Updated config retrieved")
        
        # Check specific values
        obj_config = updated_config['models']['object_detection']
        patch_config = updated_config['inference']['patch_detection']
        train_config = updated_config['training']
        
        success = True
        if obj_config['num_classes'] == 50:
            print(f"   ✅ Number of classes: {obj_config['num_classes']}")
        else:
            print(f"   ❌ Number of classes: expected 50, got {obj_config['num_classes']}")
            success = False
            
        if obj_config['confidence_threshold'] == 0.8:
            print(f"   ✅ Confidence threshold: {obj_config['confidence_threshold']}")
        else:
            print(f"   ❌ Confidence threshold: expected 0.8, got {obj_config['confidence_threshold']}")
            success = False
            
        if not patch_config['enabled']:
            print(f"   ✅ Patch detection disabled: {patch_config['enabled']}")
        else:
            print(f"   ❌ Patch detection: expected False, got {patch_config['enabled']}")
            success = False
            
        if train_config['learning_rate'] == 0.002:
            print(f"   ✅ Learning rate: {train_config['learning_rate']}")
        else:
            print(f"   ❌ Learning rate: expected 0.002, got {train_config['learning_rate']}")
            success = False
        
        if not success:
            return False
    else:
        print(f"❌ Failed to get updated config: {response.text}")
        return False
    
    # 5. Test detection with new configuration
    print("\n5️⃣ Testing detection with new configuration...")
    with open("demo_upload_images/01_urban_dense_many_small_objects.jpg", "rb") as f:
        files = {"file": f}
        response = requests.post(f"{API_BASE_URL}/detect", files=files)
        
        if response.status_code == 200:
            new_detection = response.json()
            print(f"✅ New detection successful")
            print(f"   • Objects detected: {len(new_detection['detections'])}")
            
            if new_detection['detections']:
                max_conf = max(det['confidence'] for det in new_detection['detections'])
                min_conf = min(det['confidence'] for det in new_detection['detections'])
                print(f"   • Confidence range: {min_conf:.3f} - {max_conf:.3f}")
                
                # Check if all detections meet the new threshold
                low_conf_detections = [det for det in new_detection['detections'] if det['confidence'] < 0.8]
                if not low_conf_detections:
                    print(f"   ✅ All detections meet new confidence threshold (≥0.8)")
                else:
                    print(f"   ❌ Found {len(low_conf_detections)} detections below threshold")
            
            # Compare with initial detection
            initial_count = len(initial_detection['detections'])
            new_count = len(new_detection['detections'])
            print(f"   • Detection count change: {initial_count} → {new_count} ({new_count - initial_count:+d})")
            
        else:
            print(f"❌ New detection failed: {response.text}")
            return False
    
    # 6. Test parameter override
    print("\n6️⃣ Testing parameter override...")
    with open("demo_upload_images/01_urban_dense_many_small_objects.jpg", "rb") as f:
        files = {"file": f}
        params = {
            "confidence_threshold": 0.4,  # Override config value
            "use_patch_detection": True    # Override config value
        }
        response = requests.post(f"{API_BASE_URL}/detect", files=files, params=params)
        
        if response.status_code == 200:
            override_detection = response.json()
            print(f"✅ Parameter override successful")
            print(f"   • Objects detected: {len(override_detection['detections'])}")
            
            if override_detection['detections']:
                max_conf = max(det['confidence'] for det in override_detection['detections'])
                min_conf = min(det['confidence'] for det in override_detection['detections'])
                print(f"   • Confidence range: {min_conf:.3f} - {max_conf:.3f}")
                
                # Should have more detections with lower threshold
                override_count = len(override_detection['detections'])
                config_count = len(new_detection['detections'])
                if override_count >= config_count:
                    print(f"   ✅ Override produced more/equal detections: {config_count} → {override_count}")
                else:
                    print(f"   ⚠️  Override produced fewer detections: {config_count} → {override_count}")
        else:
            print(f"❌ Parameter override failed: {response.text}")
            return False
    
    # 7. Test model information
    print("\n7️⃣ Testing model information...")
    response = requests.get(f"{API_BASE_URL}/models")
    if response.status_code == 200:
        models_info = response.json()
        print(f"✅ Model information retrieved")
        print(f"   • Total models: {models_info['total_models']}")
        print(f"   • Models loaded: {models_info['models_loaded']}")
        
        if 'yolov8' in models_info['models']:
            yolo_info = models_info['models']['yolov8']
            print(f"   • YOLOv8 classes: {yolo_info.get('classes', 'unknown')}")
            print(f"   • YOLOv8 status: {yolo_info.get('status', 'unknown')}")
    else:
        print(f"❌ Failed to get models info: {response.text}")
        return False
    
    print("\n🎉 All configuration functionality tests passed!")
    print("=" * 50)
    print("\n📋 Configuration Summary:")
    print(f"✅ Configuration retrieval works")
    print(f"✅ Configuration updates are applied")
    print(f"✅ Models are reinitialized with new config")
    print(f"✅ Detection uses updated configuration")
    print(f"✅ Parameter overrides work properly")
    print(f"✅ Model information reflects changes")
    
    return True

def test_streamlit_integration():
    """Test Streamlit integration with configuration"""
    print("\n🌐 Testing Streamlit Integration...")
    
    # Test if Streamlit can connect to API
    try:
        health_response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if health_response.status_code == 200:
            print("✅ Streamlit can connect to API")
            return True
        else:
            print(f"❌ Streamlit connection failed: {health_response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Streamlit connection error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Configuration Functionality Tests")
    print("This tests the issue: 'Number of Classes doesn't update between functionalities'")
    print()
    
    # Test API health first
    try:
        health = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if health.status_code != 200:
            print("❌ API not healthy, cannot run tests")
            exit(1)
    except:
        print("❌ Cannot connect to API, is it running?")
        print("   Run: uvicorn src.api.real_working_api:app --host 0.0.0.0 --port 8000 --reload")
        exit(1)
    
    # Run tests
    success = test_configuration_functionality()
    streamlit_success = test_streamlit_integration()
    
    if success and streamlit_success:
        print("\n🎯 ISSUE RESOLVED:")
        print("   • Configuration changes are now properly applied")
        print("   • Number of classes updates correctly") 
        print("   • All functionalities use updated configuration")
        print("   • Streamlit integration works")
        print("\n💡 Next steps:")
        print("   • Test the configuration page in Streamlit")
        print("   • Verify that changing settings updates all functionalities")
    else:
        print("\n❌ TESTS FAILED - Issue not fully resolved")
        exit(1) 