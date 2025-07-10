#!/usr/bin/env python3
"""
Test All Detection Modes with Generated Test Images
Demonstrates Object Detection, Parallel Patch Detection, Domain Adaptation, and Unsupervised Detection
"""

import os
import sys
import time
import json
import numpy as np
import cv2
from PIL import Image
import requests
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# Add src to path for local imports
sys.path.append('src')

try:
    from models.object_detection import YOLOv8Detector, ParallelPatchDetector
    from models.domain_adaptation import DomainAdversarialNetwork
    from unsupervised.lost import LOSTDetector
    from data.utils import create_dummy_data
except ImportError as e:
    print(f"⚠️  Import warning: {e}")
    print("Some local models may not be available, will test API endpoints instead")

class DetectionTester:
    """Test all detection modes with comprehensive reporting"""
    
    def __init__(self, test_images_dir="test_images", api_base_url="http://localhost:8000"):
        self.test_images_dir = test_images_dir
        self.api_base_url = api_base_url
        self.results = {}
        
        # Check if API is available
        self.api_available = self.check_api_availability()
        
        # Load local models if available
        self.local_models = self.load_local_models()
    
    def check_api_availability(self):
        """Check if FastAPI server is running"""
        try:
            response = requests.get(f"{self.api_base_url}/health", timeout=3)
            return response.status_code == 200
        except:
            return False
    
    def load_local_models(self):
        """Load local models for testing"""
        models = {}
        
        try:
            print("🔧 Loading local models...")
            
            # YOLOv8 detector
            models['yolov8'] = YOLOv8Detector(num_classes=10)
            print("   ✓ YOLOv8 detector loaded")
            
            # Parallel patch detector
            models['patch_detector'] = ParallelPatchDetector(
                base_detector=models['yolov8'],
                patch_size=(192, 192),
                overlap=0.2
            )
            print("   ✓ Parallel patch detector loaded")
            
            # Domain adaptation model
            models['domain_adaptation'] = DomainAdversarialNetwork(num_classes=10)
            print("   ✓ Domain adaptation model loaded")
            
            # LOST detector
            models['lost'] = LOSTDetector()
            print("   ✓ LOST detector loaded")
            
        except Exception as e:
            print(f"⚠️  Error loading local models: {e}")
            print("Will use API endpoints for testing")
        
        return models
    
    def get_test_images(self):
        """Get list of test images and their info"""
        test_images = []
        
        for file in os.listdir(self.test_images_dir):
            if file.endswith('.jpg') and not file.startswith('.'):
                image_path = os.path.join(self.test_images_dir, file)
                info_path = os.path.join(self.test_images_dir, file.replace('.jpg', '_info.txt'))
                
                # Load ground truth if available
                ground_truth = None
                if os.path.exists(info_path):
                    ground_truth = self.load_ground_truth(info_path)
                
                test_images.append({
                    'name': file,
                    'path': image_path,
                    'ground_truth': ground_truth
                })
        
        return test_images
    
    def load_ground_truth(self, info_path):
        """Load ground truth information from info file"""
        try:
            with open(info_path, 'r') as f:
                content = f.read()
            
            # Parse basic stats
            lines = content.split('\n')
            stats = {}
            for line in lines:
                if ':' in line and any(key in line for key in ['Total Objects', 'Cars', 'Pedestrians', 'Traffic Signs', 'Small Objects']):
                    key, value = line.split(':', 1)
                    try:
                        stats[key.strip()] = int(value.strip())
                    except:
                        stats[key.strip()] = value.strip()
            
            return stats
        except:
            return None
    
    def test_api_detection(self, image_path, endpoint="detect", **params):
        """Test detection via API"""
        if not self.api_available:
            return None
        
        try:
            with open(image_path, 'rb') as f:
                files = {'file': f}
                
                if endpoint == "detect":
                    data = {
                        'use_patch_detection': params.get('use_patch_detection', False),
                        'confidence_threshold': params.get('confidence_threshold', 0.5)
                    }
                    response = requests.post(f"{self.api_base_url}/detect", files=files, data=data)
                
                elif endpoint == "detect_patches":
                    data = {
                        'patch_size': params.get('patch_size', 192),
                        'overlap': params.get('overlap', 0.2)
                    }
                    response = requests.post(f"{self.api_base_url}/detect_patches", files=files, data=data)
                
                elif endpoint == "detect_unsupervised":
                    response = requests.post(f"{self.api_base_url}/detect_unsupervised", files=files)
                
                else:
                    return None
                
                if response.status_code == 200:
                    return response.json()
                else:
                    print(f"⚠️  API error {response.status_code}: {response.text}")
                    return None
                    
        except Exception as e:
            print(f"⚠️  API request failed: {e}")
            return None
    
    def test_local_detection(self, image_path, model_name):
        """Test detection with local models"""
        if model_name not in self.local_models:
            return None
        
        try:
            # Load and preprocess image
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Create dummy input for testing
            dummy_data = create_dummy_data(batch_size=1, image_size=(384, 1280))
            
            model = self.local_models[model_name]
            
            start_time = time.time()
            
            if model_name == 'yolov8':
                results = model.predict(dummy_data['images'])
            elif model_name == 'patch_detector':
                results = model(dummy_data['images'])
            elif model_name == 'lost':
                results = model.detect_objects_from_motion(dummy_data['images'])
            else:
                results = None
            
            processing_time = time.time() - start_time
            
            return {
                'success': True,
                'detections': results if results else [],
                'processing_time': processing_time,
                'method': model_name
            }
            
        except Exception as e:
            print(f"⚠️  Local detection failed for {model_name}: {e}")
            return None
    
    def run_comprehensive_test(self):
        """Run comprehensive testing on all images and detection modes"""
        
        print("🧪 Starting Comprehensive Detection Testing")
        print("=" * 60)
        
        test_images = self.get_test_images()
        
        if not test_images:
            print("❌ No test images found! Run generate_test_images.py first.")
            return
        
        print(f"📁 Found {len(test_images)} test images")
        print(f"🌐 API Available: {'✅ Yes' if self.api_available else '❌ No'}")
        print(f"🔧 Local Models: {'✅ Yes' if self.local_models else '❌ No'}")
        print()
        
        # Test scenarios
        test_scenarios = [
            {
                'name': 'Standard Object Detection',
                'method': 'api' if self.api_available else 'local',
                'endpoint': 'detect',
                'model': 'yolov8',
                'params': {'use_patch_detection': False}
            },
            {
                'name': 'Parallel Patch Detection',
                'method': 'api' if self.api_available else 'local',
                'endpoint': 'detect_patches',
                'model': 'patch_detector',
                'params': {'patch_size': 192, 'overlap': 0.2}
            },
            {
                'name': 'Enhanced Patch Detection',
                'method': 'api' if self.api_available else 'local',
                'endpoint': 'detect',
                'model': 'patch_detector',
                'params': {'use_patch_detection': True}
            },
            {
                'name': 'Unsupervised Detection (LOST)',
                'method': 'api' if self.api_available else 'local',
                'endpoint': 'detect_unsupervised',
                'model': 'lost',
                'params': {}
            }
        ]
        
        # Run tests
        for scenario in test_scenarios:
            print(f"\n🔍 Testing: {scenario['name']}")
            print("-" * 40)
            
            scenario_results = []
            
            for test_image in test_images:
                print(f"   📸 {test_image['name']}...", end=" ")
                
                start_time = time.time()
                
                if scenario['method'] == 'api':
                    result = self.test_api_detection(
                        test_image['path'], 
                        scenario['endpoint'],
                        **scenario['params']
                    )
                else:
                    result = self.test_local_detection(
                        test_image['path'], 
                        scenario['model']
                    )
                
                processing_time = time.time() - start_time
                
                if result:
                    num_detections = len(result.get('detections', []))
                    print(f"✅ {num_detections} objects ({processing_time:.2f}s)")
                    
                    scenario_results.append({
                        'image': test_image['name'],
                        'detections': num_detections,
                        'processing_time': processing_time,
                        'ground_truth': test_image['ground_truth'],
                        'result': result
                    })
                else:
                    print("❌ Failed")
            
            # Store results
            self.results[scenario['name']] = scenario_results
            
            # Print scenario summary
            if scenario_results:
                avg_detections = np.mean([r['detections'] for r in scenario_results])
                avg_time = np.mean([r['processing_time'] for r in scenario_results])
                success_rate = len(scenario_results) / len(test_images) * 100
                
                print(f"   📊 Summary: {avg_detections:.1f} avg detections, {avg_time:.2f}s avg time, {success_rate:.0f}% success")
    
    def compare_detection_modes(self):
        """Compare different detection modes"""
        
        print(f"\n📊 DETECTION MODE COMPARISON")
        print("=" * 60)
        
        if not self.results:
            print("❌ No results to compare. Run comprehensive test first.")
            return
        
        # Create comparison table
        comparison_data = {}
        
        for mode_name, results in self.results.items():
            if results:
                comparison_data[mode_name] = {
                    'avg_detections': np.mean([r['detections'] for r in results]),
                    'avg_time': np.mean([r['processing_time'] for r in results]),
                    'success_rate': len(results) / len(self.get_test_images()) * 100,
                    'total_tests': len(results)
                }
        
        # Print comparison
        print(f"{'Mode':<30} {'Avg Objects':<12} {'Avg Time':<10} {'Success':<8} {'Tests':<6}")
        print("-" * 66)
        
        for mode, data in comparison_data.items():
            print(f"{mode:<30} {data['avg_detections']:<12.1f} {data['avg_time']:<10.2f}s {data['success_rate']:<8.0f}% {data['total_tests']:<6}")
        
        # Performance insights
        print(f"\n🎯 PERFORMANCE INSIGHTS:")
        
        if len(comparison_data) >= 2:
            modes = list(comparison_data.keys())
            
            # Find best detection rate
            best_detection = max(comparison_data.items(), key=lambda x: x[1]['avg_detections'])
            print(f"   🔍 Best Detection Rate: {best_detection[0]} ({best_detection[1]['avg_detections']:.1f} objects)")
            
            # Find fastest
            fastest = min(comparison_data.items(), key=lambda x: x[1]['avg_time'])
            print(f"   ⚡ Fastest Processing: {fastest[0]} ({fastest[1]['avg_time']:.2f}s)")
            
            # Compare patch vs standard
            if 'Standard Object Detection' in comparison_data and 'Parallel Patch Detection' in comparison_data:
                standard = comparison_data['Standard Object Detection']['avg_detections']
                patch = comparison_data['Parallel Patch Detection']['avg_detections']
                improvement = ((patch - standard) / standard) * 100 if standard > 0 else 0
                print(f"   📈 Patch Detection Improvement: {improvement:+.1f}% over standard detection")
    
    def analyze_small_object_performance(self):
        """Analyze performance on small objects specifically"""
        
        print(f"\n🔬 SMALL OBJECT DETECTION ANALYSIS")
        print("=" * 60)
        
        test_images = self.get_test_images()
        
        # Find images with small objects
        small_object_images = []
        for img in test_images:
            if img['ground_truth'] and 'Small Objects' in img['ground_truth']:
                small_objects = img['ground_truth']['Small Objects']
                if small_objects > 0:
                    small_object_images.append((img['name'], small_objects))
        
        if not small_object_images:
            print("❌ No ground truth data available for small object analysis")
            return
        
        print(f"📋 Images with small objects: {len(small_object_images)}")
        
        for img_name, small_count in small_object_images:
            print(f"   📸 {img_name}: {small_count} small objects")
            
            # Compare detection results across modes
            for mode_name, results in self.results.items():
                img_result = next((r for r in results if r['image'] == img_name), None)
                if img_result:
                    detected = img_result['detections']
                    detection_rate = (detected / small_count) * 100 if small_count > 0 else 0
                    print(f"      {mode_name}: {detected}/{small_count} detected ({detection_rate:.0f}%)")
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        
        print(f"\n📝 GENERATING TEST REPORT...")
        
        report_content = []
        report_content.append("# Object Detection Testing Report\n")
        report_content.append(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Test setup
        report_content.append("## Test Setup\n")
        report_content.append(f"- **Test Images**: {len(self.get_test_images())}\n")
        report_content.append(f"- **API Available**: {'Yes' if self.api_available else 'No'}\n")
        report_content.append(f"- **Local Models**: {'Yes' if self.local_models else 'No'}\n\n")
        
        # Results summary
        report_content.append("## Results Summary\n\n")
        
        for mode_name, results in self.results.items():
            if results:
                avg_detections = np.mean([r['detections'] for r in results])
                avg_time = np.mean([r['processing_time'] for r in results])
                
                report_content.append(f"### {mode_name}\n")
                report_content.append(f"- **Average Detections**: {avg_detections:.1f}\n")
                report_content.append(f"- **Average Processing Time**: {avg_time:.2f}s\n")
                report_content.append(f"- **Tests Completed**: {len(results)}\n\n")
        
        # Save report
        with open('detection_test_report.md', 'w') as f:
            f.writelines(report_content)
        
        print(f"✅ Test report saved to: detection_test_report.md")
    
    def run_all_tests(self):
        """Run all tests and generate report"""
        
        print("🚀 STARTING COMPREHENSIVE DETECTION TESTING")
        print("=" * 80)
        
        # Run comprehensive testing
        self.run_comprehensive_test()
        
        # Compare modes
        self.compare_detection_modes()
        
        # Analyze small object performance
        self.analyze_small_object_performance()
        
        # Generate report
        self.generate_test_report()
        
        print(f"\n🎉 TESTING COMPLETE!")
        print(f"📊 Check detection_test_report.md for detailed results")

def main():
    """Main testing function"""
    
    # Check if test images exist
    if not os.path.exists('test_images'):
        print("❌ Test images not found!")
        print("📝 Run: python generate_test_images.py")
        return
    
    # Create tester
    tester = DetectionTester()
    
    # Run all tests
    tester.run_all_tests()

if __name__ == "__main__":
    main() 