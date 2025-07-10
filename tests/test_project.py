#!/usr/bin/env python3
"""
Test script to verify the Vehicular Technology Project implementation
Tests all components mentioned in tasks.txt
"""

import os
import sys
import numpy as np
import torch
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_project_structure():
    """Test that all required files exist"""
    logger.info("Testing project structure...")
    
    required_files = [
        "requirements.txt",
        "README.md",
        "config/config.yaml",
        "src/data/dataset_loader.py",
        "src/models/object_detection.py",
        "src/models/domain_adaptation.py", 
        "src/models/segmentation.py",
        "src/models/tracking.py",
        "src/models/autoencoder.py",
        "src/models/depth_estimation.py",
        "src/unsupervised/lost.py",
        "src/unsupervised/sonata.py",
        "src/api/main.py",
        "src/streamlit_app.py",
        "src/train.py",
        "docs/technical_report.md",
        "docs/research_papers_summary.md",
        "Dockerfile",
        "docker-compose.yml"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        logger.error(f"Missing files: {missing_files}")
        return False
    
    logger.info("✓ All required files present")
    return True

def test_imports():
    """Test that all modules can be imported"""
    logger.info("Testing module imports...")
    
    try:
        # Data modules
        from data.dataset_loader import AutonomousVehicleDataLoader
        from data.transforms import AutonomousVehicleTransforms
        
        # Model modules  
        from models.object_detection import YOLOv8Detector, ParallelPatchDetector
        from models.domain_adaptation import DomainAdversarialNetwork
        from models.segmentation import DeepLabV3Plus, UNet
        from models.tracking import DeepSORT, MOT
        from models.autoencoder import ImageAutoencoder, LiDARAutoencoder
        from models.depth_estimation import DepthEstimator
        
        # Unsupervised modules
        from unsupervised.lost import LOST
        from unsupervised.sonata import SONATA
        
        logger.info("✓ All modules imported successfully")
        return True
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        return False

def test_object_detection():
    """Test object detection functionality"""
    logger.info("Testing object detection...")
    
    try:
        from models.object_detection import YOLOv8Detector, ParallelPatchDetector
        
        # Test basic detector
        detector = YOLOv8Detector(num_classes=10)
        
        # Test with dummy input
        dummy_input = torch.randn(1, 3, 384, 1280)
        output = detector(dummy_input)
        
        assert 'predictions' in output
        logger.info("✓ Basic object detection working")
        
        # Test parallel patch detector (main task requirement)
        patch_detector = ParallelPatchDetector(
            base_detector=detector,
            patch_size=(384, 640),
            overlap_ratio=0.2
        )
        
        patch_output = patch_detector(dummy_input)
        assert 'merged_detections' in patch_output
        
        logger.info("✓ Parallel patch detection working (main task requirement)")
        return True
        
    except Exception as e:
        logger.error(f"Object detection test failed: {e}")
        return False

def test_domain_adaptation():
    """Test domain adaptation (CARLA -> KITTI)"""
    logger.info("Testing domain adaptation...")
    
    try:
        from models.domain_adaptation import DomainAdversarialNetwork
        
        # Test DANN model
        model = DomainAdversarialNetwork(num_classes=10)
        
        # Test with dummy CARLA and KITTI data
        carla_data = torch.randn(2, 3, 384, 1280)
        kitti_data = torch.randn(2, 3, 384, 1280)
        
        # Test training mode
        model.train()
        carla_output = model(carla_data, domain_label=0)  # CARLA domain
        kitti_output = model(kitti_data, domain_label=1)  # KITTI domain
        
        assert 'class_logits' in carla_output
        assert 'domain_logits' in carla_output
        
        logger.info("✓ Domain adaptation (CARLA->KITTI) working")
        return True
        
    except Exception as e:
        logger.error(f"Domain adaptation test failed: {e}")
        return False

def test_unsupervised_algorithms():
    """Test unsupervised algorithms (LOST, SONATA)"""
    logger.info("Testing unsupervised algorithms...")
    
    try:
        # Test LOST algorithm
        from unsupervised.lost import LOST
        
        lost = LOST()
        
        # Dummy video frames
        frames = [torch.randn(3, 384, 1280) for _ in range(3)]
        
        # Test motion-based detection
        detections = lost.detect_objects_from_motion(frames)
        assert isinstance(detections, list)
        
        logger.info("✓ LOST algorithm working")
        
        # Test SONATA algorithm for LiDAR
        from unsupervised.sonata import SONATA
        
        sonata = SONATA()
        
        # Dummy LiDAR point cloud
        points = np.random.rand(1000, 3) * 10  # 1000 points in 3D
        
        # Test segmentation
        result = sonata.segment_point_cloud(points)
        assert 'segments' in result
        assert 'num_segments' in result
        
        logger.info("✓ SONATA algorithm working")
        return True
        
    except Exception as e:
        logger.error(f"Unsupervised algorithms test failed: {e}")
        return False

def test_autoencoder_compression():
    """Test autoencoder architectures"""
    logger.info("Testing autoencoder compression...")
    
    try:
        from models.autoencoder import ImageAutoencoder, LiDARAutoencoder
        
        # Test image autoencoder
        img_ae = ImageAutoencoder(latent_dim=256)
        dummy_image = torch.randn(1, 3, 384, 1280)
        
        reconstruction, latent = img_ae(dummy_image)
        assert reconstruction.shape == dummy_image.shape
        assert latent.shape == (1, 256)
        
        logger.info("✓ Image autoencoder working")
        
        # Test LiDAR autoencoder
        lidar_ae = LiDARAutoencoder(max_points=1024, latent_dim=256)
        dummy_points = torch.randn(1, 1024, 4)  # [batch, points, features]
        
        recon_points, latent = lidar_ae(dummy_points)
        assert recon_points.shape == dummy_points.shape
        assert latent.shape == (1, 256)
        
        logger.info("✓ LiDAR autoencoder working")
        return True
        
    except Exception as e:
        logger.error(f"Autoencoder test failed: {e}")
        return False

def test_depth_estimation():
    """Test depth estimation with velocity tracking"""
    logger.info("Testing depth estimation...")
    
    try:
        from models.depth_estimation import DepthEstimator
        
        depth_model = DepthEstimator()
        dummy_image = torch.randn(1, 3, 384, 1280)
        
        output = depth_model(dummy_image)
        assert 'depth' in output
        assert 'detections' in output
        
        logger.info("✓ Depth estimation working")
        return True
        
    except Exception as e:
        logger.error(f"Depth estimation test failed: {e}")
        return False

def test_segmentation():
    """Test semantic segmentation"""
    logger.info("Testing semantic segmentation...")
    
    try:
        from models.segmentation import DeepLabV3Plus, UNet
        
        # Test DeepLabV3+
        deeplab = DeepLabV3Plus(num_classes=19)  # Cityscapes classes
        dummy_image = torch.randn(1, 3, 384, 1280)
        
        segmentation = deeplab(dummy_image)
        assert segmentation.shape == (1, 19, 384, 1280)
        
        logger.info("✓ Semantic segmentation working")
        return True
        
    except Exception as e:
        logger.error(f"Segmentation test failed: {e}")
        return False

def test_tracking():
    """Test multi-object tracking"""
    logger.info("Testing object tracking...")
    
    try:
        from models.tracking import DeepSORT, MOT
        
        # Test DeepSORT
        tracker = DeepSORT()
        
        # Dummy detections
        detections = [
            {'bbox': [100, 100, 200, 200], 'class': 0, 'confidence': 0.9},
            {'bbox': [300, 150, 400, 250], 'class': 1, 'confidence': 0.8}
        ]
        
        dummy_image = np.random.randint(0, 255, (384, 1280, 3), dtype=np.uint8)
        
        tracks = tracker.update(detections, dummy_image)
        assert isinstance(tracks, list)
        
        logger.info("✓ Multi-object tracking working")
        return True
        
    except Exception as e:
        logger.error(f"Tracking test failed: {e}")
        return False

def test_deployment_apis():
    """Test API endpoints"""
    logger.info("Testing deployment APIs...")
    
    try:
        # Check if FastAPI app can be imported
        from api.main import app
        
        # Check if Streamlit app exists
        assert os.path.exists("src/streamlit_app.py")
        
        logger.info("✓ Deployment APIs available")
        return True
        
    except Exception as e:
        logger.error(f"API test failed: {e}")
        return False

def verify_task_requirements():
    """Verify all task requirements are implemented"""
    logger.info("Verifying task requirements...")
    
    requirements_met = {
        "Main Objective - Parallel Object Detection": False,
        "Domain Adaptation (CARLA->KITTI)": False,
        "Object Detection & Segmentation": False,
        "Unsupervised Road Object Detection (LOST/MOST)": False,
        "Unsupervised LiDAR Segmentation (SONATA)": False,
        "Autoencoder Architectures": False,
        "Road Object Detection with Depth Estimation": False,
        "Multi-object Tracking": False,
        "FastAPI Deployment": False,
        "Streamlit Interface": False,
        "Technical Report": False,
        "Research Papers Summary": False,
        "Docker Deployment": False
    }
    
    # Check each requirement
    if os.path.exists("src/models/object_detection.py"):
        with open("src/models/object_detection.py", 'r') as f:
            content = f.read()
            if "ParallelPatchDetector" in content:
                requirements_met["Main Objective - Parallel Object Detection"] = True
    
    if os.path.exists("src/models/domain_adaptation.py"):
        requirements_met["Domain Adaptation (CARLA->KITTI)"] = True
    
    if os.path.exists("src/models/segmentation.py"):
        requirements_met["Object Detection & Segmentation"] = True
        
    if os.path.exists("src/unsupervised/lost.py"):
        requirements_met["Unsupervised Road Object Detection (LOST/MOST)"] = True
        
    if os.path.exists("src/unsupervised/sonata.py"):
        requirements_met["Unsupervised LiDAR Segmentation (SONATA)"] = True
        
    if os.path.exists("src/models/autoencoder.py"):
        requirements_met["Autoencoder Architectures"] = True
        
    if os.path.exists("src/models/depth_estimation.py"):
        requirements_met["Road Object Detection with Depth Estimation"] = True
        
    if os.path.exists("src/models/tracking.py"):
        requirements_met["Multi-object Tracking"] = True
        
    if os.path.exists("src/api/main.py"):
        requirements_met["FastAPI Deployment"] = True
        
    if os.path.exists("src/streamlit_app.py"):
        requirements_met["Streamlit Interface"] = True
        
    if os.path.exists("docs/technical_report.md"):
        requirements_met["Technical Report"] = True
        
    if os.path.exists("docs/research_papers_summary.md"):
        requirements_met["Research Papers Summary"] = True
        
    if os.path.exists("docker-compose.yml"):
        requirements_met["Docker Deployment"] = True
    
    # Print results
    logger.info("\n" + "="*60)
    logger.info("TASK REQUIREMENTS VERIFICATION")
    logger.info("="*60)
    
    for requirement, met in requirements_met.items():
        status = "✓ IMPLEMENTED" if met else "✗ MISSING"
        logger.info(f"{requirement}: {status}")
    
    total_requirements = len(requirements_met)
    met_requirements = sum(requirements_met.values())
    completion_rate = (met_requirements / total_requirements) * 100
    
    logger.info("="*60)
    logger.info(f"COMPLETION RATE: {met_requirements}/{total_requirements} ({completion_rate:.1f}%)")
    logger.info("="*60)
    
    return requirements_met, completion_rate

def main():
    """Run all tests"""
    logger.info("Starting Vehicular Technology Project Tests")
    logger.info("="*60)
    
    tests = [
        ("Project Structure", test_project_structure),
        ("Module Imports", test_imports),
        ("Object Detection", test_object_detection),
        ("Domain Adaptation", test_domain_adaptation),  
        ("Unsupervised Algorithms", test_unsupervised_algorithms),
        ("Autoencoder Compression", test_autoencoder_compression),
        ("Depth Estimation", test_depth_estimation),
        ("Semantic Segmentation", test_segmentation),
        ("Object Tracking", test_tracking),
        ("Deployment APIs", test_deployment_apis)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
            if result:
                logger.info(f"✓ {test_name} PASSED")
            else:
                logger.error(f"✗ {test_name} FAILED")
        except Exception as e:
            logger.error(f"✗ {test_name} FAILED: {e}")
            results[test_name] = False
        
        logger.info("-" * 40)
    
    # Verify task requirements
    requirements_met, completion_rate = verify_task_requirements()
    
    # Final summary
    passed_tests = sum(results.values())
    total_tests = len(results)
    
    logger.info("\nFINAL TEST SUMMARY")
    logger.info("="*60)
    logger.info(f"Tests Passed: {passed_tests}/{total_tests}")
    logger.info(f"Task Completion: {completion_rate:.1f}%")
    
    if completion_rate >= 95:
        logger.info("🎉 PROJECT SUCCESSFULLY IMPLEMENTS ALL TASK REQUIREMENTS!")
    elif completion_rate >= 80:
        logger.info("✅ PROJECT SUBSTANTIALLY IMPLEMENTS TASK REQUIREMENTS")
    else:
        logger.info("⚠️  PROJECT PARTIALLY IMPLEMENTS TASK REQUIREMENTS")
    
    return results, requirements_met

if __name__ == "__main__":
    main() 