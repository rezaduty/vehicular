#!/usr/bin/env python3
"""
Demo Script for Testing Image Upload with Generated Test Images
Shows how to use the test images with the Streamlit interface
"""

import os
import shutil
from pathlib import Path

def copy_test_images_for_demo():
    """Copy test images to a demo directory for easy access"""
    
    print("📋 Setting up test images for demo...")
    
    # Create demo directory
    demo_dir = "demo_upload_images"
    os.makedirs(demo_dir, exist_ok=True)
    
    # Copy test images with descriptive names
    test_mappings = {
        "urban_dense.jpg": "01_urban_dense_many_small_objects.jpg",
        "highway_sparse.jpg": "02_highway_sparse_large_objects.jpg", 
        "mixed_objects.jpg": "03_mixed_comprehensive_test.jpg",
        "small_objects_challenge.jpg": "04_small_objects_challenge.jpg",
        "domain_adaptation_sim.jpg": "05_simulation_style_scene.jpg",
        "kitti_style_scene.jpg": "06_kitti_real_world_style.jpg",
        "carla_style_scene.jpg": "07_carla_simulation_style.jpg"
    }
    
    for original, demo_name in test_mappings.items():
        src_path = f"test_images/{original}"
        dest_path = f"{demo_dir}/{demo_name}"
        
        if os.path.exists(src_path):
            shutil.copy2(src_path, dest_path)
            print(f"✓ Copied {original} → {demo_name}")
    
    # Copy ground truth info files
    for file in os.listdir("test_images"):
        if file.endswith("_info.txt"):
            src_path = f"test_images/{file}"
            dest_path = f"{demo_dir}/{file}"
            shutil.copy2(src_path, dest_path)
    
    # Create demo instructions
    instructions = """# Demo Upload Images

This directory contains test images for demonstrating the various object detection modes.

## 🧪 Test Images Available:

### 1. Urban Dense Scene (many small objects)
**File**: `01_urban_dense_many_small_objects.jpg`
- **Best for**: Testing parallel patch detection
- **Contains**: 12 objects (6 small)
- **Challenge**: Small object detection

### 2. Highway Sparse Scene (larger objects)
**File**: `02_highway_sparse_large_objects.jpg`
- **Best for**: Standard object detection
- **Contains**: 3 objects (1 small)
- **Challenge**: Standard detection

### 3. Mixed Comprehensive Test
**File**: `03_mixed_comprehensive_test.jpg`
- **Best for**: General testing
- **Contains**: 9 objects (5 small)
- **Challenge**: Mixed scenarios

### 4. Small Objects Challenge
**File**: `04_small_objects_challenge.jpg`
- **Best for**: Parallel patch detection comparison
- **Contains**: 16 objects (9 small)
- **Challenge**: Maximum small object density

### 5. Simulation Style Scene
**File**: `05_simulation_style_scene.jpg`
- **Best for**: Domain adaptation testing
- **Contains**: 6 objects (3 small)
- **Challenge**: Simulation data style

### 6. KITTI Real-World Style
**File**: `06_kitti_real_world_style.jpg`
- **Best for**: Domain adaptation comparison
- **Contains**: Realistic vehicle scene
- **Challenge**: Real-world data style

### 7. CARLA Simulation Style
**File**: `07_carla_simulation_style.jpg`
- **Best for**: Domain adaptation comparison
- **Contains**: Perfect simulation scene
- **Challenge**: Sim-to-real gap

## 🚀 How to Use:

### With Streamlit Interface (http://localhost:8501):
1. Navigate to the Object Detection tab
2. Upload one of the test images
3. Try different detection modes:
   - **Standard Detection**: Use for highway/sparse scenes
   - **Parallel Patch Detection**: Use for urban/dense scenes
   - **Enhanced Detection**: Compare with standard
   - **Unsupervised (LOST)**: Test without labels

### Recommended Test Sequence:

1. **Start with Standard Detection**:
   - Upload `02_highway_sparse_large_objects.jpg`
   - Note detection results

2. **Test Parallel Patch Enhancement**:
   - Upload `04_small_objects_challenge.jpg`
   - Compare Standard vs Parallel Patch Detection
   - Should see improvement on small objects

3. **Domain Adaptation Demo**:
   - Upload `06_kitti_real_world_style.jpg` (real-world style)
   - Upload `07_carla_simulation_style.jpg` (simulation style)
   - Compare detection differences

4. **Comprehensive Testing**:
   - Upload `03_mixed_comprehensive_test.jpg`
   - Test all detection modes
   - Compare results

## 📊 Expected Results:

- **Parallel Patch Detection** should perform better on images with many small objects
- **Standard Detection** should be faster on simpler scenes
- **Domain Adaptation** differences should be visible between KITTI vs CARLA styles
- **Unsupervised Detection** should work without requiring labeled training data

## 🎯 Performance Metrics to Watch:

- **Detection Count**: Number of objects found
- **Processing Time**: Speed of detection
- **Confidence Scores**: Quality of detections
- **Small Object Performance**: Specifically for patch detection

Enjoy testing the autonomous driving perception system! 🚗
"""
    
    with open(f"{demo_dir}/README.md", "w") as f:
        f.write(instructions)
    
    print(f"\n✅ Demo setup complete!")
    print(f"📁 Test images copied to: {demo_dir}/")
    print(f"📖 Instructions: {demo_dir}/README.md")
    
    return demo_dir

def show_test_image_summary():
    """Show summary of available test images"""
    
    print("\n📊 TEST IMAGE SUMMARY")
    print("=" * 50)
    
    # Read ground truth files
    for file in sorted(os.listdir("test_images")):
        if file.endswith("_info.txt"):
            print(f"\n📸 {file.replace('_info.txt', '.jpg')}")
            
            try:
                with open(f"test_images/{file}", 'r') as f:
                    content = f.read()
                
                # Extract key info
                lines = content.split('\n')
                for line in lines[:7]:  # First 7 lines have the summary
                    if line.strip():
                        print(f"   {line}")
                        
            except:
                print("   (Info file reading error)")

def main():
    """Main demo setup function"""
    
    print("🎨 DEMO IMAGE SETUP FOR OBJECT DETECTION TESTING")
    print("=" * 60)
    
    # Check if test images exist
    if not os.path.exists("test_images"):
        print("❌ Test images not found!")
        print("📝 Please run: python generate_test_images.py first")
        return
    
    # Show summary of test images
    show_test_image_summary()
    
    # Copy images for demo
    demo_dir = copy_test_images_for_demo()
    
    print(f"\n🌐 NOW YOU CAN:")
    print(f"1. 🚀 Open Streamlit: http://localhost:8501")
    print(f"2. 📁 Browse to upload images from: {demo_dir}/")
    print(f"3. 🧪 Test different detection modes")
    print(f"4. 📊 Compare results across modes")
    
    print(f"\n💡 TIP:")
    print(f"   Use images 01-04 for testing small object detection improvements")
    print(f"   Use images 06-07 for testing domain adaptation differences")
    
    print(f"\n🎯 RECOMMENDED TEST ORDER:")
    print(f"   1. 02_highway_sparse (baseline)")
    print(f"   2. 04_small_objects_challenge (patch detection test)")
    print(f"   3. 06_kitti vs 07_carla (domain adaptation)")
    print(f"   4. 03_mixed_comprehensive (full test)")

if __name__ == "__main__":
    main() 