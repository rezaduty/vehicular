# Demo Upload Images

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
