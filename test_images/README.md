# Test Images for Object Detection

This directory contains synthetic test images for various object detection scenarios.

## Test Scenarios

### urban_dense.jpg
- **Description**: Dense urban scene with many small objects
- **Scene Type**: urban
- **Objects**: 5 cars, 3 pedestrians, 4 signs
- **Info File**: urban_dense_info.txt

### highway_sparse.jpg
- **Description**: Highway scene with sparse larger objects
- **Scene Type**: highway
- **Objects**: 2 cars, 0 pedestrians, 1 signs
- **Info File**: highway_sparse_info.txt

### mixed_objects.jpg
- **Description**: Mixed scene for comprehensive testing
- **Scene Type**: urban
- **Objects**: 4 cars, 2 pedestrians, 3 signs
- **Info File**: mixed_objects_info.txt

### small_objects_challenge.jpg
- **Description**: Scene designed to test small object detection
- **Scene Type**: urban
- **Objects**: 6 cars, 4 pedestrians, 6 signs
- **Info File**: small_objects_challenge_info.txt

### domain_adaptation_sim.jpg
- **Description**: Simulation-style scene for domain adaptation testing
- **Scene Type**: highway
- **Objects**: 3 cars, 1 pedestrians, 2 signs
- **Info File**: domain_adaptation_sim_info.txt

## Reference Images

- **kitti_style_scene.jpg**: KITTI dataset style image
- **carla_style_scene.jpg**: CARLA simulation style image

## Usage

These images can be used to test:
- Standard object detection
- Parallel patch detection (especially for small objects)
- Domain adaptation (comparing CARLA vs KITTI styles)
- Unsupervised detection algorithms
