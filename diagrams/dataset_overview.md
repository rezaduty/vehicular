# Dataset Overview - Autonomous Driving Perception Project

## 📊 Dataset Structure

This project uses a comprehensive collection of datasets for training and evaluating autonomous driving perception models, with a focus on domain adaptation from simulation to real-world scenarios.

### 🌍 Real-World Datasets

#### KITTI Dataset
- **Type**: Real-world driving scenes
- **Images**: 15 high-quality images
- **Domain Label**: 1 (target domain)
- **Characteristics**: 
  - Noisy conditions
  - Varied lighting
  - Real-world imperfections
  - German highway/urban scenes

#### nuScenes Dataset
- **Type**: Multi-modal autonomous driving
- **Images**: 15 images
- **Domain Label**: 1 (target domain)
- **Characteristics**:
  - Camera, LiDAR, radar data
  - Boston/Singapore scenes
  - Complex urban environments

### 🎮 Simulation Datasets

#### CARLA Simulator
- **Type**: Synthetic simulation data
- **Images**: 20 synthetic images
- **Domain Label**: 0 (source domain)
- **Characteristics**:
  - Perfect geometric conditions
  - Controlled lighting
  - No sensor noise
  - Unlimited data generation

#### AirSim Simulator
- **Type**: Microsoft simulation platform
- **Images**: 20 synthetic images
- **Domain Label**: 0 (source domain)
- **Characteristics**:
  - Photorealistic rendering
  - Physics-based simulation
  - Configurable environments

### 🧪 Test Datasets

#### Test Scenarios Overview

| Scenario | Total Objects | Cars | Pedestrians | Signs | Small Objects | Challenge Type |
|----------|---------------|------|-------------|-------|---------------|----------------|
| Urban Dense | 12 | 5 | 3 | 4 | 6 | Small object detection |
| Highway Sparse | 3 | 2 | 0 | 1 | 1 | Standard detection |
| Mixed Scene | 9 | 4 | 2 | 3 | 5 | General testing |
| Small Objects | 16 | 6 | 4 | 6 | 9 | Maximum density |
| Simulation Style | 6 | 3 | 1 | 2 | 3 | Domain adaptation |
| KITTI Style | Variable | Variable | Variable | Variable | Variable | Real-world style |
| CARLA Style | Variable | Variable | Variable | Variable | Variable | Perfect simulation |

#### Detailed Test Scenarios

**1. Urban Dense Scene**
- **File**: `01_urban_dense_many_small_objects.jpg`
- **Purpose**: Testing parallel patch detection
- **Object Distribution**: 
  - Cars: 5 (2 small, 3 large)
  - Pedestrians: 3 (medium size)
  - Traffic Signs: 4 (all small)
- **Challenge**: Dense urban environment with many small objects

**2. Highway Sparse Scene**
- **File**: `02_highway_sparse_large_objects.jpg`
- **Purpose**: Standard object detection baseline
- **Object Distribution**:
  - Cars: 2 (large objects)
  - Traffic Signs: 1 (small)
- **Challenge**: Sparse highway environment

**3. Mixed Comprehensive Test**
- **File**: `03_mixed_comprehensive_test.jpg`
- **Purpose**: General algorithm testing
- **Object Distribution**:
  - Cars: 4 (2 small, 2 large)
  - Pedestrians: 2 (medium)
  - Traffic Signs: 3 (small)
- **Challenge**: Balanced scenario for comprehensive evaluation

**4. Small Objects Challenge**
- **File**: `04_small_objects_challenge.jpg`
- **Purpose**: Maximum small object density testing
- **Object Distribution**:
  - Cars: 6 (3 small, 3 large)
  - Pedestrians: 4 (medium)
  - Traffic Signs: 6 (all small)
- **Challenge**: Highest density of small objects

**5. Simulation Style Scene**
- **File**: `05_simulation_style_scene.jpg`
- **Purpose**: Domain adaptation testing
- **Characteristics**: Perfect simulation rendering
- **Challenge**: Sim-to-real domain gap

**6. KITTI Real-World Style**
- **File**: `06_kitti_real_world_style.jpg`
- **Purpose**: Real-world comparison
- **Characteristics**: Realistic vehicle scene
- **Challenge**: Real-world noise and conditions

**7. CARLA Simulation Style**
- **File**: `07_carla_simulation_style.jpg`
- **Purpose**: Domain adaptation baseline
- **Characteristics**: Perfect CARLA rendering
- **Challenge**: Simulation domain baseline

## 🔄 Domain Adaptation Strategy

### Source Domain: CARLA Simulation
- **Advantages**: 
  - Perfect labels
  - Unlimited data
  - Controlled conditions
  - No annotation cost
- **Limitations**:
  - Domain gap to real world
  - Lack of real-world noise
  - Perfect geometric conditions

### Target Domain: KITTI Real-World
- **Advantages**:
  - Real-world conditions
  - Actual sensor noise
  - Realistic lighting
  - True deployment conditions
- **Limitations**:
  - Limited labeled data
  - Expensive annotation
  - Weather/lighting variations

### Adaptation Techniques
1. **DANN** (Domain Adversarial Neural Networks)
2. **CORAL** (Correlation Alignment)
3. **MMD** (Maximum Mean Discrepancy)

## 📈 Object Distribution Analysis

### Object Size Categories
- **Small Objects**: < 32x32 pixels (traffic signs, distant cars)
- **Medium Objects**: 32x64 pixels (pedestrians, cyclists)
- **Large Objects**: > 64x64 pixels (close vehicles, trucks)

### Challenge Levels
- **Level 1**: Highway sparse (3 objects, 1 small)
- **Level 2**: Mixed scene (9 objects, 5 small)
- **Level 3**: Urban dense (12 objects, 6 small)
- **Level 4**: Small objects challenge (16 objects, 9 small)

## 🎯 Dataset Usage Strategy

### Training Strategy
1. **Pre-training**: CARLA simulation data (unlimited)
2. **Domain Adaptation**: CARLA → KITTI transfer
3. **Fine-tuning**: Limited KITTI real-world data
4. **Validation**: Test scenarios with known ground truth

### Evaluation Strategy
1. **Standard Detection**: Highway sparse scenes
2. **Small Object Detection**: Urban dense scenes
3. **Domain Adaptation**: Simulation vs real-world comparison
4. **Comprehensive Testing**: Mixed scenarios

### Performance Metrics
- **mAP@0.5**: Mean Average Precision at 0.5 IoU
- **Small Object mAP**: Specific metric for objects < 32x32
- **Domain Accuracy**: Classification accuracy across domains
- **Adaptation Gap**: Performance difference between domains

## 🔍 Data Quality Assurance

### Ground Truth Verification
- Manual annotation verification
- Bounding box accuracy validation
- Object category consistency
- Size classification verification

### Dataset Balance
- Equal representation across scenarios
- Balanced object size distribution
- Diverse environmental conditions
- Comprehensive challenge coverage

This dataset structure provides a comprehensive foundation for training and evaluating autonomous driving perception models with particular emphasis on domain adaptation and small object detection challenges. 