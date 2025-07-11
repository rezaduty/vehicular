# Project Structure - Autonomous Driving Perception System

## 📁 Project Organization Overview

This document provides a comprehensive overview of the autonomous driving perception project structure, detailing the organization of code, data, documentation, and configuration files.

## 🏗️ Root Directory Structure

```
vehi/                                    # Project root directory
├── config/                             # Configuration files
├── src/                                # Source code
├── docs/                               # Documentation
├── tests/                              # Test files
├── domain_adaptation_data/             # Training datasets
├── test_images/                        # Test images
├── demo_upload_images/                 # Demo images
├── video_sample_file/                  # Video processing
├── diagrams/                           # System diagrams
├── Dockerfile                          # Container configuration
├── docker-compose.yml                  # Multi-service setup
├── requirements.txt                    # Python dependencies
├── README.md                           # Project overview
├── start_services.sh                   # Service startup script
├── MODEL_SELECTION_GUIDE.md            # Model selection guide
├── README_MULTI_MODEL.md               # Multi-model documentation
├── yolov8n.pt                          # YOLOv8 model weights
└── tasks.txt                           # Project tasks
```

## 🔧 Source Code Architecture (`src/`)

### API Layer (`src/api/`)
The API layer provides RESTful endpoints for the autonomous driving perception system.

**Files:**
- `main.py` - Basic FastAPI implementation
- `real_working_api.py` - Production-ready API with real YOLOv8
- `multi_model_api.py` - Multi-model ensemble API
- `real_domain_adaptation.py` - Domain adaptation training API
- `static/` - Static assets and files

**Key Features:**
- FastAPI with async support
- CORS middleware for web compatibility
- Video processing with streaming
- Model management and selection
- Real-time progress monitoring

### Model Architecture (`src/models/`)
Comprehensive collection of machine learning models for autonomous driving.

**Core Models:**
- `object_detection.py` - YOLOv8, EfficientDet, ParallelPatchDetector
- `domain_adaptation.py` - DANN, CORAL, MMD implementations
- `segmentation.py` - DeepLabV3+, UNet for semantic segmentation
- `tracking.py` - DeepSORT, MOT for object tracking
- `autoencoder.py` - Image/LiDAR compression models
- `depth_estimation.py` - MonoDepth2 for depth estimation

**Advanced Models:**
- `tensorflow_models.py` - TensorFlow/Keras implementations
- `pytorch_models.py` - Advanced PyTorch models
- `model_ensemble.py` - Multi-model ensemble system
- `domain_adaptation_pipeline.py` - CARLA→KITTI adaptation pipeline
- `__init__.py` - Model registry and factory functions

### Data Processing (`src/data/`)
Data loading, preprocessing, and augmentation utilities.

**Components:**
- `dataset_loader.py` - Multi-format dataset loading
- `transforms.py` - Data augmentation and preprocessing
- `utils.py` - Utility functions for data handling
- `__init__.py` - Data module initialization

### Unsupervised Learning (`src/unsupervised/`)
Self-supervised and unsupervised learning implementations.

**Algorithms:**
- `lost.py` - LOST (Localization from Self-supervised Tracking)
- `most.py` - MOST (Multi-Object Self-supervised Tracking)
- `sonata.py` - SONATA (LiDAR point cloud segmentation)
- `__init__.py` - Unsupervised module initialization

### User Interfaces
- `streamlit_app.py` - Main Streamlit web interface
- `streamlit_multi_model.py` - Multi-model interface
- `train.py` - Training script for model training

## ⚙️ Configuration System (`config/`)

### Main Configuration (`config/config.yaml`)
Comprehensive YAML configuration file controlling all aspects of the system.

**Configuration Sections:**
- **Project Settings**: Name, version, description
- **Data Configuration**: Dataset paths, image dimensions, augmentation
- **Model Configuration**: Individual model settings, ensemble parameters
- **Training Configuration**: Learning rates, batch sizes, epochs
- **Inference Configuration**: Patch detection, parallel processing
- **Deployment Configuration**: API settings, Streamlit configuration
- **Evaluation Configuration**: Metrics, cross-validation, benchmarking

**Key Features:**
- Multi-model configuration support
- Domain adaptation parameters
- Ensemble method configuration
- Performance optimization settings
- Hardware configuration options

## 📚 Documentation (`docs/`)

### Technical Documentation
- `README.md` - Project overview and setup instructions
- `technical_report.md` - Detailed technical specifications
- `research_papers_summary.md` - Related research summary

### Project Status Documentation
- `FINAL_PROJECT_STATUS.md` - Complete project status
- `TESTING_COMPLETE_SUMMARY.md` - Comprehensive test results
- `PROJECT_VERIFICATION_REPORT.md` - Verification and validation

### Implementation Guides
- `WORKING_DETECTION_GUIDE.md` - Object detection usage guide
- `DOMAIN_ADAPTATION_WORKING.md` - Domain adaptation tutorial
- `STREAMLIT_FIXED_GUIDE.md` - Streamlit interface guide

### Issue Resolution
- `CONFIGURATION_ISSUE_FIXED.md` - Configuration troubleshooting
- `PARAMETERS_FIXED_CONFIRMED.md` - Parameter validation
- `REAL_FUNCTIONALITY_CONFIRMED.md` - Functionality verification

## 🧪 Testing Framework (`tests/`)

### Core Tests
- `test_project.py` - Main project functionality tests
- `test_detection_modes.py` - Detection mode validation
- `test_configuration_functionality.py` - Configuration testing
- `test_real_functionality.py` - Real functionality verification

### Specialized Tests
- `test_real_domain_adaptation.py` - Domain adaptation testing
- `test_streamlit_config_integration.py` - UI integration tests
- `test_parameter_functionality.py` - Parameter validation

### Demo Scripts
- `demo.py` - Basic demonstration scripts
- `simple_demo.py` - Simplified demo for quick testing
- `demo_upload_test.py` - Upload functionality testing

## 📊 Data Organization

### Training Data (`domain_adaptation_data/`)
Organized datasets for domain adaptation training.

**Dataset Structure:**
```
domain_adaptation_data/
├── kitti/                              # Real-world data
│   ├── real_000.jpg to real_014.jpg    # 15 KITTI images
├── carla/                              # Simulation data
│   ├── sim_000.jpg to sim_019.jpg      # 20 CARLA images
├── nuScenes/                           # Multi-modal data
│   ├── real_000.jpg to real_014.jpg    # 15 nuScenes images
└── airsim/                             # AirSim data
    ├── sim_000.jpg to sim_019.jpg      # 20 AirSim images
```

### Test Images (`test_images/`)
Curated test images for algorithm evaluation.

**Test Scenarios:**
- `urban_dense.jpg` - Dense urban scene (12 objects)
- `highway_sparse.jpg` - Sparse highway scene (3 objects)
- `mixed_objects.jpg` - Mixed scenario (9 objects)
- `small_objects_challenge.jpg` - Small objects challenge (16 objects)
- `domain_adaptation_sim.jpg` - Simulation style (6 objects)
- `kitti_style_scene.jpg` - Real-world style
- `carla_style_scene.jpg` - Perfect simulation style

### Demo Images (`demo_upload_images/`)
Demonstration images with detailed annotations.

**Demo Categories:**
- Urban dense scenes with many small objects
- Highway sparse scenes with large objects
- Mixed comprehensive test scenarios
- Small objects challenge scenarios
- Domain adaptation comparison images

## 🎥 Video Processing (`video_sample_file/`)

### Video Processing Components
- `video_detect.py` - Video processing script with full project integration
- `streamlit_app.py` - Video-specific Streamlit interface

**Features:**
- Real-time video processing
- Multi-model ensemble support
- H.264 encoding for web compatibility
- Progress monitoring and streaming

## 📈 System Diagrams (`diagrams/`)

### Comprehensive Diagram Collection
- `dataset_overview.md` - Dataset structure and organization
- `network_architecture.md` - Model architecture documentation
- `result_visualization.md` - Performance metrics and results
- `workflow_diagram.md` - System workflow documentation
- `project_structure.md` - Project organization (this document)

**Diagram Types:**
- Mermaid diagrams for system architecture
- PNG exports for presentations
- Detailed markdown documentation

## 🐳 Deployment Configuration

### Docker Configuration
- `Dockerfile` - Container configuration for the application
- `docker-compose.yml` - Multi-service deployment setup

### Service Management
- `start_services.sh` - Automated service startup script
- `requirements.txt` - Python dependency management

### Model Management
- `yolov8n.pt` - Pre-trained YOLOv8 model weights
- `MODEL_SELECTION_GUIDE.md` - Model selection documentation
- `README_MULTI_MODEL.md` - Multi-model system documentation

## 🔍 Key Project Characteristics

### Modular Architecture
- **Separation of Concerns**: Clear separation between API, models, data, and UI
- **Pluggable Components**: Easy to add new models or modify existing ones
- **Configuration-Driven**: Centralized configuration management
- **Scalable Design**: Supports horizontal and vertical scaling

### Comprehensive Testing
- **Unit Tests**: Individual component testing
- **Integration Tests**: Cross-component functionality testing
- **System Tests**: End-to-end system validation
- **Performance Tests**: Speed and accuracy benchmarking

### Documentation Standards
- **Technical Documentation**: Detailed implementation guides
- **User Documentation**: Usage instructions and tutorials
- **API Documentation**: Endpoint specifications and examples
- **Code Documentation**: Inline comments and docstrings

### Development Workflow
- **Version Control**: Git-based development workflow
- **Issue Tracking**: Documented issue resolution
- **Testing Strategy**: Comprehensive test coverage
- **Deployment Strategy**: Containerized deployment

## 🎯 Project Complexity Metrics

### Code Organization
- **Total Files**: 100+ files across all directories
- **Lines of Code**: 15,000+ lines of Python code
- **Configuration**: 400+ configuration parameters
- **Documentation**: 50+ pages of documentation

### Model Complexity
- **Object Detection**: 4 different architectures
- **Domain Adaptation**: 3 adaptation methods
- **Ensemble Methods**: 4 combination strategies
- **Unsupervised Learning**: 3 self-supervised algorithms

### Data Management
- **Training Data**: 70 images across 4 datasets
- **Test Data**: 20+ test scenarios
- **Demo Data**: 15 demonstration images
- **Video Data**: Full video processing pipeline

### System Integration
- **API Endpoints**: 15+ REST endpoints
- **Web Interface**: 7 interactive pages
- **Background Processing**: Async task management
- **Real-time Monitoring**: Performance tracking

This comprehensive project structure demonstrates a well-organized, scalable, and maintainable autonomous driving perception system with extensive documentation, testing, and deployment capabilities. 