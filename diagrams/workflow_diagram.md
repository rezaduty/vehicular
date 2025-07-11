# Workflow Diagram - Autonomous Driving Perception System

## 🔄 System Workflow Overview

This document outlines the complete workflow of the autonomous driving perception system, from user interaction through model inference to result delivery.

## 🎯 User Interface Layer

### Streamlit Web Interface
- **Port**: 8501
- **Architecture**: Multi-page application
- **Features**: 
  - Object Detection page
  - Video Processing page
  - Domain Adaptation interface
  - Model Comparison dashboard
  - Configuration management
  - Analytics dashboard

### Web Browser Interface
- **User Interaction**: File uploads, parameter configuration
- **Supported Formats**: PNG, JPG, JPEG for images; MP4, AVI, MOV for videos
- **Real-time Updates**: Progress monitoring, status updates
- **Visualization**: Interactive result display

## 🌐 API Gateway Layer

### FastAPI Server
- **Port**: 8000
- **Architecture**: RESTful API with async support
- **Key Endpoints**:
  - `/detect` - Image object detection
  - `/upload_video` - Video file upload
  - `/process_video/{video_id}` - Video processing
  - `/stream_processed_video/{video_id}` - Video streaming
  - `/models` - Model information
  - `/config` - Configuration management

### CORS Middleware
- **Purpose**: Cross-origin request handling
- **Security**: Configurable origin policies
- **Headers**: Security headers for web compatibility

## 🔧 Core Processing Pipeline

### 1. Image/Video Upload
**Process Flow:**
1. **File Validation**: Check file format, size limits
2. **Format Conversion**: Convert to standard formats
3. **Metadata Extraction**: Extract image/video properties
4. **Storage**: Save to temporary storage with unique ID

**Supported Formats:**
- **Images**: PNG, JPG, JPEG (max 10MB)
- **Videos**: MP4, AVI, MOV, MKV, WMV, FLV (max 500MB)

### 2. Model Selection
**Available Models:**
- **YOLOv8**: Real-time detection (45 FPS)
- **RetinaNet**: Small object specialist (25 FPS)
- **EfficientDet**: Efficiency optimized (32 FPS)
- **Ensemble**: Maximum accuracy (15 FPS)
- **Domain Adaptation**: DANN, CORAL, MMD

**Selection Criteria:**
- Performance requirements (speed vs accuracy)
- Object size distribution
- Domain characteristics (simulation vs real-world)
- Hardware constraints

### 3. Preprocessing
**Image Preprocessing:**
1. **Resize**: Scale to model input size (384×1280)
2. **Normalize**: Apply ImageNet normalization
3. **Tensor Conversion**: Convert to PyTorch/TensorFlow tensors
4. **Batch Formation**: Create batches for parallel processing

**Video Preprocessing:**
1. **Frame Extraction**: Extract frames at specified FPS
2. **Batch Creation**: Group frames for efficient processing
3. **Memory Management**: Optimize for GPU memory usage

### 4. Model Inference
**Inference Pipeline:**
1. **Model Loading**: Load selected model(s) to GPU
2. **Parallel Processing**: Utilize multiple workers
3. **Batch Optimization**: Process multiple images simultaneously
4. **Memory Management**: Efficient GPU memory utilization

**Performance Optimization:**
- Mixed precision inference (FP16)
- Dynamic batching
- Model caching
- GPU memory optimization

### 5. Postprocessing
**Detection Postprocessing:**
1. **Non-Maximum Suppression (NMS)**: Remove duplicate detections
2. **Coordinate Mapping**: Convert to original image coordinates
3. **Confidence Filtering**: Apply confidence thresholds
4. **Class Mapping**: Map class indices to names

**Ensemble Postprocessing:**
1. **Prediction Fusion**: Combine multiple model predictions
2. **Weighted Averaging**: Apply model-specific weights
3. **Cross-model NMS**: Remove duplicates across models
4. **Confidence Calibration**: Adjust ensemble confidence scores

## 🌉 Domain Adaptation Workflow

### Training Data Preparation
**Source Domain (CARLA):**
1. **Data Loading**: Load CARLA simulation images
2. **Label Processing**: Perfect ground truth labels
3. **Augmentation**: Apply domain-specific augmentations
4. **Batch Creation**: Create training batches

**Target Domain (KITTI):**
1. **Data Loading**: Load KITTI real-world images
2. **Unlabeled Processing**: Process without labels
3. **Feature Extraction**: Extract domain-specific features
4. **Batch Creation**: Create adaptation batches

### Domain Training Process
**DANN (Domain Adversarial Neural Network):**
1. **Feature Extraction**: Shared feature learning
2. **Task Classification**: Object detection on source domain
3. **Domain Classification**: Source vs target classification
4. **Adversarial Training**: Gradient reversal layer optimization

**CORAL (Correlation Alignment):**
1. **Feature Alignment**: Statistical moment matching
2. **Covariance Matching**: Minimize covariance differences
3. **Domain Alignment**: Align feature distributions

**MMD (Maximum Mean Discrepancy):**
1. **Kernel Methods**: RBF kernel-based alignment
2. **Distribution Matching**: Minimize distribution differences
3. **Feature Space Alignment**: RKHS-based alignment

### Adaptation Evaluation
**Performance Metrics:**
- Source domain accuracy
- Target domain accuracy
- Domain gap measurement
- Adaptation effectiveness

## 🎥 Video Processing Pipeline

### Video Upload and Validation
1. **File Upload**: Multi-format video support
2. **Format Validation**: Check codec compatibility
3. **Metadata Extraction**: Duration, FPS, resolution
4. **Storage**: Temporary storage with unique identifier

### Frame-by-Frame Processing
1. **Frame Extraction**: Extract frames at input FPS
2. **Sequential Processing**: Process frames in order
3. **Batch Optimization**: Group frames for efficiency
4. **Memory Management**: Optimize GPU memory usage

### Video Generation
1. **Frame Assembly**: Combine processed frames
2. **H.264 Encoding**: Web-compatible encoding
3. **Compression**: Optimize file size
4. **Metadata Preservation**: Maintain video properties

### Streaming API
1. **HTTP Range Requests**: Support partial content
2. **Progressive Download**: Enable streaming playback
3. **Web Compatibility**: Browser-friendly format
4. **Caching**: Efficient content delivery

## 🎛️ Model Management System

### Model Registry
**Dynamic Loading:**
- Load models on demand
- Version control and management
- Configuration-based selection
- Memory optimization

**Model Categories:**
- Object Detection models
- Domain Adaptation models
- Segmentation models
- Tracking models
- Ensemble models

### Ensemble Logic
**Ensemble Methods:**
1. **Weighted Average**: Fixed model weights
2. **Confidence Weighting**: Dynamic confidence-based weights
3. **Adaptive Weighting**: Performance-based weight adjustment

**Ensemble Pipeline:**
1. **Parallel Inference**: Run multiple models simultaneously
2. **Prediction Collection**: Gather all model predictions
3. **Weight Application**: Apply ensemble weights
4. **Fusion**: Combine predictions using selected method
5. **Final NMS**: Remove duplicates across models

### Cache System
**Model Caching:**
- GPU memory caching
- Model weight caching
- Configuration caching

**Prediction Caching:**
- Result caching for repeated requests
- Intermediate result storage
- Performance optimization

## 📊 Background Processing

### Task Queue System
**Async Processing:**
- Background task execution
- Status tracking
- Progress monitoring
- Error handling

**Task Types:**
- Video processing tasks
- Domain adaptation training
- Batch inference tasks
- Model training tasks

### Worker Pool
**Parallel Processing:**
- 4 parallel workers
- Load balancing
- Task distribution
- Resource management

**Worker Responsibilities:**
- Model inference
- Video processing
- Data preprocessing
- Result postprocessing

### Progress Monitoring
**Real-time Updates:**
- Processing status
- Progress percentage
- Estimated completion time
- Error reporting

## 📈 Monitoring & Analytics

### Performance Monitoring
**Real-time Metrics:**
- Frame rate (FPS)
- Memory usage
- GPU utilization
- Processing latency

**Performance Tracking:**
- Model-specific performance
- Ensemble performance
- Domain adaptation metrics
- Video processing metrics

### Metrics Collection
**Accuracy Metrics:**
- mAP (Mean Average Precision)
- Precision, Recall, F1-score
- IoU (Intersection over Union)
- Domain adaptation accuracy

**Efficiency Metrics:**
- Inference time
- Memory usage
- Power consumption
- Throughput

### Logging System
**Error Tracking:**
- Exception logging
- Error classification
- Debug information
- Performance logs

**System Logs:**
- API request logs
- Model loading logs
- Processing status logs
- Performance metrics logs

## 🔄 Complete Workflow Example

### Image Detection Workflow
1. **User Upload**: User uploads image via Streamlit
2. **API Request**: Streamlit sends POST request to FastAPI
3. **File Processing**: FastAPI validates and processes file
4. **Model Selection**: User-selected model loaded
5. **Preprocessing**: Image resized and normalized
6. **Inference**: Model performs object detection
7. **Postprocessing**: Apply NMS and coordinate mapping
8. **Response**: Return detection results to Streamlit
9. **Visualization**: Display results with bounding boxes
10. **Metrics**: Log performance metrics

### Video Processing Workflow
1. **Video Upload**: User uploads video file
2. **Validation**: Check format and extract metadata
3. **Background Task**: Create async processing task
4. **Frame Extraction**: Extract frames from video
5. **Batch Processing**: Process frames in parallel
6. **Progress Updates**: Real-time status updates
7. **Video Assembly**: Combine processed frames
8. **H.264 Encoding**: Encode for web playback
9. **Streaming Setup**: Enable HTTP range requests
10. **Completion**: Notify user of completion

### Domain Adaptation Workflow
1. **Data Preparation**: Load CARLA and KITTI datasets
2. **Model Initialization**: Initialize DANN components
3. **Training Loop**: Alternating source/target training
4. **Loss Calculation**: Task loss + domain loss
5. **Gradient Reversal**: Apply adversarial training
6. **Validation**: Evaluate on target domain
7. **Adaptation Metrics**: Measure domain gap
8. **Model Saving**: Save adapted model
9. **Evaluation**: Test on real-world data
10. **Deployment**: Deploy adapted model

## 🎯 Key Workflow Characteristics

### Scalability
- Horizontal scaling with multiple workers
- Vertical scaling with GPU optimization
- Load balancing across resources
- Efficient memory management

### Reliability
- Error handling and recovery
- Graceful degradation
- Status monitoring
- Comprehensive logging

### Performance
- Real-time processing capability
- Parallel processing optimization
- Memory efficiency
- GPU acceleration

### Usability
- Intuitive web interface
- Real-time progress updates
- Comprehensive visualization
- Easy configuration management

This workflow ensures efficient, reliable, and scalable processing of autonomous driving perception tasks while maintaining high performance and user experience standards. 