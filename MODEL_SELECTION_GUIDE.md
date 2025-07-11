# 🎛️ Model Selection Guide

## Overview
Your autonomous driving perception system now supports multiple detection models with easy switching capabilities in the web interface.

## 🚀 Quick Start

### Option 1: Use the Startup Script (Recommended)
```bash
./start_services.sh
```

### Option 2: Manual Start
```bash
# Terminal 1: Start API
uvicorn src.api.real_working_api:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Start Streamlit
streamlit run src/streamlit_app.py --server.port 8501
```

## 🎯 Available Models

### 1. 🤖 YOLOv8 (Primary)
- **Type**: Real-time object detection
- **Performance**: ~45 FPS on CPU
- **Parameters**: ~3.2M parameters
- **Input Size**: 640x640
- **Best For**: Fast, accurate general object detection

### 2. 🔄 Domain Adaptation (DANN)
- **Type**: Domain Adversarial Neural Network
- **Method**: Gradient Reversal Layer
- **Components**: Feature Extractor + Object Detector + Domain Classifier
- **Best For**: Adapting from simulation (CARLA) to real-world (KITTI)

### 3. 🔍 Patch Detection
- **Type**: Parallel Patch Processing
- **Patch Size**: 192x192 pixels
- **Overlap**: 20%
- **Best For**: Enhanced small object detection

### 4. 🎯 Unsupervised LOST
- **Type**: Self-supervised learning
- **Method**: Localization-based Object STructure
- **Best For**: Detection without labeled data

## 🎨 How to Change Models

### In Image Processing:
1. Go to **🎯 Object Detection** page
2. Select **📷 Image Processing**
3. In the sidebar, find **🎛️ Model Selection**
4. Choose your preferred model from the dropdown
5. Upload an image and click **🔍 Detect Objects**

### In Video Processing:
1. Go to **🎯 Object Detection** page
2. Select **🎥 Video Processing**
3. In the sidebar, find **🎛️ Model Selection**
4. Choose your preferred model from the dropdown
5. Upload a video and click **🚀 Start Video Processing**

## 📊 Model Comparison

Visit the **📊 Model Comparison** page to:
- View detailed information about each model
- Compare performance characteristics
- See technical specifications
- Analyze speed vs accuracy tradeoffs

## 🔧 Configuration

### Detection Parameters:
- **Confidence Threshold**: Minimum confidence for detections (0.1-1.0)
- **NMS Threshold**: Non-maximum suppression threshold (0.1-1.0)
- **Patch Detection**: Enable/disable patch-based processing

### Video Processing Options:
- **Object Tracking**: Enable DeepSORT tracking
- **Visualization**: Show confidence scores, class names, tracking IDs
- **Bounding Box Thickness**: Adjust visualization thickness

## 🌟 Advanced Features

### Domain Adaptation:
- Train models to adapt from simulation to real-world data
- Use CARLA simulator data to improve KITTI performance
- Gradient reversal for domain-invariant features

### Ensemble Methods:
- Combine multiple models for better accuracy
- Weighted averaging of predictions
- Confidence-based model selection

## 🎯 Best Practices

### For Real-time Applications:
- Use **YOLOv8** for fastest processing
- Lower confidence threshold for more detections
- Disable patch detection for speed

### For Maximum Accuracy:
- Use **Patch Detection** for small objects
- Higher confidence threshold for precision
- Enable object tracking for video

### For Research/Experimentation:
- Try **Domain Adaptation** for sim-to-real transfer
- Use **Unsupervised LOST** for unlabeled data
- Compare multiple models on same data

## 🚨 Troubleshooting

### Model Not Loading:
- Check API is running on port 8000
- Verify model files are present
- Check console for error messages

### Slow Processing:
- Reduce image/video resolution
- Lower confidence threshold
- Disable patch detection

### Poor Detection Quality:
- Increase confidence threshold
- Try different models
- Enable patch detection for small objects

## 🔗 API Endpoints

- **GET /models**: List available models
- **POST /detect**: Run object detection
- **POST /visualize**: Get visualization with bounding boxes
- **POST /upload_video**: Upload video for processing
- **GET /health**: Check API status

## 💡 Tips

1. **Start with YOLOv8** for general use cases
2. **Use Patch Detection** for traffic signs and small objects
3. **Try Domain Adaptation** if your data differs from training
4. **Check Model Comparison** page for detailed specs
5. **Use the startup script** for easy deployment

---

🎉 **Enjoy exploring different models and finding the best one for your autonomous driving perception tasks!** 