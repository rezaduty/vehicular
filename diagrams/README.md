# Autonomous Driving Perception Project - Diagrams

This directory contains comprehensive visual diagrams for the autonomous driving perception project in both Mermaid source format (.mmd) and high-resolution PNG images.

## 📊 Available Diagrams

### 1. Dataset Overview (`dataset_overview.*`)
- **Purpose**: Visualizes the comprehensive dataset structure used for training and evaluation
- **Content**: Real-world datasets (KITTI, nuScenes), simulation datasets (CARLA, AirSim), test scenarios, and domain adaptation strategy
- **Key Features**: Object categories, challenge levels, and dataset usage strategy

### 2. Network Architecture (`network_architecture.*`)
- **Purpose**: Illustrates the multi-model architecture and ensemble system
- **Content**: YOLOv8, RetinaNet, EfficientDet models, domain adaptation methods (DANN, CORAL, MMD), and parallel patch detection
- **Key Features**: Model specifications, performance metrics, and ensemble logic

### 3. Workflow Diagram (`workflow_diagram.*`)
- **Purpose**: Shows the complete system workflow from user interface to result delivery
- **Content**: Streamlit interface, FastAPI server, processing pipeline, video processing, and model management
- **Key Features**: API endpoints, processing steps, and data flow

### 4. Project Structure (`project_structure.*`)
- **Purpose**: Displays the comprehensive project organization and file structure
- **Content**: Source code organization, configuration files, documentation, tests, and data directories
- **Key Features**: Detailed file descriptions and module relationships

### 5. Result Visualization (`result_visualization.*`)
- **Purpose**: Presents performance results and comparison metrics
- **Content**: Detection results by scenario, model performance comparison, domain adaptation results, and visualization features
- **Key Features**: Performance metrics, accuracy comparisons, and processing times

### 6. Proof of Tasks (`proof_of_tasks.*`)
- **Purpose**: Demonstrates completion of all required project tasks
- **Content**: Core functionality achievements, advanced model implementations, and overall project status
- **Key Features**: Task completion status, performance summaries, and success metrics

## 🎨 Image Specifications

### PNG Images
- **Resolution**: 1920×1080 pixels (Full HD)
- **Format**: PNG with white background
- **Theme**: Neutral color scheme with light, professional colors
- **Quality**: High-resolution for presentations and documentation

### Mermaid Source Files
- **Format**: `.mmd` files containing Mermaid diagram syntax
- **Purpose**: Source files for generating diagrams and future modifications
- **Features**: Light color styling, clear node structure, and professional appearance

## 🔧 Usage Instructions

### Viewing Diagrams
1. **PNG Images**: Can be viewed directly in any image viewer or embedded in documents
2. **Mermaid Source**: Can be rendered using Mermaid-compatible tools or online editors

### Regenerating Images
To regenerate PNG images from Mermaid source files:
```bash
npx @mermaid-js/mermaid-cli -i diagrams/[filename].mmd -o diagrams/[filename].png -t neutral -w 1920 -H 1080 -b white
```

### Editing Diagrams
1. Edit the `.mmd` source files using any text editor
2. Use Mermaid syntax for modifications
3. Regenerate PNG images using the command above

## 📋 File Structure

```
diagrams/
├── dataset_overview.mmd        # Dataset structure diagram source
├── dataset_overview.png        # Dataset structure diagram image
├── network_architecture.mmd    # Network architecture diagram source
├── network_architecture.png    # Network architecture diagram image
├── workflow_diagram.mmd        # System workflow diagram source
├── workflow_diagram.png        # System workflow diagram image
├── project_structure.mmd       # Project structure diagram source
├── project_structure.png       # Project structure diagram image
├── result_visualization.mmd    # Results visualization diagram source
├── result_visualization.png    # Results visualization diagram image
├── proof_of_tasks.mmd         # Task completion diagram source
├── proof_of_tasks.png         # Task completion diagram image
├── [original].md files        # Original markdown documentation
└── README.md                  # This file
```

## 🎯 Integration with Project

These diagrams are designed to:
- Provide clear visual documentation of the project architecture
- Support technical presentations and reports
- Facilitate understanding of complex system interactions
- Demonstrate project completion and achievements

## 📈 Performance Highlights Visualized

- **Object Detection**: 45 FPS real-time processing with YOLOv8
- **Domain Adaptation**: 56% improvement in CARLA→KITTI transfer
- **Patch Detection**: +22% improvement in small object detection
- **Video Processing**: Real-time 15 FPS output with H.264 encoding
- **Multi-Model Ensemble**: 0.891 mAP with 4-model integration

All diagrams use consistent styling and color schemes to maintain professional presentation quality while clearly communicating the project's technical achievements and architectural complexity. 