"""
FastAPI backend for autonomous driving perception system
Provides endpoints for object detection, tracking, domain adaptation, and visualization
"""

import asyncio
import io
import json
import numpy as np
import cv2
from typing import List, Dict, Optional, Union
from PIL import Image
import torch
import yaml

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.object_detection import YOLOv8Detector, ParallelPatchDetector
from models.domain_adaptation import DomainAdversarialNetwork, DomainAdapter
from unsupervised.lost import LOST as SelfSupervisedObjectDetection
from data.dataset_loader import DatasetLoader
from data.transforms import get_transforms, PatchExtractor


# Initialize FastAPI app
app = FastAPI(
    title="Autonomous Driving Perception API",
    description="Advanced perception system for autonomous vehicles",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
import os
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


# Global model instances
models = {}
config = {}


class DetectionRequest(BaseModel):
    use_patch_detection: bool = True
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4
    model_type: str = "yolov8"


class DomainAdaptationRequest(BaseModel):
    source_dataset: str = "carla"
    target_dataset: str = "kitti"
    epochs: int = 10
    learning_rate: float = 0.001


class DetectionResponse(BaseModel):
    success: bool
    detections: List[Dict]
    processing_time: float
    image_shape: List[int]


class TrainingResponse(BaseModel):
    success: bool
    message: str
    epoch: int
    loss: float


@app.on_event("startup")
async def startup_event():
    """Initialize models and configuration on startup"""
    global models, config
    
    # Load configuration
    try:
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        # Default configuration
        config = {
            'models': {
                'object_detection': {
                    'architecture': 'yolov8',
                    'backbone': 'efficientnet-b3',
                    'num_classes': 10,
                    'confidence_threshold': 0.5,
                    'nms_threshold': 0.4
                },
                'autoencoder': {
                    'latent_dim': 256
                },
                'domain_adaptation': {
                    'lambda_grl': 1.0
                }
            },
            'inference': {
                'patch_detection': {
                    'enabled': True,
                    'patch_size': [192, 192],
                    'overlap': 0.2,
                    'min_object_size': 20
                },
                'parallel_processing': {
                    'num_workers': 4,
                    'batch_size': 16
                }
            },
            'training': {
                'learning_rate': 0.001,
                'weight_decay': 0.01,
                'epochs': 100
            }
        }
    
    # Initialize models
    await initialize_models()


async def initialize_models():
    """Initialize all models"""
    global models, config
    
    try:
        # Object detection model
        models['yolov8'] = YOLOv8Detector(
            num_classes=config['models']['object_detection']['num_classes'],
            confidence_threshold=config['models']['object_detection']['confidence_threshold'],
            nms_threshold=config['models']['object_detection']['nms_threshold']
        )
        
        # Parallel patch detector
        models['patch_detector'] = ParallelPatchDetector(
            base_detector=models['yolov8'],
            patch_size=tuple(config['inference']['patch_detection']['patch_size']),
            overlap=config['inference']['patch_detection']['overlap'],
            min_object_size=config['inference']['patch_detection']['min_object_size']
        )
        
        # Domain adaptation model
        models['domain_adaptation'] = DomainAdversarialNetwork(
            num_classes=config['models']['object_detection']['num_classes'],
            lambda_grl=config['models']['domain_adaptation']['lambda_grl']
        )
        
        # Unsupervised detection model
        models['lost'] = SelfSupervisedObjectDetection(config)
        
        # Move models to GPU if available
        if torch.cuda.is_available():
            for model_name, model in models.items():
                if hasattr(model, 'cuda'):
                    models[model_name] = model.cuda()
        
        print("Models initialized successfully")
        
    except Exception as e:
        print(f"Error initializing models: {e}")


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocess uploaded image for model inference"""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to numpy array
    image_np = np.array(image)
    
    # Resize to model input size
    image_resized = cv2.resize(image_np, (1280, 384))
    
    # Convert to tensor and normalize
    image_tensor = torch.from_numpy(image_resized.transpose(2, 0, 1)).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    
    return image_tensor


def postprocess_detections(detections: List[Dict], image_shape: tuple) -> List[Dict]:
    """Postprocess detections for API response"""
    processed_detections = []
    
    for detection in detections:
        if isinstance(detection, dict) and 'boxes' in detection:
            boxes = detection['boxes']
            scores = detection['scores']
            classes = detection['classes']
            
            for box, score, cls in zip(boxes, scores, classes):
                processed_detections.append({
                    'bbox': box.tolist() if hasattr(box, 'tolist') else list(box),
                    'confidence': float(score),
                    'class': int(cls),
                    'class_name': get_class_name(int(cls))
                })
    
    return processed_detections


def get_class_name(class_id: int) -> str:
    """Get class name from class ID"""
    class_names = [
        'Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting',
        'Cyclist', 'Tram', 'Misc', 'DontCare', 'Traffic_Sign'
    ]
    
    if 0 <= class_id < len(class_names):
        return class_names[class_id]
    else:
        return 'Unknown'


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Autonomous Driving Perception API",
        "version": "1.0.0",
        "endpoints": {
            "detection": "/detect",
            "patch_detection": "/detect_patches",
            "domain_adaptation": "/domain_adapt",
            "unsupervised_detection": "/detect_unsupervised",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": len(models),
        "gpu_available": torch.cuda.is_available()
    }


@app.post("/detect", response_model=DetectionResponse)
async def detect_objects(
    file: UploadFile = File(...),
    detection_params: DetectionRequest = DetectionRequest()
):
    """Detect objects in uploaded image"""
    import time
    start_time = time.time()
    
    try:
        # Read and preprocess image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        image_tensor = preprocess_image(image)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            image_tensor = image_tensor.cuda()
        
        # Select model based on request
        if detection_params.use_patch_detection:
            model = models['patch_detector']
            detections = model(image_tensor)
        else:
            model = models['yolov8']
            detections = model.predict(image_tensor)
        
        # Postprocess results
        processed_detections = postprocess_detections(detections, image.size)
        
        processing_time = time.time() - start_time
        
        return DetectionResponse(
            success=True,
            detections=processed_detections,
            processing_time=processing_time,
            image_shape=[image.width, image.height]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@app.post("/detect_patches")
async def detect_with_patches(
    file: UploadFile = File(...),
    patch_size: int = 192,
    overlap: float = 0.2
):
    """Detect objects using parallel patch processing"""
    import time
    start_time = time.time()
    
    try:
        # Read and preprocess image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        image_np = np.array(image)
        
        # Extract patches
        patch_extractor = PatchExtractor(
            patch_size=(patch_size, patch_size),
            overlap=overlap
        )
        
        patches = patch_extractor.extract_patches(image_np)
        
        # Process each patch
        patch_predictions = []
        
        for patch_data in patches:
            patch_tensor = preprocess_image(Image.fromarray(patch_data['patch']))
            
            if torch.cuda.is_available():
                patch_tensor = patch_tensor.cuda()
            
            # Get predictions
            with torch.no_grad():
                pred = models['yolov8'].predict(patch_tensor)[0]
            
            # Add patch offset information
            pred['x_offset'] = patch_data['x_offset']
            pred['y_offset'] = patch_data['y_offset']
            patch_predictions.append(pred)
        
        # Merge predictions
        merged_predictions = patch_extractor.merge_predictions(
            patch_predictions, image_np.shape[:2]
        )
        
        # Postprocess
        processed_detections = postprocess_detections([merged_predictions], image.size)
        
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "detections": processed_detections,
            "num_patches": len(patches),
            "processing_time": processing_time,
            "patch_info": {
                "patch_size": patch_size,
                "overlap": overlap,
                "total_patches": len(patches)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Patch detection failed: {str(e)}")


@app.post("/detect_unsupervised")
async def detect_unsupervised(file: UploadFile = File(...)):
    """Detect objects using unsupervised LOST algorithm"""
    import time
    start_time = time.time()
    
    try:
        # Read and preprocess image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        image_tensor = preprocess_image(image)
        
        if torch.cuda.is_available():
            image_tensor = image_tensor.cuda()
        
        # Use LOST detector
        detections = models['lost'].detect(image_tensor)
        
        # Postprocess
        processed_detections = postprocess_detections(detections, image.size)
        
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "detections": processed_detections,
            "processing_time": processing_time,
            "method": "LOST (unsupervised)"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unsupervised detection failed: {str(e)}")


@app.post("/domain_adapt")
async def start_domain_adaptation(
    background_tasks: BackgroundTasks,
    params: DomainAdaptationRequest = DomainAdaptationRequest()
):
    """Start domain adaptation training"""
    
    # Add training task to background
    background_tasks.add_task(
        run_domain_adaptation,
        params.source_dataset,
        params.target_dataset,
        params.epochs,
        params.learning_rate
    )
    
    return {
        "success": True,
        "message": "Domain adaptation training started",
        "parameters": {
            "source_dataset": params.source_dataset,
            "target_dataset": params.target_dataset,
            "epochs": params.epochs,
            "learning_rate": params.learning_rate
        }
    }


async def run_domain_adaptation(
    source_dataset: str,
    target_dataset: str,
    epochs: int,
    learning_rate: float
):
    """Run domain adaptation training in background"""
    try:
        # This would normally load actual datasets
        # For demo purposes, we'll simulate training
        
        print(f"Starting domain adaptation: {source_dataset} -> {target_dataset}")
        
        # Initialize domain adapter
        model = models['domain_adaptation']
        
        # Simulate training epochs
        for epoch in range(epochs):
            # Simulate training step
            await asyncio.sleep(1)  # Simulate training time
            
            loss = np.random.uniform(0.1, 1.0)  # Simulate loss
            
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")
        
        print("Domain adaptation training completed")
        
    except Exception as e:
        print(f"Domain adaptation failed: {e}")


@app.get("/models")
async def get_available_models():
    """Get list of available models"""
    model_info = {}
    
    for name, model in models.items():
        model_info[name] = {
            "type": type(model).__name__,
            "parameters": getattr(model, 'num_parameters', lambda: 0)() if hasattr(model, 'num_parameters') else 0,
            "device": "cuda" if next(model.parameters()).is_cuda else "cpu" if hasattr(model, 'parameters') else "unknown"
        }
    
    return {
        "models": model_info,
        "total_models": len(models)
    }


@app.get("/config")
async def get_configuration():
    """Get current configuration"""
    return config


@app.post("/config")
async def update_configuration(new_config: Dict):
    """Update configuration"""
    global config
    
    try:
        # Update configuration
        config.update(new_config)
        
        # Reinitialize models with new config
        await initialize_models()
        
        return {
            "success": True,
            "message": "Configuration updated successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Configuration update failed: {str(e)}")


@app.get("/datasets")
async def get_supported_datasets():
    """Get list of supported datasets"""
    return {
        "datasets": [
            {
                "name": "KITTI",
                "type": "real_world",
                "modalities": ["camera", "lidar"],
                "description": "Real-world driving data from Germany"
            },
            {
                "name": "CARLA",
                "type": "simulation",
                "modalities": ["camera", "lidar", "semantic"],
                "description": "High-fidelity simulation data"
            },
            {
                "name": "nuScenes",
                "type": "real_world",
                "modalities": ["camera", "lidar", "radar"],
                "description": "360-degree sensor data from Boston and Singapore"
            },
            {
                "name": "Waymo",
                "type": "real_world",
                "modalities": ["camera", "lidar"],
                "description": "Multi-city autonomous driving data"
            }
        ]
    }


@app.post("/visualize")
async def visualize_detections(
    file: UploadFile = File(...),
    detection_params: DetectionRequest = DetectionRequest()
):
    """Visualize object detections on image"""
    import tempfile
    import os
    
    try:
        # Get detections
        detection_response = await detect_objects(file, detection_params)
        
        if not detection_response.success:
            raise HTTPException(status_code=500, detail="Detection failed")
        
        # Read image again for visualization
        await file.seek(0)
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        image_np = np.array(image)
        
        # Draw bounding boxes
        for detection in detection_response.detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # Draw rectangle
            cv2.rectangle(
                image_np,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                (0, 255, 0),
                2
            )
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(
                image_np,
                label,
                (int(bbox[0]), int(bbox[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            result_image = Image.fromarray(image_np)
            result_image.save(tmp.name, 'JPEG')
            tmp_path = tmp.name
        
        return FileResponse(
            tmp_path,
            media_type="image/jpeg",
            filename="detection_result.jpg"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Visualization failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True
    ) 