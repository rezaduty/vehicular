"""
Multi-Model API for Autonomous Driving Perception
Supports ensemble predictions, domain adaptation, and comprehensive model management
"""

from fastapi import FastAPI, HTTPException, File, UploadFile, BackgroundTasks, Query
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
import cv2
import asyncio
import uuid
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import yaml
from datetime import datetime
import tempfile
import os

# Import model components
from ..models.model_ensemble import ModelEnsemble, create_model_ensemble
from ..models.domain_adaptation_pipeline import DomainAdaptationPipeline, create_domain_adaptation_pipeline
from ..models.object_detection import YOLOv8Detector, ParallelPatchDetector
from ..models.tensorflow_models import create_retinanet, create_efficientdet
from ..models.pytorch_models import create_coral_model, create_mmd_model, create_monodepth2_model
from ..data.dataset_loader import DatasetLoader
from ..data.transforms import get_transforms


# Initialize FastAPI app
app = FastAPI(
    title="Multi-Model Autonomous Driving Perception API",
    description="API for multi-model object detection, domain adaptation, and ensemble predictions",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
config = None
model_ensemble = None
domain_adaptation_pipeline = None
video_processing_status = {}
logger = logging.getLogger(__name__)


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global config, model_ensemble, domain_adaptation_pipeline
    
    # Load configuration
    config_path = Path("config/config.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        raise RuntimeError("Configuration file not found")
    
    logger.info("🚀 Starting Multi-Model Autonomous Driving Perception API...")
    
    # Initialize model ensemble
    enabled_models = []
    if config['models']['object_detection']['yolov8']['enabled']:
        enabled_models.append('yolov8')
    if config['models']['object_detection']['retinanet']['enabled']:
        enabled_models.append('retinanet')
    if config['models']['object_detection']['efficientdet']['enabled']:
        enabled_models.append('efficientdet')
    
    if config['models']['domain_adaptation']['dann']['enabled']:
        enabled_models.append('dann')
    if config['models']['domain_adaptation']['coral']['enabled']:
        enabled_models.append('coral')
    if config['models']['domain_adaptation']['mmd']['enabled']:
        enabled_models.append('mmd')
    
    model_ensemble = create_model_ensemble(
        config=config,
        model_types=enabled_models,
        ensemble_method=config['models']['object_detection']['ensemble']['method']
    )
    
    # Initialize domain adaptation pipeline
    adaptation_methods = []
    if config['models']['domain_adaptation']['dann']['enabled']:
        adaptation_methods.append('dann')
    if config['models']['domain_adaptation']['coral']['enabled']:
        adaptation_methods.append('coral')
    if config['models']['domain_adaptation']['mmd']['enabled']:
        adaptation_methods.append('mmd')
    
    if adaptation_methods:
        domain_adaptation_pipeline = create_domain_adaptation_pipeline(
            config=config,
            adaptation_methods=adaptation_methods
        )
    
    logger.info("✅ Multi-Model API initialized successfully")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "models_loaded": len(model_ensemble.models) if model_ensemble else 0,
        "domain_adaptation_enabled": domain_adaptation_pipeline is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/models/info")
async def get_models_info():
    """Get information about loaded models"""
    if not model_ensemble:
        raise HTTPException(status_code=500, detail="Model ensemble not initialized")
    
    models_info = {
        "ensemble_method": model_ensemble.ensemble_method,
        "models": {},
        "weights": model_ensemble.weights,
        "domain_adaptation": domain_adaptation_pipeline is not None
    }
    
    for model_type in model_ensemble.model_types:
        model_config = config['models'].get('object_detection', {}).get(model_type, {})
        if not model_config:
            # Check other model categories
            for category in ['domain_adaptation', 'segmentation', 'depth_estimation']:
                if model_type in config['models'].get(category, {}):
                    model_config = config['models'][category][model_type]
                    break
        
        models_info["models"][model_type] = {
            "enabled": model_config.get('enabled', False),
            "architecture": model_config.get('architecture', model_type),
            "weight": model_ensemble.weights.get(model_type, 1.0)
        }
    
    return models_info


@app.post("/predict/ensemble")
async def predict_ensemble(
    file: UploadFile = File(...),
    use_domain_adaptation: bool = Query(True, description="Use domain adaptation"),
    confidence_threshold: float = Query(0.5, description="Confidence threshold"),
    return_individual_predictions: bool = Query(False, description="Return individual model predictions")
):
    """Make ensemble predictions on uploaded image"""
    
    if not model_ensemble:
        raise HTTPException(status_code=500, detail="Model ensemble not initialized")
    
    try:
        # Read and process image
        image_bytes = await file.read()
        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Resize image to expected dimensions
        target_height = config['data']['image']['height']
        target_width = config['data']['image']['width']
        image_resized = cv2.resize(image, (target_width, target_height))
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image_resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        
        # Make predictions
        if use_domain_adaptation and domain_adaptation_pipeline:
            predictions = domain_adaptation_pipeline.predict_with_domain_adaptation(
                image_tensor, 
                use_ensemble=True
            )
        else:
            predictions = model_ensemble.predict(
                image_tensor,
                return_individual_predictions=return_individual_predictions
            )
        
        # Process predictions for JSON response
        processed_predictions = _process_predictions_for_json(predictions)
        
        return {
            "status": "success",
            "predictions": processed_predictions,
            "image_shape": image.shape,
            "processing_info": {
                "ensemble_method": model_ensemble.ensemble_method,
                "domain_adaptation_used": use_domain_adaptation,
                "confidence_threshold": confidence_threshold,
                "models_used": list(model_ensemble.models.keys())
            }
        }
        
    except Exception as e:
        logger.error(f"Error in ensemble prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/model/{model_type}")
async def predict_single_model(
    model_type: str,
    file: UploadFile = File(...),
    confidence_threshold: float = Query(0.5, description="Confidence threshold")
):
    """Make prediction using a specific model"""
    
    if not model_ensemble or model_type not in model_ensemble.models:
        raise HTTPException(status_code=404, detail=f"Model {model_type} not found")
    
    try:
        # Read and process image
        image_bytes = await file.read()
        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Resize image
        target_height = config['data']['image']['height']
        target_width = config['data']['image']['width']
        image_resized = cv2.resize(image, (target_width, target_height))
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image_resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        
        # Get specific model prediction
        model = model_ensemble.models[model_type]
        prediction = model_ensemble._get_model_prediction(model, model_type, image_tensor)
        
        # Process prediction for JSON response
        processed_prediction = _process_predictions_for_json(prediction)
        
        return {
            "status": "success",
            "model_type": model_type,
            "prediction": processed_prediction,
            "image_shape": image.shape,
            "confidence_threshold": confidence_threshold
        }
        
    except Exception as e:
        logger.error(f"Error in {model_type} prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/domain_adaptation/train")
async def train_domain_adaptation(
    background_tasks: BackgroundTasks,
    source_dataset: str = Query("carla", description="Source domain dataset"),
    target_dataset: str = Query("kitti", description="Target domain dataset"),
    num_epochs: int = Query(50, description="Number of training epochs"),
    adaptation_methods: List[str] = Query(["dann", "coral", "mmd"], description="Adaptation methods to use")
):
    """Train domain adaptation models"""
    
    if not domain_adaptation_pipeline:
        raise HTTPException(status_code=500, detail="Domain adaptation pipeline not initialized")
    
    training_id = str(uuid.uuid4())
    
    # Start training in background
    background_tasks.add_task(
        _train_domain_adaptation_background,
        training_id,
        source_dataset,
        target_dataset,
        num_epochs,
        adaptation_methods
    )
    
    return {
        "status": "training_started",
        "training_id": training_id,
        "source_dataset": source_dataset,
        "target_dataset": target_dataset,
        "num_epochs": num_epochs,
        "adaptation_methods": adaptation_methods
    }


@app.get("/domain_adaptation/status/{training_id}")
async def get_training_status(training_id: str):
    """Get domain adaptation training status"""
    
    # This would typically check a database or cache for training status
    # For now, return a placeholder response
    return {
        "training_id": training_id,
        "status": "training",
        "progress": 0.5,
        "current_epoch": 25,
        "total_epochs": 50,
        "losses": {
            "task_loss": 0.45,
            "domain_loss": 0.32,
            "coral_loss": 0.28,
            "mmd_loss": 0.31
        }
    }


@app.post("/models/ensemble/update_weights")
async def update_ensemble_weights(weights: Dict[str, float]):
    """Update ensemble model weights"""
    
    if not model_ensemble:
        raise HTTPException(status_code=500, detail="Model ensemble not initialized")
    
    try:
        # Validate weights
        for model_type in weights:
            if model_type not in model_ensemble.models:
                raise HTTPException(status_code=400, detail=f"Model {model_type} not found")
        
        # Update weights
        model_ensemble.weights.update(weights)
        
        # Normalize weights
        total_weight = sum(model_ensemble.weights.values())
        model_ensemble.weights = {k: v / total_weight for k, v in model_ensemble.weights.items()}
        
        return {
            "status": "success",
            "updated_weights": model_ensemble.weights
        }
        
    except Exception as e:
        logger.error(f"Error updating ensemble weights: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Weight update error: {str(e)}")


@app.post("/video/process_ensemble")
async def process_video_ensemble(
    file: UploadFile = File(...),
    use_domain_adaptation: bool = Query(True, description="Use domain adaptation"),
    confidence_threshold: float = Query(0.5, description="Confidence threshold"),
    enable_tracking: bool = Query(True, description="Enable object tracking"),
    output_format: str = Query("mp4v", description="Output video format")
):
    """Process video with ensemble models"""
    
    if not model_ensemble:
        raise HTTPException(status_code=500, detail="Model ensemble not initialized")
    
    try:
        # Generate unique video ID
        video_id = f"video_{int(datetime.now().timestamp())}"
        
        # Save uploaded video
        temp_dir = Path(tempfile.gettempdir()) / "video_processing"
        temp_dir.mkdir(exist_ok=True)
        
        input_path = temp_dir / f"{video_id}_input.mp4"
        with open(input_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Initialize video processing status
        video_processing_status[video_id] = {
            "status": "uploaded",
            "progress": 0.0,
            "total_frames": 0,
            "processed_frames": 0,
            "input_path": str(input_path),
            "output_path": None,
            "processing_info": {
                "ensemble_method": model_ensemble.ensemble_method,
                "domain_adaptation": use_domain_adaptation,
                "confidence_threshold": confidence_threshold,
                "tracking_enabled": enable_tracking
            }
        }
        
        return {
            "status": "uploaded",
            "video_id": video_id,
            "message": "Video uploaded successfully. Use /video/process/{video_id} to start processing."
        }
        
    except Exception as e:
        logger.error(f"Error uploading video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Video upload error: {str(e)}")


@app.post("/video/process/{video_id}")
async def start_video_processing(video_id: str, background_tasks: BackgroundTasks):
    """Start video processing with ensemble models"""
    
    if video_id not in video_processing_status:
        raise HTTPException(status_code=404, detail="Video not found")
    
    # Start processing in background
    background_tasks.add_task(_process_video_background, video_id)
    
    return {
        "status": "processing_started",
        "video_id": video_id,
        "message": "Video processing started in background"
    }


@app.get("/video/status/{video_id}")
async def get_video_status(video_id: str):
    """Get video processing status"""
    
    if video_id not in video_processing_status:
        raise HTTPException(status_code=404, detail="Video not found")
    
    return video_processing_status[video_id]


@app.get("/stream_processed_video/{video_id}")
async def stream_processed_video(video_id: str):
    """Stream processed video"""
    
    if video_id not in video_processing_status:
        raise HTTPException(status_code=404, detail="Video not found")
    
    status = video_processing_status[video_id]
    
    if status["status"] != "completed" or not status["output_path"]:
        raise HTTPException(status_code=404, detail="Processed video not available")
    
    output_path = Path(status["output_path"])
    
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Processed video file not found")
    
    def generate_video():
        with open(output_path, "rb") as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                yield chunk
    
    return StreamingResponse(
        generate_video(),
        media_type="video/mp4",
        headers={"Content-Disposition": f"attachment; filename=processed_{video_id}.mp4"}
    )


@app.get("/benchmark/models")
async def benchmark_models():
    """Benchmark all loaded models"""
    
    if not model_ensemble:
        raise HTTPException(status_code=500, detail="Model ensemble not initialized")
    
    try:
        # Create dummy input
        dummy_input = torch.randn(1, 3, 384, 1280)
        
        benchmark_results = {}
        
        for model_type, model in model_ensemble.models.items():
            # Measure inference time
            import time
            
            model.eval()
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(10):  # Average over 10 runs
                    _ = model_ensemble._get_model_prediction(model, model_type, dummy_input)
            
            end_time = time.time()
            avg_inference_time = (end_time - start_time) / 10
            
            benchmark_results[model_type] = {
                "avg_inference_time": avg_inference_time,
                "fps": 1.0 / avg_inference_time,
                "model_parameters": sum(p.numel() for p in model.parameters() if hasattr(model, 'parameters'))
            }
        
        return {
            "status": "success",
            "benchmark_results": benchmark_results,
            "test_input_shape": list(dummy_input.shape)
        }
        
    except Exception as e:
        logger.error(f"Error in model benchmarking: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Benchmarking error: {str(e)}")


# Background task functions
async def _train_domain_adaptation_background(training_id: str, source_dataset: str, 
                                            target_dataset: str, num_epochs: int, 
                                            adaptation_methods: List[str]):
    """Background task for domain adaptation training"""
    
    try:
        # This would implement the actual training logic
        # For now, simulate training progress
        for epoch in range(num_epochs):
            await asyncio.sleep(1)  # Simulate training time
            
            # Update training status (would typically use a database)
            progress = (epoch + 1) / num_epochs
            
            logger.info(f"Training {training_id}: Epoch {epoch+1}/{num_epochs} ({progress:.2%})")
        
        logger.info(f"Domain adaptation training {training_id} completed")
        
    except Exception as e:
        logger.error(f"Error in domain adaptation training {training_id}: {str(e)}")


async def _process_video_background(video_id: str):
    """Background task for video processing"""
    
    try:
        status = video_processing_status[video_id]
        input_path = Path(status["input_path"])
        
        # Open video
        cap = cv2.VideoCapture(str(input_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Update status
        status["total_frames"] = total_frames
        status["status"] = "processing"
        
        # Setup output video
        output_path = input_path.parent / f"{video_id}_processed.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        processed_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame with ensemble
            try:
                # Resize frame for model input
                target_height = config['data']['image']['height']
                target_width = config['data']['image']['width']
                frame_resized = cv2.resize(frame, (target_width, target_height))
                
                # Convert to tensor
                frame_tensor = torch.from_numpy(frame_resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                
                # Get ensemble predictions
                predictions = model_ensemble.predict(frame_tensor)
                
                # Draw predictions on frame
                processed_frame = _draw_predictions_on_frame(frame, predictions, width, height)
                
                # Write frame
                out.write(processed_frame)
                
            except Exception as e:
                logger.warning(f"Error processing frame {processed_frames}: {str(e)}")
                # Write original frame if processing fails
                out.write(frame)
            
            processed_frames += 1
            
            # Update progress
            progress = processed_frames / total_frames
            status["processed_frames"] = processed_frames
            status["progress"] = progress
            
            # Log progress every 100 frames
            if processed_frames % 100 == 0:
                logger.info(f"Video {video_id}: {processed_frames}/{total_frames} frames processed ({progress:.1%})")
        
        # Cleanup
        cap.release()
        out.release()
        
        # Update final status
        status["status"] = "completed"
        status["output_path"] = str(output_path)
        status["progress"] = 1.0
        
        logger.info(f"Video processing completed for {video_id}")
        
    except Exception as e:
        logger.error(f"Error processing video {video_id}: {str(e)}")
        status["status"] = "error"
        status["error"] = str(e)


def _process_predictions_for_json(predictions):
    """Process predictions to be JSON serializable"""
    
    if isinstance(predictions, dict):
        processed = {}
        for key, value in predictions.items():
            if isinstance(value, torch.Tensor):
                processed[key] = value.cpu().numpy().tolist()
            elif isinstance(value, np.ndarray):
                processed[key] = value.tolist()
            elif isinstance(value, dict):
                processed[key] = _process_predictions_for_json(value)
            elif isinstance(value, list):
                processed[key] = [_process_predictions_for_json(item) if isinstance(item, dict) else item for item in value]
            else:
                processed[key] = value
        return processed
    elif isinstance(predictions, (torch.Tensor, np.ndarray)):
        return predictions.cpu().numpy().tolist() if isinstance(predictions, torch.Tensor) else predictions.tolist()
    else:
        return predictions


def _draw_predictions_on_frame(frame, predictions, width, height):
    """Draw ensemble predictions on video frame"""
    
    processed_frame = frame.copy()
    
    try:
        if 'ensemble' in predictions:
            ensemble_pred = predictions['ensemble']
            
            if 'detection' in ensemble_pred:
                detection_pred = ensemble_pred['detection']
                
                boxes = detection_pred.get('boxes', [])
                scores = detection_pred.get('scores', [])
                classes = detection_pred.get('classes', [])
                model_sources = detection_pred.get('model_sources', [])
                
                # Scale boxes to frame dimensions
                target_height = config['data']['image']['height']
                target_width = config['data']['image']['width']
                
                scale_x = width / target_width
                scale_y = height / target_height
                
                for i, (box, score, class_id) in enumerate(zip(boxes, scores, classes)):
                    if score > 0.5:  # Confidence threshold
                        x1, y1, x2, y2 = box
                        x1 = int(x1 * scale_x)
                        y1 = int(y1 * scale_y)
                        x2 = int(x2 * scale_x)
                        y2 = int(y2 * scale_y)
                        
                        # Get color based on model source
                        model_id = model_sources[i] if i < len(model_sources) else 0
                        colors = config['inference']['video_processing']['visualization']['colors']
                        color = colors[model_id % len(colors)]
                        
                        # Draw bounding box
                        cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw label
                        label = f"Class {class_id}: {score:.2f}"
                        cv2.putText(processed_frame, label, (x1, y1 - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    except Exception as e:
        logger.warning(f"Error drawing predictions: {str(e)}")
    
    return processed_frame


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 