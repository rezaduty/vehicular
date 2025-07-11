"""
Model Ensemble System for Autonomous Driving Perception
Combines predictions from multiple models for improved performance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import cv2
from pathlib import Path
import logging

from .object_detection import YOLOv8Detector, ParallelPatchDetector
from .domain_adaptation import DomainAdversarialNetwork
from .pytorch_models import CORALDomainAdaptation, MMDDomainAdaptation, MonoDepth2
from .tensorflow_models import create_retinanet, create_efficientdet, create_deeplabv3plus
from .segmentation import DeepLabV3Plus, UNet
from .depth_estimation import DepthEstimator
from .tracking import DeepSORT


class ModelEnsemble:
    """Ensemble of multiple models for robust perception"""
    
    def __init__(self, 
                 config: Dict,
                 model_types: List[str] = ['yolov8', 'retinanet', 'efficientdet'],
                 ensemble_method: str = 'weighted_average',
                 weights: Optional[Dict[str, float]] = None,
                 use_domain_adaptation: bool = True):
        
        self.config = config
        self.model_types = model_types
        self.ensemble_method = ensemble_method
        self.use_domain_adaptation = use_domain_adaptation
        
        # Initialize models
        self.models = self._initialize_models()
        
        # Set ensemble weights
        if weights is None:
            self.weights = {model_type: 1.0 for model_type in model_types}
        else:
            self.weights = weights
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        self.weights = {k: v / total_weight for k, v in self.weights.values()}
        
        # Performance tracking
        self.performance_history = {model_type: [] for model_type in model_types}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def _initialize_models(self) -> Dict[str, nn.Module]:
        """Initialize all models in the ensemble"""
        models = {}
        
        num_classes = self.config['models']['object_detection']['num_classes']
        confidence_threshold = self.config['models']['object_detection']['confidence_threshold']
        
        # YOLOv8 model
        if 'yolov8' in self.model_types:
            yolov8_model = YOLOv8Detector(
                num_classes=num_classes,
                confidence_threshold=confidence_threshold
            )
            
            # Wrap with parallel patch detector if enabled
            if self.config['inference']['patch_detection']['enabled']:
                yolov8_model = ParallelPatchDetector(
                    base_detector=yolov8_model,
                    patch_size=tuple(self.config['inference']['patch_detection']['patch_size']),
                    overlap=self.config['inference']['patch_detection']['overlap']
                )
            
            models['yolov8'] = yolov8_model
        
        # RetinaNet model (TensorFlow)
        if 'retinanet' in self.model_types:
            models['retinanet'] = create_retinanet(
                num_classes=num_classes,
                backbone='resnet50',
                input_shape=(384, 1280, 3)
            )
        
        # EfficientDet model (TensorFlow)
        if 'efficientdet' in self.model_types:
            models['efficientdet'] = create_efficientdet(
                num_classes=num_classes,
                compound_coef=0,
                input_shape=(384, 1280, 3)
            )
        
        # Domain adaptation models
        if self.use_domain_adaptation:
            if 'dann' in self.model_types:
                models['dann'] = DomainAdversarialNetwork(
                    num_classes=num_classes,
                    lambda_grl=self.config['models']['domain_adaptation']['lambda_grl']
                )
            
            if 'coral' in self.model_types:
                models['coral'] = CORALDomainAdaptation(
                    num_classes=num_classes,
                    coral_weight=1.0
                )
            
            if 'mmd' in self.model_types:
                models['mmd'] = MMDDomainAdaptation(
                    num_classes=num_classes,
                    mmd_weight=1.0
                )
        
        # Segmentation models
        if 'deeplabv3plus' in self.model_types:
            models['deeplabv3plus'] = DeepLabV3Plus(
                num_classes=self.config['models']['segmentation']['num_classes']
            )
        
        if 'unet' in self.model_types:
            models['unet'] = UNet(
                num_classes=self.config['models']['segmentation']['num_classes']
            )
        
        # Depth estimation
        if 'depth_estimator' in self.model_types:
            models['depth_estimator'] = DepthEstimator()
        
        if 'monodepth2' in self.model_types:
            models['monodepth2'] = MonoDepth2()
        
        # Tracking
        if 'deepsort' in self.model_types:
            models['deepsort'] = DeepSORT()
        
        return models
    
    def predict(self, images: Union[torch.Tensor, np.ndarray], 
                return_individual_predictions: bool = False) -> Dict:
        """Make ensemble predictions"""
        
        # Convert to appropriate format
        if isinstance(images, np.ndarray):
            images_torch = torch.from_numpy(images).float()
            if len(images_torch.shape) == 3:
                images_torch = images_torch.unsqueeze(0)
        else:
            images_torch = images
        
        # Get predictions from each model
        individual_predictions = {}
        
        for model_type, model in self.models.items():
            try:
                pred = self._get_model_prediction(model, model_type, images_torch)
                individual_predictions[model_type] = pred
            except Exception as e:
                self.logger.warning(f"Error in {model_type} prediction: {str(e)}")
                continue
        
        # Combine predictions
        ensemble_prediction = self._combine_predictions(individual_predictions)
        
        # Add confidence scores
        ensemble_prediction = self._add_confidence_scores(ensemble_prediction, individual_predictions)
        
        if return_individual_predictions:
            return {
                'ensemble': ensemble_prediction,
                'individual': individual_predictions
            }
        
        return ensemble_prediction
    
    def _get_model_prediction(self, model, model_type: str, images: torch.Tensor) -> Dict:
        """Get prediction from a specific model"""
        
        model.eval()
        
        with torch.no_grad():
            if model_type in ['yolov8']:
                # YOLOv8 prediction
                predictions = model.predict(images)
                return self._format_yolo_prediction(predictions)
            
            elif model_type in ['dann', 'coral', 'mmd']:
                # Domain adaptation models
                outputs = model(images)
                if 'task_predictions' in outputs:
                    pred = outputs['task_predictions']
                elif 'predictions' in outputs:
                    pred = outputs['predictions']
                else:
                    pred = outputs
                
                return self._format_classification_prediction(pred)
            
            elif model_type in ['retinanet', 'efficientdet']:
                # TensorFlow models (simplified)
                import tensorflow as tf
                
                # Convert to TensorFlow tensor
                if isinstance(images, torch.Tensor):
                    images_tf = tf.convert_to_tensor(images.numpy())
                else:
                    images_tf = images
                
                outputs = model(images_tf)
                return self._format_detection_prediction(outputs)
            
            elif model_type in ['deeplabv3plus', 'unet']:
                # Segmentation models
                outputs = model(images)
                return self._format_segmentation_prediction(outputs)
            
            elif model_type in ['depth_estimator', 'monodepth2']:
                # Depth estimation models
                outputs = model(images)
                return self._format_depth_prediction(outputs)
            
            else:
                # Generic prediction
                outputs = model(images)
                return {'raw_output': outputs}
    
    def _format_yolo_prediction(self, predictions: List[Dict]) -> Dict:
        """Format YOLOv8 predictions"""
        
        formatted_predictions = []
        
        for pred in predictions:
            if len(pred['boxes']) > 0:
                formatted_pred = {
                    'boxes': pred['boxes'],
                    'scores': pred['scores'],
                    'classes': pred['classes']
                }
            else:
                formatted_pred = {
                    'boxes': np.array([]),
                    'scores': np.array([]),
                    'classes': np.array([])
                }
            
            formatted_predictions.append(formatted_pred)
        
        return {
            'type': 'detection',
            'predictions': formatted_predictions
        }
    
    def _format_classification_prediction(self, predictions: torch.Tensor) -> Dict:
        """Format classification predictions"""
        
        # Apply softmax for probabilities
        probabilities = F.softmax(predictions, dim=1)
        
        # Get predicted classes and confidence scores
        confidence_scores, predicted_classes = torch.max(probabilities, dim=1)
        
        return {
            'type': 'classification',
            'predictions': predicted_classes.cpu().numpy(),
            'confidence_scores': confidence_scores.cpu().numpy(),
            'probabilities': probabilities.cpu().numpy()
        }
    
    def _format_detection_prediction(self, outputs: Dict) -> Dict:
        """Format detection predictions from TensorFlow models"""
        
        # This is a simplified version - actual implementation would depend on model output format
        return {
            'type': 'detection',
            'raw_output': outputs
        }
    
    def _format_segmentation_prediction(self, outputs: torch.Tensor) -> Dict:
        """Format segmentation predictions"""
        
        # Apply softmax for class probabilities
        probabilities = F.softmax(outputs, dim=1)
        
        # Get predicted classes
        predicted_classes = torch.argmax(probabilities, dim=1)
        
        return {
            'type': 'segmentation',
            'predictions': predicted_classes.cpu().numpy(),
            'probabilities': probabilities.cpu().numpy()
        }
    
    def _format_depth_prediction(self, outputs: Dict) -> Dict:
        """Format depth estimation predictions"""
        
        if 'depth' in outputs:
            depth_map = outputs['depth']
        elif 'disp' in outputs:
            # Convert disparity to depth
            depth_map = 1.0 / (outputs['disp'] + 1e-6)
        else:
            depth_map = outputs
        
        return {
            'type': 'depth',
            'depth_map': depth_map.cpu().numpy() if isinstance(depth_map, torch.Tensor) else depth_map
        }
    
    def _combine_predictions(self, individual_predictions: Dict[str, Dict]) -> Dict:
        """Combine predictions from multiple models"""
        
        if self.ensemble_method == 'weighted_average':
            return self._weighted_average_combination(individual_predictions)
        elif self.ensemble_method == 'max_voting':
            return self._max_voting_combination(individual_predictions)
        elif self.ensemble_method == 'confidence_weighted':
            return self._confidence_weighted_combination(individual_predictions)
        elif self.ensemble_method == 'adaptive_weighted':
            return self._adaptive_weighted_combination(individual_predictions)
        else:
            return self._weighted_average_combination(individual_predictions)
    
    def _weighted_average_combination(self, individual_predictions: Dict[str, Dict]) -> Dict:
        """Combine predictions using weighted average"""
        
        # Separate predictions by type
        detection_predictions = {}
        classification_predictions = {}
        segmentation_predictions = {}
        depth_predictions = {}
        
        for model_type, pred in individual_predictions.items():
            if pred['type'] == 'detection':
                detection_predictions[model_type] = pred
            elif pred['type'] == 'classification':
                classification_predictions[model_type] = pred
            elif pred['type'] == 'segmentation':
                segmentation_predictions[model_type] = pred
            elif pred['type'] == 'depth':
                depth_predictions[model_type] = pred
        
        combined_prediction = {}
        
        # Combine detection predictions
        if detection_predictions:
            combined_prediction['detection'] = self._combine_detection_predictions(detection_predictions)
        
        # Combine classification predictions
        if classification_predictions:
            combined_prediction['classification'] = self._combine_classification_predictions(classification_predictions)
        
        # Combine segmentation predictions
        if segmentation_predictions:
            combined_prediction['segmentation'] = self._combine_segmentation_predictions(segmentation_predictions)
        
        # Combine depth predictions
        if depth_predictions:
            combined_prediction['depth'] = self._combine_depth_predictions(depth_predictions)
        
        return combined_prediction
    
    def _combine_detection_predictions(self, detection_predictions: Dict[str, Dict]) -> Dict:
        """Combine object detection predictions"""
        
        # Non-Maximum Suppression (NMS) across all models
        all_boxes = []
        all_scores = []
        all_classes = []
        all_model_ids = []
        
        for model_id, (model_type, pred) in enumerate(detection_predictions.items()):
            for batch_pred in pred['predictions']:
                if len(batch_pred['boxes']) > 0:
                    weight = self.weights.get(model_type, 1.0)
                    
                    all_boxes.extend(batch_pred['boxes'])
                    all_scores.extend(batch_pred['scores'] * weight)
                    all_classes.extend(batch_pred['classes'])
                    all_model_ids.extend([model_id] * len(batch_pred['boxes']))
        
        if not all_boxes:
            return {
                'boxes': np.array([]),
                'scores': np.array([]),
                'classes': np.array([]),
                'model_sources': np.array([])
            }
        
        # Apply NMS
        boxes = np.array(all_boxes)
        scores = np.array(all_scores)
        classes = np.array(all_classes)
        model_ids = np.array(all_model_ids)
        
        # Group by class and apply NMS
        final_boxes = []
        final_scores = []
        final_classes = []
        final_model_ids = []
        
        for class_id in np.unique(classes):
            class_mask = classes == class_id
            class_boxes = boxes[class_mask]
            class_scores = scores[class_mask]
            class_model_ids = model_ids[class_mask]
            
            # Apply NMS
            keep_indices = self._apply_nms(class_boxes, class_scores, iou_threshold=0.5)
            
            final_boxes.extend(class_boxes[keep_indices])
            final_scores.extend(class_scores[keep_indices])
            final_classes.extend([class_id] * len(keep_indices))
            final_model_ids.extend(class_model_ids[keep_indices])
        
        return {
            'boxes': np.array(final_boxes),
            'scores': np.array(final_scores),
            'classes': np.array(final_classes),
            'model_sources': np.array(final_model_ids)
        }
    
    def _combine_classification_predictions(self, classification_predictions: Dict[str, Dict]) -> Dict:
        """Combine classification predictions"""
        
        # Weighted average of probabilities
        weighted_probabilities = None
        total_weight = 0
        
        for model_type, pred in classification_predictions.items():
            weight = self.weights.get(model_type, 1.0)
            
            if weighted_probabilities is None:
                weighted_probabilities = weight * pred['probabilities']
            else:
                weighted_probabilities += weight * pred['probabilities']
            
            total_weight += weight
        
        # Normalize
        weighted_probabilities /= total_weight
        
        # Get final predictions
        predicted_classes = np.argmax(weighted_probabilities, axis=1)
        confidence_scores = np.max(weighted_probabilities, axis=1)
        
        return {
            'predictions': predicted_classes,
            'confidence_scores': confidence_scores,
            'probabilities': weighted_probabilities
        }
    
    def _combine_segmentation_predictions(self, segmentation_predictions: Dict[str, Dict]) -> Dict:
        """Combine segmentation predictions"""
        
        # Weighted average of probabilities
        weighted_probabilities = None
        total_weight = 0
        
        for model_type, pred in segmentation_predictions.items():
            weight = self.weights.get(model_type, 1.0)
            
            if weighted_probabilities is None:
                weighted_probabilities = weight * pred['probabilities']
            else:
                weighted_probabilities += weight * pred['probabilities']
            
            total_weight += weight
        
        # Normalize
        weighted_probabilities /= total_weight
        
        # Get final predictions
        predicted_classes = np.argmax(weighted_probabilities, axis=1)
        
        return {
            'predictions': predicted_classes,
            'probabilities': weighted_probabilities
        }
    
    def _combine_depth_predictions(self, depth_predictions: Dict[str, Dict]) -> Dict:
        """Combine depth predictions"""
        
        # Weighted average of depth maps
        weighted_depth = None
        total_weight = 0
        
        for model_type, pred in depth_predictions.items():
            weight = self.weights.get(model_type, 1.0)
            
            if weighted_depth is None:
                weighted_depth = weight * pred['depth_map']
            else:
                weighted_depth += weight * pred['depth_map']
            
            total_weight += weight
        
        # Normalize
        weighted_depth /= total_weight
        
        return {
            'depth_map': weighted_depth
        }
    
    def _apply_nms(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5) -> List[int]:
        """Apply Non-Maximum Suppression"""
        
        if len(boxes) == 0:
            return []
        
        # Sort by scores
        sorted_indices = np.argsort(scores)[::-1]
        
        keep_indices = []
        
        while len(sorted_indices) > 0:
            # Take the box with highest score
            current_idx = sorted_indices[0]
            keep_indices.append(current_idx)
            
            if len(sorted_indices) == 1:
                break
            
            # Calculate IoU with remaining boxes
            current_box = boxes[current_idx]
            remaining_boxes = boxes[sorted_indices[1:]]
            
            ious = self._calculate_iou_batch(current_box, remaining_boxes)
            
            # Keep boxes with IoU below threshold
            keep_mask = ious < iou_threshold
            sorted_indices = sorted_indices[1:][keep_mask]
        
        return keep_indices
    
    def _calculate_iou_batch(self, box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Calculate IoU between one box and multiple boxes"""
        
        # Calculate intersection
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        # Calculate areas
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        # Calculate IoU
        union = box_area + boxes_area - intersection
        iou = intersection / (union + 1e-6)
        
        return iou
    
    def _add_confidence_scores(self, ensemble_prediction: Dict, individual_predictions: Dict) -> Dict:
        """Add confidence scores to ensemble predictions"""
        
        # Calculate ensemble confidence based on agreement between models
        if 'detection' in ensemble_prediction:
            # For detection, confidence is based on score and model agreement
            detection_pred = ensemble_prediction['detection']
            if len(detection_pred['scores']) > 0:
                # Boost confidence for predictions from multiple models
                model_agreement = np.bincount(detection_pred['model_sources'])
                max_agreement = np.max(model_agreement)
                
                # Adjust scores based on agreement
                agreement_boost = 1.0 + 0.1 * (max_agreement - 1)
                detection_pred['scores'] *= agreement_boost
                detection_pred['ensemble_confidence'] = detection_pred['scores']
        
        if 'classification' in ensemble_prediction:
            # For classification, confidence is already calculated
            pass
        
        return ensemble_prediction
    
    def update_weights_based_on_performance(self, performance_metrics: Dict[str, float]):
        """Update ensemble weights based on performance metrics"""
        
        # Simple performance-based weight update
        total_performance = sum(performance_metrics.values())
        
        for model_type in self.weights:
            if model_type in performance_metrics:
                # Higher performance gets higher weight
                self.weights[model_type] = performance_metrics[model_type] / total_performance
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        self.weights = {k: v / total_weight for k, v in self.weights.items()}
        
        self.logger.info(f"Updated ensemble weights: {self.weights}")
    
    def save_ensemble(self, save_path: str):
        """Save ensemble models and configuration"""
        
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save individual models
        for model_type, model in self.models.items():
            model_path = save_dir / f"{model_type}_model.pth"
            
            if hasattr(model, 'state_dict'):
                torch.save(model.state_dict(), model_path)
            else:
                # For TensorFlow models
                try:
                    model.save(str(save_dir / f"{model_type}_model"))
                except:
                    self.logger.warning(f"Could not save {model_type} model")
        
        # Save ensemble configuration
        ensemble_config = {
            'model_types': self.model_types,
            'ensemble_method': self.ensemble_method,
            'weights': self.weights,
            'config': self.config
        }
        
        config_path = save_dir / "ensemble_config.json"
        import json
        with open(config_path, 'w') as f:
            json.dump(ensemble_config, f, indent=2)
        
        self.logger.info(f"Saved ensemble to {save_path}")
    
    def load_ensemble(self, load_path: str):
        """Load ensemble models and configuration"""
        
        load_dir = Path(load_path)
        
        # Load ensemble configuration
        config_path = load_dir / "ensemble_config.json"
        if config_path.exists():
            import json
            with open(config_path, 'r') as f:
                ensemble_config = json.load(f)
            
            self.model_types = ensemble_config['model_types']
            self.ensemble_method = ensemble_config['ensemble_method']
            self.weights = ensemble_config['weights']
        
        # Load individual models
        for model_type, model in self.models.items():
            model_path = load_dir / f"{model_type}_model.pth"
            
            if model_path.exists() and hasattr(model, 'load_state_dict'):
                model.load_state_dict(torch.load(model_path))
                self.logger.info(f"Loaded {model_type} model")
        
        self.logger.info(f"Loaded ensemble from {load_path}")


# Factory function
def create_model_ensemble(config: Dict, 
                         model_types: List[str] = ['yolov8', 'retinanet', 'efficientdet'],
                         ensemble_method: str = 'weighted_average') -> ModelEnsemble:
    """Create model ensemble"""
    
    ensemble = ModelEnsemble(
        config=config,
        model_types=model_types,
        ensemble_method=ensemble_method
    )
    
    return ensemble 