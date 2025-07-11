"""
Model architectures for autonomous driving perception
Includes PyTorch, TensorFlow, ensemble, and domain adaptation models
"""

# PyTorch models
from .object_detection import YOLOv8Detector, EfficientDetector, ParallelPatchDetector
from .segmentation import DeepLabV3Plus, UNet
from .tracking import DeepSORT, MOT
from .domain_adaptation import DomainAdversarialNetwork, DANN
from .autoencoder import ImageAutoencoder, LiDARAutoencoder, VariationalAutoencoder, ConditionalAutoencoder
from .depth_estimation import DepthEstimator, MonoDepth, DepthVelocityTracker

# Advanced PyTorch models
from .pytorch_models import (
    CORALDomainAdaptation, 
    MMDDomainAdaptation, 
    MonoDepth2, 
    AdvancedDomainAdversarialNetwork,
    MultiScaleFeatureExtractor,
    create_coral_model,
    create_mmd_model,
    create_monodepth2_model,
    create_advanced_dann_model
)

# TensorFlow models
from .tensorflow_models import (
    RetinaNet,
    EfficientDet, 
    DeepLabV3PlusTF,
    create_retinanet,
    create_efficientdet,
    create_deeplabv3plus
)

# Ensemble and domain adaptation
from .model_ensemble import ModelEnsemble, create_model_ensemble
from .domain_adaptation_pipeline import (
    DomainAdaptationPipeline, 
    DomainAdaptationTrainer,
    create_domain_adaptation_pipeline
)

# Unsupervised models
from .unsupervised.lost import LOSTDetector, SelfSupervisedObjectDetection
from .unsupervised.most import MOSTTracker, MultiObjectSelfSupervisedTracking
from .unsupervised.sonata import SONATA, SONATAVisualizer, SONATAEvaluator

__all__ = [
    # PyTorch Object Detection
    'YOLOv8Detector',
    'EfficientDetector',
    'ParallelPatchDetector',
    
    # PyTorch Segmentation
    'DeepLabV3Plus',
    'UNet',
    
    # PyTorch Tracking
    'DeepSORT',
    'MOT',
    
    # PyTorch Domain Adaptation
    'DomainAdversarialNetwork',
    'DANN',
    'CORALDomainAdaptation',
    'MMDDomainAdaptation',
    'AdvancedDomainAdversarialNetwork',
    'MultiScaleFeatureExtractor',
    
    # PyTorch Autoencoder
    'ImageAutoencoder',
    'LiDARAutoencoder',
    'VariationalAutoencoder',
    'ConditionalAutoencoder',
    
    # PyTorch Depth Estimation
    'DepthEstimator',
    'MonoDepth',
    'MonoDepth2',
    'DepthVelocityTracker',
    
    # TensorFlow Models
    'RetinaNet',
    'EfficientDet',
    'DeepLabV3PlusTF',
    
    # Ensemble and Pipeline
    'ModelEnsemble',
    'DomainAdaptationPipeline',
    'DomainAdaptationTrainer',
    
    # Unsupervised Models
    'LOSTDetector',
    'SelfSupervisedObjectDetection',
    'MOSTTracker',
    'MultiObjectSelfSupervisedTracking',
    'SONATA',
    'SONATAVisualizer',
    'SONATAEvaluator',
    
    # Factory Functions
    'create_coral_model',
    'create_mmd_model',
    'create_monodepth2_model',
    'create_advanced_dann_model',
    'create_retinanet',
    'create_efficientdet',
    'create_deeplabv3plus',
    'create_model_ensemble',
    'create_domain_adaptation_pipeline'
]


# Model registry for dynamic loading
MODEL_REGISTRY = {
    # Object Detection
    'yolov8': YOLOv8Detector,
    'efficientdet_pytorch': EfficientDetector,
    'parallel_patch_detector': ParallelPatchDetector,
    
    # TensorFlow Object Detection
    'retinanet': RetinaNet,
    'efficientdet': EfficientDet,
    
    # Segmentation
    'deeplabv3plus': DeepLabV3Plus,
    'deeplabv3plus_tf': DeepLabV3PlusTF,
    'unet': UNet,
    
    # Tracking
    'deepsort': DeepSORT,
    'mot': MOT,
    
    # Domain Adaptation
    'dann': DomainAdversarialNetwork,
    'advanced_dann': AdvancedDomainAdversarialNetwork,
    'coral': CORALDomainAdaptation,
    'mmd': MMDDomainAdaptation,
    
    # Depth Estimation
    'depth_estimator': DepthEstimator,
    'monodepth': MonoDepth,
    'monodepth2': MonoDepth2,
    
    # Autoencoder
    'image_autoencoder': ImageAutoencoder,
    'lidar_autoencoder': LiDARAutoencoder,
    'vae': VariationalAutoencoder,
    'conditional_autoencoder': ConditionalAutoencoder,
    
    # Unsupervised
    'lost': LOSTDetector,
    'most': MOSTTracker,
    'sonata': SONATA,
    
    # Ensemble
    'model_ensemble': ModelEnsemble,
    'domain_adaptation_pipeline': DomainAdaptationPipeline
}


def get_model(model_name: str, config: dict = None, **kwargs):
    """
    Get model by name from registry
    
    Args:
        model_name: Name of the model
        config: Configuration dictionary
        **kwargs: Additional arguments for model initialization
        
    Returns:
        Model instance
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not found in registry. Available models: {list(MODEL_REGISTRY.keys())}")
    
    model_class = MODEL_REGISTRY[model_name]
    
    # Handle different model initialization patterns
    if model_name in ['yolov8', 'efficientdet_pytorch', 'parallel_patch_detector']:
        # Object detection models
        if config:
            num_classes = config.get('models', {}).get('object_detection', {}).get('num_classes', 10)
            confidence_threshold = config.get('models', {}).get('object_detection', {}).get('confidence_threshold', 0.5)
            return model_class(num_classes=num_classes, confidence_threshold=confidence_threshold, **kwargs)
        else:
            return model_class(**kwargs)
    
    elif model_name in ['retinanet', 'efficientdet']:
        # TensorFlow models
        if config:
            num_classes = config.get('models', {}).get('object_detection', {}).get('num_classes', 10)
            input_shape = (
                config.get('data', {}).get('image', {}).get('height', 384),
                config.get('data', {}).get('image', {}).get('width', 1280),
                config.get('data', {}).get('image', {}).get('channels', 3)
            )
            return model_class(num_classes=num_classes, input_shape=input_shape, **kwargs)
        else:
            return model_class(**kwargs)
    
    elif model_name in ['deeplabv3plus', 'deeplabv3plus_tf', 'unet']:
        # Segmentation models
        if config:
            num_classes = config.get('models', {}).get('segmentation', {}).get('num_classes', 19)
            return model_class(num_classes=num_classes, **kwargs)
        else:
            return model_class(**kwargs)
    
    elif model_name in ['dann', 'advanced_dann', 'coral', 'mmd']:
        # Domain adaptation models
        if config:
            num_classes = config.get('models', {}).get('object_detection', {}).get('num_classes', 10)
            return model_class(num_classes=num_classes, **kwargs)
        else:
            return model_class(**kwargs)
    
    elif model_name == 'model_ensemble':
        # Ensemble model
        if config:
            return create_model_ensemble(config, **kwargs)
        else:
            return model_class(**kwargs)
    
    elif model_name == 'domain_adaptation_pipeline':
        # Domain adaptation pipeline
        if config:
            return create_domain_adaptation_pipeline(config, **kwargs)
        else:
            return model_class(**kwargs)
    
    else:
        # Default initialization
        return model_class(**kwargs)


def list_available_models():
    """List all available models in the registry"""
    return list(MODEL_REGISTRY.keys())


def get_model_info(model_name: str):
    """Get information about a specific model"""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not found in registry")
    
    model_class = MODEL_REGISTRY[model_name]
    
    return {
        'name': model_name,
        'class': model_class.__name__,
        'module': model_class.__module__,
        'docstring': model_class.__doc__
    }


# Model categories for organization
MODEL_CATEGORIES = {
    'object_detection': [
        'yolov8', 'efficientdet_pytorch', 'parallel_patch_detector',
        'retinanet', 'efficientdet'
    ],
    'segmentation': [
        'deeplabv3plus', 'deeplabv3plus_tf', 'unet'
    ],
    'tracking': [
        'deepsort', 'mot'
    ],
    'domain_adaptation': [
        'dann', 'advanced_dann', 'coral', 'mmd'
    ],
    'depth_estimation': [
        'depth_estimator', 'monodepth', 'monodepth2'
    ],
    'autoencoder': [
        'image_autoencoder', 'lidar_autoencoder', 'vae', 'conditional_autoencoder'
    ],
    'unsupervised': [
        'lost', 'most', 'sonata'
    ],
    'ensemble': [
        'model_ensemble', 'domain_adaptation_pipeline'
    ]
}


def get_models_by_category(category: str):
    """Get models by category"""
    if category not in MODEL_CATEGORIES:
        raise ValueError(f"Category '{category}' not found. Available categories: {list(MODEL_CATEGORIES.keys())}")
    
    return MODEL_CATEGORIES[category]


def get_all_categories():
    """Get all available model categories"""
    return list(MODEL_CATEGORIES.keys()) 