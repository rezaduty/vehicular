"""
Model architectures for autonomous driving perception
"""

from .object_detection import YOLOv8Detector, EfficientDetector
from .segmentation import DeepLabV3Plus, UNet
from .tracking import DeepSORT, MOT
from .domain_adaptation import DomainAdversarialNetwork, DANN
from .autoencoder import ImageAutoencoder, LiDARAutoencoder
from .depth_estimation import DepthEstimator, MonoDepth

__all__ = [
    'YOLOv8Detector',
    'EfficientDetector', 
    'DeepLabV3Plus',
    'UNet',
    'DeepSORT',
    'MOT',
    'DomainAdversarialNetwork',
    'DANN',
    'ImageAutoencoder',
    'LiDARAutoencoder',
    'DepthEstimator',
    'MonoDepth'
] 