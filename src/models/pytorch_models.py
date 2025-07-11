"""
Advanced PyTorch models for autonomous driving perception
Includes CORAL, MMD, MonoDepth2, and other domain adaptation techniques
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import math


class CORALDomainAdaptation(nn.Module):
    """CORAL (Correlation Alignment) for domain adaptation"""
    
    def __init__(self, 
                 backbone: str = 'resnet50',
                 num_classes: int = 10,
                 feature_dim: int = 256,
                 coral_weight: float = 1.0):
        super().__init__()
        
        self.coral_weight = coral_weight
        
        # Feature extractor
        self.feature_extractor = self._build_backbone(backbone)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def _build_backbone(self, backbone_name: str):
        """Build backbone network"""
        if backbone_name == 'resnet50':
            backbone = models.resnet50(pretrained=True)
            # Remove final classification layer
            backbone = nn.Sequential(*list(backbone.children())[:-1])
        elif backbone_name == 'resnet18':
            backbone = models.resnet18(pretrained=True)
            backbone = nn.Sequential(*list(backbone.children())[:-1])
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        return backbone
    
    def forward(self, source_data, target_data=None):
        """Forward pass for CORAL domain adaptation"""
        
        # Extract features
        source_features = self.feature_extractor(source_data)
        source_features = source_features.view(source_features.size(0), -1)
        
        # Classification
        source_predictions = self.classifier(source_features)
        
        if target_data is not None and self.training:
            # Extract target features
            target_features = self.feature_extractor(target_data)
            target_features = target_features.view(target_features.size(0), -1)
            
            # Compute CORAL loss
            coral_loss = self.coral_loss(source_features, target_features)
            
            return {
                'predictions': source_predictions,
                'coral_loss': coral_loss,
                'source_features': source_features,
                'target_features': target_features
            }
        
        return {'predictions': source_predictions}
    
    def coral_loss(self, source_features, target_features):
        """Compute CORAL loss"""
        # Compute covariance matrices
        source_cov = self._compute_covariance(source_features)
        target_cov = self._compute_covariance(target_features)
        
        # Frobenius norm of difference
        loss = torch.norm(source_cov - target_cov, p='fro') ** 2
        loss = loss / (4 * source_features.size(1) ** 2)
        
        return self.coral_weight * loss
    
    def _compute_covariance(self, features):
        """Compute covariance matrix"""
        n = features.size(0)
        
        # Center the features
        mean = torch.mean(features, dim=0, keepdim=True)
        centered = features - mean
        
        # Compute covariance
        cov = torch.mm(centered.t(), centered) / (n - 1)
        
        return cov


class MMDDomainAdaptation(nn.Module):
    """Maximum Mean Discrepancy (MMD) for domain adaptation"""
    
    def __init__(self,
                 backbone: str = 'resnet50',
                 num_classes: int = 10,
                 feature_dim: int = 256,
                 mmd_weight: float = 1.0,
                 kernel_type: str = 'rbf',
                 kernel_num: int = 5):
        super().__init__()
        
        self.mmd_weight = mmd_weight
        self.kernel_type = kernel_type
        self.kernel_num = kernel_num
        
        # Feature extractor
        self.feature_extractor = self._build_backbone(backbone)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # Kernel parameters
        if kernel_type == 'rbf':
            self.kernel_bandwidth = nn.Parameter(torch.ones(kernel_num))
        
    def _build_backbone(self, backbone_name: str):
        """Build backbone network"""
        if backbone_name == 'resnet50':
            backbone = models.resnet50(pretrained=True)
            backbone = nn.Sequential(*list(backbone.children())[:-1])
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        return backbone
    
    def forward(self, source_data, target_data=None):
        """Forward pass for MMD domain adaptation"""
        
        # Extract features
        source_features = self.feature_extractor(source_data)
        source_features = source_features.view(source_features.size(0), -1)
        
        # Classification
        source_predictions = self.classifier(source_features)
        
        if target_data is not None and self.training:
            # Extract target features
            target_features = self.feature_extractor(target_data)
            target_features = target_features.view(target_features.size(0), -1)
            
            # Compute MMD loss
            mmd_loss = self.mmd_loss(source_features, target_features)
            
            return {
                'predictions': source_predictions,
                'mmd_loss': mmd_loss,
                'source_features': source_features,
                'target_features': target_features
            }
        
        return {'predictions': source_predictions}
    
    def mmd_loss(self, source_features, target_features):
        """Compute MMD loss"""
        if self.kernel_type == 'rbf':
            return self._rbf_mmd(source_features, target_features)
        elif self.kernel_type == 'linear':
            return self._linear_mmd(source_features, target_features)
        else:
            raise ValueError(f"Unsupported kernel type: {self.kernel_type}")
    
    def _rbf_mmd(self, source, target):
        """Compute RBF MMD"""
        source_size = source.size(0)
        target_size = target.size(0)
        
        # Compute pairwise distances
        source_source = self._compute_kernel_matrix(source, source)
        target_target = self._compute_kernel_matrix(target, target)
        source_target = self._compute_kernel_matrix(source, target)
        
        # MMD computation
        mmd = (torch.sum(source_source) / (source_size ** 2) +
               torch.sum(target_target) / (target_size ** 2) -
               2 * torch.sum(source_target) / (source_size * target_size))
        
        return self.mmd_weight * mmd
    
    def _compute_kernel_matrix(self, x, y):
        """Compute RBF kernel matrix"""
        x_size = x.size(0)
        y_size = y.size(0)
        dim = x.size(1)
        
        # Expand dimensions for broadcasting
        x = x.unsqueeze(1).expand(x_size, y_size, dim)
        y = y.unsqueeze(0).expand(x_size, y_size, dim)
        
        # Compute squared distances
        distances = torch.sum((x - y) ** 2, dim=2)
        
        # Apply RBF kernel with multiple bandwidths
        kernel_matrix = torch.zeros_like(distances)
        for bandwidth in self.kernel_bandwidth:
            kernel_matrix += torch.exp(-distances / bandwidth)
        
        return kernel_matrix / self.kernel_num
    
    def _linear_mmd(self, source, target):
        """Compute linear MMD"""
        source_mean = torch.mean(source, dim=0)
        target_mean = torch.mean(target, dim=0)
        
        mmd = torch.norm(source_mean - target_mean, p=2) ** 2
        
        return self.mmd_weight * mmd


class MonoDepth2(nn.Module):
    """MonoDepth2 implementation for self-supervised depth estimation"""
    
    def __init__(self,
                 num_layers: int = 18,
                 pretrained: bool = True,
                 num_input_images: int = 1,
                 num_poses: int = 1,
                 frame_ids: List[int] = [0, -1, 1]):
        super().__init__()
        
        self.num_layers = num_layers
        self.num_input_images = num_input_images
        self.num_poses = num_poses
        self.frame_ids = frame_ids
        
        # Depth encoder
        self.depth_encoder = DepthEncoder(num_layers, pretrained)
        
        # Depth decoder
        self.depth_decoder = DepthDecoder(self.depth_encoder.num_ch_enc)
        
        # Pose encoder
        self.pose_encoder = PoseEncoder(num_input_images, num_layers, pretrained)
        
        # Pose decoder
        self.pose_decoder = PoseDecoder(self.pose_encoder.num_ch_enc, num_poses)
        
    def forward(self, inputs):
        """Forward pass for MonoDepth2"""
        outputs = {}
        
        # Depth prediction
        features = self.depth_encoder(inputs[("color", 0, 0)])
        outputs.update(self.depth_decoder(features))
        
        if self.training:
            # Pose prediction
            pose_inputs = [inputs[("color", i, 0)] for i in self.frame_ids]
            pose_inputs = torch.cat(pose_inputs, 1)
            
            pose_features = self.pose_encoder(pose_inputs)
            outputs.update(self.pose_decoder(pose_features))
        
        return outputs


class DepthEncoder(nn.Module):
    """Depth encoder for MonoDepth2"""
    
    def __init__(self, num_layers, pretrained):
        super().__init__()
        
        self.num_layers = num_layers
        
        if num_layers == 18:
            self.encoder = models.resnet18(pretrained=pretrained)
        elif num_layers == 34:
            self.encoder = models.resnet34(pretrained=pretrained)
        elif num_layers == 50:
            self.encoder = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported num_layers: {num_layers}")
        
        # Remove final layers
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])
        
        # Number of channels at each layer
        if num_layers in [18, 34]:
            self.num_ch_enc = [64, 64, 128, 256, 512]
        else:
            self.num_ch_enc = [64, 256, 512, 1024, 2048]
    
    def forward(self, input_image):
        """Forward pass"""
        features = []
        x = input_image
        
        # Extract features at multiple scales
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i in [0, 4, 5, 6, 7]:  # After conv1, layer1, layer2, layer3, layer4
                features.append(x)
        
        return features


class DepthDecoder(nn.Module):
    """Depth decoder for MonoDepth2"""
    
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1):
        super().__init__()
        
        self.num_output_channels = num_output_channels
        self.scales = scales
        
        self.num_ch_dec = [16, 32, 64, 128, 256]
        
        # Decoder layers
        self.convs = nn.ModuleDict()
        
        for i in range(4, -1, -1):
            # Upconv layers
            num_ch_in = num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            
            self.convs[f"upconv_{i}_0"] = ConvBlock(num_ch_in, num_ch_out)
            
            # Skip connection layers
            num_ch_in = self.num_ch_dec[i]
            if i > 0:
                num_ch_in += num_ch_enc[i - 1]
            
            num_ch_out = self.num_ch_dec[i]
            self.convs[f"upconv_{i}_1"] = ConvBlock(num_ch_in, num_ch_out)
        
        # Disparity layers
        for s in self.scales:
            self.convs[f"dispconv_{s}"] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)
    
    def forward(self, input_features):
        """Forward pass"""
        outputs = {}
        
        # Decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[f"upconv_{i}_0"](x)
            x = [F.interpolate(x, scale_factor=2, mode="nearest")]
            
            if i > 0:
                x += [input_features[i - 1]]
            
            x = torch.cat(x, 1)
            x = self.convs[f"upconv_{i}_1"](x)
            
            # Generate disparity at this scale
            if i in self.scales:
                outputs[("disp", i)] = torch.sigmoid(self.convs[f"dispconv_{i}"](x))
        
        return outputs


class PoseEncoder(nn.Module):
    """Pose encoder for MonoDepth2"""
    
    def __init__(self, num_input_images, num_layers, pretrained):
        super().__init__()
        
        self.num_input_images = num_input_images
        
        if num_layers == 18:
            self.encoder = models.resnet18(pretrained=pretrained)
        elif num_layers == 34:
            self.encoder = models.resnet34(pretrained=pretrained)
        elif num_layers == 50:
            self.encoder = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported num_layers: {num_layers}")
        
        # Modify first layer for multiple input images
        if num_input_images > 1:
            self.encoder.conv1 = nn.Conv2d(
                num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        
        # Remove final layers
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])
        
        # Number of channels
        if num_layers in [18, 34]:
            self.num_ch_enc = [64, 64, 128, 256, 512]
        else:
            self.num_ch_enc = [64, 256, 512, 1024, 2048]
    
    def forward(self, input_image):
        """Forward pass"""
        features = []
        x = input_image
        
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i in [0, 4, 5, 6, 7]:
                features.append(x)
        
        return features


class PoseDecoder(nn.Module):
    """Pose decoder for MonoDepth2"""
    
    def __init__(self, num_ch_enc, num_input_features=1, num_frames_to_predict_for=1):
        super().__init__()
        
        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features
        self.num_frames_to_predict_for = num_frames_to_predict_for
        
        self.convs = nn.ModuleDict()
        
        # Squeeze layers
        self.convs["squeeze"] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        
        # Pose prediction layers
        self.convs["pose_0"] = nn.Conv2d(256, 256, 3, 1, 1)
        self.convs["pose_1"] = nn.Conv2d(256, 256, 3, 1, 1)
        self.convs["pose_2"] = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)
        
        self.relu = nn.ReLU()
        
    def forward(self, input_features):
        """Forward pass"""
        last_features = input_features[-1]
        
        # Squeeze
        x = self.convs["squeeze"](last_features)
        x = self.relu(x)
        
        # Pose prediction
        x = self.convs["pose_0"](x)
        x = self.relu(x)
        
        x = self.convs["pose_1"](x)
        x = self.relu(x)
        
        x = self.convs["pose_2"](x)
        
        # Global average pooling
        x = x.mean(3).mean(2)
        
        # Reshape for output
        x = x.view(-1, self.num_frames_to_predict_for, 1, 6)
        
        return {("axisangle", 0, 1): x}


class AdvancedDomainAdversarialNetwork(nn.Module):
    """Advanced DANN with multiple domain discriminators"""
    
    def __init__(self,
                 num_classes: int = 10,
                 num_domains: int = 2,
                 backbone: str = 'resnet50',
                 feature_dim: int = 256,
                 lambda_grl: float = 1.0,
                 use_multiple_discriminators: bool = True):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.lambda_grl = lambda_grl
        self.use_multiple_discriminators = use_multiple_discriminators
        
        # Feature extractor
        self.feature_extractor = self._build_backbone(backbone)
        
        # Task classifier
        self.task_classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # Domain discriminators
        if use_multiple_discriminators:
            # Multiple discriminators for different feature levels
            self.domain_discriminators = nn.ModuleList([
                DomainDiscriminator(feature_dim, num_domains)
                for _ in range(3)  # Low, mid, high level features
            ])
        else:
            self.domain_discriminator = DomainDiscriminator(feature_dim, num_domains)
        
        # Gradient reversal layer
        self.gradient_reversal = GradientReversalLayer(lambda_grl)
        
    def _build_backbone(self, backbone_name: str):
        """Build backbone network"""
        if backbone_name == 'resnet50':
            backbone = models.resnet50(pretrained=True)
            backbone = nn.Sequential(*list(backbone.children())[:-1])
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        return backbone
    
    def forward(self, x, domain_labels=None):
        """Forward pass"""
        # Extract features
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        
        # Task prediction
        task_predictions = self.task_classifier(features)
        
        outputs = {'task_predictions': task_predictions}
        
        if domain_labels is not None and self.training:
            # Domain discrimination
            reversed_features = self.gradient_reversal(features)
            
            if self.use_multiple_discriminators:
                domain_predictions = []
                for discriminator in self.domain_discriminators:
                    domain_pred = discriminator(reversed_features)
                    domain_predictions.append(domain_pred)
                outputs['domain_predictions'] = domain_predictions
            else:
                domain_predictions = self.domain_discriminator(reversed_features)
                outputs['domain_predictions'] = domain_predictions
        
        return outputs


class DomainDiscriminator(nn.Module):
    """Domain discriminator for adversarial training"""
    
    def __init__(self, input_dim: int, num_domains: int):
        super().__init__()
        
        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_domains)
        )
    
    def forward(self, x):
        return self.discriminator(x)


class GradientReversalLayer(torch.autograd.Function):
    """Gradient Reversal Layer"""
    
    @staticmethod
    def forward(ctx, x, lambda_grl):
        ctx.lambda_grl = lambda_grl
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_grl * grad_output, None


class GradientReversalLayerModule(nn.Module):
    """Gradient Reversal Layer as a module"""
    
    def __init__(self, lambda_grl: float = 1.0):
        super().__init__()
        self.lambda_grl = lambda_grl
    
    def forward(self, x):
        return GradientReversalLayer.apply(x, self.lambda_grl)


class MultiScaleFeatureExtractor(nn.Module):
    """Multi-scale feature extractor for domain adaptation"""
    
    def __init__(self, backbone: str = 'resnet50'):
        super().__init__()
        
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
            
            # Extract features at multiple scales
            self.layer1 = nn.Sequential(*list(self.backbone.children())[:5])
            self.layer2 = self.backbone.layer2
            self.layer3 = self.backbone.layer3
            self.layer4 = self.backbone.layer4
            
            # Adaptive pooling for consistent feature sizes
            self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
            
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
    
    def forward(self, x):
        """Extract multi-scale features"""
        # Low-level features
        low_features = self.layer1(x)
        low_features_pooled = self.adaptive_pool(low_features).view(low_features.size(0), -1)
        
        # Mid-level features
        mid_features = self.layer2(low_features)
        mid_features_pooled = self.adaptive_pool(mid_features).view(mid_features.size(0), -1)
        
        # High-level features
        high_features = self.layer3(mid_features)
        high_features = self.layer4(high_features)
        high_features_pooled = self.adaptive_pool(high_features).view(high_features.size(0), -1)
        
        return {
            'low_level': low_features_pooled,
            'mid_level': mid_features_pooled,
            'high_level': high_features_pooled
        }


# Helper layers
class ConvBlock(nn.Module):
    """Convolution block with batch normalization and ReLU"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)
        
    def forward(self, x):
        return self.nonlin(self.conv(x))


class Conv3x3(nn.Module):
    """3x3 convolution with padding"""
    
    def __init__(self, in_channels, out_channels, use_refl=True):
        super().__init__()
        
        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        
        self.conv = nn.Conv2d(in_channels, out_channels, 3)
        
    def forward(self, x):
        return self.conv(self.pad(x))


# Model factory functions
def create_coral_model(num_classes=10, backbone='resnet50', feature_dim=256):
    """Create CORAL domain adaptation model"""
    return CORALDomainAdaptation(
        backbone=backbone,
        num_classes=num_classes,
        feature_dim=feature_dim
    )


def create_mmd_model(num_classes=10, backbone='resnet50', feature_dim=256):
    """Create MMD domain adaptation model"""
    return MMDDomainAdaptation(
        backbone=backbone,
        num_classes=num_classes,
        feature_dim=feature_dim
    )


def create_monodepth2_model(num_layers=18, pretrained=True):
    """Create MonoDepth2 model"""
    return MonoDepth2(
        num_layers=num_layers,
        pretrained=pretrained
    )


def create_advanced_dann_model(num_classes=10, num_domains=2, backbone='resnet50'):
    """Create advanced DANN model"""
    return AdvancedDomainAdversarialNetwork(
        num_classes=num_classes,
        num_domains=num_domains,
        backbone=backbone
    ) 