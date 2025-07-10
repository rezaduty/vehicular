"""
Domain adaptation models for autonomous driving
Transfer learning from simulation (CARLA) to real-world data (KITTI)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
from typing import Dict, List, Tuple, Optional


class GradientReversalFunction(Function):
    """Gradient Reversal Layer implementation"""
    
    @staticmethod
    def forward(ctx, x, lambda_grl):
        ctx.lambda_grl = lambda_grl
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lambda_grl
        return output, None


class GradientReversalLayer(nn.Module):
    """Gradient Reversal Layer for domain adversarial training"""
    
    def __init__(self, lambda_grl: float = 1.0):
        super().__init__()
        self.lambda_grl = lambda_grl
        
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_grl)
    
    def set_lambda(self, lambda_grl: float):
        self.lambda_grl = lambda_grl


class DomainClassifier(nn.Module):
    """Domain classifier for adversarial training"""
    
    def __init__(self, 
                 input_dim: int = 256,
                 hidden_dim: int = 512,
                 num_domains: int = 2):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, num_domains)
        )
        
    def forward(self, x):
        # Global average pooling if input is feature maps
        if len(x.shape) == 4:
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = x.view(x.size(0), -1)
        
        return self.classifier(x)


class FeatureExtractor(nn.Module):
    """Shared feature extractor for domain adaptation"""
    
    def __init__(self, 
                 backbone: str = 'resnet50',
                 pretrained: bool = True,
                 feature_dim: int = 256):
        super().__init__()
        
        if backbone == 'resnet50':
            import torchvision.models as models
            self.backbone = models.resnet50(pretrained=pretrained)
            # Remove the final classification layer
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
            backbone_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Feature projection layer
        self.feature_proj = nn.Sequential(
            nn.Conv2d(backbone_dim, feature_dim, 1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        features = self.feature_proj(features)
        return features


class DomainAdversarialNetwork(nn.Module):
    """Domain Adversarial Neural Network (DANN) for object detection"""
    
    def __init__(self,
                 num_classes: int = 10,
                 num_domains: int = 2,
                 backbone: str = 'resnet50',
                 feature_dim: int = 256,
                 lambda_grl: float = 1.0):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.lambda_grl = lambda_grl
        
        # Shared feature extractor
        self.feature_extractor = FeatureExtractor(
            backbone=backbone,
            feature_dim=feature_dim
        )
        
        # Task-specific predictor (object detection)
        self.object_detector = ObjectDetectionHead(
            feature_dim=feature_dim,
            num_classes=num_classes
        )
        
        # Domain classifier with gradient reversal
        self.gradient_reversal = GradientReversalLayer(lambda_grl)
        self.domain_classifier = DomainClassifier(
            input_dim=feature_dim,
            num_domains=num_domains
        )
        
        # Losses
        self.detection_loss = DetectionLoss()
        self.domain_loss = nn.CrossEntropyLoss()
        
    def forward(self, x, domain_labels=None, targets=None):
        """Forward pass for training or inference"""
        
        # Extract shared features
        features = self.feature_extractor(x)
        
        # Object detection predictions
        detection_outputs = self.object_detector(features)
        
        if self.training and domain_labels is not None:
            # Domain classification with gradient reversal
            domain_features = self.gradient_reversal(features)
            domain_outputs = self.domain_classifier(domain_features)
            
            # Compute losses
            losses = {}
            
            if targets is not None:
                losses['detection_loss'] = self.detection_loss(
                    detection_outputs, targets
                )
            
            losses['domain_loss'] = self.domain_loss(
                domain_outputs, domain_labels
            )
            
            return detection_outputs, losses
        else:
            return detection_outputs
    
    def set_lambda_grl(self, lambda_grl: float):
        """Update gradient reversal lambda parameter"""
        self.lambda_grl = lambda_grl
        self.gradient_reversal.set_lambda(lambda_grl)


class ObjectDetectionHead(nn.Module):
    """Object detection head for DANN"""
    
    def __init__(self, 
                 feature_dim: int = 256,
                 num_classes: int = 10,
                 num_anchors: int = 9):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Classification head
        self.cls_head = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, num_anchors * num_classes, 3, padding=1)
        )
        
        # Regression head
        self.reg_head = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, num_anchors * 4, 3, padding=1)
        )
        
    def forward(self, features):
        """Forward pass"""
        cls_logits = self.cls_head(features)
        bbox_regression = self.reg_head(features)
        
        # Reshape outputs
        batch_size = features.size(0)
        
        cls_logits = cls_logits.view(
            batch_size, self.num_anchors, self.num_classes, -1
        ).permute(0, 3, 1, 2).contiguous()
        
        bbox_regression = bbox_regression.view(
            batch_size, self.num_anchors, 4, -1
        ).permute(0, 3, 1, 2).contiguous()
        
        return {
            'cls_logits': cls_logits,
            'bbox_regression': bbox_regression
        }


class DetectionLoss(nn.Module):
    """Detection loss for object detection head"""
    
    def __init__(self, 
                 alpha: float = 0.25,
                 gamma: float = 2.0,
                 beta: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        
    def forward(self, predictions, targets):
        """Compute detection loss"""
        cls_logits = predictions['cls_logits']
        bbox_regression = predictions['bbox_regression']
        
        # Focal loss for classification
        cls_loss = self._focal_loss(cls_logits, targets)
        
        # Smooth L1 loss for regression
        reg_loss = self._smooth_l1_loss(bbox_regression, targets)
        
        total_loss = cls_loss + self.beta * reg_loss
        
        return {
            'total_loss': total_loss,
            'cls_loss': cls_loss,
            'reg_loss': reg_loss
        }
    
    def _focal_loss(self, inputs, targets):
        """Focal loss implementation"""
        # Simplified focal loss - would need proper implementation
        # with anchor matching for real training
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
    
    def _smooth_l1_loss(self, inputs, targets):
        """Smooth L1 loss for bounding box regression"""
        # Simplified - would need proper anchor matching
        return F.smooth_l1_loss(inputs, targets, reduction='mean')


class DANN(DomainAdversarialNetwork):
    """Alias for DomainAdversarialNetwork for backward compatibility"""
    pass


class DomainAdapter:
    """Domain adaptation trainer and utilities"""
    
    def __init__(self, 
                 model: DomainAdversarialNetwork,
                 source_dataloader,
                 target_dataloader,
                 config: Dict):
        self.model = model
        self.source_dataloader = source_dataloader
        self.target_dataloader = target_dataloader
        self.config = config
        
        # Optimizers
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Scheduler for lambda_grl
        self.lambda_scheduler = LambdaScheduler()
        
    def train_epoch(self, epoch: int):
        """Train one epoch with domain adaptation"""
        self.model.train()
        
        source_iter = iter(self.source_dataloader)
        target_iter = iter(self.target_dataloader)
        
        total_loss = 0.0
        num_batches = min(len(self.source_dataloader), len(self.target_dataloader))
        
        for batch_idx in range(num_batches):
            # Update lambda for gradient reversal
            p = float(batch_idx + epoch * num_batches) / (
                self.config['training']['epochs'] * num_batches
            )
            lambda_grl = self.lambda_scheduler.get_lambda(p)
            self.model.set_lambda_grl(lambda_grl)
            
            # Get source batch
            try:
                source_batch = next(source_iter)
            except StopIteration:
                source_iter = iter(self.source_dataloader)
                source_batch = next(source_iter)
            
            # Get target batch
            try:
                target_batch = next(target_iter)
            except StopIteration:
                target_iter = iter(self.target_dataloader)
                target_batch = next(target_iter)
            
            # Combine batches
            images = torch.cat([source_batch['images'], target_batch['images']])
            domain_labels = torch.cat([
                torch.zeros(len(source_batch['images'])),  # Source domain = 0
                torch.ones(len(target_batch['images']))    # Target domain = 1
            ]).long()
            
            # Forward pass
            if torch.cuda.is_available():
                images = images.cuda()
                domain_labels = domain_labels.cuda()
            
            self.optimizer.zero_grad()
            
            outputs, losses = self.model(
                images, 
                domain_labels=domain_labels, 
                targets=source_batch.get('labels')
            )
            
            # Backward pass
            total_batch_loss = sum(losses.values())
            total_batch_loss.backward()
            self.optimizer.step()
            
            total_loss += total_batch_loss.item()
            
        return total_loss / num_batches
    
    def evaluate(self, dataloader):
        """Evaluate model on given dataloader"""
        self.model.eval()
        
        total_loss = 0.0
        predictions = []
        
        with torch.no_grad():
            for batch in dataloader:
                images = batch['images']
                if torch.cuda.is_available():
                    images = images.cuda()
                
                outputs = self.model(images)
                predictions.extend(outputs)
                
        return predictions


class LambdaScheduler:
    """Lambda scheduler for gradient reversal layer"""
    
    def __init__(self, gamma: float = 10.0):
        self.gamma = gamma
        
    def get_lambda(self, p: float) -> float:
        """Get lambda value based on training progress p ∈ [0, 1]"""
        return 2.0 / (1.0 + np.exp(-self.gamma * p)) - 1.0


class DomainDiscrepancyLoss(nn.Module):
    """Maximum Mean Discrepancy (MMD) loss for domain adaptation"""
    
    def __init__(self, kernel_type: str = 'rbf', gamma: float = 1.0):
        super().__init__()
        self.kernel_type = kernel_type
        self.gamma = gamma
        
    def forward(self, source_features, target_features):
        """Compute MMD loss between source and target features"""
        return self.mmd_loss(source_features, target_features)
    
    def gaussian_kernel(self, x, y, gamma):
        """Gaussian RBF kernel"""
        x_size = x.size(0)
        y_size = y.size(0)
        dim = x.size(1)
        
        x = x.unsqueeze(1)  # (x_size, 1, dim)
        y = y.unsqueeze(0)  # (1, y_size, dim)
        
        tiled_x = x.expand(x_size, y_size, dim)
        tiled_y = y.expand(x_size, y_size, dim)
        
        kernel_input = (tiled_x - tiled_y).pow(2).mean(2) / float(dim)
        return torch.exp(-gamma * kernel_input)
    
    def mmd_loss(self, source, target):
        """Maximum Mean Discrepancy loss"""
        # Flatten features if needed
        if len(source.shape) > 2:
            source = source.view(source.size(0), -1)
            target = target.view(target.size(0), -1)
        
        delta = source.size(0) - target.size(0)
        if delta > 0:
            target = F.pad(target, (0, 0, 0, delta))
        elif delta < 0:
            source = F.pad(source, (0, 0, 0, -delta))
        
        xx = self.gaussian_kernel(source, source, self.gamma)
        yy = self.gaussian_kernel(target, target, self.gamma)
        xy = self.gaussian_kernel(source, target, self.gamma)
        
        mmd = xx.mean() + yy.mean() - 2 * xy.mean()
        return mmd 