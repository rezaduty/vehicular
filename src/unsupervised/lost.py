"""
LOST: Localized Object detection using Self-supervised Training
Implementation for unsupervised road object detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
import random
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA


class LOSTDetector(nn.Module):
    """LOST: Localized Object detection using Self-supervised Training"""
    
    def __init__(self,
                 backbone: str = 'resnet50',
                 feature_dim: int = 256,
                 num_proposals: int = 100,
                 confidence_threshold: float = 0.5):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_proposals = num_proposals
        self.confidence_threshold = confidence_threshold
        
        # Feature extractor backbone
        self.backbone = self._build_backbone(backbone)
        
        # Feature projection head
        self.projection_head = nn.Sequential(
            nn.Conv2d(2048, feature_dim, 1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )
        
        # Self-supervised learning components
        self.motion_predictor = MotionPredictor(feature_dim)
        self.temporal_consistency = TemporalConsistency(feature_dim)
        self.pseudo_labeler = PseudoLabeler()
        
        # Object proposal generator
        self.proposal_generator = ObjectProposalGenerator(
            feature_dim=feature_dim,
            num_proposals=num_proposals
        )
        
    def _build_backbone(self, backbone_name: str):
        """Build feature extraction backbone"""
        if backbone_name == 'resnet50':
            import torchvision.models as models
            backbone = models.resnet50(pretrained=True)
            # Remove final classification layers
            backbone = nn.Sequential(*list(backbone.children())[:-2])
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        return backbone
    
    def forward(self, x: torch.Tensor, prev_frame: Optional[torch.Tensor] = None):
        """Forward pass for self-supervised learning"""
        
        # Extract features from current frame
        features = self.backbone(x)
        features = self.projection_head(features)
        
        # Generate object proposals
        proposals = self.proposal_generator(features)
        
        if self.training and prev_frame is not None:
            # Self-supervised learning with temporal consistency
            prev_features = self.backbone(prev_frame)
            prev_features = self.projection_head(prev_features)
            
            # Motion prediction loss
            motion_loss = self.motion_predictor(features, prev_features)
            
            # Temporal consistency loss
            consistency_loss = self.temporal_consistency(features, prev_features)
            
            return {
                'proposals': proposals,
                'features': features,
                'motion_loss': motion_loss,
                'consistency_loss': consistency_loss
            }
        else:
            return {
                'proposals': proposals,
                'features': features
            }
    
    def detect_objects(self, x: torch.Tensor) -> List[Dict]:
        """Detect objects using learned features"""
        self.eval()
        
        with torch.no_grad():
            outputs = self.forward(x)
            proposals = outputs['proposals']
            features = outputs['features']
            
            # Generate pseudo-labels using clustering
            pseudo_labels = self.pseudo_labeler.generate_labels(
                features, proposals
            )
            
            # Filter proposals by confidence
            detections = []
            for i, (proposal_batch, label_batch) in enumerate(zip(proposals, pseudo_labels)):
                valid_proposals = []
                
                for proposal, label in zip(proposal_batch, label_batch):
                    if proposal['confidence'] > self.confidence_threshold:
                        detection = {
                            'bbox': proposal['bbox'],
                            'confidence': proposal['confidence'],
                            'class': label,
                            'features': proposal['features']
                        }
                        valid_proposals.append(detection)
                
                detections.append(valid_proposals)
            
            return detections


class MotionPredictor(nn.Module):
    """Predicts motion between consecutive frames"""
    
    def __init__(self, feature_dim: int = 256):
        super().__init__()
        
        self.motion_net = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, 3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim // 2, 3, padding=1),
            nn.BatchNorm2d(feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim // 2, 2, 3, padding=1)  # 2D motion vectors
        )
        
    def forward(self, current_features: torch.Tensor, prev_features: torch.Tensor):
        """Predict motion vectors between frames"""
        
        # Concatenate current and previous features
        combined = torch.cat([current_features, prev_features], dim=1)
        
        # Predict motion vectors
        motion_vectors = self.motion_net(combined)
        
        # Warp previous features using predicted motion
        warped_prev = self._warp_features(prev_features, motion_vectors)
        
        # Motion consistency loss
        motion_loss = F.mse_loss(warped_prev, current_features)
        
        return motion_loss
    
    def _warp_features(self, features: torch.Tensor, motion_vectors: torch.Tensor):
        """Warp features using motion vectors"""
        B, C, H, W = features.shape
        
        # Create coordinate grid
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, device=features.device),
            torch.arange(W, device=features.device),
            indexing='ij'
        )
        
        coords = torch.stack([x_coords, y_coords], dim=0).float()
        coords = coords.unsqueeze(0).repeat(B, 1, 1, 1)
        
        # Add motion vectors to coordinates
        new_coords = coords + motion_vectors
        
        # Normalize coordinates to [-1, 1] for grid_sample
        new_coords[:, 0] = 2.0 * new_coords[:, 0] / (W - 1) - 1.0
        new_coords[:, 1] = 2.0 * new_coords[:, 1] / (H - 1) - 1.0
        
        # Rearrange for grid_sample: [B, H, W, 2]
        grid = new_coords.permute(0, 2, 3, 1)
        
        # Warp features
        warped = F.grid_sample(features, grid, align_corners=True)
        
        return warped


class TemporalConsistency(nn.Module):
    """Enforces temporal consistency in feature representations"""
    
    def __init__(self, feature_dim: int = 256):
        super().__init__()
        self.feature_dim = feature_dim
        
    def forward(self, current_features: torch.Tensor, prev_features: torch.Tensor):
        """Compute temporal consistency loss"""
        
        # Global average pooling to get frame-level representations
        current_global = F.adaptive_avg_pool2d(current_features, (1, 1))
        prev_global = F.adaptive_avg_pool2d(prev_features, (1, 1))
        
        current_global = current_global.view(current_global.size(0), -1)
        prev_global = prev_global.view(prev_global.size(0), -1)
        
        # Normalize features
        current_global = F.normalize(current_global, p=2, dim=1)
        prev_global = F.normalize(prev_global, p=2, dim=1)
        
        # Temporal consistency loss (maximize similarity)
        consistency_loss = 1.0 - F.cosine_similarity(current_global, prev_global).mean()
        
        return consistency_loss


class ObjectProposalGenerator(nn.Module):
    """Generates object proposals from features"""
    
    def __init__(self, feature_dim: int = 256, num_proposals: int = 100):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_proposals = num_proposals
        
        # Objectness score predictor
        self.objectness_head = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim // 2, 1, 1),
            nn.Sigmoid()
        )
        
        # Bounding box regression head
        self.bbox_head = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim // 2, 4, 1)
        )
        
    def forward(self, features: torch.Tensor) -> List[List[Dict]]:
        """Generate object proposals"""
        
        # Predict objectness scores
        objectness = self.objectness_head(features)
        
        # Predict bounding box deltas
        bbox_deltas = self.bbox_head(features)
        
        batch_proposals = []
        
        for i in range(features.size(0)):
            obj_scores = objectness[i, 0]  # [H, W]
            bbox_pred = bbox_deltas[i]      # [4, H, W]
            
            # Get top proposals based on objectness scores
            proposals = self._extract_proposals(
                obj_scores, bbox_pred, features[i]
            )
            
            batch_proposals.append(proposals)
        
        return batch_proposals
    
    def _extract_proposals(self, 
                          objectness: torch.Tensor,
                          bbox_deltas: torch.Tensor,
                          features: torch.Tensor) -> List[Dict]:
        """Extract top proposals from feature maps"""
        
        H, W = objectness.shape
        
        # Flatten and get top-k proposals
        flat_scores = objectness.view(-1)
        top_indices = torch.topk(flat_scores, min(self.num_proposals, len(flat_scores)))[1]
        
        proposals = []
        
        for idx in top_indices:
            y = idx // W
            x = idx % W
            
            confidence = flat_scores[idx].item()
            
            # Extract bounding box (simplified)
            dx, dy, dw, dh = bbox_deltas[:, y, x]
            
            # Convert to absolute coordinates (simplified)
            scale_x = 8  # Assuming stride of 8
            scale_y = 8
            
            center_x = x * scale_x + dx * scale_x
            center_y = y * scale_y + dy * scale_y
            width = torch.exp(dw) * scale_x
            height = torch.exp(dh) * scale_y
            
            bbox = [
                center_x - width / 2,
                center_y - height / 2,
                center_x + width / 2,
                center_y + height / 2
            ]
            
            # Extract feature vector for this proposal
            feature_vector = features[:, y, x]
            
            proposal = {
                'bbox': [coord.item() for coord in bbox],
                'confidence': confidence,
                'features': feature_vector.detach().cpu().numpy(),
                'location': (x.item(), y.item())
            }
            
            proposals.append(proposal)
        
        return proposals


class PseudoLabeler:
    """Generates pseudo-labels for detected objects using clustering"""
    
    def __init__(self, 
                 num_clusters: int = 5,
                 clustering_method: str = 'kmeans'):
        self.num_clusters = num_clusters
        self.clustering_method = clustering_method
        
    def generate_labels(self, 
                       features: torch.Tensor,
                       proposals: List[List[Dict]]) -> List[List[int]]:
        """Generate pseudo-labels for proposals using clustering"""
        
        all_features = []
        feature_indices = []
        
        # Collect all proposal features
        for batch_idx, batch_proposals in enumerate(proposals):
            for prop_idx, proposal in enumerate(batch_proposals):
                all_features.append(proposal['features'])
                feature_indices.append((batch_idx, prop_idx))
        
        if not all_features:
            return [[] for _ in proposals]
        
        # Stack features
        feature_matrix = np.stack(all_features)
        
        # Apply clustering
        if self.clustering_method == 'kmeans':
            clusterer = KMeans(n_clusters=self.num_clusters, random_state=42)
        elif self.clustering_method == 'dbscan':
            clusterer = DBSCAN(eps=0.5, min_samples=3)
        else:
            raise ValueError(f"Unknown clustering method: {self.clustering_method}")
        
        cluster_labels = clusterer.fit_predict(feature_matrix)
        
        # Organize labels back into batch structure
        batch_labels = [[] for _ in proposals]
        
        for (batch_idx, prop_idx), label in zip(feature_indices, cluster_labels):
            if len(batch_labels[batch_idx]) <= prop_idx:
                batch_labels[batch_idx].extend([0] * (prop_idx + 1 - len(batch_labels[batch_idx])))
            batch_labels[batch_idx][prop_idx] = max(0, label)  # Handle -1 from DBSCAN
        
        return batch_labels


class SelfSupervisedObjectDetection:
    """Main class for self-supervised object detection training"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Initialize model
        self.model = LOSTDetector(
            backbone=config['models']['object_detection']['backbone'],
            feature_dim=config['models']['autoencoder']['latent_dim'],
            num_proposals=config['inference']['parallel_processing']['batch_size'],
            confidence_threshold=config['models']['object_detection']['confidence_threshold']
        )
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Data augmentation for self-supervision
        self.augmentations = self._setup_augmentations()
        
    def _setup_augmentations(self):
        """Setup data augmentations for self-supervised learning"""
        return T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.RandomRotation(degrees=5),
            T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1))
        ])
    
    def train_epoch(self, dataloader):
        """Train one epoch"""
        self.model.train()
        total_loss = 0.0
        
        prev_batch = None
        
        for batch_idx, batch in enumerate(dataloader):
            images = batch['images']
            
            if torch.cuda.is_available():
                images = images.cuda()
            
            self.optimizer.zero_grad()
            
            if prev_batch is not None:
                # Use temporal consistency with previous batch
                outputs = self.model(images, prev_batch)
                
                # Compute total loss
                loss = outputs['motion_loss'] + outputs['consistency_loss']
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            else:
                # First batch - just forward pass
                outputs = self.model(images)
            
            prev_batch = images.detach()
        
        return total_loss / max(1, len(dataloader) - 1)
    
    def detect(self, images: torch.Tensor) -> List[Dict]:
        """Detect objects in images"""
        self.model.eval()
        
        with torch.no_grad():
            detections = self.model.detect_objects(images)
        
        return detections
    
    def save_model(self, path: str):
        """Save trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, path)
    
    def load_model(self, path: str):
        """Load trained model"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


# Aliases for compatibility
LOST = LOSTDetector 