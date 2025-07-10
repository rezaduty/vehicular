"""
Depth estimation models for autonomous driving
Combines object detection with depth estimation and velocity tracking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional


class DepthEstimator(nn.Module):
    """Monocular depth estimation for autonomous driving"""
    
    def __init__(self, 
                 backbone: str = 'resnet50',
                 max_depth: float = 100.0):
        super().__init__()
        self.max_depth = max_depth
        
        # Backbone encoder
        if backbone == 'resnet50':
            import torchvision.models as models
            resnet = models.resnet50(pretrained=True)
            self.encoder = nn.Sequential(*list(resnet.children())[:-2])
            backbone_channels = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Decoder for depth prediction
        self.decoder = DepthDecoder(backbone_channels)
        
        # Object detection head
        self.detection_head = DetectionWithDepthHead(backbone_channels)
        
    def forward(self, x):
        # Extract features
        features = self.encoder(x)
        
        # Predict depth
        depth = self.decoder(features)
        
        # Object detection with depth
        detections = self.detection_head(features, depth)
        
        return {
            'depth': depth,
            'detections': detections
        }


class DepthDecoder(nn.Module):
    """Decoder for depth estimation"""
    
    def __init__(self, in_channels: int = 2048):
        super().__init__()
        
        # Upsampling layers
        self.upconv4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 512, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        self.upconv3 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.upconv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.upconv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.upconv0 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Final depth prediction
        self.depth_pred = nn.Sequential(
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Progressive upsampling
        x = self.upconv4(x)
        x = self.upconv3(x)
        x = self.upconv2(x)
        x = self.upconv1(x)
        x = self.upconv0(x)
        
        # Depth prediction
        depth = self.depth_pred(x)
        
        return depth


class DetectionWithDepthHead(nn.Module):
    """Detection head that incorporates depth information"""
    
    def __init__(self, 
                 feature_channels: int = 2048,
                 num_classes: int = 10,
                 num_anchors: int = 9):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Feature adaptation
        self.feature_adapter = nn.Conv2d(feature_channels, 256, 1)
        
        # Classification head
        self.cls_head = nn.Sequential(
            nn.Conv2d(256 + 1, 256, 3, padding=1),  # +1 for depth channel
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_anchors * num_classes, 1)
        )
        
        # Regression head
        self.reg_head = nn.Sequential(
            nn.Conv2d(256 + 1, 256, 3, padding=1),  # +1 for depth channel
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_anchors * 4, 1)
        )
        
        # Depth regression head (per-object depth refinement)
        self.depth_head = nn.Sequential(
            nn.Conv2d(256 + 1, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_anchors * 1, 1)
        )
        
    def forward(self, features, depth_map):
        # Adapt features
        adapted_features = self.feature_adapter(features)
        
        # Resize depth map to match feature size
        depth_resized = F.interpolate(
            depth_map, 
            size=adapted_features.shape[-2:], 
            mode='bilinear', 
            align_corners=True
        )
        
        # Concatenate features with depth
        combined_features = torch.cat([adapted_features, depth_resized], dim=1)
        
        # Predictions
        cls_logits = self.cls_head(combined_features)
        bbox_regression = self.reg_head(combined_features)
        depth_regression = self.depth_head(combined_features)
        
        return {
            'cls_logits': cls_logits,
            'bbox_regression': bbox_regression,
            'depth_regression': depth_regression
        }


class MonoDepth(nn.Module):
    """MonoDepth-style self-supervised depth estimation"""
    
    def __init__(self):
        super().__init__()
        
        # Depth encoder
        self.depth_encoder = DepthEncoder()
        
        # Pose encoder
        self.pose_encoder = PoseEncoder()
        
    def forward(self, current_image, target_image):
        # Predict depth for current image
        depth = self.depth_encoder(current_image)
        
        # Predict relative pose
        pose = self.pose_encoder(torch.cat([current_image, target_image], dim=1))
        
        return depth, pose


class DepthEncoder(nn.Module):
    """Encoder for depth prediction"""
    
    def __init__(self):
        super().__init__()
        
        # ResNet-like encoder
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = self._make_layer(64, 128, 2, stride=2)
        self.conv3 = self._make_layer(128, 256, 2, stride=2)
        self.conv4 = self._make_layer(256, 512, 2, stride=2)
        self.conv5 = self._make_layer(512, 512, 2, stride=2)
        
        # Decoder
        self.decoder = DepthDecoder(512)
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(1, blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        
        depth = self.decoder(x5)
        return depth


class PoseEncoder(nn.Module):
    """Encoder for camera pose estimation"""
    
    def __init__(self):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(6, 64, 7, stride=2, padding=3),  # 6 channels for two RGB images
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.pose_pred = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 6)  # 6 DOF pose
        )
        
    def forward(self, x):
        features = self.conv_layers(x)
        pose = self.pose_pred(features)
        return pose


class DepthVelocityTracker:
    """Combines depth estimation with velocity tracking"""
    
    def __init__(self, depth_model: DepthEstimator):
        self.depth_model = depth_model
        self.previous_detections = None
        self.previous_timestamp = None
        
    def update(self, image: torch.Tensor, timestamp: float) -> Dict:
        """Update with new frame and estimate velocities"""
        
        # Get current detections with depth
        current_output = self.depth_model(image)
        current_detections = current_output['detections']
        depth_map = current_output['depth']
        
        # Process detections
        processed_detections = []
        
        if 'cls_logits' in current_detections:
            # Convert raw outputs to detections
            detections = self._process_raw_detections(
                current_detections, depth_map, image.shape[-2:]
            )
        else:
            detections = current_detections
        
        # Estimate velocities if we have previous frame
        if self.previous_detections is not None and self.previous_timestamp is not None:
            dt = timestamp - self.previous_timestamp
            detections = self._estimate_velocities(detections, dt)
        
        # Store for next frame
        self.previous_detections = detections
        self.previous_timestamp = timestamp
        
        return {
            'detections': detections,
            'depth_map': depth_map
        }
    
    def _process_raw_detections(self, raw_detections, depth_map, image_shape):
        """Convert raw model outputs to detection format"""
        # This would typically involve:
        # 1. Applying NMS
        # 2. Converting to bounding boxes
        # 3. Extracting depth values
        # Simplified implementation here
        
        detections = []
        # Placeholder processing - in real implementation would decode anchors
        return detections
    
    def _estimate_velocities(self, current_detections, dt):
        """Estimate object velocities based on previous frame"""
        
        for detection in current_detections:
            # Find matching detection in previous frame
            matched_prev = self._find_matching_detection(detection)
            
            if matched_prev is not None:
                # Calculate 3D velocity
                velocity_3d = self._calculate_3d_velocity(
                    detection, matched_prev, dt
                )
                detection['velocity_3d'] = velocity_3d
                detection['speed'] = np.linalg.norm(velocity_3d)
            else:
                detection['velocity_3d'] = [0, 0, 0]
                detection['speed'] = 0
        
        return current_detections
    
    def _find_matching_detection(self, current_detection):
        """Find matching detection in previous frame"""
        if not self.previous_detections:
            return None
        
        best_match = None
        best_iou = 0
        
        for prev_detection in self.previous_detections:
            if prev_detection['class'] == current_detection['class']:
                iou = self._calculate_iou(
                    current_detection['bbox'], 
                    prev_detection['bbox']
                )
                if iou > best_iou and iou > 0.3:
                    best_iou = iou
                    best_match = prev_detection
        
        return best_match
    
    def _calculate_3d_velocity(self, current, previous, dt):
        """Calculate 3D velocity from depth and 2D motion"""
        
        # Get 3D positions
        current_3d = self._backproject_to_3d(current)
        previous_3d = self._backproject_to_3d(previous)
        
        # Calculate velocity
        velocity = [(current_3d[i] - previous_3d[i]) / dt for i in range(3)]
        
        return velocity
    
    def _backproject_to_3d(self, detection):
        """Backproject 2D detection to 3D using depth"""
        bbox = detection['bbox']
        depth = detection.get('depth', 10.0)  # Default depth if not available
        
        # Center of bounding box
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        
        # Simplified camera intrinsics (would use actual calibration)
        fx, fy = 721.5377, 721.5377  # KITTI-like intrinsics
        cx, cy = 609.5593, 172.8540
        
        # Backproject to 3D
        x_3d = (center_x - cx) * depth / fx
        y_3d = (center_y - cy) * depth / fy
        z_3d = depth
        
        return [x_3d, y_3d, z_3d]
    
    def _calculate_iou(self, bbox1, bbox2):
        """Calculate IoU between two bounding boxes"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0 