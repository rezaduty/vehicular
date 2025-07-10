"""
Object detection models for autonomous driving
Includes YOLOv8, EfficientDet, and custom architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from ultralytics import YOLO
import timm
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import cv2


class YOLOv8Detector(nn.Module):
    """YOLOv8-based object detector with custom backbone"""
    
    def __init__(self, 
                 num_classes: int = 10,
                 model_size: str = 'n',
                 pretrained: bool = True,
                 confidence_threshold: float = 0.5,
                 nms_threshold: float = 0.4):
        super().__init__()
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        
        # Load YOLOv8 model
        model_name = f'yolov8{model_size}.pt' if pretrained else f'yolov8{model_size}.yaml'
        self.model = YOLO(model_name)
        
        # Modify for custom number of classes if needed
        if num_classes != 80:  # COCO has 80 classes
            self.model.model[-1].nc = num_classes
            self.model.model[-1].anchors = self.model.model[-1].anchors.clone()
            
        self.class_names = [
            'Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting',
            'Cyclist', 'Tram', 'Misc', 'DontCare', 'Traffic_Sign'
        ]
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for training"""
        if self.training:
            return self.model(x)
        else:
            return self.predict(x)
            
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Inference with post-processing"""
        results = self.model(x)
        
        predictions = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                pred = {
                    'boxes': boxes.xyxy.cpu().numpy(),
                    'scores': boxes.conf.cpu().numpy(),
                    'classes': boxes.cls.cpu().numpy().astype(int)
                }
            else:
                pred = {
                    'boxes': np.array([]),
                    'scores': np.array([]),
                    'classes': np.array([])
                }
            predictions.append(pred)
            
        return predictions
        
    def train_step(self, 
                   images: torch.Tensor,
                   targets: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Training step"""
        # Convert targets to YOLO format
        yolo_targets = self._convert_targets_to_yolo(targets, images.shape)
        
        # Forward pass
        loss = self.model(images, yolo_targets)
        
        return {'loss': loss}
        
    def _convert_targets_to_yolo(self, 
                                targets: List[Dict[str, torch.Tensor]], 
                                image_shape: Tuple[int, ...]) -> torch.Tensor:
        """Convert targets to YOLO format"""
        batch_size = len(targets)
        _, _, height, width = image_shape
        
        yolo_targets = []
        for i, target in enumerate(targets):
            if len(target['boxes']) == 0:
                continue
                
            boxes = target['boxes']
            classes = target['classes']
            
            # Convert to normalized center format
            x_center = (boxes[:, 0] + boxes[:, 2]) / 2.0 / width
            y_center = (boxes[:, 1] + boxes[:, 3]) / 2.0 / height
            box_width = (boxes[:, 2] - boxes[:, 0]) / width
            box_height = (boxes[:, 3] - boxes[:, 1]) / height
            
            # Create YOLO format: [batch_idx, class, x_center, y_center, width, height]
            batch_idx = torch.full((len(boxes), 1), i, dtype=torch.float32)
            classes_expanded = classes.unsqueeze(1).float()
            
            yolo_target = torch.cat([
                batch_idx,
                classes_expanded,
                x_center.unsqueeze(1),
                y_center.unsqueeze(1),
                box_width.unsqueeze(1),
                box_height.unsqueeze(1)
            ], dim=1)
            
            yolo_targets.append(yolo_target)
            
        if yolo_targets:
            return torch.cat(yolo_targets, dim=0)
        else:
            return torch.empty((0, 6))


class EfficientDetector(nn.Module):
    """EfficientDet-based object detector"""
    
    def __init__(self,
                 num_classes: int = 10,
                 compound_coef: int = 0,
                 pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.compound_coef = compound_coef
        
        # EfficientNet backbone
        self.backbone = timm.create_model(
            f'efficientnet_b{compound_coef}',
            pretrained=pretrained,
            features_only=True,
            out_indices=(2, 3, 4)
        )
        
        # Feature pyramid network
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=self._get_backbone_channels(),
            out_channels=256
        )
        
        # Detection heads
        self.classification_head = ClassificationHead(
            in_channels=256,
            num_anchors=9,
            num_classes=num_classes
        )
        
        self.regression_head = RegressionHead(
            in_channels=256,
            num_anchors=9
        )
        
        # Anchor generator
        self.anchor_generator = AnchorGenerator()
        
    def _get_backbone_channels(self) -> List[int]:
        """Get number of channels from backbone feature maps"""
        if self.compound_coef == 0:
            return [40, 112, 320]
        elif self.compound_coef == 1:
            return [40, 112, 320]
        elif self.compound_coef == 2:
            return [48, 120, 352]
        elif self.compound_coef == 3:
            return [48, 136, 384]
        else:
            return [56, 160, 448]
            
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Extract features
        features = self.backbone(x)
        
        # Feature pyramid
        fpn_features = self.fpn(features)
        
        # Generate anchors
        anchors = self.anchor_generator(fpn_features, x.shape[-2:])
        
        # Predictions
        class_logits = []
        bbox_regression = []
        
        for feature in fpn_features:
            class_logits.append(self.classification_head(feature))
            bbox_regression.append(self.regression_head(feature))
            
        return {
            'class_logits': class_logits,
            'bbox_regression': bbox_regression,
            'anchors': anchors
        }


class FeaturePyramidNetwork(nn.Module):
    """Feature Pyramid Network for multi-scale feature extraction"""
    
    def __init__(self, in_channels_list: List[int], out_channels: int = 256):
        super().__init__()
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        
        for in_channels in in_channels_list:
            inner_block = nn.Conv2d(in_channels, out_channels, 1)
            layer_block = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)
            
    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        # Start from the top level
        last_inner = self.inner_blocks[-1](x[-1])
        results = [self.layer_blocks[-1](last_inner)]
        
        # Process remaining levels top-down
        for i in range(len(x) - 2, -1, -1):
            inner_lateral = self.inner_blocks[i](x[i])
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.layer_blocks[i](last_inner))
            
        return results


class ClassificationHead(nn.Module):
    """Classification head for object detection"""
    
    def __init__(self, in_channels: int, num_anchors: int, num_classes: int, num_layers: int = 4):
        super().__init__()
        
        layers = []
        for _ in range(num_layers):
            layers.extend([
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.ReLU(inplace=True)
            ])
            
        layers.append(nn.Conv2d(in_channels, num_anchors * num_classes, 3, padding=1))
        
        self.classification_head = nn.Sequential(*layers)
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.classification_head(x)
        
        # Reshape for classification
        N, _, H, W = x.shape
        x = x.view(N, self.num_anchors, self.num_classes, H, W)
        x = x.permute(0, 3, 4, 1, 2)  # [N, H, W, num_anchors, num_classes]
        
        return x.contiguous().view(N, -1, self.num_classes)


class RegressionHead(nn.Module):
    """Regression head for bounding box regression"""
    
    def __init__(self, in_channels: int, num_anchors: int, num_layers: int = 4):
        super().__init__()
        
        layers = []
        for _ in range(num_layers):
            layers.extend([
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.ReLU(inplace=True)
            ])
            
        layers.append(nn.Conv2d(in_channels, num_anchors * 4, 3, padding=1))
        
        self.regression_head = nn.Sequential(*layers)
        self.num_anchors = num_anchors
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.regression_head(x)
        
        # Reshape for regression
        N, _, H, W = x.shape
        x = x.view(N, self.num_anchors, 4, H, W)
        x = x.permute(0, 3, 4, 1, 2)  # [N, H, W, num_anchors, 4]
        
        return x.contiguous().view(N, -1, 4)


class AnchorGenerator(nn.Module):
    """Generate anchors for object detection"""
    
    def __init__(self, 
                 sizes: List[int] = [32, 64, 128, 256, 512],
                 aspect_ratios: List[float] = [0.5, 1.0, 2.0],
                 scales: List[float] = [1.0, 1.26, 1.59]):
        super().__init__()
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.scales = scales
        
    def forward(self, 
                feature_maps: List[torch.Tensor], 
                image_size: Tuple[int, int]) -> List[torch.Tensor]:
        """Generate anchors for each feature map level"""
        anchors = []
        
        for i, feature_map in enumerate(feature_maps):
            size = self.sizes[i]
            height, width = feature_map.shape[-2:]
            
            # Generate base anchors
            base_anchors = self._generate_base_anchors(size)
            
            # Generate grid
            shifts = self._generate_shifts(height, width, feature_map.device)
            
            # Generate all anchors for this level
            level_anchors = self._generate_level_anchors(base_anchors, shifts)
            anchors.append(level_anchors)
            
        return anchors
        
    def _generate_base_anchors(self, size: int) -> torch.Tensor:
        """Generate base anchors for a given size"""
        anchors = []
        
        for scale in self.scales:
            for aspect_ratio in self.aspect_ratios:
                area = size * size * scale * scale
                width = (area / aspect_ratio) ** 0.5
                height = area / width
                
                # Center at origin
                anchor = [-width/2, -height/2, width/2, height/2]
                anchors.append(anchor)
                
        return torch.tensor(anchors, dtype=torch.float32)
        
    def _generate_shifts(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        """Generate coordinate shifts for feature map locations"""
        shift_x = torch.arange(0, width, device=device) * 8  # Assuming stride 8
        shift_y = torch.arange(0, height, device=device) * 8
        
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing='ij')
        shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=2)
        
        return shifts.view(-1, 4)
        
    def _generate_level_anchors(self, 
                               base_anchors: torch.Tensor, 
                               shifts: torch.Tensor) -> torch.Tensor:
        """Generate all anchors for a feature map level"""
        num_base_anchors = base_anchors.shape[0]
        num_shifts = shifts.shape[0]
        
        # Expand dimensions for broadcasting
        base_anchors = base_anchors.to(shifts.device)
        base_anchors = base_anchors.view(1, num_base_anchors, 4)
        shifts = shifts.view(num_shifts, 1, 4)
        
        # Add shifts to base anchors
        anchors = base_anchors + shifts
        
        return anchors.view(-1, 4)


class ParallelPatchDetector(nn.Module):
    """Parallel patch-based detector for enhanced small object detection"""
    
    def __init__(self, 
                 base_detector: nn.Module,
                 patch_size: Tuple[int, int] = (192, 192),
                 overlap: float = 0.2,
                 min_object_size: int = 20):
        super().__init__()
        self.base_detector = base_detector
        self.patch_size = patch_size
        self.overlap = overlap
        self.min_object_size = min_object_size
        
    def forward(self, x: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        """Forward pass with patch-based detection"""
        batch_predictions = []
        
        for i in range(x.shape[0]):
            image = x[i].cpu().numpy().transpose(1, 2, 0)
            
            # Extract patches
            patches = self._extract_patches(image)
            
            # Run detection on patches
            patch_predictions = []
            for patch_data in patches:
                patch_tensor = torch.from_numpy(
                    patch_data['patch'].transpose(2, 0, 1)
                ).unsqueeze(0).to(x.device)
                
                with torch.no_grad():
                    pred = self.base_detector.predict(patch_tensor)[0]
                    
                # Add patch offset information
                pred['x_offset'] = patch_data['x_offset']
                pred['y_offset'] = patch_data['y_offset']
                patch_predictions.append(pred)
                
            # Merge patch predictions
            merged_pred = self._merge_predictions(patch_predictions, image.shape[:2])
            batch_predictions.append(merged_pred)
            
        return batch_predictions
        
    def _extract_patches(self, image: np.ndarray) -> List[Dict]:
        """Extract overlapping patches from image"""
        h, w = image.shape[:2]
        patch_h, patch_w = self.patch_size
        
        stride_h = int(patch_h * (1 - self.overlap))
        stride_w = int(patch_w * (1 - self.overlap))
        
        patches = []
        
        for y in range(0, h - patch_h + 1, stride_h):
            for x in range(0, w - patch_w + 1, stride_w):
                patch = image[y:y+patch_h, x:x+patch_w]
                
                if patch.shape[:2] != self.patch_size:
                    patch = cv2.resize(patch, (patch_w, patch_h))
                
                patches.append({
                    'patch': patch,
                    'x_offset': x,
                    'y_offset': y
                })
                
        return patches
        
    def _merge_predictions(self, 
                          patch_predictions: List[Dict],
                          image_shape: Tuple[int, int]) -> Dict[str, np.ndarray]:
        """Merge predictions from multiple patches"""
        all_boxes = []
        all_scores = []
        all_classes = []
        
        for pred in patch_predictions:
            if len(pred['boxes']) > 0:
                # Adjust coordinates
                boxes = pred['boxes'].copy()
                boxes[:, [0, 2]] += pred['x_offset']
                boxes[:, [1, 3]] += pred['y_offset']
                
                # Filter by size
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                valid_mask = areas >= self.min_object_size ** 2
                
                if np.any(valid_mask):
                    all_boxes.append(boxes[valid_mask])
                    all_scores.append(pred['scores'][valid_mask])
                    all_classes.append(pred['classes'][valid_mask])
        
        if not all_boxes:
            return {
                'boxes': np.array([]),
                'scores': np.array([]),
                'classes': np.array([])
            }
            
        # Concatenate and apply NMS
        all_boxes = np.vstack(all_boxes)
        all_scores = np.concatenate(all_scores)
        all_classes = np.concatenate(all_classes)
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            [list(box) for box in all_boxes],
            list(all_scores),
            score_threshold=0.3,
            nms_threshold=0.5
        )
        
        if len(indices) > 0:
            indices = indices.flatten()
            return {
                'boxes': all_boxes[indices],
                'scores': all_scores[indices],
                'classes': all_classes[indices]
            }
        else:
            return {
                'boxes': np.array([]),
                'scores': np.array([]),
                'classes': np.array([])
            } 