"""
Data transformations and augmentation for autonomous driving datasets
"""

import numpy as np
import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as T
from typing import Dict, List, Tuple, Any, Optional
import random


class AugmentationPipeline:
    """Comprehensive augmentation pipeline for autonomous driving data"""
    
    def __init__(self, 
                 image_size: Tuple[int, int] = (384, 1280),
                 augment_prob: float = 0.5,
                 normalize: bool = True):
        self.image_size = image_size
        self.augment_prob = augment_prob
        self.normalize = normalize
        
        # Define augmentation pipeline
        self.train_transform = A.Compose([
            A.Resize(height=image_size[0], width=image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, 
                contrast_limit=0.2, 
                p=0.5
            ),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.3
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.MotionBlur(blur_limit=3, p=0.2),
            A.RandomRain(
                slant_lower=-10,
                slant_upper=10,
                drop_length=20,
                drop_width=1,
                drop_color=(200, 200, 200),
                blur_value=1,
                brightness_coefficient=0.8,
                rain_type=None,
                p=0.1
            ),
            A.RandomSunFlare(
                flare_roi=(0, 0, 1, 0.5),
                angle_lower=0,
                angle_upper=1,
                num_flare_circles_lower=6,
                num_flare_circles_upper=10,
                src_radius=160,
                src_color=(255, 255, 255),
                p=0.1
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ) if normalize else A.NoOp(),
            ToTensorV2()
        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['class_labels'],
            min_area=0,
            min_visibility=0.1
        ))
        
        self.val_transform = A.Compose([
            A.Resize(height=image_size[0], width=image_size[1]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ) if normalize else A.NoOp(),
            ToTensorV2()
        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['class_labels'],
            min_area=0,
            min_visibility=0.1
        ))
        
    def __call__(self, sample: Dict[str, Any], training: bool = True) -> Dict[str, Any]:
        """Apply transformations to a sample"""
        image = sample['image']
        
        # Extract bounding boxes if available
        bboxes = []
        class_labels = []
        
        if 'boxes' in sample and sample['boxes']:
            for i, box_data in enumerate(sample['boxes']):
                if 'bbox_2d' in box_data:
                    bbox = box_data['bbox_2d']  # [left, top, right, bottom]
                    bboxes.append(bbox)
                    class_labels.append(sample['classes'][i])
        
        # Apply augmentations
        if training and len(bboxes) > 0:
            transformed = self.train_transform(
                image=image,
                bboxes=bboxes,
                class_labels=class_labels
            )
        elif training:
            transformed = self.train_transform(image=image)
        else:
            if len(bboxes) > 0:
                transformed = self.val_transform(
                    image=image,
                    bboxes=bboxes,
                    class_labels=class_labels
                )
            else:
                transformed = self.val_transform(image=image)
        
        # Update sample
        sample['image'] = transformed['image']
        
        # Update bounding boxes if they exist and were transformed
        if 'bboxes' in transformed and transformed['bboxes']:
            for i, bbox in enumerate(transformed['bboxes']):
                sample['boxes'][i]['bbox_2d'] = list(bbox)
                
        return sample


class LiDARTransforms:
    """Transformations for LiDAR point cloud data"""
    
    def __init__(self, 
                 max_points: int = 16384,
                 voxel_size: float = 0.1,
                 point_range: List[float] = [-40, -40, -3, 40, 40, 1]):
        self.max_points = max_points
        self.voxel_size = voxel_size
        self.point_range = point_range
        
    def filter_points_by_range(self, points: np.ndarray) -> np.ndarray:
        """Filter points by distance range"""
        x_min, y_min, z_min, x_max, y_max, z_max = self.point_range
        
        mask = (
            (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
            (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
            (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
        )
        
        return points[mask]
        
    def random_sampling(self, points: np.ndarray) -> np.ndarray:
        """Randomly sample points to fixed size"""
        if len(points) > self.max_points:
            indices = np.random.choice(len(points), self.max_points, replace=False)
            return points[indices]
        elif len(points) < self.max_points:
            # Pad with zeros
            padding = np.zeros((self.max_points - len(points), points.shape[1]))
            return np.vstack([points, padding])
        return points
        
    def random_rotation(self, points: np.ndarray, angle_range: float = 0.1) -> np.ndarray:
        """Apply random rotation around Z-axis"""
        angle = np.random.uniform(-angle_range, angle_range)
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        
        rotation_matrix = np.array([
            [cos_angle, -sin_angle, 0],
            [sin_angle, cos_angle, 0],
            [0, 0, 1]
        ])
        
        points[:, :3] = points[:, :3] @ rotation_matrix.T
        return points
        
    def random_scaling(self, points: np.ndarray, scale_range: Tuple[float, float] = (0.95, 1.05)) -> np.ndarray:
        """Apply random scaling"""
        scale = np.random.uniform(scale_range[0], scale_range[1])
        points[:, :3] *= scale
        return points
        
    def random_translation(self, points: np.ndarray, translation_range: float = 0.2) -> np.ndarray:
        """Apply random translation"""
        translation = np.random.uniform(-translation_range, translation_range, 3)
        points[:, :3] += translation
        return points
        
    def __call__(self, points: np.ndarray, training: bool = True) -> np.ndarray:
        """Apply LiDAR transformations"""
        # Filter points by range
        points = self.filter_points_by_range(points)
        
        if training:
            # Apply augmentations
            if np.random.random() < 0.5:
                points = self.random_rotation(points)
            if np.random.random() < 0.3:
                points = self.random_scaling(points)
            if np.random.random() < 0.3:
                points = self.random_translation(points)
                
        # Sample to fixed size
        points = self.random_sampling(points)
        
        return points


class PatchExtractor:
    """Extract patches from images for parallel processing"""
    
    def __init__(self,
                 patch_size: Tuple[int, int] = (192, 192),
                 overlap: float = 0.2,
                 min_object_size: int = 20):
        self.patch_size = patch_size
        self.overlap = overlap
        self.min_object_size = min_object_size
        
    def extract_patches(self, image: np.ndarray) -> List[Dict]:
        """Extract overlapping patches from image"""
        h, w = image.shape[:2]
        patch_h, patch_w = self.patch_size
        
        stride_h = int(patch_h * (1 - self.overlap))
        stride_w = int(patch_w * (1 - self.overlap))
        
        patches = []
        
        for y in range(0, h - patch_h + 1, stride_h):
            for x in range(0, w - patch_w + 1, stride_w):
                # Ensure we don't go out of bounds
                y_end = min(y + patch_h, h)
                x_end = min(x + patch_w, w)
                
                # Extract patch
                patch = image[y:y_end, x:x_end]
                
                # Resize to target size if needed
                if patch.shape[:2] != self.patch_size:
                    patch = cv2.resize(patch, (patch_w, patch_h))
                
                patches.append({
                    'patch': patch,
                    'x_offset': x,
                    'y_offset': y,
                    'x_end': x_end,
                    'y_end': y_end
                })
                
        return patches
        
    def merge_predictions(self, 
                         patch_predictions: List[Dict],
                         image_shape: Tuple[int, int]) -> Dict:
        """Merge predictions from multiple patches"""
        h, w = image_shape[:2]
        
        # Collect all detections
        all_boxes = []
        all_scores = []
        all_classes = []
        
        for pred in patch_predictions:
            if 'boxes' in pred and len(pred['boxes']) > 0:
                # Adjust box coordinates to global image coordinates
                boxes = pred['boxes'].copy()
                boxes[:, [0, 2]] += pred['x_offset']  # x coordinates
                boxes[:, [1, 3]] += pred['y_offset']  # y coordinates
                
                # Clip to image boundaries
                boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
                boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h)
                
                # Filter out small boxes
                box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                valid_mask = box_areas >= self.min_object_size ** 2
                
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
            
        # Concatenate all detections
        all_boxes = np.vstack(all_boxes)
        all_scores = np.concatenate(all_scores)
        all_classes = np.concatenate(all_classes)
        
        # Apply Non-Maximum Suppression
        final_boxes, final_scores, final_classes = self.apply_nms(
            all_boxes, all_scores, all_classes
        )
        
        return {
            'boxes': final_boxes,
            'scores': final_scores,
            'classes': final_classes
        }
        
    def apply_nms(self, 
                  boxes: np.ndarray,
                  scores: np.ndarray,
                  classes: np.ndarray,
                  iou_threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply Non-Maximum Suppression"""
        if len(boxes) == 0:
            return boxes, scores, classes
            
        # Convert to format expected by cv2.dnn.NMSBoxes
        boxes_list = [list(box) for box in boxes]
        scores_list = list(scores)
        
        indices = cv2.dnn.NMSBoxes(
            boxes_list, scores_list, 
            score_threshold=0.3, 
            nms_threshold=iou_threshold
        )
        
        if len(indices) > 0:
            indices = indices.flatten()
            return boxes[indices], scores[indices], classes[indices]
        else:
            return np.array([]), np.array([]), np.array([])


def get_transforms(config: Dict, training: bool = True) -> Dict:
    """Get appropriate transforms based on configuration"""
    image_transforms = AugmentationPipeline(
        image_size=(config['data']['image']['height'], config['data']['image']['width']),
        augment_prob=config['data']['augmentation'].get('probability', 0.5),
        normalize=True
    )
    
    lidar_transforms = None
    if 'lidar' in config['data']:
        lidar_transforms = LiDARTransforms(
            max_points=config['data']['lidar']['max_points'],
            point_range=config['data']['lidar'].get('point_range', [-40, -40, -3, 40, 40, 1])
        )
    
    patch_extractor = None
    if config.get('inference', {}).get('patch_detection', {}).get('enabled', False):
        patch_config = config['inference']['patch_detection']
        patch_extractor = PatchExtractor(
            patch_size=tuple(patch_config['patch_size']),
            overlap=patch_config['overlap'],
            min_object_size=patch_config['min_object_size']
        )
    
    return {
        'image': image_transforms,
        'lidar': lidar_transforms,
        'patch': patch_extractor
    } 