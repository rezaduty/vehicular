"""
Utility functions for data processing and handling
"""

import numpy as np
import torch
import cv2
from typing import List, Dict, Tuple, Optional


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image to [0, 1] range"""
    return image.astype(np.float32) / 255.0


def denormalize_image(image: np.ndarray) -> np.ndarray:
    """Denormalize image from [0, 1] to [0, 255] range"""
    return (image * 255).astype(np.uint8)


def resize_with_aspect_ratio(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize image while maintaining aspect ratio"""
    h, w = image.shape[:2]
    target_h, target_w = target_size
    
    # Calculate scaling factor
    scale = min(target_w / w, target_h / h)
    
    # Calculate new dimensions
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h))
    
    # Create padded image
    padded = np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype)
    
    # Center the resized image
    start_y = (target_h - new_h) // 2
    start_x = (target_w - new_w) // 2
    padded[start_y:start_y + new_h, start_x:start_x + new_w] = resized
    
    return padded


def create_collate_fn(pad_value: float = 0.0):
    """Create collate function for DataLoader"""
    
    def collate_fn(batch):
        images = []
        targets = []
        
        for item in batch:
            if isinstance(item, dict):
                images.append(item['image'])
                targets.append(item.get('target', {}))
            else:
                images.append(item[0])
                targets.append(item[1] if len(item) > 1 else {})
        
        # Stack images
        images = torch.stack(images, dim=0)
        
        return {
            'images': images,
            'targets': targets
        }
    
    return collate_fn


def apply_augmentation(image: np.ndarray, augmentation_params: Dict) -> np.ndarray:
    """Apply data augmentation to image"""
    
    if augmentation_params.get('horizontal_flip', False):
        if np.random.random() > 0.5:
            image = cv2.flip(image, 1)
    
    if 'brightness' in augmentation_params:
        factor = np.random.uniform(*augmentation_params['brightness'])
        image = cv2.convertScaleAbs(image, alpha=factor, beta=0)
    
    if 'contrast' in augmentation_params:
        factor = np.random.uniform(*augmentation_params['contrast'])
        image = cv2.convertScaleAbs(image, alpha=factor, beta=0)
    
    return image


def extract_patches(image: np.ndarray, patch_size: Tuple[int, int], stride: Tuple[int, int]) -> List[np.ndarray]:
    """Extract patches from image"""
    patches = []
    h, w = image.shape[:2]
    patch_h, patch_w = patch_size
    stride_h, stride_w = stride
    
    for y in range(0, h - patch_h + 1, stride_h):
        for x in range(0, w - patch_w + 1, stride_w):
            patch = image[y:y + patch_h, x:x + patch_w]
            patches.append(patch)
    
    return patches


def merge_patch_predictions(predictions: List[Dict], patch_coords: List[Tuple[int, int]], 
                          image_size: Tuple[int, int], iou_threshold: float = 0.5) -> List[Dict]:
    """Merge predictions from multiple patches"""
    
    all_detections = []
    
    # Adjust coordinates for each patch
    for pred, (start_y, start_x) in zip(predictions, patch_coords):
        if 'detections' in pred:
            for detection in pred['detections']:
                # Adjust bounding box coordinates
                if 'bbox' in detection:
                    bbox = detection['bbox'].copy()
                    bbox[0] += start_x  # x1
                    bbox[1] += start_y  # y1
                    bbox[2] += start_x  # x2
                    bbox[3] += start_y  # y2
                    detection['bbox'] = bbox
                
                all_detections.append(detection)
    
    # Apply NMS to merge overlapping detections
    if all_detections:
        merged_detections = non_max_suppression(all_detections, iou_threshold)
    else:
        merged_detections = []
    
    return merged_detections


def non_max_suppression(detections: List[Dict], iou_threshold: float) -> List[Dict]:
    """Apply Non-Maximum Suppression"""
    
    if not detections:
        return []
    
    # Sort by confidence
    detections = sorted(detections, key=lambda x: x.get('confidence', 0), reverse=True)
    
    keep = []
    
    while detections:
        # Take detection with highest confidence
        current = detections.pop(0)
        keep.append(current)
        
        # Remove detections with high IoU
        remaining = []
        for det in detections:
            if calculate_iou(current['bbox'], det['bbox']) < iou_threshold:
                remaining.append(det)
        
        detections = remaining
    
    return keep


def calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """Calculate Intersection over Union"""
    
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


def convert_to_tensor(data, dtype=torch.float32):
    """Convert numpy array or list to PyTorch tensor"""
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).type(dtype)
    elif isinstance(data, list):
        return torch.tensor(data, dtype=dtype)
    elif isinstance(data, torch.Tensor):
        return data.type(dtype)
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")


def create_dummy_data(batch_size: int = 1, image_size: Tuple[int, int] = (384, 1280)) -> Dict:
    """Create dummy data for testing"""
    
    height, width = image_size
    
    images = torch.randn(batch_size, 3, height, width)
    
    # Create dummy targets
    targets = []
    for _ in range(batch_size):
        num_objects = np.random.randint(1, 5)
        target = {
            'boxes': torch.rand(num_objects, 4) * torch.tensor([width, height, width, height]),
            'labels': torch.randint(0, 10, (num_objects,)),
            'scores': torch.rand(num_objects)
        }
        targets.append(target)
    
    return {
        'images': images,
        'targets': targets
    }


def download_datasets(dataset_names: List[str], data_dir: str = "./data") -> bool:
    """Download datasets for autonomous driving"""
    
    import os
    
    # Create data directory
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"Dataset download initiated for: {dataset_names}")
    print(f"Data directory: {data_dir}")
    
    supported_datasets = {
        'kitti': {
            'url': 'http://www.cvlibs.net/datasets/kitti/',
            'description': 'KITTI Vision Benchmark Suite'
        },
        'carla': {
            'url': 'https://carla.org/',
            'description': 'CARLA Autonomous Driving Simulator'
        },
        'nuscenes': {
            'url': 'https://www.nuscenes.org/',
            'description': 'nuScenes Dataset'
        },
        'waymo': {
            'url': 'https://waymo.com/open/',
            'description': 'Waymo Open Dataset'
        }
    }
    
    for dataset_name in dataset_names:
        if dataset_name.lower() in supported_datasets:
            dataset_info = supported_datasets[dataset_name.lower()]
            print(f"✓ {dataset_name}: {dataset_info['description']}")
            print(f"  URL: {dataset_info['url']}")
            
            # Create dataset directory
            dataset_dir = os.path.join(data_dir, dataset_name.lower())
            os.makedirs(dataset_dir, exist_ok=True)
            
            # Create placeholder files (in real implementation, would download actual data)
            placeholder_file = os.path.join(dataset_dir, "README.md")
            with open(placeholder_file, 'w') as f:
                f.write(f"# {dataset_info['description']}\n\n")
                f.write(f"Dataset: {dataset_name}\n")
                f.write(f"Source: {dataset_info['url']}\n")
                f.write(f"Status: Placeholder (download implementation required)\n")
        else:
            print(f"✗ {dataset_name}: Not supported")
            return False
    
    print("Dataset download simulation completed successfully!")
    return True


def verify_dataset(dataset_path: str, dataset_type: str = "kitti") -> bool:
    """Verify dataset integrity and structure"""
    
    import os
    
    if not os.path.exists(dataset_path):
        print(f"Dataset path does not exist: {dataset_path}")
        return False
    
    print(f"Verifying {dataset_type} dataset at: {dataset_path}")
    
    # Define expected structure for different datasets
    expected_structure = {
        'kitti': {
            'directories': ['training', 'testing'],
            'subdirs': ['image_2', 'label_2', 'velodyne'],
            'extensions': ['.png', '.txt', '.bin']
        },
        'carla': {
            'directories': ['episodes'],
            'subdirs': ['rgb', 'semantic', 'depth', 'lidar'],
            'extensions': ['.png', '.ply', '.json']
        },
        'nuscenes': {
            'directories': ['samples', 'sweeps', 'maps'],
            'subdirs': ['CAM_FRONT', 'LIDAR_TOP'],
            'extensions': ['.jpg', '.pcd', '.json']
        }
    }
    
    if dataset_type.lower() not in expected_structure:
        print(f"Unknown dataset type: {dataset_type}")
        return False
    
    structure = expected_structure[dataset_type.lower()]
    verification_passed = True
    
    # Check main directories
    for directory in structure['directories']:
        dir_path = os.path.join(dataset_path, directory)
        if os.path.exists(dir_path):
            print(f"✓ Found directory: {directory}")
        else:
            print(f"✗ Missing directory: {directory}")
            verification_passed = False
    
    # Check for some files (simplified verification)
    file_count = 0
    for root, dirs, files in os.walk(dataset_path):
        file_count += len(files)
    
    print(f"Total files found: {file_count}")
    
    if file_count == 0:
        print("⚠️  No files found in dataset directory")
        verification_passed = False
    
    if verification_passed:
        print("✅ Dataset verification passed!")
    else:
        print("❌ Dataset verification failed!")
    
    return verification_passed


def get_dataset_info(dataset_type: str) -> Dict:
    """Get information about supported datasets"""
    
    dataset_info = {
        'kitti': {
            'name': 'KITTI Vision Benchmark Suite',
            'type': 'real_world',
            'modalities': ['camera', 'lidar'],
            'classes': ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram'],
            'image_size': (375, 1242),
            'description': 'Real-world autonomous driving data collected in Karlsruhe, Germany'
        },
        'carla': {
            'name': 'CARLA Simulator',
            'type': 'simulation',
            'modalities': ['camera', 'lidar', 'semantic', 'depth'],
            'classes': ['Vehicle', 'Pedestrian', 'TrafficSign', 'TrafficLight'],
            'image_size': (600, 800),
            'description': 'High-fidelity autonomous driving simulation environment'
        },
        'nuscenes': {
            'name': 'nuScenes Dataset',
            'type': 'real_world',
            'modalities': ['camera', 'lidar', 'radar'],
            'classes': ['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle', 'motorcycle', 'pedestrian'],
            'image_size': (900, 1600),
            'description': 'Large-scale autonomous driving dataset with 360° view'
        },
        'waymo': {
            'name': 'Waymo Open Dataset',
            'type': 'real_world',
            'modalities': ['camera', 'lidar'],
            'classes': ['Vehicle', 'Pedestrian', 'Cyclist', 'Sign'],
            'image_size': (886, 1920),
            'description': 'Large-scale real-world autonomous driving dataset from Waymo'
        }
    }
    
    return dataset_info.get(dataset_type.lower(), {
        'name': 'Unknown Dataset',
        'type': 'unknown',
        'modalities': [],
        'classes': [],
        'image_size': (384, 1280),
        'description': 'Dataset information not available'
    }) 