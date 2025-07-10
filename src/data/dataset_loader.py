"""
Dataset loader for autonomous driving datasets (KITTI, CARLA, etc.)
Supports multi-modal data: RGB images, LiDAR point clouds, semantic segmentation
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import open3d as o3d
import albumentations as A
from pathlib import Path


class BaseDataset(Dataset):
    """Base dataset class for autonomous driving data"""
    
    def __init__(self, 
                 data_root: str,
                 split: str = 'train',
                 transform=None,
                 modalities: List[str] = ['camera'],
                 config: Dict = None):
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.modalities = modalities
        self.config = config or {}
        
        # Supported object classes
        self.class_names = [
            'Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting',
            'Cyclist', 'Tram', 'Misc', 'DontCare', 'Traffic_Sign'
        ]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
    def __len__(self):
        raise NotImplementedError
        
    def __getitem__(self, idx):
        raise NotImplementedError
        
    def load_image(self, image_path: str) -> np.ndarray:
        """Load RGB image"""
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
        
    def load_lidar(self, lidar_path: str) -> np.ndarray:
        """Load LiDAR point cloud"""
        points = np.fromfile(str(lidar_path), dtype=np.float32).reshape(-1, 4)
        return points
        
    def load_calibration(self, calib_path: str) -> Dict:
        """Load camera calibration parameters"""
        calib = {}
        with open(calib_path, 'r') as f:
            for line in f.readlines():
                if ':' in line:
                    key, value = line.split(':', 1)
                    calib[key] = np.array([float(x) for x in value.split()])
        return calib


class KITTIDataset(BaseDataset):
    """KITTI dataset loader for object detection and tracking"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_dir = self.data_root / 'training' / 'image_2'
        self.lidar_dir = self.data_root / 'training' / 'velodyne'
        self.label_dir = self.data_root / 'training' / 'label_2'
        self.calib_dir = self.data_root / 'training' / 'calib'
        
        self.image_files = sorted(list(self.image_dir.glob('*.png')))
        
    def __len__(self):
        return len(self.image_files)
        
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        file_id = image_path.stem
        
        # Load image
        image = self.load_image(image_path)
        
        # Initialize sample dictionary
        sample = {
            'image': image,
            'file_id': file_id,
            'dataset': 'kitti'
        }
        
        # Load LiDAR if requested
        if 'lidar' in self.modalities:
            lidar_path = self.lidar_dir / f"{file_id}.bin"
            if lidar_path.exists():
                sample['lidar'] = self.load_lidar(lidar_path)
                
        # Load labels if available
        label_path = self.label_dir / f"{file_id}.txt"
        if label_path.exists():
            sample.update(self.load_labels(label_path))
            
        # Load calibration
        calib_path = self.calib_dir / f"{file_id}.txt"
        if calib_path.exists():
            sample['calibration'] = self.load_calibration(calib_path)
            
        # Apply transforms
        if self.transform:
            sample = self.transform(sample)
            
        return sample
        
    def load_labels(self, label_path: str) -> Dict:
        """Load KITTI format labels"""
        boxes = []
        classes = []
        
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split(' ')
                if len(parts) >= 15:
                    class_name = parts[0]
                    if class_name in self.class_to_idx:
                        # 2D bounding box (left, top, right, bottom)
                        bbox_2d = [float(parts[i]) for i in range(4, 8)]
                        
                        # 3D bounding box
                        dimensions = [float(parts[i]) for i in range(8, 11)]  # h, w, l
                        location = [float(parts[i]) for i in range(11, 14)]   # x, y, z
                        rotation_y = float(parts[14])
                        
                        boxes.append({
                            'bbox_2d': bbox_2d,
                            'dimensions': dimensions,
                            'location': location,
                            'rotation_y': rotation_y
                        })
                        classes.append(self.class_to_idx[class_name])
                        
        return {
            'boxes': boxes,
            'classes': classes,
            'num_objects': len(boxes)
        }


class CARLADataset(BaseDataset):
    """CARLA simulation dataset loader"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_dir = self.data_root / 'CameraRGB'
        self.lidar_dir = self.data_root / 'Lidar'
        self.semantic_dir = self.data_root / 'CameraSemSeg'
        self.metadata_file = self.data_root / 'metadata.json'
        
        # Load metadata
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
            
        self.image_files = sorted(list(self.image_dir.glob('*.png')))
        
    def __len__(self):
        return len(self.image_files)
        
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        file_id = image_path.stem
        
        # Load image
        image = self.load_image(image_path)
        
        sample = {
            'image': image,
            'file_id': file_id,
            'dataset': 'carla'
        }
        
        # Load LiDAR if requested
        if 'lidar' in self.modalities:
            lidar_path = self.lidar_dir / f"{file_id}.ply"
            if lidar_path.exists():
                sample['lidar'] = self.load_carla_lidar(lidar_path)
                
        # Load semantic segmentation if requested
        if 'semantic' in self.modalities:
            semantic_path = self.semantic_dir / f"{file_id}.png"
            if semantic_path.exists():
                sample['semantic'] = self.load_semantic(semantic_path)
                
        # Load metadata for this frame
        if file_id in self.metadata:
            sample['metadata'] = self.metadata[file_id]
            
        # Apply transforms
        if self.transform:
            sample = self.transform(sample)
            
        return sample
        
    def load_carla_lidar(self, lidar_path: str) -> np.ndarray:
        """Load CARLA LiDAR point cloud (PLY format)"""
        pcd = o3d.io.read_point_cloud(str(lidar_path))
        points = np.asarray(pcd.points)
        # Add dummy intensity channel for consistency
        intensities = np.ones((points.shape[0], 1))
        return np.hstack([points, intensities])
        
    def load_semantic(self, semantic_path: str) -> np.ndarray:
        """Load semantic segmentation mask"""
        semantic = cv2.imread(str(semantic_path), cv2.IMREAD_GRAYSCALE)
        return semantic


class DatasetLoader:
    """Unified dataset loader for multiple autonomous driving datasets"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.datasets = {}
        
    def load_dataset(self, 
                    dataset_name: str,
                    split: str = 'train',
                    batch_size: int = 8,
                    num_workers: int = 4,
                    shuffle: bool = True) -> DataLoader:
        """Load a specific dataset"""
        
        if dataset_name == 'kitti':
            dataset = KITTIDataset(
                data_root=self.config['data']['datasets']['kitti']['path'],
                split=split,
                modalities=self.config['data']['datasets']['kitti']['modalities'],
                config=self.config
            )
        elif dataset_name == 'carla':
            dataset = CARLADataset(
                data_root=self.config['data']['datasets']['carla']['path'],
                split=split,
                modalities=self.config['data']['datasets']['carla']['modalities'],
                config=self.config
            )
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
            
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=self.collate_fn
        )
        
    def load_mixed_dataset(self, 
                          dataset_names: List[str],
                          split: str = 'train',
                          batch_size: int = 8,
                          num_workers: int = 4) -> DataLoader:
        """Load mixed dataset from multiple sources"""
        datasets = []
        
        for name in dataset_names:
            if name == 'kitti':
                dataset = KITTIDataset(
                    data_root=self.config['data']['datasets']['kitti']['path'],
                    split=split,
                    modalities=self.config['data']['datasets']['kitti']['modalities'],
                    config=self.config
                )
            elif name == 'carla':
                dataset = CARLADataset(
                    data_root=self.config['data']['datasets']['carla']['path'],
                    split=split,
                    modalities=self.config['data']['datasets']['carla']['modalities'],
                    config=self.config
                )
            datasets.append(dataset)
            
        # Concatenate datasets
        mixed_dataset = torch.utils.data.ConcatDataset(datasets)
        
        return DataLoader(
            mixed_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=self.collate_fn
        )
        
    @staticmethod
    def collate_fn(batch):
        """Custom collate function for handling variable-size data"""
        # Separate different data types
        images = []
        lidar_data = []
        labels = []
        metadata = []
        
        for sample in batch:
            images.append(sample['image'])
            
            if 'lidar' in sample:
                lidar_data.append(sample['lidar'])
                
            if 'boxes' in sample:
                labels.append({
                    'boxes': sample['boxes'],
                    'classes': sample['classes']
                })
                
            metadata.append({
                'file_id': sample['file_id'],
                'dataset': sample['dataset']
            })
            
        # Stack images
        images = torch.stack([torch.from_numpy(img.transpose(2, 0, 1)) for img in images])
        
        result = {
            'images': images,
            'metadata': metadata
        }
        
        if lidar_data:
            result['lidar'] = lidar_data
            
        if labels:
            result['labels'] = labels
            
        return result 