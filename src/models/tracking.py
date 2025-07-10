"""
Object tracking models for autonomous driving
Includes DeepSORT, MOT, and other tracking algorithms
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.optimize import linear_sum_assignment
import cv2


class DeepSORT:
    """Deep SORT tracker implementation"""
    
    def __init__(self, 
                 max_age: int = 30,
                 min_hits: int = 3,
                 max_cosine_distance: float = 0.2):
        self.max_age = max_age
        self.min_hits = min_hits
        self.max_cosine_distance = max_cosine_distance
        
        # Track management
        self.tracks = []
        self.track_id_counter = 0
        
        # Feature encoder for appearance
        self.encoder = FeatureEncoder()
        
    def update(self, detections: List[Dict], image: np.ndarray) -> List[Dict]:
        """Update tracks with new detections"""
        
        # Predict existing tracks
        for track in self.tracks:
            track.predict()
        
        # Extract appearance features
        features = self._extract_features(detections, image)
        
        # Associate detections to tracks
        matches, unmatched_dets, unmatched_trks = self._associate(
            detections, self.tracks, features
        )
        
        # Update matched tracks
        for det_idx, trk_idx in matches:
            self.tracks[trk_idx].update(detections[det_idx], features[det_idx])
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            track = Track(
                detections[det_idx], 
                self.track_id_counter,
                features[det_idx]
            )
            self.tracks.append(track)
            self.track_id_counter += 1
        
        # Mark unmatched tracks
        for trk_idx in unmatched_trks:
            self.tracks[trk_idx].mark_missed()
        
        # Remove dead tracks
        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        
        # Return current tracks
        results = []
        for track in self.tracks:
            if track.is_confirmed():
                results.append({
                    'track_id': track.track_id,
                    'bbox': track.get_state(),
                    'class': track.class_id,
                    'confidence': track.confidence
                })
        
        return results
    
    def _extract_features(self, detections: List[Dict], image: np.ndarray) -> List[np.ndarray]:
        """Extract appearance features from detections"""
        features = []
        
        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            
            # Crop and resize
            crop = image[y1:y2, x1:x2]
            if crop.size > 0:
                crop_resized = cv2.resize(crop, (64, 128))
                feature = self.encoder.encode(crop_resized)
                features.append(feature)
            else:
                # Fallback for invalid crops
                features.append(np.zeros(128))
        
        return features
    
    def _associate(self, detections: List[Dict], tracks: List['Track'], features: List[np.ndarray]):
        """Associate detections to tracks using appearance and motion"""
        
        if len(tracks) == 0:
            return [], list(range(len(detections))), []
        
        # Compute cost matrix
        cost_matrix = np.zeros((len(detections), len(tracks)))
        
        for det_idx, (detection, feature) in enumerate(zip(detections, features)):
            for trk_idx, track in enumerate(tracks):
                if not track.is_confirmed():
                    cost_matrix[det_idx, trk_idx] = 1e6
                    continue
                
                # Appearance cost
                appearance_cost = self._cosine_distance(feature, track.feature)
                
                # Motion cost (IoU)
                motion_cost = 1.0 - self._iou(detection['bbox'], track.get_state())
                
                # Combined cost
                if appearance_cost > self.max_cosine_distance:
                    cost_matrix[det_idx, trk_idx] = 1e6
                else:
                    cost_matrix[det_idx, trk_idx] = 0.2 * appearance_cost + 0.8 * motion_cost
        
        # Hungarian assignment
        det_indices, trk_indices = linear_sum_assignment(cost_matrix)
        
        # Filter out high-cost assignments
        matches = []
        for det_idx, trk_idx in zip(det_indices, trk_indices):
            if cost_matrix[det_idx, trk_idx] < 0.7:
                matches.append((det_idx, trk_idx))
        
        unmatched_dets = [i for i in range(len(detections)) if i not in det_indices]
        unmatched_trks = [i for i in range(len(tracks)) if i not in trk_indices]
        
        return matches, unmatched_dets, unmatched_trks
    
    def _cosine_distance(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Compute cosine distance between features"""
        return 1.0 - np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))
    
    def _iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Compute IoU between two bounding boxes"""
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


class Track:
    """Individual track for multi-object tracking"""
    
    def __init__(self, detection: Dict, track_id: int, feature: np.ndarray):
        self.track_id = track_id
        self.class_id = detection['class']
        self.confidence = detection['confidence']
        self.feature = feature
        
        # State: [x, y, w, h, vx, vy, vw, vh]
        bbox = detection['bbox']
        x = (bbox[0] + bbox[2]) / 2
        y = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        
        self.state = np.array([x, y, w, h, 0, 0, 0, 0], dtype=np.float32)
        
        # Kalman filter
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        self._init_kalman_filter()
        
        # Track management
        self.hits = 1
        self.hit_streak = 1
        self.age = 1
        self.time_since_update = 0
        
        # Track states
        self.state_enum = {'Tentative': 1, 'Confirmed': 2, 'Deleted': 3}
        self.track_state = self.state_enum['Tentative']
        
    def _init_kalman_filter(self):
        """Initialize Kalman filter for tracking"""
        # State transition matrix
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement matrix
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ])
        
        # Process noise
        self.kf.Q = np.eye(8) * 0.1
        
        # Measurement noise
        self.kf.R = np.eye(4) * 1.0
        
        # Initial state
        self.kf.x = self.state.copy()
        self.kf.P = np.eye(8) * 1000
    
    def predict(self):
        """Predict next state"""
        self.age += 1
        self.time_since_update += 1
        self.kf.predict()
        self.state = self.kf.x.copy()
    
    def update(self, detection: Dict, feature: np.ndarray):
        """Update track with detection"""
        bbox = detection['bbox']
        x = (bbox[0] + bbox[2]) / 2
        y = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        
        measurement = np.array([x, y, w, h])
        self.kf.update(measurement)
        self.state = self.kf.x.copy()
        
        # Update track state
        self.hits += 1
        self.hit_streak += 1
        self.time_since_update = 0
        self.confidence = detection['confidence']
        self.feature = 0.9 * self.feature + 0.1 * feature  # EMA update
        
        # Confirm track
        if self.track_state == self.state_enum['Tentative'] and self.hits >= 3:
            self.track_state = self.state_enum['Confirmed']
    
    def mark_missed(self):
        """Mark track as missed"""
        self.hit_streak = 0
        if self.track_state == self.state_enum['Tentative']:
            self.track_state = self.state_enum['Deleted']
    
    def is_confirmed(self) -> bool:
        """Check if track is confirmed"""
        return self.track_state == self.state_enum['Confirmed']
    
    def is_deleted(self) -> bool:
        """Check if track should be deleted"""
        if self.track_state == self.state_enum['Deleted']:
            return True
        if self.time_since_update > 30:  # max_age
            return True
        return False
    
    def get_state(self) -> List[float]:
        """Get current bounding box"""
        x, y, w, h = self.state[:4]
        return [x - w/2, y - h/2, x + w/2, y + h/2]


class FeatureEncoder(nn.Module):
    """Feature encoder for appearance-based tracking"""
    
    def __init__(self, feature_dim: int = 128):
        super().__init__()
        
        self.backbone = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(3, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Conv Block 2
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Conv Block 3
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Global average pooling
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            
            # FC layers
            nn.Linear(128, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        # L2 normalize
        features = nn.functional.normalize(features, p=2, dim=1)
        return features
    
    def encode(self, image: np.ndarray) -> np.ndarray:
        """Encode single image to feature vector"""
        self.eval()
        
        # Preprocess
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0)
        
        with torch.no_grad():
            features = self.forward(image)
            return features.cpu().numpy().flatten()


class KalmanFilter:
    """Simple Kalman filter implementation"""
    
    def __init__(self, dim_x: int, dim_z: int):
        self.dim_x = dim_x
        self.dim_z = dim_z
        
        self.x = np.zeros(dim_x)  # state
        self.P = np.eye(dim_x)    # covariance
        self.F = np.eye(dim_x)    # state transition
        self.H = np.eye(dim_z, dim_x)  # measurement function
        self.Q = np.eye(dim_x)    # process noise
        self.R = np.eye(dim_z)    # measurement noise
    
    def predict(self):
        """Prediction step"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
    
    def update(self, z):
        """Update step"""
        y = z - self.H @ self.x  # residual
        S = self.H @ self.P @ self.H.T + self.R  # residual covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        
        self.x = self.x + K @ y
        self.P = self.P - K @ self.H @ self.P


class MOT:
    """Multi-Object Tracker (simplified SORT-like implementation)"""
    
    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        
        self.tracks = []
        self.track_id_counter = 0
    
    def update(self, detections: List[Dict]) -> List[Dict]:
        """Update tracker with new detections"""
        
        # Predict existing tracks
        for track in self.tracks:
            track.predict()
        
        # Association
        if len(self.tracks) == 0:
            matches = []
            unmatched_dets = list(range(len(detections)))
            unmatched_trks = []
        else:
            matches, unmatched_dets, unmatched_trks = self._associate_detections_to_trackers(
                detections, self.tracks
            )
        
        # Update matched tracks
        for det_idx, trk_idx in matches:
            self.tracks[trk_idx].update(detections[det_idx], np.zeros(128))
        
        # Create new tracks
        for det_idx in unmatched_dets:
            track = Track(detections[det_idx], self.track_id_counter, np.zeros(128))
            self.tracks.append(track)
            self.track_id_counter += 1
        
        # Mark unmatched tracks
        for trk_idx in unmatched_trks:
            self.tracks[trk_idx].mark_missed()
        
        # Remove dead tracks
        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        
        # Return results
        results = []
        for track in self.tracks:
            if track.is_confirmed() and track.time_since_update < 1:
                results.append({
                    'track_id': track.track_id,
                    'bbox': track.get_state(),
                    'class': track.class_id,
                    'confidence': track.confidence
                })
        
        return results
    
    def _associate_detections_to_trackers(self, detections, tracks):
        """Associate detections to tracks using IoU"""
        
        iou_matrix = np.zeros((len(detections), len(tracks)))
        
        for det_idx, detection in enumerate(detections):
            for trk_idx, track in enumerate(tracks):
                iou_matrix[det_idx, trk_idx] = self._iou(detection['bbox'], track.get_state())
        
        # Convert to cost matrix
        cost_matrix = 1.0 - iou_matrix
        
        # Hungarian assignment
        det_indices, trk_indices = linear_sum_assignment(cost_matrix)
        
        # Filter low IoU matches
        matches = []
        for det_idx, trk_idx in zip(det_indices, trk_indices):
            if iou_matrix[det_idx, trk_idx] >= self.iou_threshold:
                matches.append((det_idx, trk_idx))
        
        unmatched_dets = [i for i in range(len(detections)) if i not in det_indices]
        unmatched_trks = [i for i in range(len(tracks)) if i not in trk_indices]
        
        return matches, unmatched_dets, unmatched_trks
    
    def _iou(self, bbox1, bbox2):
        """Compute IoU between two bounding boxes"""
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