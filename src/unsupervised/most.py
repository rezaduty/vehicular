"""
MOST: Multi-Object Self-supervised Tracking
Implementation for autonomous driving perception
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import cv2
from scipy.optimize import linear_sum_assignment


class MOSTTracker:
    """Multi-Object Self-supervised Tracking implementation"""
    
    def __init__(self, 
                 max_age: int = 30,
                 min_hits: int = 3,
                 iou_threshold: float = 0.3,
                 feature_dim: int = 256):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.feature_dim = feature_dim
        
        # Track management
        self.tracks = []
        self.track_id_counter = 0
        
        # Feature extractor for self-supervised learning
        self.feature_extractor = SelfSupervisedFeatureExtractor(feature_dim)
        
        # Self-supervised learning components
        self.temporal_consistency_loss = TemporalConsistencyLoss()
        self.contrastive_loss = ContrastiveLoss()
        
    def update(self, detections: List[Dict], image: np.ndarray) -> List[Dict]:
        """Update tracks with new detections using self-supervised learning"""
        
        # Extract self-supervised features
        features = self._extract_self_supervised_features(detections, image)
        
        # Predict existing tracks
        for track in self.tracks:
            track.predict()
        
        # Associate detections to tracks
        matches, unmatched_dets, unmatched_trks = self._associate_with_self_supervision(
            detections, self.tracks, features
        )
        
        # Update matched tracks
        for det_idx, trk_idx in matches:
            self.tracks[trk_idx].update(detections[det_idx], features[det_idx])
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            track = SelfSupervisedTrack(
                detections[det_idx], 
                self.track_id_counter,
                features[det_idx]
            )
            self.tracks.append(track)
            self.track_id_counter += 1
        
        # Mark unmatched tracks
        for trk_idx in unmatched_trks:
            self.tracks[trk_idx].mark_missed()
        
        # Self-supervised learning update
        self._update_self_supervised_learning(image, detections, features)
        
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
                    'confidence': track.confidence,
                    'feature': track.feature
                })
        
        return results
    
    def _extract_self_supervised_features(self, detections: List[Dict], image: np.ndarray) -> List[np.ndarray]:
        """Extract self-supervised features from detections"""
        features = []
        
        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            
            # Crop object region
            crop = image[y1:y2, x1:x2]
            if crop.size > 0:
                crop_resized = cv2.resize(crop, (128, 64))
                
                # Extract self-supervised features
                feature = self.feature_extractor.extract_features(crop_resized)
                features.append(feature)
            else:
                # Fallback for invalid crops
                features.append(np.zeros(self.feature_dim))
        
        return features
    
    def _associate_with_self_supervision(self, detections: List[Dict], tracks: List['SelfSupervisedTrack'], 
                                       features: List[np.ndarray]):
        """Associate detections to tracks using self-supervised features"""
        
        if len(tracks) == 0:
            return [], list(range(len(detections))), []
        
        # Compute cost matrix combining IoU and self-supervised features
        cost_matrix = np.zeros((len(detections), len(tracks)))
        
        for det_idx, (detection, feature) in enumerate(zip(detections, features)):
            for trk_idx, track in enumerate(tracks):
                if not track.is_confirmed():
                    cost_matrix[det_idx, trk_idx] = 1e6
                    continue
                
                # IoU cost
                iou_cost = 1.0 - self._calculate_iou(detection['bbox'], track.get_state())
                
                # Self-supervised feature similarity
                feature_cost = self._compute_feature_distance(feature, track.feature)
                
                # Temporal consistency cost
                temporal_cost = self._compute_temporal_consistency(track, detection)
                
                # Combined cost
                cost_matrix[det_idx, trk_idx] = (0.4 * iou_cost + 
                                               0.4 * feature_cost + 
                                               0.2 * temporal_cost)
        
        # Hungarian assignment
        det_indices, trk_indices = linear_sum_assignment(cost_matrix)
        
        # Filter out high-cost assignments
        matches = []
        for det_idx, trk_idx in zip(det_indices, trk_indices):
            if cost_matrix[det_idx, trk_idx] < 0.8:
                matches.append((det_idx, trk_idx))
        
        unmatched_dets = [i for i in range(len(detections)) if i not in det_indices]
        unmatched_trks = [i for i in range(len(tracks)) if i not in trk_indices]
        
        return matches, unmatched_dets, unmatched_trks
    
    def _compute_feature_distance(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Compute distance between self-supervised features"""
        return 1.0 - np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2) + 1e-8)
    
    def _compute_temporal_consistency(self, track: 'SelfSupervisedTrack', detection: Dict) -> float:
        """Compute temporal consistency cost"""
        if len(track.history) < 2:
            return 0.0
        
        # Simple motion prediction
        predicted_center = track.predict_next_center()
        current_center = self._get_bbox_center(detection['bbox'])
        
        distance = np.linalg.norm(np.array(predicted_center) - np.array(current_center))
        
        # Normalize by image size (assuming 1280x384)
        normalized_distance = distance / np.sqrt(1280**2 + 384**2)
        
        return min(normalized_distance, 1.0)
    
    def _get_bbox_center(self, bbox: List[float]) -> Tuple[float, float]:
        """Get center of bounding box"""
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
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
    
    def _update_self_supervised_learning(self, image: np.ndarray, detections: List[Dict], 
                                       features: List[np.ndarray]):
        """Update self-supervised learning components"""
        
        # Update feature extractor with temporal consistency
        if len(self.tracks) > 1:
            self._update_temporal_consistency(features)
        
        # Update contrastive learning
        if len(features) > 1:
            self._update_contrastive_learning(features)
    
    def _update_temporal_consistency(self, features: List[np.ndarray]):
        """Update temporal consistency learning"""
        # Simplified temporal consistency update
        for track in self.tracks:
            if len(track.feature_history) > 1:
                current_feature = track.feature
                previous_feature = track.feature_history[-2]
                
                # Temporal consistency loss
                consistency_loss = self.temporal_consistency_loss(current_feature, previous_feature)
                
                # Update feature extractor (simplified)
                # In practice, this would involve gradient updates
                pass
    
    def _update_contrastive_learning(self, features: List[np.ndarray]):
        """Update contrastive learning"""
        # Simplified contrastive learning update
        for i, feat1 in enumerate(features):
            for j, feat2 in enumerate(features):
                if i != j:
                    # Contrastive loss for different objects
                    contrastive_loss = self.contrastive_loss(feat1, feat2, target=0)
                    
                    # Update feature extractor (simplified)
                    # In practice, this would involve gradient updates
                    pass


class SelfSupervisedTrack:
    """Individual track for self-supervised multi-object tracking"""
    
    def __init__(self, detection: Dict, track_id: int, feature: np.ndarray):
        self.track_id = track_id
        self.class_id = detection['class']
        self.confidence = detection['confidence']
        self.feature = feature
        
        # State: [x, y, w, h, vx, vy]
        bbox = detection['bbox']
        x = (bbox[0] + bbox[2]) / 2
        y = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        
        self.state = np.array([x, y, w, h, 0, 0], dtype=np.float32)
        
        # History for self-supervised learning
        self.history = [self.state.copy()]
        self.feature_history = [feature.copy()]
        
        # Track management
        self.hits = 1
        self.hit_streak = 1
        self.age = 1
        self.time_since_update = 0
        
        # Track states
        self.state_enum = {'Tentative': 1, 'Confirmed': 2, 'Deleted': 3}
        self.track_state = self.state_enum['Tentative']
    
    def predict(self):
        """Predict next state using motion model"""
        self.age += 1
        self.time_since_update += 1
        
        # Simple constant velocity model
        self.state[0] += self.state[4]  # x += vx
        self.state[1] += self.state[5]  # y += vy
    
    def update(self, detection: Dict, feature: np.ndarray):
        """Update track with detection"""
        bbox = detection['bbox']
        x = (bbox[0] + bbox[2]) / 2
        y = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        
        # Update velocity
        if len(self.history) > 0:
            prev_state = self.history[-1]
            self.state[4] = x - prev_state[0]  # vx
            self.state[5] = y - prev_state[1]  # vy
        
        # Update position and size
        self.state[0] = x
        self.state[1] = y
        self.state[2] = w
        self.state[3] = h
        
        # Update feature with exponential moving average
        self.feature = 0.8 * self.feature + 0.2 * feature
        
        # Update history
        self.history.append(self.state.copy())
        self.feature_history.append(feature.copy())
        
        # Keep limited history
        if len(self.history) > 10:
            self.history.pop(0)
            self.feature_history.pop(0)
        
        # Update track state
        self.hits += 1
        self.hit_streak += 1
        self.time_since_update = 0
        self.confidence = detection['confidence']
        
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
    
    def predict_next_center(self) -> Tuple[float, float]:
        """Predict next center position"""
        if len(self.history) < 2:
            return (self.state[0], self.state[1])
        
        # Use velocity to predict
        next_x = self.state[0] + self.state[4]
        next_y = self.state[1] + self.state[5]
        
        return (next_x, next_y)


class SelfSupervisedFeatureExtractor:
    """Self-supervised feature extractor for tracking"""
    
    def __init__(self, feature_dim: int = 256):
        self.feature_dim = feature_dim
        # Simple feature extraction using traditional methods
        # In practice, this would be a neural network
        
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extract self-supervised features from image patch"""
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Extract multiple types of features
        features = []
        
        # 1. Histogram of Oriented Gradients (HOG)
        hog_features = self._extract_hog_features(gray)
        features.extend(hog_features)
        
        # 2. Local Binary Patterns (LBP)
        lbp_features = self._extract_lbp_features(gray)
        features.extend(lbp_features)
        
        # 3. Color histogram (if color image)
        if len(image.shape) == 3:
            color_features = self._extract_color_features(image)
            features.extend(color_features)
        
        # Normalize and pad/truncate to desired dimension
        features = np.array(features)
        
        if len(features) > self.feature_dim:
            features = features[:self.feature_dim]
        elif len(features) < self.feature_dim:
            features = np.pad(features, (0, self.feature_dim - len(features)))
        
        # L2 normalize
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        return features
    
    def _extract_hog_features(self, image: np.ndarray) -> List[float]:
        """Extract HOG features"""
        # Resize image for consistent feature extraction
        image_resized = cv2.resize(image, (64, 32))
        
        # Compute gradients
        grad_x = cv2.Sobel(image_resized, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image_resized, cv2.CV_64F, 0, 1, ksize=3)
        
        # Compute magnitude and angle
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        angle = np.arctan2(grad_y, grad_x)
        
        # Simple histogram of gradients (simplified HOG)
        hist, _ = np.histogram(angle.flatten(), bins=9, range=(-np.pi, np.pi), weights=magnitude.flatten())
        
        return hist.tolist()
    
    def _extract_lbp_features(self, image: np.ndarray) -> List[float]:
        """Extract Local Binary Pattern features"""
        # Resize image
        image_resized = cv2.resize(image, (32, 16))
        
        # Simple LBP implementation
        lbp = np.zeros_like(image_resized)
        
        for i in range(1, image_resized.shape[0] - 1):
            for j in range(1, image_resized.shape[1] - 1):
                center = image_resized[i, j]
                code = 0
                
                # 8-neighborhood
                neighbors = [
                    image_resized[i-1, j-1], image_resized[i-1, j], image_resized[i-1, j+1],
                    image_resized[i, j+1], image_resized[i+1, j+1], image_resized[i+1, j],
                    image_resized[i+1, j-1], image_resized[i, j-1]
                ]
                
                for k, neighbor in enumerate(neighbors):
                    if neighbor >= center:
                        code |= (1 << k)
                
                lbp[i, j] = code
        
        # Histogram of LBP codes
        hist, _ = np.histogram(lbp.flatten(), bins=32, range=(0, 256))
        
        return hist.tolist()
    
    def _extract_color_features(self, image: np.ndarray) -> List[float]:
        """Extract color histogram features"""
        # Convert to HSV for better color representation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Compute histograms for each channel
        hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256])
        
        # Concatenate and normalize
        color_features = np.concatenate([hist_h.flatten(), hist_s.flatten(), hist_v.flatten()])
        
        return color_features.tolist()


class TemporalConsistencyLoss:
    """Temporal consistency loss for self-supervised learning"""
    
    def __call__(self, feature_t: np.ndarray, feature_t_minus_1: np.ndarray) -> float:
        """Compute temporal consistency loss"""
        # Simple L2 distance (encouraging temporal smoothness)
        diff = feature_t - feature_t_minus_1
        loss = np.mean(diff**2)
        return loss


class ContrastiveLoss:
    """Contrastive loss for self-supervised learning"""
    
    def __init__(self, margin: float = 1.0):
        self.margin = margin
    
    def __call__(self, feature1: np.ndarray, feature2: np.ndarray, target: int) -> float:
        """Compute contrastive loss"""
        # Euclidean distance
        distance = np.linalg.norm(feature1 - feature2)
        
        if target == 1:  # Same object
            loss = 0.5 * distance**2
        else:  # Different objects
            loss = 0.5 * max(0, self.margin - distance)**2
        
        return loss


class MultiObjectSelfSupervisedTracking:
    """Complete multi-object self-supervised tracking system"""
    
    def __init__(self, config: Optional[Dict] = None):
        if config is None:
            config = {
                'max_age': 30,
                'min_hits': 3,
                'iou_threshold': 0.3,
                'feature_dim': 256
            }
        
        self.tracker = MOSTTracker(**config)
        
    def track_objects(self, detections: List[Dict], image: np.ndarray) -> List[Dict]:
        """Track objects across frames using self-supervised learning"""
        return self.tracker.update(detections, image)
    
    def reset(self):
        """Reset tracker state"""
        self.tracker.tracks = []
        self.tracker.track_id_counter = 0 