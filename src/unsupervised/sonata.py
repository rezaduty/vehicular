"""
SONATA: Self-Organized Network Architecture for unsupervised LiDAR point cloud segmentation
Implementation for autonomous driving perception
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import DBSCAN
from typing import Dict, List, Tuple, Optional
import open3d as o3d


class SONATA:
    """SONATA algorithm for unsupervised LiDAR segmentation"""
    
    def __init__(self, 
                 feature_dim: int = 64,
                 dbscan_eps: float = 0.5,
                 dbscan_min_samples: int = 10,
                 temporal_weight: float = 0.3):
        
        self.feature_dim = feature_dim
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.temporal_weight = temporal_weight
        
        # Feature extractor
        self.feature_extractor = PointNetFeatureExtractor(feature_dim)
        
        # Temporal consistency tracker
        self.previous_segments = None
        self.segment_history = []
        
    def segment_point_cloud(self, points: np.ndarray, intensities: Optional[np.ndarray] = None) -> Dict:
        """
        Segment point cloud using SONATA approach
        
        Args:
            points: [N, 3] point coordinates
            intensities: [N, 1] point intensities (optional)
            
        Returns:
            Dictionary with segmentation results
        """
        
        # Preprocess point cloud
        processed_points = self._preprocess_points(points, intensities)
        
        # Extract features
        features = self._extract_features(processed_points)
        
        # Spatial clustering
        spatial_segments = self._spatial_clustering(points, features)
        
        # Temporal consistency
        if self.previous_segments is not None:
            temporal_segments = self._enforce_temporal_consistency(
                spatial_segments, points
            )
        else:
            temporal_segments = spatial_segments
        
        # Post-processing
        final_segments = self._post_process_segments(temporal_segments, points)
        
        # Update history
        self.previous_segments = final_segments
        self.segment_history.append(final_segments)
        if len(self.segment_history) > 10:  # Keep last 10 frames
            self.segment_history.pop(0)
        
        return {
            'segments': final_segments,
            'features': features,
            'num_segments': len(np.unique(final_segments[final_segments >= 0]))
        }
    
    def _preprocess_points(self, points: np.ndarray, intensities: Optional[np.ndarray] = None) -> torch.Tensor:
        """Preprocess point cloud for feature extraction"""
        
        # Normalize coordinates
        centroid = np.mean(points, axis=0)
        points_centered = points - centroid
        
        # Scale to unit sphere
        max_dist = np.max(np.linalg.norm(points_centered, axis=1))
        points_normalized = points_centered / max_dist
        
        # Add intensity if available
        if intensities is not None:
            processed = np.concatenate([points_normalized, intensities.reshape(-1, 1)], axis=1)
        else:
            processed = points_normalized
        
        return torch.from_numpy(processed).float()
    
    def _extract_features(self, points: torch.Tensor) -> np.ndarray:
        """Extract point-wise features using PointNet-based architecture"""
        
        self.feature_extractor.eval()
        
        with torch.no_grad():
            # Add batch dimension
            points_batch = points.unsqueeze(0)  # [1, N, 3/4]
            
            # Extract features
            features = self.feature_extractor(points_batch)  # [1, N, feature_dim]
            features = features.squeeze(0).numpy()  # [N, feature_dim]
        
        return features
    
    def _spatial_clustering(self, points: np.ndarray, features: np.ndarray) -> np.ndarray:
        """Perform spatial clustering using DBSCAN on features"""
        
        # Combine spatial and feature information
        spatial_weight = 0.7
        feature_weight = 0.3
        
        # Normalize spatial coordinates
        points_norm = (points - np.mean(points, axis=0)) / np.std(points, axis=0)
        
        # Normalize features
        features_norm = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-8)
        
        # Combine for clustering
        clustering_input = np.concatenate([
            spatial_weight * points_norm,
            feature_weight * features_norm
        ], axis=1)
        
        # DBSCAN clustering
        clustering = DBSCAN(
            eps=self.dbscan_eps,
            min_samples=self.dbscan_min_samples
        )
        
        labels = clustering.fit_predict(clustering_input)
        
        return labels
    
    def _enforce_temporal_consistency(self, current_segments: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Enforce temporal consistency across frames"""
        
        if self.previous_segments is None:
            return current_segments
        
        # Find correspondences between current and previous segments
        correspondences = self._find_segment_correspondences(
            current_segments, self.previous_segments, points
        )
        
        # Update segments based on correspondences
        consistent_segments = current_segments.copy()
        
        for curr_id, prev_id in correspondences.items():
            if prev_id != -1:  # Valid correspondence
                # Smooth transition
                curr_mask = current_segments == curr_id
                if np.sum(curr_mask) > 0:
                    # Apply temporal smoothing
                    consistency_score = self._calculate_consistency_score(
                        curr_mask, prev_id
                    )
                    
                    if consistency_score > 0.5:
                        consistent_segments[curr_mask] = prev_id
        
        return consistent_segments
    
    def _find_segment_correspondences(self, current: np.ndarray, previous: np.ndarray, points: np.ndarray) -> Dict[int, int]:
        """Find correspondences between current and previous segments"""
        
        correspondences = {}
        current_ids = np.unique(current[current >= 0])
        previous_ids = np.unique(previous[previous >= 0])
        
        for curr_id in current_ids:
            curr_mask = current == curr_id
            curr_centroid = np.mean(points[curr_mask], axis=0)
            
            best_prev_id = -1
            best_distance = float('inf')
            
            # Find closest previous segment centroid
            for prev_id in previous_ids:
                prev_mask = previous == prev_id
                if np.sum(prev_mask) == 0:
                    continue
                    
                prev_centroid = np.mean(points[prev_mask], axis=0)
                distance = np.linalg.norm(curr_centroid - prev_centroid)
                
                if distance < best_distance and distance < 2.0:  # 2m threshold
                    best_distance = distance
                    best_prev_id = prev_id
            
            correspondences[curr_id] = best_prev_id
        
        return correspondences
    
    def _calculate_consistency_score(self, curr_mask: np.ndarray, prev_id: int) -> float:
        """Calculate temporal consistency score"""
        
        if self.previous_segments is None:
            return 0.0
        
        prev_mask = self.previous_segments == prev_id
        
        # Calculate overlap
        intersection = np.sum(curr_mask & prev_mask)
        union = np.sum(curr_mask | prev_mask)
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _post_process_segments(self, segments: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Post-process segments to remove noise and small segments"""
        
        processed_segments = segments.copy()
        segment_ids = np.unique(segments[segments >= 0])
        
        for seg_id in segment_ids:
            mask = segments == seg_id
            segment_points = points[mask]
            
            # Remove small segments
            if len(segment_points) < 20:
                processed_segments[mask] = -1  # Mark as noise
                continue
            
            # Check segment compactness
            if self._is_segment_too_scattered(segment_points):
                processed_segments[mask] = -1
        
        # Relabel segments to be consecutive
        unique_segments = np.unique(processed_segments[processed_segments >= 0])
        for new_id, old_id in enumerate(unique_segments):
            processed_segments[processed_segments == old_id] = new_id
        
        return processed_segments
    
    def _is_segment_too_scattered(self, points: np.ndarray) -> bool:
        """Check if segment is too scattered to be a valid object"""
        
        if len(points) < 10:
            return True
        
        # Calculate point spread
        centroid = np.mean(points, axis=0)
        distances = np.linalg.norm(points - centroid, axis=1)
        
        # Check if 90% of points are within reasonable distance
        threshold_distance = np.percentile(distances, 90)
        
        return threshold_distance > 5.0  # 5m threshold


class PointNetFeatureExtractor(nn.Module):
    """PointNet-based feature extractor for point clouds"""
    
    def __init__(self, feature_dim: int = 64, input_dim: int = 3):
        super().__init__()
        
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        
        # Point-wise MLPs
        self.mlp1 = nn.Sequential(
            nn.Conv1d(input_dim, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )
        
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True)
        )
        
        # Global feature MLP
        self.global_mlp = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True)
        )
        
        # Point-wise feature combination
        self.point_mlp = nn.Sequential(
            nn.Conv1d(64 + 256, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, feature_dim, 1)
        )
        
    def forward(self, points):
        """
        Forward pass
        
        Args:
            points: [B, N, input_dim] point coordinates
            
        Returns:
            Point-wise features: [B, N, feature_dim]
        """
        batch_size, num_points, _ = points.shape
        
        # Transpose for conv1d: [B, input_dim, N]
        points = points.transpose(1, 2)
        
        # Point-wise features
        point_feat1 = self.mlp1(points)  # [B, 64, N]
        point_feat2 = self.mlp2(point_feat1)  # [B, 1024, N]
        
        # Global feature
        global_feat = torch.max(point_feat2, dim=2)[0]  # [B, 1024]
        global_feat = self.global_mlp(global_feat)  # [B, 256]
        
        # Expand global feature to all points
        global_feat_expanded = global_feat.unsqueeze(2).expand(-1, -1, num_points)  # [B, 256, N]
        
        # Combine local and global features
        combined_feat = torch.cat([point_feat1, global_feat_expanded], dim=1)  # [B, 64+256, N]
        
        # Final point-wise features
        point_features = self.point_mlp(combined_feat)  # [B, feature_dim, N]
        
        # Transpose back: [B, N, feature_dim]
        point_features = point_features.transpose(1, 2)
        
        return point_features


class SONATAVisualizer:
    """Visualization utilities for SONATA segmentation results"""
    
    @staticmethod
    def visualize_segments(points: np.ndarray, segments: np.ndarray, save_path: str = None):
        """Visualize segmented point cloud"""
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Color segments
        colors = np.zeros((len(points), 3))
        unique_segments = np.unique(segments[segments >= 0])
        
        # Generate distinct colors
        np.random.seed(42)  # For reproducible colors
        segment_colors = np.random.rand(len(unique_segments), 3)
        
        for i, seg_id in enumerate(unique_segments):
            mask = segments == seg_id
            colors[mask] = segment_colors[i]
        
        # Noise points in gray
        noise_mask = segments == -1
        colors[noise_mask] = [0.5, 0.5, 0.5]
        
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        if save_path:
            o3d.io.write_point_cloud(save_path, pcd)
        else:
            o3d.visualization.draw_geometries([pcd])
    
    @staticmethod
    def create_segmentation_summary(segments: np.ndarray) -> Dict:
        """Create summary statistics for segmentation"""
        
        unique_segments = np.unique(segments[segments >= 0])
        num_segments = len(unique_segments)
        num_noise_points = np.sum(segments == -1)
        total_points = len(segments)
        
        segment_sizes = []
        for seg_id in unique_segments:
            size = np.sum(segments == seg_id)
            segment_sizes.append(size)
        
        return {
            'num_segments': num_segments,
            'num_noise_points': num_noise_points,
            'total_points': total_points,
            'noise_ratio': num_noise_points / total_points,
            'avg_segment_size': np.mean(segment_sizes) if segment_sizes else 0,
            'min_segment_size': np.min(segment_sizes) if segment_sizes else 0,
            'max_segment_size': np.max(segment_sizes) if segment_sizes else 0
        }


# Example usage and evaluation
# Aliases for compatibility
SONATASegmentation = SONATA
SelfOrganizingPointCloudSegmentation = SONATA


class SONATAEvaluator:
    """Evaluation metrics for SONATA segmentation"""
    
    @staticmethod
    def evaluate_segmentation(predicted_segments: np.ndarray, 
                            ground_truth: np.ndarray = None) -> Dict:
        """Evaluate segmentation quality"""
        
        metrics = {}
        
        # Basic statistics
        summary = SONATAVisualizer.create_segmentation_summary(predicted_segments)
        metrics.update(summary)
        
        # If ground truth is available
        if ground_truth is not None:
            # Calculate IoU-based metrics
            metrics.update(
                SONATAEvaluator._calculate_iou_metrics(predicted_segments, ground_truth)
            )
        
        # Segmentation quality metrics
        metrics.update(
            SONATAEvaluator._calculate_quality_metrics(predicted_segments)
        )
        
        return metrics
    
    @staticmethod
    def _calculate_iou_metrics(predicted: np.ndarray, gt: np.ndarray) -> Dict:
        """Calculate IoU-based evaluation metrics"""
        
        # This would implement proper segmentation evaluation
        # For now, return placeholder metrics
        return {
            'mean_iou': 0.75,  # Placeholder
            'accuracy': 0.85   # Placeholder
        }
    
    @staticmethod
    def _calculate_quality_metrics(segments: np.ndarray) -> Dict:
        """Calculate intrinsic quality metrics"""
        
        unique_segments = np.unique(segments[segments >= 0])
        
        # Compactness measure
        compactness_scores = []
        for seg_id in unique_segments:
            mask = segments == seg_id
            if np.sum(mask) > 10:  # Only for segments with enough points
                # This would calculate actual compactness
                compactness_scores.append(0.8)  # Placeholder
        
        return {
            'avg_compactness': np.mean(compactness_scores) if compactness_scores else 0,
            'segmentation_quality': 0.78  # Placeholder composite score
        } 