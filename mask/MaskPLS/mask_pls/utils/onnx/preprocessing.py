"""
Preprocessing pipeline for MaskPLS ONNX model
Converts point clouds to voxel grids and normalized coordinates
"""

import numpy as np
import torch
from typing import Tuple, List, Dict


class PointCloudPreprocessor:
    """
    Preprocesses point clouds for the simplified MaskPLS ONNX model
    """
    def __init__(self, 
                 spatial_shape=(32, 32, 16),
                 coordinate_bounds=[[-48.0, 48.0], [-48.0, 48.0], [-4.0, 1.5]],
                 max_points=50000):
        """
        Args:
            spatial_shape: (D, H, W) voxel grid dimensions
            coordinate_bounds: [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
            max_points: Maximum number of points (for padding/truncation)
        """
        self.spatial_shape = spatial_shape
        self.coordinate_bounds = np.array(coordinate_bounds)
        self.max_points = max_points
        
        # Precompute normalization factors
        self.coord_min = self.coordinate_bounds[:, 0]
        self.coord_range = self.coordinate_bounds[:, 1] - self.coordinate_bounds[:, 0]
        
    def preprocess_single(self, points: np.ndarray, features: np.ndarray) -> Dict:
        """
        Preprocess a single point cloud
        
        Args:
            points: [N, 3] point coordinates (x, y, z)
            features: [N, C] point features (e.g., intensity, rgb)
        
        Returns:
            Dictionary with:
                - voxel_features: [C, D, H, W] voxelized features
                - point_coords: [N', 3] normalized coordinates
                - point_indices: [N'] indices of kept points
        """
        # Filter points within bounds
        valid_mask = np.all(
            (points >= self.coord_min) & (points < self.coord_min + self.coord_range),
            axis=1
        )
        valid_points = points[valid_mask]
        valid_features = features[valid_mask]
        valid_indices = np.where(valid_mask)[0]
        
        # Subsample if too many points
        if len(valid_points) > self.max_points:
            indices = np.random.choice(len(valid_points), self.max_points, replace=False)
            valid_points = valid_points[indices]
            valid_features = valid_features[indices]
            valid_indices = valid_indices[indices]
        
        # Normalize coordinates to [0, 1]
        norm_coords = (valid_points - self.coord_min) / self.coord_range
        norm_coords = np.clip(norm_coords, 0, 0.999)
        
        # Voxelize
        voxel_features = self.voxelize(valid_points, valid_features)
        
        # Pad coordinates if needed
        if len(norm_coords) < self.max_points:
            pad_size = self.max_points - len(norm_coords)
            norm_coords = np.pad(norm_coords, ((0, pad_size), (0, 0)), mode='constant')
            # Padding mask can be derived from comparing against zero
        
        return {
            'voxel_features': voxel_features,
            'point_coords': norm_coords,
            'point_indices': valid_indices,
            'num_points': len(valid_points)
        }
    
    def voxelize(self, points: np.ndarray, features: np.ndarray) -> np.ndarray:
        """
        Convert points to voxel grid
        
        Args:
            points: [N, 3] point coordinates
            features: [N, C] point features
        
        Returns:
            voxel_grid: [C, D, H, W] voxelized features
        """
        D, H, W = self.spatial_shape
        C = features.shape[1]
        
        # Initialize voxel grid
        voxel_grid = np.zeros((C, D, H, W), dtype=np.float32)
        voxel_count = np.zeros((D, H, W), dtype=np.int32)
        
        # Normalize coordinates to voxel indices
        norm_coords = (points - self.coord_min) / self.coord_range
        voxel_coords = (norm_coords * np.array([D, H, W])).astype(np.int32)
        
        # Clip to valid range
        voxel_coords = np.clip(voxel_coords, 0, np.array([D-1, H-1, W-1]))
        
        # Accumulate features
        for i in range(len(points)):
            d, h, w = voxel_coords[i]
            voxel_grid[:, d, h, w] += features[i]
            voxel_count[d, h, w] += 1
        
        # Average pooling
        mask = voxel_count > 0
        voxel_grid[:, mask] /= voxel_count[mask]
        
        return voxel_grid
    
    def preprocess_batch(self, batch: List[Tuple[np.ndarray, np.ndarray]]) -> Dict:
        """
        Preprocess a batch of point clouds
        
        Args:
            batch: List of (points, features) tuples
        
        Returns:
            Dictionary with batched arrays
        """
        results = [self.preprocess_single(points, features) for points, features in batch]
        
        # Stack results
        voxel_features = np.stack([r['voxel_features'] for r in results])
        point_coords = np.stack([r['point_coords'] for r in results])
        
        return {
            'voxel_features': voxel_features,
            'point_coords': point_coords,
            'num_points': [r['num_points'] for r in results]
        }


class SemanticKITTIPreprocessor(PointCloudPreprocessor):
    """
    Preprocessor specifically for SemanticKITTI dataset
    """
    def __init__(self, dataset='KITTI'):
        if dataset == 'KITTI':
            super().__init__(
                spatial_shape=(32, 32, 16),
                coordinate_bounds=[[-48.0, 48.0], [-48.0, 48.0], [-4.0, 1.5]],
                max_points=50000
            )
        else:  # NuScenes
            super().__init__(
                spatial_shape=(32, 32, 16),
                coordinate_bounds=[[-50.0, 50.0], [-50.0, 50.0], [-5.0, 3.0]],
                max_points=40000
            )
    
    def load_pointcloud(self, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load point cloud from .bin file
        
        Args:
            filepath: Path to .bin file
        
        Returns:
            points: [N, 3] coordinates
            features: [N, 4] XYZI features
        """
        # Load point cloud
        pointcloud = np.fromfile(filepath, dtype=np.float32).reshape(-1, 4)
        points = pointcloud[:, :3]
        intensity = pointcloud[:, 3:4]
        
        # Create features (xyz + intensity)
        features = np.concatenate([points, intensity], axis=1)
        
        return points, features


def postprocess_predictions(outputs: Dict, preprocessor: PointCloudPreprocessor) -> Dict:
    """
    Postprocess ONNX model outputs
    
    Args:
        outputs: Model outputs (pred_logits, pred_masks, sem_logits)
        preprocessor: The preprocessor used (for denormalization)
    
    Returns:
        Dictionary with semantic and instance predictions
    """
    pred_logits = outputs['pred_logits']  # [B, Q, C+1]
    pred_masks = outputs['pred_masks']    # [B, N, Q]
    sem_logits = outputs['sem_logits']    # [B, N, C]
    
    batch_size = pred_logits.shape[0]
    results = []
    
    for b in range(batch_size):
        # Semantic segmentation
        sem_pred = np.argmax(sem_logits[b], axis=-1)  # [N]
        
        # Instance segmentation (simplified)
        # Get class predictions for each query
        query_classes = np.argmax(pred_logits[b], axis=-1)  # [Q]
        query_scores = np.max(pred_logits[b], axis=-1)     # [Q]
        
        # Filter out no-object predictions
        valid_queries = query_classes < preprocessor.num_classes
        
        if np.any(valid_queries):
            # Get mask predictions
            mask_probs = 1 / (1 + np.exp(-pred_masks[b]))  # Sigmoid
            
            # Assign each point to highest scoring mask
            instance_scores = mask_probs[:, valid_queries] * query_scores[valid_queries]
            ins_pred = np.argmax(instance_scores, axis=1) + 1
            
            # Points with low scores get instance 0
            low_score_mask = np.max(instance_scores, axis=1) < 0.5
            ins_pred[low_score_mask] = 0
        else:
            ins_pred = np.zeros_like(sem_pred)
        
        results.append({
            'semantic': sem_pred,
            'instance': ins_pred
        })
    
    return results


# Example usage
if __name__ == "__main__":
    # Create preprocessor
    preprocessor = SemanticKITTIPreprocessor(dataset='KITTI')
    
    # Example: preprocess a single point cloud
    # Generate dummy data
    num_points = 100000
    points = np.random.randn(num_points, 3) * 20  # Random points
    intensity = np.random.rand(num_points, 1)
    features = np.concatenate([points, intensity], axis=1)
    
    # Preprocess
    result = preprocessor.preprocess_single(points, features)
    
    print("Preprocessing results:")
    print(f"  Voxel features shape: {result['voxel_features'].shape}")
    print(f"  Point coords shape: {result['point_coords'].shape}")
    print(f"  Valid points: {result['num_points']}")
    
    # Example: batch preprocessing
    batch = [(points, features) for _ in range(4)]
    batch_result = preprocessor.preprocess_batch(batch)
    
    print("\nBatch preprocessing:")
    print(f"  Voxel features shape: {batch_result['voxel_features'].shape}")
    print(f"  Point coords shape: {batch_result['point_coords'].shape}")
    
    # Convert to torch tensors for model input
    voxel_tensor = torch.from_numpy(batch_result['voxel_features'])
    coord_tensor = torch.from_numpy(batch_result['point_coords'])
    
    print("\nReady for ONNX model input!")
    print(f"  voxel_features: {voxel_tensor.shape} ({voxel_tensor.dtype})")
    print(f"  point_coords: {coord_tensor.shape} ({coord_tensor.dtype})")
