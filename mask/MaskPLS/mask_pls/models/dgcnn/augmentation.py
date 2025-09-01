# mask_pls/models/dgcnn/augmentation.py
"""
Data augmentation for point clouds
"""

import numpy as np
import torch


class PointCloudAugmentation:
    """
    Augmentation pipeline for point clouds
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.augmentations = []
        
        if cfg.get('rotation', True):
            self.augmentations.append(RandomRotation())
        if cfg.get('scaling', True):
            self.augmentations.append(RandomScaling())
        if cfg.get('flipping', True):
            self.augmentations.append(RandomFlip())
        if cfg.get('jittering', True):
            self.augmentations.append(RandomJitter())
        if cfg.get('dropout', True):
            self.augmentations.append(RandomDropout())
    
    def __call__(self, points, features=None):
        for aug in self.augmentations:
            if np.random.random() < 0.5:  # 50% chance for each augmentation
                points, features = aug(points, features)
        return points, features


class RandomRotation:
    """Random rotation around z-axis"""
    def __init__(self, angle_range=(-np.pi, np.pi)):
        self.angle_range = angle_range
    
    def __call__(self, points, features=None):
        angle = np.random.uniform(*self.angle_range)
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        
        rotation_matrix = np.array([
            [cos_angle, -sin_angle, 0],
            [sin_angle, cos_angle, 0],
            [0, 0, 1]
        ])
        
        points = points @ rotation_matrix.T
        return points, features


class RandomScaling:
    """Random scaling"""
    def __init__(self, scale_range=(0.95, 1.05)):
        self.scale_range = scale_range
    
    def __call__(self, points, features=None):
        scale = np.random.uniform(*self.scale_range)
        points = points * scale
        return points, features


class RandomFlip:
    """Random flipping along x or y axis"""
    def __init__(self, axis=['x', 'y']):
        self.axis = axis
    
    def __call__(self, points, features=None):
        for ax in self.axis:
            if np.random.random() < 0.5:
                if ax == 'x':
                    points[:, 0] = -points[:, 0]
                elif ax == 'y':
                    points[:, 1] = -points[:, 1]
        return points, features


class RandomJitter:
    """Add random noise to points"""
    def __init__(self, sigma=0.01, clip=0.05):
        self.sigma = sigma
        self.clip = clip
    
    def __call__(self, points, features=None):
        noise = np.clip(np.random.normal(0, self.sigma, points.shape), 
                       -self.clip, self.clip)
        points = points + noise
        return points, features


class RandomDropout:
    """Randomly drop points"""
    def __init__(self, dropout_ratio=0.1):
        self.dropout_ratio = dropout_ratio
    
    def __call__(self, points, features=None):
        n_points = points.shape[0]
        n_keep = int(n_points * (1 - self.dropout_ratio))
        
        indices = np.random.choice(n_points, n_keep, replace=False)
        points = points[indices]
        
        if features is not None:
            features = features[indices]
        
        return points, features