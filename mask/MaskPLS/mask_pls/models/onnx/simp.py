def forward(self, voxel_features, point_coords):
    B, C, D, H, W = voxel_features.shape
    B_p, N, _ = point_coords.shape
    
    # Project features
    voxel_features = self.feature_proj(voxel_features)
    C_out = voxel_features.shape[1]
    
    # Flatten voxel features
    voxel_flat = voxel_features.view(B, C_out, -1)
    
    # Convert coordinates to indices
    voxel_coords = point_coords * torch.tensor([D, H, W], device=point_coords.device, dtype=point_coords.dtype)
    
    # Clamp each dimension separately for ONNX compatibility
    voxel_coords[..., 0] = torch.clamp(voxel_coords[..., 0], 0, D-1)
    voxel_coords[..., 1] = torch.clamp(voxel_coords[..., 1], 0, H-1)
    voxel_coords[..., 2] = torch.clamp(voxel_coords[..., 2], 0, W-1)
    
    voxel_coords = voxel_coords.long()
    
    # Flat indices
    flat_indices = (voxel_coords[..., 0] * H * W + 
                   voxel_coords[..., 1] * W + 
                   voxel_coords[..., 2])
    flat_indices = flat_indices.unsqueeze(1).expand(-1, C_out, -1)
    
    # Gather features
    point_features = torch.gather(voxel_flat, 2, flat_indices)
    point_features = point_features.transpose(1, 2)
    
    # MLP
    point_features = self.mlp(point_features)
    return point_features
